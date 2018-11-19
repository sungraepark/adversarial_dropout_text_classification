# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Virtual adversarial text models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

# Dependency imports

import tensorflow as tf

import adversarial_losses as adv_lib
import inputs as inputs_lib
import layers as layers_lib
from utilities import batch_noise

flags = tf.app.flags
FLAGS = flags.FLAGS

# Flags governing adversarial training are defined in adversarial_losses.py.

# Classifier
flags.DEFINE_integer('num_classes', 2, 'Number of classes for classification')

# Data path
flags.DEFINE_string('data_dir', '/tmp/IMDB',
                    'Directory path to preprocessed text dataset.')
flags.DEFINE_string('vocab_freq_path', None,
                    'Path to pre-calculated vocab frequency data. If '
                    'None, use FLAGS.data_dir/vocab_freq.txt.')
flags.DEFINE_integer('batch_size', 64, 'Size of the batch.')
flags.DEFINE_integer('num_timesteps', 100, 'Number of timesteps for BPTT')

# Model architechture
flags.DEFINE_bool('bidir_lstm', False, 'Whether to build a bidirectional LSTM.')
flags.DEFINE_bool('single_label', True, 'Whether the sequence has a single '
                  'label, for optimization.')
flags.DEFINE_integer('rnn_num_layers', 1, 'Number of LSTM layers.')
flags.DEFINE_integer('rnn_cell_size', 512,
                     'Number of hidden units in the LSTM.')
flags.DEFINE_integer('cl_num_layers', 1,
                     'Number of hidden layers of classification model.')
flags.DEFINE_integer('cl_hidden_size', 30,
                     'Number of hidden units in classification layer.')
flags.DEFINE_integer('num_candidate_samples', -1,
                     'Num samples used in the sampled output layer.')
flags.DEFINE_bool('use_seq2seq_autoencoder', False,
                  'If True, seq2seq auto-encoder is used to pretrain. '
                  'If False, standard language model is used.')

# Vocabulary and embeddings
flags.DEFINE_integer('embedding_dims', 256, 'Dimensions of embedded vector.')
flags.DEFINE_integer('vocab_size', 86934,
                     'The size of the vocaburary. This value '
                     'should be exactly same as the number of the '
                     'vocabulary used in dataset. Because the last '
                     'indexed vocabulary of the dataset preprocessed by '
                     'my preprocessed code, is always <eos> and here we '
                     'specify the <eos> with the the index.')
flags.DEFINE_bool('normalize_embeddings', True,
                  'Normalize word embeddings by vocab frequency')
flags.DEFINE_bool('stop_gradient_adt', True,
                  'Normalize word embeddings by vocab frequency')

# Optimization
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate while fine-tuning.')
flags.DEFINE_float('learning_rate_decay_factor', 1.0,
                   'Learning rate decay factor')
flags.DEFINE_boolean('sync_replicas', False, 'sync_replica or not')
flags.DEFINE_integer('replicas_to_aggregate', 1,
                     'The number of replicas to aggregate')

# Regularization
flags.DEFINE_float('max_grad_norm', 1.0,
                   'Clip the global gradient norm to this value.')
flags.DEFINE_float('keep_prob_emb', 1.0, 'keep probability on embedding layer. '
                   '0.5 is optimal on IMDB with virtual adversarial training.')
flags.DEFINE_float('keep_prob_emb2', 1.0, 'keep probability on embedding layer. '
                   '0.5 is optimal on IMDB with virtual adversarial training.')
flags.DEFINE_float('keep_prob_lstm_out', 1.0,
                   'keep probability on lstm output.')
flags.DEFINE_float('keep_prob_lstm_hh', 1.0,
                   'keep probability on lstm output.')
flags.DEFINE_float('keep_prob_cl_hidden', 1.0,
                   'keep probability on classification hidden layer')

def get_model():
  if FLAGS.bidir_lstm:
    return VatxtBidirModel()
  else:
    return VatxtModel()


class VatxtModel(object):
  """Constructs training and evaluation graphs.

  Main methods: `classifier_training()`, `language_model_training()`,
  and `eval_graph()`.

  Variable reuse is a critical part of the model, both for sharing variables
  between the language model and the classifier, and for reusing variables for
  the adversarial loss calculation. To ensure correct variable reuse, all
  variables are created in Keras-style layers, wherein stateful layers (i.e.
  layers with variables) are represented as callable instances of the Layer
  class. Each time the Layer instance is called, it is using the same variables.

  All Layers are constructed in the __init__ method and reused in the various
  graph-building functions.
  """

  def __init__(self, cl_logits_input_dim=None):
      
    print(FLAGS.normalize_embeddings, FLAGS.stop_gradient_adt)
    self.global_step = tf.train.get_or_create_global_step()
    self.vocab_freqs = _get_vocab_freqs()

    # Cache VatxtInput objects
    self.cl_inputs = None
    self.lm_inputs = None

    # Cache intermediate Tensors that are reused
    self.tensors = {}

    # Construct layers which are reused in constructing the LM and
    # Classification graphs. Instantiating them all once here ensures that
    # variable reuse works correctly.
    self.layers = {}
    self.layers['embedding'] = layers_lib.Embedding(
        FLAGS.vocab_size, FLAGS.embedding_dims, FLAGS.normalize_embeddings,
        self.vocab_freqs, FLAGS.keep_prob_emb)
    self.layers['lstm'] = layers_lib.dropout_LSTM(
        FLAGS.rnn_cell_size, FLAGS.rnn_num_layers, FLAGS.keep_prob_lstm_out)
    self.layers['lm_loss'] = layers_lib.SoftmaxLoss(
        FLAGS.vocab_size,
        FLAGS.num_candidate_samples,
        self.vocab_freqs,
        name='LM_loss')

    cl_logits_input_dim = cl_logits_input_dim or FLAGS.rnn_cell_size
    self.layers['cl_logits'] = layers_lib.cl_logits_subgraph(
        [FLAGS.cl_hidden_size] * FLAGS.cl_num_layers, cl_logits_input_dim,
        FLAGS.num_classes, FLAGS.keep_prob_cl_hidden)

  @property
  def pretrained_variables(self):
    return (self.layers['embedding'].trainable_weights +
            self.layers['lstm'].trainable_weights)

  def classifier_training(self):
    loss = self.classifier_graph()
    train_op = optimize(loss, self.global_step)
    return train_op, loss, self.global_step

  def language_model_training(self):
    loss = self.language_model_graph()
    train_op = optimize(loss, self.global_step)
    return train_op, loss, self.global_step

  def classifier_graph(self):
    """Constructs classifier graph from inputs to classifier loss.

    * Caches the VatxtInput object in `self.cl_inputs`
    * Caches tensors: `cl_embedded`, `cl_logits`, `cl_loss`

    Returns:
      loss: scalar float.
    """
    inputs = _inputs('train', pretrain=False)
    self.cl_inputs = inputs
    embedded = self.layers['embedding'](inputs.tokens)
    
    if FLAGS.keep_prob_emb2 < 1.:
      shape = embedded.get_shape().as_list()
      embedded = tf.nn.dropout(embedded, FLAGS.keep_prob_emb2, noise_shape=(shape[0], 1, shape[2]))
    
    self.tensors['cl_embedded'] = embedded
    
    one_mask = None
    #if FLAGS.keep_prob_lstm_hh != 1.0:
        #one_mask = batch_noise([FLAGS.batch_size, FLAGS.rnn_cell_size], inner_seed=123, keep_prob=FLAGS.keep_prob_lstm_hh)
        
    _, next_state, logits, loss = self.cl_loss_from_embedding(
        embedded, one_mask, return_intermediates=True)
    tf.summary.scalar('classification_loss', loss)
    self.tensors['cl_logits'] = logits
    self.tensors['cl_loss'] = loss

    if FLAGS.single_label:
      indices = tf.stack([tf.range(FLAGS.batch_size), inputs.length - 1], 1)
      labels = tf.expand_dims(tf.gather_nd(inputs.labels, indices), 1)
      weights = tf.expand_dims(tf.gather_nd(inputs.weights, indices), 1)
    else:
      labels = inputs.labels
      weights = inputs.weights
    acc = layers_lib.accuracy(logits, labels, weights)
    tf.summary.scalar('accuracy', acc)

    adv_loss = (self.adversarial_loss() * tf.constant(
        FLAGS.adv_reg_coeff, name='adv_reg_coeff'))
    tf.summary.scalar('adversarial_loss', adv_loss)

    total_loss = loss + adv_loss

    with tf.control_dependencies([inputs.save_state(next_state)]):
      total_loss = tf.identity(total_loss)
      tf.summary.scalar('total_classification_loss', total_loss)
    return total_loss

  def language_model_graph(self, compute_loss=True):
    """Constructs LM graph from inputs to LM loss.

    * Caches the VatxtInput object in `self.lm_inputs`
    * Caches tensors: `lm_embedded`

    Args:
      compute_loss: bool, whether to compute and return the loss or stop after
        the LSTM computation.

    Returns:
      loss: scalar float.
    """
    inputs = _inputs('train', pretrain=True)
    self.lm_inputs = inputs
    return self._lm_loss(inputs, compute_loss=compute_loss)

  def _lm_loss(self,
               inputs,
               emb_key='lm_embedded',
               lstm_layer='lstm',
               lm_loss_layer='lm_loss',
               loss_name='lm_loss',
               compute_loss=True):
    embedded = self.layers['embedding'](inputs.tokens)
    self.tensors[emb_key] = embedded
    lstm_out, next_state = self.layers[lstm_layer](embedded, inputs.state,
                                                   inputs.length, None)
    if compute_loss:
      loss = self.layers[lm_loss_layer](
          [lstm_out, inputs.labels, inputs.weights])
      with tf.control_dependencies([inputs.save_state(next_state)]):
        loss = tf.identity(loss)
        tf.summary.scalar(loss_name, loss)

      return loss

  def eval_graph(self, dataset='test'):
    """Constructs classifier evaluation graph.

    Args:
      dataset: the labeled dataset to evaluate, {'train', 'test', 'valid'}.

    Returns:
      eval_ops: dict<metric name, tuple(value, update_op)>
      var_restore_dict: dict mapping variable restoration names to variables.
        Trainable variables will be mapped to their moving average names.
    """
    inputs = _inputs(dataset, pretrain=False)
    embedded = self.layers['embedding'](inputs.tokens)
    _, next_state, logits, _ = self.cl_loss_from_embedding(
        embedded, None, inputs=inputs, return_intermediates=True)

    if FLAGS.single_label:
      indices = tf.stack([tf.range(FLAGS.batch_size), inputs.length - 1], 1)
      labels = tf.expand_dims(tf.gather_nd(inputs.labels, indices), 1)
      weights = tf.expand_dims(tf.gather_nd(inputs.weights, indices), 1)
    else:
      labels = inputs.labels
      weights = inputs.weights
    eval_ops = {
        'accuracy':
            tf.contrib.metrics.streaming_accuracy(
                layers_lib.predictions(logits), labels, weights)
    }

    with tf.control_dependencies([inputs.save_state(next_state)]):
      acc, acc_update = eval_ops['accuracy']
      acc_update = tf.identity(acc_update)
      eval_ops['accuracy'] = (acc, acc_update)

    var_restore_dict = make_restore_average_vars_dict()
    return eval_ops, var_restore_dict

  def cl_loss_from_embedding(self,
                             embedded,
                             state_mask=None,
                             inputs=None,
                             return_intermediates=False):
    """Compute classification loss from embedding.

    Args:
      embedded: 3-D float Tensor [batch_size, num_timesteps, embedding_dim]
      inputs: VatxtInput, defaults to self.cl_inputs.
      return_intermediates: bool, whether to return intermediate tensors or only
        the final loss.

    Returns:
      If return_intermediates is True:
        lstm_out, next_state, logits, loss
      Else:
        loss
    """
    if inputs is None:
      inputs = self.cl_inputs
    
    if FLAGS.keep_prob_emb2 < 1. and state_mask is not None :
      shape = embedded.get_shape().as_list()

      # Use same dropout masks at each timestep with specifying noise_shape.
      # This slightly improves performance.
      # Please see https://arxiv.org/abs/1512.05287 for the theoretical
      # explanation.
      #embedded = tf.nn.dropout(embedded, FLAGS.keep_prob_emb2, noise_shape=(shape[0], 1, shape[2]))
      
    
    lstm_out, next_state = self.layers['lstm'](embedded, inputs.state,
                                               inputs.length, state_mask)
    if FLAGS.single_label:
      indices = tf.stack([tf.range(FLAGS.batch_size), inputs.length - 1], 1)
      lstm_out = tf.expand_dims(tf.gather_nd(lstm_out, indices), 1)
      labels = tf.expand_dims(tf.gather_nd(inputs.labels, indices), 1)
      weights = tf.expand_dims(tf.gather_nd(inputs.weights, indices), 1)
    else:
      labels = inputs.labels
      weights = inputs.weights
    logits = self.layers['cl_logits'](lstm_out)
    loss = layers_lib.classification_loss(logits, labels, weights)

    if return_intermediates:
      return lstm_out, next_state, logits, loss
    else:
      return loss

  def adversarial_loss(self):
    """Compute adversarial loss based on FLAGS.adv_training_method."""

    def random_perturbation_loss():
      return adv_lib.random_perturbation_loss(self.tensors['cl_embedded'],
                                              self.cl_inputs.length,
                                              self.cl_loss_from_embedding)

    def adversarial_loss():
      return adv_lib.adversarial_loss(self.tensors['cl_embedded'],
                                      self.tensors['cl_loss'],
                                      self.cl_loss_from_embedding)

    def virtual_adversarial_loss():
      """Computes virtual adversarial loss.

      Uses lm_inputs and constructs the language model graph if it hasn't yet
      been constructed.

      Also ensures that the LM input states are saved for LSTM state-saving
      BPTT.

      Returns:
        loss: float scalar.
      """
      if self.lm_inputs is None:
        self.language_model_graph(compute_loss=False)

      def logits_from_embedding(embedded, mask=None, return_next_state=False):
        _, next_state, logits, _ = self.cl_loss_from_embedding(
            embedded, mask, inputs=self.lm_inputs, return_intermediates=True)
        if return_next_state:
          return next_state, logits
        else:
          return logits

      next_state, lm_cl_logits = logits_from_embedding(
          self.tensors['lm_embedded'], None, return_next_state=True)

      va_loss = adv_lib.virtual_adversarial_loss(
          lm_cl_logits, self.tensors['lm_embedded'], None, self.lm_inputs,
          logits_from_embedding)

      with tf.control_dependencies([self.lm_inputs.save_state(next_state)]):
        va_loss = tf.identity(va_loss)

      return va_loss
    
    def adversarial_per_and_dropout_loss():
      """Computes adversarial dropout loss.

      Returns:
        loss: float scalar.
      """
      if self.lm_inputs is None:
        self.language_model_graph(compute_loss=False)

      def logits_from_embedding(embedded, mask, return_next_state=False):
        lstm_out, next_state, logits, _ = self.cl_loss_from_embedding(
            embedded, mask, inputs=self.lm_inputs, return_intermediates=True)
        if return_next_state:
          return next_state, logits
        else:
          return logits
      
      embedded = self.tensors['lm_embedded']
      embedded2 = self.tensors['lm_embedded']
      if FLAGS.keep_prob_emb2 < 1.:
          shape = embedded.get_shape().as_list()
          embedded = tf.nn.dropout(embedded, FLAGS.keep_prob_emb2, noise_shape=(shape[0], 1, shape[2]))
          embedded2 = tf.nn.dropout(embedded2, FLAGS.keep_prob_emb2, noise_shape=(shape[0], 1, shape[2]))
          
      next_state, lm_cl_logits = logits_from_embedding(
          embedded, None, return_next_state=True)
      
      vat_loss = adv_lib.virtual_adversarial_loss(
          lm_cl_logits, embedded, None, self.lm_inputs,
          logits_from_embedding)
      if FLAGS.stop_gradient_adt:
          lm_cl_logits = tf.stop_gradient(lm_cl_logits)
      one_mask = tf.ones([FLAGS.batch_size, FLAGS.rnn_cell_size], dtype=tf.float32)
      adt_loss = adv_lib.iterative_adversarial_dropout_loss(
          lm_cl_logits, embedded2, one_mask, self.lm_inputs,
          logits_from_embedding)

      with tf.control_dependencies([self.lm_inputs.save_state(next_state)]):
        vat_loss = tf.identity(vat_loss)
        adt_loss = tf.identity(adt_loss)

      return vat_loss + adt_loss
    
    def adversarial_dropout_loss():
      """Computes adversarial dropout loss.

      Returns:
        loss: float scalar.
      """
      if self.lm_inputs is None:
        self.language_model_graph(compute_loss=False)

      def logits_from_embedding(embedded, mask, return_next_state=False):
        lstm_out, next_state, logits, _ = self.cl_loss_from_embedding(
            embedded, mask, inputs=self.lm_inputs, return_intermediates=True)
        if return_next_state:
          return next_state, logits
        else:
          return logits
      
      one_mask = tf.ones([FLAGS.batch_size, FLAGS.rnn_cell_size], dtype=tf.float32)
      #one_mask = batch_noise([FLAGS.batch_size, FLAGS.rnn_cell_size], inner_seed=1234, keep_prob=FLAGS.keep_prob_lstm_hh)
      embedded1 = self.tensors['lm_embedded']
      embedded2 = self.tensors['lm_embedded']
      if FLAGS.keep_prob_emb2 < 1.:
          shape = embedded1.get_shape().as_list()
          embedded1 = tf.nn.dropout(self.tensors['lm_embedded'], FLAGS.keep_prob_emb2, noise_shape=(shape[0], 1, shape[2]))
          embedded2 = tf.nn.dropout(self.tensors['lm_embedded'], FLAGS.keep_prob_emb2, noise_shape=(shape[0], 1, shape[2]))
          
      next_state, lm_cl_logits = logits_from_embedding(
          embedded1, None, return_next_state=True)
      if FLAGS.stop_gradient_adt:
          lm_cl_logits = tf.stop_gradient(lm_cl_logits)
      va_loss = adv_lib.adversarial_dropout_loss(
          lm_cl_logits, embedded2, one_mask, self.lm_inputs,
          logits_from_embedding)
      #va_loss = adv_lib.iterative_adversarial_dropout_loss(
      #    lm_cl_logits, embedded2, one_mask, self.lm_inputs,
      #    logits_from_embedding)

      with tf.control_dependencies([self.lm_inputs.save_state(next_state)]):
        va_loss = tf.identity(va_loss)
        
      return va_loss
    
    def fraternal_dropout_loss():
      
      if self.lm_inputs is None:
        self.language_model_graph(compute_loss=False)

      def logits_from_embedding(embedded, mask, return_next_state=False):
        _, next_state, logits, _ = self.cl_loss_from_embedding(
            embedded, mask, inputs=self.lm_inputs, return_intermediates=True)
        if return_next_state:
          return next_state, logits
        else:
          return logits

      one_mask = batch_noise([FLAGS.batch_size, FLAGS.rnn_cell_size], inner_seed=1234, keep_prob=FLAGS.keep_prob_lstm_hh)
      next_state, lm_cl_logits = logits_from_embedding(
          self.tensors['lm_embedded'], one_mask, return_next_state=True)
      
      va_loss = adv_lib.fraternal_dropout_loss(
          lm_cl_logits, self.tensors['lm_embedded'], one_mask, self.lm_inputs,
          logits_from_embedding)
      with tf.control_dependencies([self.lm_inputs.save_state(next_state)]):
          va_loss = tf.identity(va_loss)
      return va_loss
    
    def pesudo_ensemble_loss():
      
      if self.lm_inputs is None:
        self.language_model_graph(compute_loss=False)

      def logits_from_embedding(embedded, mask, return_next_state=False):
        _, next_state, logits, _ = self.cl_loss_from_embedding(
            embedded, mask, inputs=self.lm_inputs, return_intermediates=True)
        if return_next_state:
          return next_state, logits
        else:
          return logits
      
      one_mask = tf.ones([FLAGS.batch_size, FLAGS.rnn_cell_size], dtype=tf.float32)
      next_state, lm_cl_logits = logits_from_embedding(
          self.tensors['lm_embedded'], one_mask, return_next_state=True)
      
      va_loss = adv_lib.fraternal_dropout_loss(
          lm_cl_logits, self.tensors['lm_embedded'], one_mask, self.lm_inputs,
          logits_from_embedding)
      with tf.control_dependencies([self.lm_inputs.save_state(next_state)]):
          va_loss = tf.identity(va_loss)
      return va_loss
  
    
    adv_training_methods = {
        # Random perturbation
        'rp': random_perturbation_loss,
        # Adversarial training
        'at': adversarial_loss,
        # Virtual adversarial training
        'vat': virtual_adversarial_loss,
        # fraternal dropout training
        'fd': fraternal_dropout_loss,
        # pseudo-Ensemble Agreement
        'pea': pesudo_ensemble_loss,
        # adversarial dropout training
        'adt': adversarial_dropout_loss,
        # Both adt and vat
        'adtvat': adversarial_per_and_dropout_loss,
        '': lambda: tf.constant(0.),
        None: lambda: tf.constant(0.),
    }

    with tf.name_scope('adversarial_loss'):
      return adv_training_methods[FLAGS.adv_training_method]()


def _inputs(dataset='train', pretrain=False, bidir=False):
  return inputs_lib.inputs(
      data_dir=FLAGS.data_dir,
      phase=dataset,
      bidir=bidir,
      pretrain=pretrain,
      use_seq2seq=pretrain and FLAGS.use_seq2seq_autoencoder,
      state_size=FLAGS.rnn_cell_size,
      num_layers=FLAGS.rnn_num_layers,
      batch_size=FLAGS.batch_size,
      unroll_steps=FLAGS.num_timesteps,
      eos_id=FLAGS.vocab_size - 1)


def _get_vocab_freqs():
  """Returns vocab frequencies.

  Returns:
    List of integers, length=FLAGS.vocab_size.

  Raises:
    ValueError: if the length of the frequency file is not equal to the vocab
      size, or if the file is not found.
  """
  path = FLAGS.vocab_freq_path or os.path.join(FLAGS.data_dir, 'vocab_freq.txt')

  if tf.gfile.Exists(path):
    with tf.gfile.Open(path) as f:
      # Get pre-calculated frequencies of words.
      reader = csv.reader(f, quoting=csv.QUOTE_NONE)
      freqs = [int(row[-1]) for row in reader]
      if len(freqs) != FLAGS.vocab_size:
        raise ValueError('Frequency file length %d != vocab size %d' %
                         (len(freqs), FLAGS.vocab_size))
  else:
    if FLAGS.vocab_freq_path:
      raise ValueError('vocab_freq_path not found')
    freqs = [1] * FLAGS.vocab_size

  return freqs


def make_restore_average_vars_dict():
  """Returns dict mapping moving average names to variables."""
  var_restore_dict = {}
  variable_averages = tf.train.ExponentialMovingAverage(0.999)
  for v in tf.global_variables():
    if v in tf.trainable_variables():
      name = variable_averages.average_name(v)
    else:
      name = v.op.name
    var_restore_dict[name] = v
  return var_restore_dict


def optimize(loss, global_step):
  return layers_lib.optimize(
      loss, global_step, FLAGS.max_grad_norm, FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor, FLAGS.sync_replicas,
      FLAGS.replicas_to_aggregate, FLAGS.task)
