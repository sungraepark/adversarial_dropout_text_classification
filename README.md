# Adversarial Dropout for Recurrent Neural Networks (Semi-supervised Text Classification)

Tensorflow implementation in the paper "Adversarial Dropout for Reccurent Neural Networks"
This implementation is based on the tensorflow official code at https://github.com/tensorflow/models/tree/master/research/adversarial_text

## Requirements


## IMDB dataset 

Download IMDB dataset [1] at the following url.

http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

## Implementation command

### Preprocessing & Pretraining

These implementation steps are the same with the tensorflow official code. We re-note the implementation commands, but you can follow the original descriptions ( https://github.com/tensorflow/models/tree/master/research/adversarial_text ).
  
#### Preprocess

Generate vocabulary

```bash
python gen_vocab.py --output_dir={$preprocessed_dataset_path} --dataset=imdb
    --imdb_input_dir={$dataset_path} --lowercase=False
```

Generate tfrecords with the dataset

```bash
$ python gen_data.py --output_dir={$preprocessed_dataset_path} --dataset=imdb \
    --imdb_input_dir={$dataset_path} --lowercase=False --label_gain=False
```

#### Pretraining

Pretraining is important for the semi-supervised text classification tasks.

Learn a language model with the dataset

```bash
$ python pretrain.py --train_dir={$pretrained_LM_dir} --data_dir={$preprocessed_dataset_path} \
    --vocab_size=86934 --embedding_dims=256 --rnn_cell_size=1024 --num_candidate_samples=1024 \
    --batch_size=256 --learning_rate=0.001 --learning_rate_decay_factor=0.9999 --max_steps=100000 \
    --max_grad_norm=1.0 --num_timesteps=400 --keep_prob_emb=0.5 --normalize_embeddings
```

### Train classifier with the adversarial dropout

We added the additional parameters for the adversarial dropout. 

```bash
$ python train_classifier.py \
    --train_dir={$output_model_dir} \
    --pretrained_model_dir={$pretrained_LM_dir} \
    --data_dir={$preprocessed_dataset_path} \
    --vocab_size=86934 \
    --embedding_dims=256 \
    --rnn_cell_size=1024 \
    --cl_num_layers=1 \
    --cl_hidden_size=30 \
    --batch_size=64 \
    --learning_rate=0.0005 \
    --learning_rate_decay_factor=0.9998 \
    --max_steps=15000 \
    --max_grad_norm=1.0 \
    --num_timesteps=400 \
    --keep_prob_emb=0.5 \
    --normalize_embeddings \
    --adv_training_method=adt \
    --adv_drop_change_rate=0.04 \
    --adv_drop_change_iteration=2
```

## References

[1] Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

