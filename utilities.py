import math
import numpy as np
import tensorflow as tf
import collections
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.util import nest
from tensorflow.python.framework import ops

def rampup(epoch, rampup_length):
    if epoch < rampup_length:
        p = max(0.0, float(epoch)) / float(rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def rampdown(epoch, rampdown_length, total_epoch):
    if epoch >= (total_epoch - rampdown_length):
        ep = (epoch - (total_epoch - rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / rampdown_length)
    else:
        return 1.0 

def batch_noise(shape, inner_seed, keep_prob):
    noise = tf.random_uniform(shape, dtype=tf.float32)
    random_tensor = keep_prob + noise
    binary_tensor = tf.floor(random_tensor)
    binary_tensor = binary_tensor/keep_prob
    return binary_tensor

def one_drop_noise(shape, inner_seed):
    noise = tf.random_uniform(shape, dtype=tf.float32)
    max_val = tf.reduce_max(noise, axis=1, keep_dims=True)
    drop_points = tf.cast( tf.greater_equal(noise, max_val), tf.float32)
    binary_tensor = tf.ones_like(noise) - drop_points
    return binary_tensor

def adversarial_dropout(cur_mask, Jacobian, change_limit, name="ad"):
    
    dim = tf.reduce_prod(tf.shape(cur_mask)[1:])
    changed_mask = cur_mask/tf.reduce_max(cur_mask)
    
    if change_limit != 0 :
        
        dir = tf.reshape(Jacobian, [-1, dim])
        
        # mask (cur_mask=1->m=1), (cur_mask=0->m=-1)
        m = changed_mask
        m = 2.0*m - tf.ones_like(m)
        
        # sign of Jacobian  (J>0 -> s=1), (J<0 -> s= -1)
        s = tf.cast( tf.greater(dir, float(0.0)), tf.float32)
        #s = 2.0*s - tf.ones_like(s)                  
        s = s - tf.ones_like(s)     
        
        # remain (J>0, m=-1) and (J<0, m=1), which are candidates to be changed
        change_candidate = tf.cast( tf.less( s*m, float(0.0) ), tf.float32) # s = -1, m = 1
        ads_dL_dx = tf.abs(dir)
        
        # ordering abs_Jacobian for candidates
        # the maximum number of the changes is "change_limit"
        # draw top_k elements ( if the top k element is 0, the number of the changes is less than "change_limit" ) 
        left_values = change_candidate*ads_dL_dx
        with tf.device("/cpu:0"):
            min_left_values = tf.nn.top_k(left_values, change_limit)[0][:,-1]    
        change_target = tf.cast(  tf.greater(left_values, tf.expand_dims(min_left_values, -1) ), tf.float32)
        
        # changed mask with change_target
        changed_mask = (m - 2.0*m*change_target + tf.ones_like(m))*0.5 
    
    sum_binary_tensor = tf.reduce_sum(changed_mask, axis=1, keep_dims=True)
    sum_full_dims = tf.reduce_sum(tf.ones_like(changed_mask), axis=1, keep_dims=True)
    ratio_of_binary_tensor = sum_binary_tensor / sum_full_dims # tf.div(sum_binary_tensor, sum_full_dims) #
    
    changed_mask = changed_mask/ratio_of_binary_tensor
    
    return changed_mask

def adversarial_iter_dropout(cur_mask, Jacobian, change_limit, name="ad"):
    
    dim = tf.reduce_prod(tf.shape(cur_mask)[1:])
    changed_mask = cur_mask
    
    if change_limit != 0 :
        
        dir = tf.reshape(Jacobian, [-1, dim])
        
        # mask (cur_mask=1->m=1), (cur_mask=0->m=-1)
        m = cur_mask
        m = 2.0*m - tf.ones_like(m)
        
        # sign of Jacobian  (J>0 -> s=1), (J<0 -> s= -1)
        s = tf.cast( tf.greater(dir, float(0.0)), tf.float32)
        s = 2.0*s - tf.ones_like(s)                  
        
        # remain (J>0, m=-1) and (J<0, m=1), which are candidates to be changed
        change_candidate = tf.cast( tf.less( s*cur_mask, float(0.0) ), tf.float32) # s = -1, m = 1
        
        ads_dL_dx = tf.abs(dir)
        
        # ordering abs_Jacobian for candidates
        # the maximum number of the changes is "change_limit"
        # draw top_k elements ( if the top k element is 0, the number of the changes is less than "change_limit" ) 
        
        #left_values = change_candidate*ads_dL_dx
        # for test 
        left_values = cur_mask*ads_dL_dx
        
        
        with tf.device("/cpu:0"):
            min_left_values = tf.nn.top_k(left_values, change_limit )[0][:,-1]    
        
        print(tf.shape(min_left_values) )
        change_target = tf.cast(  tf.greater_equal(left_values, tf.expand_dims(min_left_values, -1) ), tf.float32)
        
        # changed mask with change_target
        changed_mask = (m - 2.0*m*change_target + tf.ones_like(m))*0.5 
    
    return changed_mask

def ce_loss(logit, y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))

def ce_loss_sum(logit, y):
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))

def qe_loss(logit1, logit2):
    logit1 = tf.nn.softmax(logit1)
    logit2 = tf.nn.softmax(logit2)
    return tf.reduce_mean( tf.squared_difference( logit1, logit2 ) )

def qe_loss_sum(logit1, logit2):
    logit1 = tf.nn.softmax(logit1)
    logit2 = tf.nn.softmax(logit2)
    return tf.reduce_sum(tf.reduce_mean( tf.squared_difference( logit1, logit2 ), axis=1 ))

def accuracy(logit, y):
    pred = tf.argmax(logit, 1)
    true = tf.argmax(y, 1)
    return tf.reduce_sum(tf.to_float(tf.equal(pred, true)))

  
def logsoftmax(x):
    xdev = x - tf.reduce_max(x, 1, keep_dims=True)
    lsm = xdev - tf.log(tf.reduce_sum(tf.exp(xdev), 1, keep_dims=True))
    return lsm
  

def kl_divergence_with_logit(q_logit, p_logit):
    q = tf.nn.softmax(q_logit)
    qlogq = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(q_logit), 1))
    qlogp = tf.reduce_mean(tf.reduce_sum(q * logsoftmax(p_logit), 1))
    return qlogq - qlogp


def entropy_y_x(logit):
    p = tf.nn.softmax(logit)
    return -tf.reduce_mean(tf.reduce_sum(p * logsoftmax(logit), 1))