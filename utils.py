import numpy as np;
import tensorflow as tf;
import numpy as np
from scipy import signal;
import librosa.filters;
import librosa;
import time;


def tf_cosine_similarity(a, b):
    # a is 1-D tensor of utterance embedding
    # b is a 1-D tensor of centroids
    normalize_a = tf.nn.l2_normalize(a, 0)
    normalize_b = tf.nn.l2_normalize(b, 0)
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
    return cos_similarity

def tf_scaled_cosine_similarity(a, b):
    # a is embedding vecter of an utterance, by default [256(proj_nodes)]
    # b is centroid, by default [64(num_spk_per_batch), 256(proj_nodes)]
    # returns similarity vector of utt for every centroid
    normalize_a = tf.reshape(tf.nn.l2_normalize(a, axis=-1), [1, -1])
    normalize_b = tf.transpose(tf.nn.l2_normalize(b, axis=-1))

    # cosine similarity vector [1,64]
    cos_similarity = tf.reshape(tf.matmul(normalize_a, normalize_b),[-1]) # [1,64] to [64]

    # w is always positive
    with tf.variable_scope("cos_params"):
        w = tf.Variable(10.0, name="scale_weight")
        bias = tf.Variable(-5.0, name="scale_bias")
        tf.assign(w, tf.clip_by_value(w, 0.0, 1000.0))

    
    # scaled cosine similarity
    scaled_cos_similarity = tf.add(tf.multiply(w, cos_similarity), bias)
    
    
    return scaled_cos_similarity

"""
def tf_scaled_cosine_similarity(a, b):
    # a is embedding matrix of utterances, by default [640, 256(proj_nodes)]
    # b is matrix of centroids, by default [64, 256(proj_nodes)]
    # returns a similiarity matrix of a and b'
    normalize_a = tf.nn.l2_normalize(a, -1)
    normalize_b = tf.transpose(tf.nn.l2_normalize(b, -1))

    # cosine similarity
    cos_similarity = tf.matmul(normalize_a, normalize_b)


    # w is always positive
    with tf.variable_scope("cos_params", reuse=tf.AUTO_REUSE):
        w = tf.Variable(10, name="scale_weight")
        bias = tf.Variable(-5, name="scale_bias")
        tf.assign(w, tf.clip_by_value(w, 0, np.infty))
    
    # scaled cosine similarity
    scaled_cos_similarity = tf.add(tf.multiply(w, cos_similarity), bias)
    
    return scaled_cos_similarity
"""

