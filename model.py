# GENERALIZED END-TO-END LOSS FOR SPEAKER VERIFICATION
# https://arxiv.org/abs/1710.10467

import tensorflow as tf
import argparse
from tensorflow.python.layers import core as layers_core
import utils
import numpy as np
import re


# 되도록이면 GE2E 클래스를 test/infer하는 데에도 쓸 수 있도록 바꿀 것

class GE2E():
    def __init__(self, hparams):

        self.hparams = hparams
        self.batch_size = self.hparams.num_utt_per_batch * self.hparams.num_spk_per_batch

    def set_up_model(self, mode):
        ge2e_graph = tf.Graph()
        with ge2e_graph.as_default():
            if mode == "train":
                # Input Batch of [N*M(batch_size), total_frames, 40(spectrogram_channel)]
                # Target Batch of [N*M(batch_size), Speaker ID]
                self.input_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, self.hparams.spectrogram_scale], name="input_batch")
                self.target_batch = tf.placeholder(dtype=tf.int32, shape=[None], name="target_batch")
                self._create_embedding()
                self._cal_loss()
                self._optimize()

            elif mode == "infer" or "test":
                self.input_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, self.hparams.spectrogram_scale], name="input_batch")
                self._create_embedding()


            else:
                raise ValueError("mode not supported")

        return ge2e_graph

    def _create_embedding(self):
        
        with tf.variable_scope("lstm_embedding"):
            # Create Embedding Using LSTM
            stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.hparams.num_lstm_cells, num_proj=self.hparams.dim_lstm_projection) for _ in range(self.hparams.num_lstm_stacks)])
            # 
            # Create Initial State
            #init_state = stacked_lstm.zero_state(self.batch_size, dtype=tf.float32)
            # Decode Using dynamic_rnn
            # outputs is a tensor of [batch_size, total_frames, output_size]
            # output_size is self.hparams.dim_lstm_projection if num_proj in LSTMCell is set
            # state is a tensor of [batch_size, state_size of the cell]
        
            outputs, state = tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=self.input_batch, dtype=tf.float32)
        
            # L2 Normalize the output of the last layer at the final frame
            # norm_out is a tensor of [batch_size, output_size], by default, [640, 256(proj_nodes)]
            self.norm_out = tf.nn.l2_normalize(outputs[:, -1, :], axis=-1)

    def _cal_centroid(self, centroid_idx, utt_idx = None):

        all_utts_for_spk = self.norm_out[centroid_idx * self.hparams.num_utt_per_batch : (centroid_idx+1) * self.hparams.num_utt_per_batch, :]  
        
        if self.hparams.loss_type == "optional_softmax":
            spk_id = (utt_idx // self.hparams.num_utt_per_batch)
            utt_idx_in_group = utt_idx % self.hparams.num_utt_per_batch
            if centroid_idx == spk_id:
                mask = np.array([False if utt == utt_idx_in_group else True for utt in range(self.hparams.num_utt_per_batch)])
                centroid = tf.reduce_mean(tf.boolean_mask(all_utts_for_spk, mask), 0)
        
        centroid = tf.reduce_mean(all_utts_for_spk, 0)

        return centroid


    def _cal_centroid_matrix(self, utt_idx = None):
        if self.hparams.loss_type == "basic_softmax":
            # From the utterances of the first speaker batch to those of the last speaker, calculate each centroid
            centroid_mat = []
            for i in range(self.hparams.num_spk_per_batch):
                # centroid is a vector, reduced mean of embeddings per speaker
                centroid = tf.reduce_mean(self.norm_out[i * self.hparams.num_utt_per_batch : (i+1) * self.hparams.num_utt_per_batch, :], 0)
                centroid_mat.append(centroid)
            # self.centroids == [c1, c2, c3 ... cn], its shape [64, 256(proj_nodes)]
            centroid_mat = tf.convert_to_tensor(centroid_mat)

        elif self.hparams.loss_type == "optional_softmax":
            centroid_mat = tf.convert_to_tensor(tf.map_fn(self._cal_centroid, tf.range(self.hparams.num_spk_per_batch), dtype=tf.float32))
        else:
            print("Loss type not supported") 
        return centroid_mat
            

    def _create_sim_per_utt(self, utt_idx):

        #utt_dvector is a tensor of shape [output_size]
        utt_dvector = self.norm_out[utt_idx, :]
        #centroids is a tensor of shape [num_spk_per_batch, output_size]
        #sim_per_utt is a tensor of shape [num_spk_per_batch]
        centroids = self._cal_centroid_matrix(utt_idx)
        sim_per_utt = utils.tf_scaled_cosine_similarity(utt_dvector, centroids)

        return sim_per_utt

    def _create_sim_mat(self):
        centroids = self._cal_centroid_matrix()
        sim_mat = tf.sigmoid(utils.tf_scaled_cosine_similarity(self.norm_out, centroids))
        return sim_mat
            

    def _cal_loss(self):
        with tf.variable_scope("loss"):
            # utt_idx // num_utt_per_batch(10) => true idx of sim_mat columns

            if self.hparams.loss_type == "basic_softmax":
                self.sim_mat = self._create_sim_mat()
                self.sim_mat_summary = tf.summary.image("sim_mat", tf.reshape(self.sim_mat,[1, self.batch_size, self.hparams.num_spk_per_batch, 1]))
                self.total_loss = tf.divide(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.sim_mat, labels=self.target_batch)), self.batch_size)
                

            elif self.hparams.loss_type == "optional_softmax":
                # sim_mat has shape of [batch_size, num_spk]
                self.sim_mat = tf.convert_to_tensor(tf.map_fn(self._create_sim_per_utt, tf.range(self.batch_size),dtype=tf.float32))
                self.sim_mat_summary = tf.summary.image("sim_mat", tf.reshape(self.sim_mat,[1, self.batch_size, self.hparams.num_spk_per_batch, 1]))
                self.total_loss = tf.divide(tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.sim_mat, labels=self.target_batch)), self.batch_size)
                
            else:
                print("Loss type not supported")         
           
    def _optimize(self):
        with tf.variable_scope("optimize"):
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
            learning_rate = tf.train.exponential_decay(self.hparams.learning_rate, self.global_step,
                                       30000000, 0.5, staircase=True)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.total_loss)
            
            clipped_grad_and_vars = []
            for grad, var in grads_and_vars:
                if re.search("cos_params", var.name):
                    grad = tf.clip_by_value(grad, -self.hparams.scale_clip, self.hparams.scale_clip)
                elif re.search("projection", var.name):
                    grad = tf.clip_by_value(grad, -self.hparams.lstm_proj_clip, self.hparams.lstm_proj_clip)
                else:
                    grad = tf.clip_by_norm(grad, self.hparams.l2_norm_clip)
                clipped_grad_and_vars.append((grad, var))
    
            self.optimize = optimizer.apply_gradients(clipped_grad_and_vars, global_step=self.global_step)
            



