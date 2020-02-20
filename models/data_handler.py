import tensorflow as tf
import numpy as np

import os

class DataHandler:
    """
    Small computation graph that randomly samples into an inputed dataset.
    It also makes sure that all the sampled small graphs are concatenated into
    a larger super-graph.
    """

    def __init__(self, 
        sess, 
        path_to_data='data/',
        minibatch_size=None,
        mode='train'):

        self.sess = sess
        self.path_to_data = path_to_data
        self.mode = mode
        self.minibatch_size = minibatch_size

        self.build()

    def build(self):

        self.B_np = np.load(os.path.join(self.path_to_data, 'B_'+self.mode+'.npy'))
        self.A_np = np.load(os.path.join(self.path_to_data, 'A_'+self.mode+'.npy'))

        self.num_samples = tf.Variable(self.B_np.shape[0], trainable=False)
        self.num_nodes = tf.Variable(self.B_np.shape[1], trainable=False)
        self.num_edges = tf.Variable(self.A_np.shape[1], trainable=False)

        try:
            self.B_dim = tf.Variable(self.B_np.shape[2], trainable=False)
        except:
            self.B_dim = tf.Variable(1, trainable=False)

        self.A_dim = tf.Variable(self.A_np.shape[2], trainable=False)

        self.A = tf.Variable(self.A_np, trainable=False)
        self.B = tf.Variable(self.B_np, trainable=False)

        if self.minibatch_size is not None:
            self.minibatch_size_tf = tf.Variable(self.minibatch_size, trainable=False)
        else:
            self.minibatch_size_tf = self.num_samples

        self.sess.run(tf.initialize_variables(
            [self.A,
            self.B,
            self.A_dim,
            self.B_dim,
            self.num_samples,
            self.num_nodes,
            self.num_edges,
            self.minibatch_size_tf]))

        

        self.scores = tf.random.uniform([self.num_samples])
        self.values, self.minibatch_indices = tf.math.top_k(
            self.scores, 
            k=self.minibatch_size_tf)

        self.B_minibatch = tf.gather(self.B, self.minibatch_indices)
        self.B_minibatch = tf.reshape(self.B_minibatch, [-1, self.B_dim])

        self.A_minibatch = tf.gather(self.A, self.minibatch_indices)

        self.indices_offset = tf.linspace(0., tf.cast(self.minibatch_size_tf-1, tf.float32), self.minibatch_size_tf)
        self.indices_offset = self.indices_offset * tf.cast(self.num_nodes, tf.float32)
        self.indices_offset = tf.reshape(self.indices_offset, [-1, 1, 1])
        self.indices_offset = self.indices_offset * tf.ones([1,1,2])

        self.indices_offset = tf.concat([self.indices_offset, tf.zeros([self.minibatch_size_tf,1,1])], axis=2) #+ tf.zeros_like(self.A)
        self.A_minibatch = self.A_minibatch + self.indices_offset
        self.A_minibatch = tf.reshape(self.A_minibatch, [-1, self.A_dim])


    def sample(self):

        return self.A_minibatch, self.B_minibatch

    def change_source(self,
        path_to_data='data',
        minibatch_size=None,
        mode='train'):

        self.B_np = np.load(os.path.join(self.path_to_data, 'B_'+self.mode+'.npy'))
        self.A_np = np.load(os.path.join(self.path_to_data, 'A_'+self.mode+'.npy'))

        self.sess.run([self.A.assign(self.A_np),
            self.B.assign(self.B_np),
            self.A_dim.assign(self.A_np.shape[2]),
            self.num_samples.assign(self.B_np.shape[0]),
            self.num_nodes.assign(self.B_np.shape[1]),
            self.num_edges.assign(self.A_np.shape[1])])

        try:
            self.sess.run(self.B_dim.assign(self.B_np.shape[2]))
        except:
            self.sess.run(self.B_dim.assign(1))

        if minibatch_size is not None:
            self.sess.run(self.minibatch_size_tf.assign(minibatch_size))
        else:
            self.sess.run(self.minibatch_size_tf.assign(self.B_np.shape[0]))

