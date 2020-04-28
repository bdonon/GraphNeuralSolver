# This file defines the interaction and counteraction forces, as well as their first order derivatives
import tensorflow as tf

class Dimensions:

    def __init__(self):

        self.type = 'Spring-like interaction'
        
        # Input dimensions
        self.d_in_A = 1
        self.d_in_B = 3

        # Output dimensions
        self.d_out = 1

        # How many equations should how for each node
        self.d_F = 1

class Forces:
    """
    Defines the elementary forces for interacting springs
    Each force should follow a similar template, that is compatible with tensorflow
    """
    def __init__(self):
        """
        Here are the inputs dimensions:
            - X :   [n_samples, n_nodes, d_out]
            - B :   [n_samples, n_nodes, d_in_B]
            - X_i : [n_samples, n_edges, d_out]
            - X_j : [n_samples, n_edges, d_out]
            - A_ij: [n_samples, n_edges, d_in_A]
        """

    def F_round(self, X, B):
        return -B[:,:,:1] * (1-B[:,:,1:2]) + B[:,:,1:2] * (X[:,:,:]-B[:,:,2:3])             
                                                        # tf.int32, [n_samples, n_nodes, d_F]

    def F_bar(self, X_i, X_j, A_ij):
        return A_ij * (X_j-X_i)                         # tf.int32, [n_samples, n_edges, d_F]