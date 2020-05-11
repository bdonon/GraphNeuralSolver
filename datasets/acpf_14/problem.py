# This file defines the interaction and counteraction forces, as well as their first order derivatives
import tensorflow as tf
import numpy as np

def custom_gather(params, indices_edges):
    """
    This computational graph module performs the gather_nd operation while taking into account
    the batch dimension.

    Inputs
        - params : tf tensor of shape [n_samples, n_nodes, d_out], and type tf.float32
        - indices_edges : tf tensor of shape [n_samples, n_edges], and type tf.int32
    Output
        - tf tensor of shape [n_samples, n_edges, d_out] and type tf.float32
    """

    # Get all relevant dimensions
    n_samples = tf.shape(params)[0]                                     # tf.int32, [1]
    n_nodes = tf.shape(params)[1]                                       # tf.int32, [1]
    n_edges = tf.shape(indices_edges)[1]                                # tf.int32, [1]
    d_out = tf.shape(params)[2]                                         # tf.int32, [1]

    # Build indices for the batch dimension
    indices_batch_float = tf.linspace(0., tf.cast(n_samples, tf.float32)-1., n_samples)         
                                                                        # tf.float32, [n_samples]
    indices_batch = tf.cast(indices_batch_float, tf.int32)              # tf.int32, [n_samples]
    indices_batch = tf.expand_dims(indices_batch, 1) * tf.ones([1, n_edges], dtype=tf.int32)    
                                                                        # tf.int32, [n_samples, n_edges]

    # Flatten the indices
    indices = n_nodes * indices_batch + indices_edges                   # tf.int32, [n_samples, n_edges]
    indices_flat = tf.reshape(indices, [-1, 1])                         # tf.int32, [n_samples * n_edges, 1]

    # Flatten the node parameters
    params_flat = tf.reshape(params, [-1, d_out])                       # tf.float32, [n_samples * n_nodes, d_out]

    # Perform the gather operation
    gathered_flat = tf.gather_nd(params_flat, indices_flat)             # tf.float32, [n_samples * n_edges, d_out]

    # Un-flatten the result of the gather operation
    gathered = tf.reshape(gathered_flat, [n_samples, n_edges, d_out])   # [n_samples , n_edges, d_out]

    return gathered

def custom_scatter(indices_edges, params, shape):
    """
    This computational graph module performs the scatter_nd operation while taking into account
    the batch dimension. Note that here we can also have d instead of d_F

    Inputs
        - indices_edges : tf tensor of shape [n_samples, n_edges], and type tf.int32
        - params : tf tensor of shape [n_samples, n_edges, d_F], and type tf.float32
        - shape : tf.tensor of shape [3]
    Output
        - tf tensor of shape [n_samples, n_nodes, n_nodes, d_F] and type tf.float32
    """

    # Get all the relevant dimensions
    n_samples = tf.shape(params)[0]                                     # tf.int32, [1]
    n_nodes = shape[1]                                                  # tf.int32, [1]
    n_edges = tf.shape(params)[1]                                       # tf.int32, [1]
    d_F = tf.shape(params)[2]                                           # tf.int32, [1]

    # Build indices for the batch dimension
    indices_batch_float = tf.linspace(0., tf.cast(n_samples, tf.float32)-1., n_samples)         
                                                                        # tf.float32, [n_samples]
    indices_batch = tf.cast(indices_batch_float, tf.int32)              # tf.int32, [n_samples]
    indices_batch = tf.expand_dims(indices_batch, 1) * tf.ones([1, n_edges], dtype=tf.int32)    
                                                                        # tf.int32, [n_samples, n_edges]

    # Stack batch and edge dimensions
    indices = n_nodes * indices_batch + indices_edges                   # tf.int32, [n_samples, n_edges]
    indices_flat = tf.reshape(indices, [-1, 1])                         # tf.int32, [n_samples * n_edges, 1]

    # Flatten the edge parameters
    params_flat = tf.reshape(params, [n_samples*n_edges, d_F])          # tf.float32, [n_samples * n_edges, d_F]

    # Perform the scatter operation
    scattered_flat = tf.scatter_nd(indices_flat, params_flat, shape=[n_samples*n_nodes, d_F])   
                                                                        # tf.float32, [n_samples * n_nodes, d_F]

    # Un-flatten the result of the scatter operation
    scattered = tf.reshape(scattered_flat, [n_samples, n_nodes, d_F])   # tf.float32, [n_samples, n_nodes, d_F]

    return scattered    



class Problem:

    def __init__(self):

        self.name = 'Discretized Poisson'
        
        # Input dimensions
        self.d_in_A = 2
        self.d_in_B = 5

        # Output dimensions
        self.d_out = 2

        # How many equations should how for each node
        self.d_F = 1

        self.initial_U = np.array([1.,0.])

        # Normalization constants
        self.B_mean = np.array([0, -5.21, 0.638, 0.374, 0.071428])
        self.B_std = np.array([61.5, 6.459, 0.47915, 0.50514, 0.2575])
        self.A_mean = np.array([0.0, 0.0, 887, 1.05978])
        self.A_std = np.array([1.0, 1.0, 870.318, 1.38223432])

    def cost_function(self, X, A, B):

        # Gather instances dimensions (samples, nodes and edges)
        n_samples = tf.shape(X)[0]                                  # tf.int32, [1]
        n_nodes = tf.shape(X)[1]                                    # tf.int32, [1]
        n_edges = tf.shape(A)[1]                                    # tf.int32, [1]

        # Extract indices from A matrix
        indices_from = tf.cast(A[:,:,0], tf.int32)                  # tf.int32, [n_samples, n_edges, 1]
        indices_to = tf.cast(A[:,:,1], tf.int32)                    # tf.int32, [n_samples, n_edges, 1]

        # Extact edge characteristics from A matrix
        A1_ij = A[:,:,2:3]                                            # tf.float32, [n_samples, n_edge, d_in_A]
        A2_ij = A[:,:,3:4]                                            # tf.float32, [n_samples, n_edge, d_in_A]


        # Gather X on both sides of each edge
        X1_i = custom_gather(B[:,:,2:3]*X[:,:,0:1] + (1-B[:,:,2:3])*B[:,:,3:4], indices_from)                        # tf.float32, [n_samples , n_edges, d_out]
        X2_i = custom_gather(X[:,:,1:2], indices_from)                        # tf.float32, [n_samples , n_edges, d_out]
        X1_j = custom_gather(B[:,:,2:3]*X[:,:,0:1] + (1-B[:,:,2:3])*B[:,:,3:4], indices_to)                        # tf.float32, [n_samples , n_edges, d_out]
        X2_j = custom_gather(X[:,:,1:2], indices_to)                        # tf.float32, [n_samples , n_edges, d_out]


        # Compute line stuff
        P_ij = A1_ij * X1_i * X1_j * tf.math.cos(X2_i - X2_j - A2_ij)
        Q_ij = A1_ij * X1_i * X1_j * tf.math.sin(X2_i - X2_j - A2_ij)


        delta_P = (1-B[:,:,4:5]) * (- B[:,:,0:1] + custom_scatter(indices_from, P_ij, [n_samples, n_nodes, 1]))**2
        #delta_P = (- B[:,:,0:1] + custom_scatter(indices_from, P_ij, [n_samples, n_nodes, 1]))**2
        delta_Q = B[:,:,2:3] * (- B[:,:,1:2] + custom_scatter(indices_from, Q_ij, [n_samples, n_nodes, 1]))**2
        delta_V = (1-B[:,:,2:3]) * (X[:,:,0:1] - B[:,:,3:4])**2

        # cost_per_sample =  tf.reduce_mean(delta_P, axis=[1,2]) / tf.reduce_mean((1-B[:,:,4:5])*B[:,:,0:1]**2, axis=[1,2])\
        #     + tf.reduce_mean(delta_Q, axis=[1,2]) / tf.reduce_mean(B[:,:,2:3] * B[:,:,1:2]**2, axis=[1,2])\
        #     + tf.reduce_mean(delta_V, axis=[1,2]) / tf.reduce_mean((1-B[:,:,2:3])*B[:,:,3:4]**2, axis=[1,2])

        cost_per_sample =  tf.reduce_mean(delta_P, axis=[1,2]) / 1e4 \
            + tf.reduce_mean(delta_Q, axis=[1,2]) / 1e4 \
            + tf.reduce_mean(delta_V, axis=[1,2]) 

        #cost_per_sample = tf.reduce_mean(delta_V, axis=[1,2]) / tf.reduce_mean((1-B[:,:,2:3])*B[:,:,3:4]**2, axis=[1,2])

        return cost_per_sample#, tf.reduce_mean(delta_P, axis=[1,2]), tf.reduce_mean(delta_Q, axis=[1,2]), tf.reduce_mean(delta_V, axis=[1,2]) #/ 1e10

















