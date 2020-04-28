# This file defines the interaction and counteraction forces, as well as their first order derivatives
import tensorflow as tf

# class Dimensions:

#     def __init__(self):

#         self.type = 'Spring-like interaction'
        
#         # Input dimensions
#         self.d_in_A = 1
#         self.d_in_B = 3

#         # Output dimensions
#         self.d_out = 1

#         # How many equations should how for each node
#         self.d_F = 1

# class Forces:
#     """
#     Defines the elementary forces for interacting springs
#     Each force should follow a similar template, that is compatible with tensorflow
#     """
#     def __init__(self):
#         """
#         Here are the inputs dimensions:
#             - X :   [n_samples, n_nodes, d_out]
#             - B :   [n_samples, n_nodes, d_in_B]
#             - X_i : [n_samples, n_edges, d_out]
#             - X_j : [n_samples, n_edges, d_out]
#             - A_ij: [n_samples, n_edges, d_in_A]
#         """

#     def F_round(self, X, B):
#         return -B[:,:,:1] * (1-B[:,:,1:2]) + B[:,:,1:2] * (X[:,:,:]-B[:,:,2:3])             
#                                                         # tf.int32, [n_samples, n_nodes, d_F]

#     def F_bar(self, X_i, X_j, A_ij):
#         return A_ij * (X_j-X_i)                         # tf.int32, [n_samples, n_edges, d_F]



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
        self.d_in_A = 1
        self.d_in_B = 3

        # Output dimensions
        self.d_out = 1

        # How many equations should how for each node
        self.d_F = 1

    def cost_function(self, X, A, B):

        # Gather instances dimensions (samples, nodes and edges)
        n_samples = tf.shape(X)[0]                                  # tf.int32, [1]
        n_nodes = tf.shape(X)[1]                                    # tf.int32, [1]
        n_edges = tf.shape(A)[1]                                    # tf.int32, [1]

        # Extract indices from A matrix
        indices_from = tf.cast(A[:,:,0], tf.int32)                  # tf.int32, [n_samples, n_edges, 1]
        indices_to = tf.cast(A[:,:,1], tf.int32)                    # tf.int32, [n_samples, n_edges, 1]

        # Extact edge characteristics from A matrix
        A_ij = A[:,:,2:]                                            # tf.float32, [n_samples, n_edge, d_in_A]

        # Gather X on both sides of each edge
        X_i = custom_gather(X, indices_from)                        # tf.float32, [n_samples , n_edges, d_out]
        X_j = custom_gather(X, indices_to)                          # tf.float32, [n_samples , n_edges, d_out]

        # Compute interaction forces for each edge
        F_bar = A_ij * (X_j-X_i)                                    # tf.float32, [n_samples, n_edges, d_F]

        # Scatter the interaction forces from edges to nodes
        F_bar_sum = custom_scatter(indices_from, F_bar, [n_samples, n_nodes, self.d_F])      
                                                                    # tf.float32, [n_samples, n_nodes, d_F]

        # Compute counteraction forces for each node
        F_round = -B[:,:,:1] * (1-B[:,:,1:2]) + B[:,:,1:2] * (X[:,:,:]-B[:,:,2:3])                         
                                                                    # tf.float32, [n_samples, n_nodes, d_F]

        # Sum all forces applied to each node
        residual = F_round + F_bar_sum                              # tf.float32, [n_samples, n_nodes, d_F]

        cost_per_sample = tf.reduce_mean(residual**2, axis=[1,2])                             # tf.float32, [1]

        return cost_per_sample

















