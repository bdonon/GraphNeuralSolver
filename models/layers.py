import sys
import os

import tensorflow as tf


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

def custom_scatter_jacobian(indices_from, indices_to, params, shape):
    """
    This computational graph module performs the scatter_nd operation while taking into account
    the batch dimension. It creates matrices useful to compute the jacobian matrix. 
    Note that here we can also have d instead of d_F

    Inputs
        - indices_from : tf tensor of shape [n_samples, n_edges], and type tf.int32
        - indices_to : tf tensor of shape [n_samples, n_edges], and type tf.int32
        - params : tf tensor of shape [n_samples, n_edges, d_F], and type tf.float32
        - shape : tf.tensor of shape [3]
    Output
        - tf tensor of shape [n_samples, n_nodes, d_F] and type tf.float32
    """

    # Get all the relevant dimensions
    n_samples = tf.shape(params)[0]                                 # tf.int32, [1]
    n_nodes = shape[1]                                              # tf.int32, [1]
    n_edges = tf.shape(params)[1]                                   # tf.int32, [1]
    d_F = tf.shape(params)[2]                                       # tf.int32, [1]

    # Build indices for the batch dimension
    indices_batch_float = tf.linspace(0., tf.cast(n_samples, tf.float32)-1., n_samples)         
                                                                    # tf.float32, [n_samples] 
    indices_batch = tf.cast(indices_batch_float, tf.int32)          # tf.int32, [n_samples]
    indices_batch = tf.expand_dims(indices_batch, 1) * tf.ones([1, n_edges], dtype=tf.int32)    
                                                                    # tf.int32, [n_samples, n_edges]

    # Stack batch and edge dimensions
    indices = n_nodes**2 * indices_batch + n_nodes * indices_from + indices_to                  
                                                                    # tf.int32, [n_samples, n_edges]
    indices_flat = tf.reshape(indices, [-1, 1])                     # tf.int32, [n_samples * n_edges, 1]

    # Flatten the edge parameters
    params_flat = tf.reshape(params, [n_samples*n_edges, d_F])      # tf.float32, [n_samples * n_edges, d_F]

    # Perform the scatter operation
    scattered_flat = tf.scatter_nd(indices_flat, params_flat, shape=[n_samples*n_nodes*n_nodes, d_F])   
                                                                    # tf.float32, [n_samples * n_nodes * n_nodes, d_F]

    # Un-flatten the result of the scatter operation
    scattered = tf.reshape(scattered_flat, [n_samples, n_nodes, n_nodes, d_F])                  
                                                                    # tf.float32, [n_samples, n_nodes, n_nodes, d_F]

    return scattered

class FullyConnected:
    """
    Simple fully connected block. Serves as an elementary learning block in our neural network architecture.

    Params
        - latent_dimension : integer, number of hidden neurons in every intermediate layer
        - hidden : integer, number of layers. If set to 1, there is no hidden layer
        - non_lin : string, chosen non linearity
        - input_dim : integer, dimension of the input; if not specified, set to latent_dimension
        - output_dim : integer, dimension of the output; if not specified, set to latent_dimension
        - name : string, name of the neural network block
    """
    
    def __init__(self, 
        latent_dimension=10,
        hidden_layers=3,
        non_lin='leaky_relu', 
        input_dim=None,
        output_dim=None, 
        name='encoder'):
        
        # Get parameters
        self.latent_dimension = latent_dimension
        self.hidden_layers = hidden_layers
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Initialize list of trainable variables of the layer
        self.trainable_variables = []
        
        # Convert str into an actual tensorflow operator
        if non_lin == 'tanh':
            self.non_lin = tf.tanh
        elif non_lin == 'leaky_relu':
            self.non_lin = tf.nn.leaky_relu

        # Build weights
        self.build()
        
    def build(self):
        """
        Builds and collects the weights of the neural network block.
        """
        
        # Initialize weights dict
        self.W = {}
        self.b = {} 

        # Iterate over all layers
        for layer in range(self.hidden_layers):

            # Make sure the dimensions are used for the weights
            left_dim = self.latent_dimension
            right_dim = self.latent_dimension
            if (layer == 0) and (self.input_dim is not None):
                left_dim = self.input_dim
            if (layer == self.hidden_layers-1) and (self.output_dim is not None):
                right_dim = self.output_dim

            # Initialize weight matrix
            self.W[str(layer)] = tf.compat.v1.get_variable(name='W_'+self.name+'_{}'.format(layer),
                shape=[1, left_dim, right_dim],
                initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False, seed=2),
                trainable=True,
                dtype=tf.float32)
            self.trainable_variables.append(self.W[str(layer)])

            # Initialize bias vector
            self.b[str(layer)] = tf.compat.v1.get_variable(name='b_'+self.name+'_{}'.format(layer),
                shape=[1, 1, right_dim],
                initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32, uniform=False, seed=2),
                trainable=True,
                dtype=tf.float32)
            self.trainable_variables.append(self.b[str(layer)])

                            
    def __call__(self, h):
        """
        Builds the computational graph.
        """

        for layer in range(self.hidden_layers):
            # Iterate over all layers

            if layer==self.hidden_layers-1:
                # If last layer, then do not apply any non linearity
                h = tf.matmul(h, self.W[str(layer)]) + self.b[str(layer)]

            else:
                h = self.non_lin(tf.matmul(h, self.W[str(layer)])+ self.b[str(layer)])

        return h

class EquilibriumViolation:
    """
    Computes the violation of the target equations defined by:
        - a local force template;
        - an hypergraph instance
    """

    def __init__(self, path_to_data):
        """
        Import a force template.

        Params:
            - path_to_data : string, path to a valid dataset that contains a forces.py file
        """

        self.path_to_data = path_to_data

        try:
            # Try importing the force template located in the dataset folder
            sys.path.append(self.path_to_data)
            from forces import Forces
            self.forces = Forces()

        except ImportError:
            print('You should provide a compatible "forces.py" file in your data folder!')


    def error_tensor(self, X, A, B):
        """
        This computes for each node of each graph the sum of all the forces.

        Inputs
            - X : tensor of shape [n_samples, n_nodes, d_out]
            - A : tensor of shape [n_samples, n_edges, 2+d_in_A]
            - B : tensor of shape [n_samples, n_nodes, d_in_B]
        Output
            - Err : tensor of shape [n_samples, n_nodes, d_F]
        """

        d_out = tf.shape(X)[2]                                      # tf.int32, [1]
        n_samples = tf.shape(X)[0]                                  # tf.int32, [1]
        n_nodes = tf.shape(X)[1]                                    # tf.int32, [1]
        n_edges = tf.shape(A)[1]                                    # tf.int32, [1]

        # Get how many equalities should hold at each node
        d_F = self.forces.d_F                                       # tf.int32, [1]

        # Extract indices from A matrix
        indices_from = tf.cast(A[:,:,0], tf.int32)                  # tf.int32, [n_samples, n_edges, 1]
        indices_to = tf.cast(A[:,:,1], tf.int32)                    # tf.int32, [n_samples, n_edges, 1]

        # Extact edge characteristics from A matrix
        A_ij = A[:,:,2:]                                            # tf.float32, [n_samples, n_edge, d_in_A]

        # Gather X on both sides of each edge
        X_i = custom_gather(X, indices_from)                        # tf.float32, [n_samples , n_edges, d_out]
        X_j = custom_gather(X, indices_to)                          # tf.float32, [n_samples , n_edges, d_out]

        # Compute interaction forces for each edge
        F_bar = self.forces.F_bar(X_i, X_j, A_ij)                   # tf.float32, [n_samples, n_edges, d_F]

        # Scatter the interaction forces from edges to nodes
        F_bar_sum = custom_scatter(indices_from, F_bar, [n_samples, n_nodes, d_F])      
                                                                    # tf.float32, [n_samples, n_nodes, d_F]

        # Compute counteraction forces for each node
        F_round = self.forces.F_round(X, B)                         # tf.float32, [n_samples, n_nodes, d_F]

        # Sum all forces applied to each node
        Err = F_round + F_bar_sum                                   # tf.float32, [n_samples, n_nodes, d_F]

        return Err


    def jacobian_tensor(self, X, A, B):
        """
        Computes the jacobian of the error

        Inputs
            - X : tensor of shape [n_samples, n_nodes, d_out]
            - A : tensor of shape [n_samples, n_edges, 2+d_in_A]
            - B : tensor of shape [n_samples, n_nodes, d_in_B]

        Output 
            - jacobian : tensor of shape [n_samples, d_F, n_nodes, n_nodes]
        """

        d_out = tf.shape(X)[2]                                      # tf.int32, [1]
        n_samples = tf.shape(X)[0]                                  # tf.int32, [1]
        n_nodes = tf.shape(X)[1]                                    # tf.int32, [1]
        n_edges = tf.shape(A)[1]                                    # tf.int32, [1]

        # Get how many equalities should hold at each node
        d_F = self.forces.d_F                                       # tf.int32, [1]

        # Extract indices from A matrix
        indices_from = tf.cast(A[:,:,0], tf.int32)                  # tf.int32, [n_samples, n_edges]
        indices_to = tf.cast(A[:,:,1], tf.int32)                    # tf.int32, [n_samples, n_edges]

        # Extact edge characteristics from A matrix
        A_ij = A[:,:,2:]                                            # tf.float32, [n_samples, n_edges, d_in_A]

        # Gather predictions on "from" and "to" sides of each edge
        X_i = custom_gather(X, indices_from)                        # tf.float32, [n_samples, n_edges, d_out]
        X_j = custom_gather(X, indices_to)                          # tf.float32, [n_samples, n_edges, d_out]


        #### First, let's focus on the diagonal part, which is somehow similar to the error tensor

        # Compute local derivative of counteraction forces
        dF_round_dX = self.forces.dF_round_dX(X, B)                 # tf.float32, [n_samples, n_nodes, d_F]

        # Compute local derivative of interaction forces w.r.t the "from" side
        dF_bar_dX_i = self.forces.dF_bar_dX_i(X_i, X_j, A_ij)       # tf.float32, [n_samples, n_edges, d_F]

        # Scatter forces from edges to nodes
        dF_bar_dX_i_sum = custom_scatter(indices_from, dF_bar_dX_i, [n_samples, n_nodes, d_F])       
                                                                    # tf.float32, [n_samples, n_nodes, d_F]

        # Sum all forces applied to each node
        dF_sum_diag = dF_round_dX + dF_bar_dX_i_sum                 # tf.float32, [n_samples, n_nodes, d_F]

        # Convert it to a diagonal tensor
        dF_sum_diag_transpose = tf.transpose(dF_sum_diag, [0,2,1])  # tf.float32, [n_samples, d_F, n_nodes]
        dF_diag = tf.linalg.diag(dF_sum_diag_transpose)             # tf.float32, [n_samples, d_F, n_nodes, n_nodes]


        #### Now let's build the non diagonal part of the jacobian tensor, which is a bit more complicated

        # Compute local derivative of interaction forces w.r.t the "to" side
        dF_bar_dX_j = self.forces.dF_bar_dX_j(X_i, X_j, A_ij)       # tf.float32, [n_samples, n_edges, d_F]

        # Scatter forces from edges to couples of nodes
        dF_sum_non_diag = custom_scatter_jacobian(indices_from, indices_to, dF_bar_dX_j, [n_samples, n_nodes, n_nodes, d_F])    
                                                                    # tf.float32, [n_samples, n_nodes, n_nodes, d_F]

        # Transpose
        dF_sum_non_diag = tf.transpose(dF_sum_non_diag, [0,3,1,2])  # tf.float32, [n_samples, d_F, n_nodes, n_nodes]


        #### Sum the diagonal and non diagonal terms:

        jacobian = dF_diag + dF_sum_non_diag                        # tf.float32, [n_samples, d_F, n_nodes, n_nodes]

        return jacobian



    def __call__(self, X, A, B, nr=True):
        """
        Computes a surrogate for the distance between our prediction and the ground truth.
        This surrogate is exact in the case of linear systems.

        Inputs
            - X : tensor of shape [n_samples, n_nodes, d_out]
            - A : tensor of shape [n_samples, n_edges, 2+d_in_A]
            - B : tensor of shape [n_samples, n_nodes, d_in_B]

        Output
            - loss : tensor of shape [1]
        """

        d_out = tf.shape(X)[2]                                      # tf.int32, [1]
        n_samples = tf.shape(X)[0]                                  # tf.int32, [1]
        n_nodes = tf.shape(X)[1]                                    # tf.int32, [1]
        n_edges = tf.shape(A)[1]                                    # tf.int32, [1]

        # Get how many equalities should hold at each node
        d_F = self.forces.d_F                                       # tf.int32, [1]

        # Get error and jacobian tensors
        error = self.error_tensor(X, A, B)                          # tf.float32, [n_samples, n_nodes, d_F]
        jacobian = self.jacobian_tensor(X, A, B)                    # tf.float32, [n_samples, d_F, n_nodes, n_nodes]

        # Transpose and add dummy dim the error for matrix compatibility
        error_transpose = tf.transpose(error, [0,2,1])              # tf.float32, [n_samples, d_F, n_nodes]
        error_dummy = tf.expand_dims(error_transpose, -1)           # tf.float32, [n_samples, d_F, n_nodes, 1]
        error_dummy_flat = tf.reshape(error_dummy, [-1, n_nodes, 1])                             
                                                                    # tf.float32, [n_samples * d_F, n_nodes, 1]

        # Inverse the jacobian
        jacobian_flat = tf.reshape(jacobian, [-1, n_nodes, n_nodes])
                                                                    # tf.float32, [n_samples * d_F, n_nodes, n_nodes]
        inverse_jacobian_flat = tf.linalg.inv(jacobian_flat)        # tf.float32, [n_samples * d_F, n_nodes, n_nodes]    

        # Compute loss
        proxy_difference_flat = tf.matmul(inverse_jacobian_flat, error_dummy_flat) 
                                                                    # tf.float32, [n_samples * d_F, n_nodes, 1]
        proxy_difference = tf.reshape(proxy_difference_flat, [n_samples, d_F, n_nodes, 1])
                                                                    # tf.float32, [n_samples, d_F, n_nodes, 1]

        # Compute the l2 norm for each node, and take the average for the dataset
        if nr:
            loss = tf.reduce_mean(proxy_difference**2)              # tf.float32, [1]
        else:
            loss = tf.reduce_mean(error**2)                         # tf.float32, [1]

        return loss, error, jacobian, proxy_difference