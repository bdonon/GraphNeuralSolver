"""
This aims at checking some properties in a dataset, so that it can work properly with
a Graph Neural Solver. See https://github.com/bdonon/GraphNeuralSolver for more info
"""

import os
import sys

import numpy as np
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


def load_numpy(path):
    """
    Loads a numpy array
    """
    return np.load(path)

def load_tfrecords(path, dims):
    """
    Loads a tfrecords array
    """

    d_in_A = dims.d_in_A
    d_in_B = dims.d_in_B
    d_out = dims.d_out

    A = []
    B = []
    X = []

    for serialized_example in tf.python_io.tf_record_iterator(path):

        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        A_sample = np.array(example.features.feature['A'].float_list.value)
        A_sample = np.reshape(A_sample, [-1, d_in_A+2])

        B_sample = np.array(example.features.feature['B'].float_list.value)
        B_sample = np.reshape(B_sample, [-1, d_in_B])

        X_sample = np.array(example.features.feature['X'].float_list.value)
        X_sample = np.reshape(X_sample, [-1, d_out])

        A.append(A_sample)
        B.append(B_sample)
        X.append(X_sample)

    return np.array(A), np.array(B), np.array(X)

def test_all_files_exist_and_import():
    """
    Tests that all the files required for the Graph Neural Solver
    """

    required_files = [
        # Numpy files
        'A_train.npy',
        'A_val.npy',
        'A_test.npy',
        'B_train.npy',
        'B_val.npy',
        'B_test.npy',
        'X_train.npy',
        'X_val.npy',
        'X_test.npy',
        'coord_train.npy',
        'coord_val.npy',
        'coord_test.npy',
        # tfrecords files
        'train.tfrecords',
        'val.tfrecords',
        'test.tfrecords',
        # Problem definition
        'problem.py'
    ]

    # Assert that all files are there
    for file in required_files:
        assert os.path.exists(file), ("Missing {} file.".format(file))

    # Import problem forces and dimensions
    sys.path.append('problem.py')
    from problem import Forces, Dimensions
    forces = Forces()
    dims = Dimensions()

    # Import data
    data = {}
    for file in required_files:

        # If numpy file
        if file[-4:] == '.npy':
            data[file] = load_numpy(file)

        # If tfrecords file
        elif file[-10:] == '.tfrecords':
            mode = file[:-10]
            data['A_'+mode+'.tfrecords'], data['B_'+mode+'.tfrecords'], data['X_'+mode+'.tfrecords'] = load_tfrecords(file, dims)

    return data, forces, dims

def test_n_samples(data):
    """
    Test that all the dataset files have the same amount of samples.
    The amount of samples is very likely to depend on the mode (train, val, test)
    """

    modes = ['train', 'val', 'test']
    arrays = ['A', 'B', 'X']
    forms = ['.npy', '.tfrecords']
    
    for mode in modes:

        # Get reference amount of samples
        n_sample_ref = np.shape(data['A_'+mode+'.npy'])[0]

        # Check that every array in the same mode has the same amount of samples
        for array in arrays:

            for form in forms:

                current_data = data[array+'_'+mode+form]
                n_sample = np.shape(current_data)[0]

                assert n_sample_ref == n_sample, \
                    ("A_train.npy and {}_{}{} do not have the same amount of samples (resp. {} and {})."\
                    .format(array, mode, form, n_sample_ref, n_sample))

        current_data = data['coord_'+mode+'.npy']
        n_sample = np.shape(current_data)[0]

        assert n_sample_ref == n_sample, \
            ("A_train.npy and coord_{}.npy do not have the same amount of samples (resp. {} and {})."\
            .format(mode, n_sample_ref, n_sample))

def test_rank(data):
    """
    Test that all arrays are rank 3
    """

    modes = ['train', 'val', 'test']
    arrays = ['A', 'B', 'X']
    forms = ['.npy', '.tfrecords']

    for mode in modes:

        for array in arrays:

            for form in forms:

                current_data = data[array+'_'+mode+form]

                assert len(current_data.shape) == 3, \
                    ("{}_{}{} is not rank 3.".format(array, mode, form))

        current_data = data['coord_'+mode+'.npy']

        assert len(current_data.shape) == 3, \
            ("coord_{}.npy is not rank 3.".format(mode))

def test_problem_dimensions(data, dims):
    """
    Tests that the dimensions provided in problem.py match the ones of the dataset
    """

    modes = ['train', 'val', 'test']
    arrays = ['A', 'B', 'X']
    forms = ['.npy', '.tfrecords']

    # Get relevant dimensions
    d_in_A = dims.d_in_A
    d_in_B = dims.d_in_B
    d_out = dims.d_out
    
    # Get dimensions of each array 
    for mode in modes:
        for form in forms:
            current_A = data['A_'+mode+form]
            current_B = data['B_'+mode+form]
            current_X = data['X_'+mode+form]
            current_d_in_A = np.shape(current_A)[-1]-2
            current_d_in_B = np.shape(current_B)[-1]
            current_d_out = np.shape(current_X)[-1]

            assert current_d_in_A == d_in_A, \
                    ("A_{}{}'s last dimension does not match the one provided in problem.py (resp. {} and {}).".\
                        format(mode, form, current_d_in_A, d_in_A))
            assert current_d_in_B == d_in_B, \
                    ("B_{}{}'s last dimension does not match the one provided in problem.py (resp. {} and {}).".\
                        format(mode, form, current_d_in_B, d_in_B))
            assert current_d_out == d_out, \
                    ("X_{}{}'s last dimension does not match the one provided in problem.py (resp. {} and {}).".\
                        format(mode, form, current_d_out, d_out))

def test_n_nodes(data):
    """
    Tests that for a given mode, B X and coord all have the same amount of nodes
    """

    modes = ['train', 'val', 'test']
    arrays = ['A', 'B', 'X']
    forms = ['.npy', '.tfrecords']

    for mode in modes:

        # Get number of nodes in B_mode.npy as a reference
        n_nodes_ref = np.shape(data['B_'+mode+'.npy'])[1]

         # Get number of nodes in every other file of the same mode
        for array in ['B', 'X']:

            for form in forms:

                n_nodes = np.shape(data[array+'_'+mode+form])[1]

                assert (n_nodes == n_nodes_ref), \
                    ('B_{}.npy and {}_{}{} do not have the same amount of nodes (resp. {} and {})'.\
                        format(mode, array, mode, form, n_nodes_ref, n_nodes))

        n_nodes = np.shape(data['coord_'+mode+'.npy'])[1]

        assert (n_nodes == n_nodes_ref), \
            ('B_{}.npy and coord_{}.npy do not have the same amount of nodes (resp. {} and {})'.\
                format(mode, mode, n_nodes_ref, n_nodes))

def test_np_tfrecords(data):
    """
    Tests that files stored in .npy and .tfrecords match to a certain precision
    """

    modes = ['train', 'val', 'test']
    arrays = ['A', 'B', 'X']
    forms = ['.npy', '.tfrecords']

    for mode in modes:

        for array in arrays:

            # Compute absolute difference between the npy and the tfrecords files
            npy = data[array+'_'+mode+'.npy']
            tfrecords = data[array+'_'+mode+'.tfrecords']
            diff = np.abs(npy-tfrecords)

            assert (np.max(diff) < 1e-3).all(), \
                ('{}_{}.npy and {}_{}.tfrecords do not match (precision = 1e-3)'.format(array, mode, array, mode))

def test_or_ex_n_nodes(data):
    """
    Tests that the origin and extremity columns of A all refer to existing nodes
    """

    modes = ['train', 'val', 'test']
    forms = ['.npy', '.tfrecords']

    for mode in modes:

        # Get the amount of nodes in array B
        current_B = data['B_'+mode+'.npy']
        n_nodes = np.shape(current_B)[1]

        for form in forms:

            # Get max origin and extremity indices listed in A
            current_A = data['A_'+mode+form]
            n_max_or = np.max(current_A[:,:,0])
            n_max_ex = np.max(current_A[:,:,1])

            assert (n_max_or < n_nodes).all(), \
                ('A_{}{} origin column refers to an unexisting node (id = {})'.format(mode, form, n_max_or))
            assert (n_max_ex < n_nodes), \
                ('A_{}{} extremity column refers to an unexisting node (id = {})'.format(mode, form, n_max_ex))

def test_d_F_equations(dims, forces):
    """
    Tests that the forces and the provided dimensions match.
    """

    # Get relevant dimensions
    d_in_A = dims.d_in_A
    d_in_B = dims.d_in_B
    d_out = dims.d_out
    d_F = dims.d_F

    # Build dummy variables
    A = tf.zeros([1,1,d_in_A])
    B = tf.zeros([1,1,d_in_B])
    X = tf.zeros([1,1,d_out])

    # Build resulting dummy forces
    F_bar = forces.F_bar(X, X, A)
    F_round = forces.F_round(X, B)

    # Initialize session
    sess = tf.Session()

    # Compute dummy forces and get last dimension
    F_bar_np, F_round_np = sess.run([F_bar, F_round])
    d_Fbar = np.shape(F_bar_np)[2]
    d_Fround = np.shape(F_round_np)[2]

    assert (d_Fbar == d_F), \
        ('The expression for F_bar in problem.py does not build d_F equalities (resp. {} and {})'.format(d_Fbar, d_F))
    assert (d_Fround == d_F), \
        ('The expression for F_round in problem.py does not build d_F equalities (resp. {} and {})'.format(d_Fround, d_F))


def test_solution(data, dims, forces):
    """
    Computes the residual and the loss for A, B, X and the provided forces.
    """

    # Get relevant dimensions
    d_in_A = dims.d_in_A
    d_in_B = dims.d_in_B
    d_out = dims.d_out
    d_F = dims.d_F 


    # BUILD COMPUTATIONAL GRAPH

    # Build placeholders for the input
    A = tf.placeholder(tf.float32, shape=(None, None, d_in_A+2))
    B = tf.placeholder(tf.float32, shape=(None, None, d_in_B))
    X = tf.placeholder(tf.float32, shape=(None, None, d_out))

    n_samples = tf.shape(A)[0]
    n_nodes = tf.shape(B)[1]

    # Extract indices from A matrix
    indices_from = tf.cast(A[:,:,0], tf.int32)                  # tf.int32, [n_samples, n_edges, 1]
    indices_to = tf.cast(A[:,:,1], tf.int32)                    # tf.int32, [n_samples, n_edges, 1]

    # Extact edge characteristics from A matrix
    A_ij = A[:,:,2:]                                            # tf.float32, [n_samples, n_edge, d_in_A]

    # Gather X on both sides of each edge
    X_i = custom_gather(X, indices_from)                        # tf.float32, [n_samples , n_edges, d_out]
    X_j = custom_gather(X, indices_to)                          # tf.float32, [n_samples , n_edges, d_out]

    # Compute interaction forces for each edge
    F_bar = forces.F_bar(X_i, X_j, A_ij)                   # tf.float32, [n_samples, n_edges, d_F]

    # Scatter the interaction forces from edges to nodes
    F_bar_sum = custom_scatter(indices_from, F_bar, [n_samples, n_nodes, d_F])      
                                                                # tf.float32, [n_samples, n_nodes, d_F]

    # Compute counteraction forces for each node
    F_round = forces.F_round(X, B)                         # tf.float32, [n_samples, n_nodes, d_F]

    # Sum all forces applied to each node to get the residual
    res = F_round + F_bar_sum                                   # tf.float32, [n_samples, n_nodes, d_F]

    # Compute the loss
    loss = tf.reduce_mean(res**2)

    
    # Initialize the session
    sess = tf.Session()

    # Compute the loss and some statistics about the residual for all 3 modes
    modes = ['train', 'val', 'test']

    for mode in modes:

        feed_dict = {
            A : data['A_'+mode+'.npy'],
            B : data['B_'+mode+'.npy'],
            X : data['X_'+mode+'.npy']
        }

        loss_np, res_np = sess.run([loss, res], feed_dict=feed_dict)

        print('In {} mode, we have :'.format(mode))
        print('    loss = {}'.format(loss_np))
        print('    mean residual = {}'.format(np.mean(res_np)))
        print('    std residual = {}'.format(np.std(res_np)))


if __name__ == '__main__':

    print('Testing the dataset...')
    print('This may take a while...')

    print('Testing that all required files exist, and importing them...')
    data, forces, dims = test_all_files_exist_and_import()
    print('    [OK] All required files are there, and loaded')

    print('Testing that each file has the same amount of samples...')
    test_n_samples(data)
    print('    [OK] All files have the same amount of samples')

    print('Testing that all matrices are rank 3...')
    test_rank(data)
    print('    [OK] All files are rank 3')

    print('Testing that dimensions specified in problem.py are consistent...')
    test_problem_dimensions(data, dims)
    print('    [OK] All provided dimensions match the data')

    print('Testing that the number of nodes is consistent...')
    test_n_nodes(data)
    print('    [OK] All number of nodes are consistent')

    print('Testing that .npy and .tfrecords files are identical...')
    test_np_tfrecords(data)
    print('    [OK] All .npy and .tfrecords are identical')

    print('Testing that all the origin and extremities of A matrices match with the amount of nodes...')
    test_or_ex_n_nodes(data)
    print('    [OK] All origins and extremities refer to existing nodes')

    print('Testing that the provided forces indeed create d_F equations...')
    test_d_F_equations(dims, forces)
    print('    [OK] Forces create d_F equations')

    print('Testing that X is solution to the problem formed by A, B and the forces...')
    test_solution(data, dims, forces)





