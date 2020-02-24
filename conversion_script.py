import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm


dir_names = ['circle', 'polygon', 'regular_square', 'irregular_square']
modes = ['train', 'val', 'test']

origin_dir = 'datasets/poisson/'
target_dir = 'datasets/poisson_alt/'

for dir_name in dir_names:
    
    for mode in modes:
        
        print('Processing {} in {} mode...'.format(dir_name, mode))

        path_origin = os.path.join(origin_dir, dir_name)
        path_target = os.path.join(target_dir, dir_name)
        
        A_np = np.load(os.path.join(path_origin, 'A_'+mode+'.npy'))
        B_np = np.load(os.path.join(path_origin, 'B_'+mode+'.npy'))
        X_np = np.load(os.path.join(path_origin, 'X_'+mode+'.npy'))
        
        B_new = []
        A_new = []
        
        for sample in tqdm(range(np.shape(A_np)[0])):
            A_sample = A_np[sample]
            B_sample = B_np[sample]
            X_sample = X_np[sample]

            indices_from = A_sample[:,0]
            indices_to = A_sample[:,1]
            a_ij = A_sample[:,2]

            n_nodes = np.shape(B_sample)[0]

            # Get list of indices for which a_ii == 1.0
            indices_1 = indices_from[a_ij * (indices_from==indices_to) == 1.0]
            indices_1 = np.unique(indices_1).astype(np.int32)

            B_new_sample = np.zeros([n_nodes, 3])
            B_new_sample[:,0] = B_sample[:,0]
            B_new_sample[indices_1,0] = 0
            B_new_sample[indices_1,1] = 1.
            B_new_sample[indices_1,2] = B_sample[indices_1,0]

            B_new.append(B_new_sample)


            indices_loop = (indices_from!=indices_to)
            A_new_sample = A_sample[indices_loop]

            A_new.append(A_new_sample)

        max_edges = 0
        for A_new_sample in A_new:
            max_edges = max(max_edges, np.shape(A_new_sample)[0])
            
            
        A_new_new = []
        for A_new_sample in A_new:
            n_edges = np.shape(A_new_sample)[0]
            n_edges_missing = max_edges - n_edges
            d_in_A = np.shape(A_new_sample)[1]
            A_new_sample = np.r_[A_new_sample, np.zeros([n_edges_missing, d_in_A])]
            
            A_new_new.append(A_new_sample)
            
        A_new = np.array(A_new_new)
        B_new = np.array(B_new)
        
        np.save(os.path.join(path_target, 'A_'+mode+'.npy'), A_new)
        np.save(os.path.join(path_target, 'B_'+mode+'.npy'), B_new)
        np.save(os.path.join(path_target, 'X_'+mode+'.npy'), X_np)


__author__ = "Sangwoong Yoon"

def np_to_tfrecords(A, B, X, file_path_prefix, verbose=True):
    """
    author : "Sangwoong Yoon"
    """
    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:  
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))
            
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2
    
    assert isinstance(B, np.ndarray)
    assert len(B.shape) == 2
    
    assert isinstance(X, np.ndarray)
    assert len(X.shape) == 2
    
    # load appropriate tf.train.Feature class depending on dtype
    dtype_feature_a = _dtype_feature(A)
    dtype_feature_b = _dtype_feature(B)
    dtype_feature_x = _dtype_feature(X)      
        
    # Generate tfrecord writer
    result_tf_file = file_path_prefix + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(result_tf_file)
    if verbose:
        print("Serializing {:d} examples into {}".format(X.shape[0], result_tf_file))
        
    # iterate over each sample,
    # and serialize it as ProtoBuf.
    for idx in tqdm.tqdm(range(A.shape[0])):
        a = A[idx]
        b = B[idx]
        x = X[idx]
        
        d_feature = {}
        d_feature['A'] = dtype_feature_a(a)
        d_feature['B'] = dtype_feature_b(b)
        d_feature['X'] = dtype_feature_x(x)
        
            
        features = tf.train.Features(feature=d_feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
    
    if verbose:
        print("Writing {} done!".format(result_tf_file))

     

for dir_name in dir_names:

    path_target = os.path.join(target_dir, dir_name)
    
    for mode in modes:

        A = np.load(os.path.join(path_target, 'A_'+mode+'.npy'), allow_pickle=True)
        B = np.load(os.path.join(path_target, 'B_'+mode+'.npy'), allow_pickle=True)
        X = np.load(os.path.join(path_target, 'X_'+mode+'.npy'), allow_pickle=True)

        n_samples = np.array(np.shape(A)[0])

        A = np.reshape(A, [n_samples, -1])
        B = np.reshape(B, [n_samples, -1])
        X = np.reshape(X, [n_samples, -1])
        
        print(A)

        np_to_tfrecords(A, B, X, os.path.join(path_target, mode), 
            verbose=True)

    # Copy the desired force template
    src_force = 'problem_templates/problem.py'
    dst_force = os.path.join(path_target, 'problem.py')
    new_dst_force = os.path.join(path_target, 'problem.py')
    shutil.copy(src_force, path_target)
    os.rename(dst_force, new_dst_force)


