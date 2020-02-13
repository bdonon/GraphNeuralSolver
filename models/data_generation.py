import sys
import tensorflow as tf

class DataGenerator():
    """
    Class of tensorflow computational graph that builds a dataset of tuples (A,X,B) such that AX=B
    """

    def __init__(self,
        sess,
        name='DataGenerator',
        n_nodes=10,
        n_edges=15,
        p_cons=0.1,
        n_samples=1,
        a_distrib=None,
        x_c_distrib=None,
        b_distrib=None
        ):

        self.sess = sess

        # Define dataset and graph sizes
        self.n_nodes = tf.Variable(n_nodes, name='n_nodes')
        self.n_edges = tf.Variable(n_edges, name='n_edges')
        self.n_samples = tf.Variable(n_samples, name='n_samples')

        # Define proba for a node to be constrained
        self.p_cons = tf.Variable(p_cons, name='p_cons')

        # Initialize those 'root' variables
        self.sess.run(tf.initialize_variables([
            self.n_nodes,
            self.n_edges,
            self.n_samples,
            self.p_cons]))

        # Build a sparse representation of a tree
        self.edges_up_left = tf.range(1, self.n_nodes)* tf.ones([self.n_samples,1], tf.int32)

        self.edges_up_right = tf.cast(tf.range(0, self.n_nodes-1, 1), tf.float32)
        self.edges_up_right = tf.reshape(self.edges_up_right, [1, -1]) * tf.ones([self.n_samples, 1])
        self.edges_up_right = self.edges_up_right * \
            tf.random.uniform(shape=[self.n_samples, self.n_nodes-1], minval=0., maxval=0.999)
        self.edges_up_right = tf.math.floor(self.edges_up_right)
        self.edges_up_right = tf.cast(self.edges_up_right, tf.int32)

        self.edges_up = tf.stack([self.edges_up_left, self.edges_up_right], 2)


        self.edges_down = tf.random.uniform([self.n_samples, self.n_edges-self.n_nodes+1, 2], 
            minval=0, 
            maxval=self.n_nodes-1, 
            dtype=tf.int32)

        self.edges = tf.concat([self.edges_up, self.edges_down], 1)

        if a_distrib[0] == 'uniform':
            self.stiffness = tf.random.uniform([self.n_samples, self.n_edges, 1], 
                minval=a_distrib[1], 
                maxval=a_distrib[2])

        elif a_distrib[0] == 'normal':
            self.stiffness = tf.random.normal([self.n_samples, self.n_edges, 1], 
                mean=a_distrib[1], 
                stddev=a_distrib[2])

        else:
            sys.exit('Not a valid distribution for a')


        # Ensure that stiffnesses are negative
        self.stiffness = - tf.math.abs(self.stiffness)

        self.A = tf.concat([tf.dtypes.cast(self.edges, tf.float32), self.stiffness], 2)

        # Add symmetric edges
        self.A_transpose = tf.stack([self.A[:,:,1], self.A[:,:,0], self.A[:,:,2]], 2)
        self.A = tf.concat([self.A, self.A_transpose], 1)

        # Add diagonal components
        self.indices_offset = tf.linspace(0., tf.cast(self.n_samples-1, tf.float32), self.n_samples)
        self.indices_offset = self.indices_offset
        self.indices_offset = tf.reshape(self.indices_offset, [-1, 1,1])
        self.indices_offset = self.indices_offset * tf.ones([1,self.n_edges*2,1])

        self.indices_scatter = tf.concat([self.indices_offset, self.A[:,:,:1]], axis=2)    

        self.A_loop = tf.scatter_nd(tf.cast(self.indices_scatter, tf.int32), -self.A[:,:,2], [tf.shape(self.A)[0], self.n_nodes])

        self.indices_loop = tf.linspace(0., tf.cast(self.n_nodes-1, tf.float32), self.n_nodes)
        self.indices_loop = tf.reshape(self.indices_loop, [1,-1]) * tf.ones([self.n_samples,1])

        self.A_loop = tf.stack([self.indices_loop, self.indices_loop, self.A_loop], axis=2)

        self.A = tf.concat([self.A, self.A_loop], axis=1)

        # Samples nodes that are constrained
        self.is_constrained = tf.random.categorical(tf.math.log([[1-self.p_cons, self.p_cons]]), self.n_nodes*self.n_samples)
        self.is_constrained = tf.reshape(self.is_constrained, [self.n_samples, self.n_nodes])
        # Let's always constrain node 0
        self.is_constrained = tf.concat([tf.cast(tf.ones([self.n_samples,1]), tf.int64), self.is_constrained[:,1:]], axis=1)

        #tf.random.uniform(minval=0, maxval=2, shape=[size, n], dtype=tf.int32)

        self.indices_offset = tf.linspace(0., tf.cast(self.n_samples-1, tf.float32), self.n_samples)
        self.indices_offset = self.indices_offset
        self.indices_offset = tf.reshape(self.indices_offset, [-1, 1,1])
        self.indices_offset = self.indices_offset * tf.ones([1,self.n_edges*2+self.n_nodes,1])

        self.indices_scatter = tf.concat([self.indices_offset, self.A[:,:,:1]], axis=2)   

        self.is_constrained_edge = tf.gather_nd(self.is_constrained, tf.cast(self.indices_scatter, tf.int32))
        self.is_self_loop_edge = 1.*tf.cast(tf.equal(self.A[:,:,0], self.A[:,:,1]), tf.float32)

        self.is_constrained_and_loop = tf.cast(self.is_constrained_edge, tf.float32) * self.is_self_loop_edge
        self.is_not_constrained_edge = 1. - tf.cast(self.is_constrained_edge, tf.float32)

        self.edges = self.A[:,:,2]*self.is_not_constrained_edge + 1.*self.is_constrained_and_loop
        self.A = tf.stack([self.A[:,:,0], self.A[:,:,1], self.edges], axis=2)

        # Sample external forces
        if b_distrib[0] == 'uniform':
            self.B = tf.random.uniform([self.n_samples, self.n_nodes, 1], 
                minval=b_distrib[1], 
                maxval=b_distrib[2])

        elif b_distrib[0] == 'normal':
            self.B = tf.random.normal([self.n_samples, self.n_nodes, 1], 
                mean=b_distrib[1], 
                stddev=b_distrib[2])

        else:
            sys.exit('Not a valid distribution for b')

        # Sample external forces
        if x_c_distrib[0] == 'uniform':
            self.X_c = tf.random.uniform([self.n_samples, self.n_nodes, 1], 
                minval=x_c_distrib[1], 
                maxval=x_c_distrib[2])

        elif x_c_distrib[0] == 'normal':
            self.X_c = tf.random.normal([self.n_samples, self.n_nodes, 1], 
                mean=x_c_distrib[1], 
                stddev=x_c_distrib[2])

        else:
            sys.exit('Not a valid distribution for x_c')

        self.is_constrained = tf.reshape(self.is_constrained, [self.n_samples, -1, 1])

        self.B = self.B * (1. - tf.cast(self.is_constrained, tf.float32)) + \
            self.X_c * tf.cast(self.is_constrained, tf.float32)

        self.sample_index = tf.linspace(0., tf.cast(self.n_samples, tf.float32)-1., self.n_samples)
        self.sample_index = tf.reshape(self.sample_index, [-1,1,1])
        self.sample_index = self.sample_index * tf.ones([1, 2*self.n_edges+self.n_nodes, 1])
        self.A_with_sample_index = tf.concat([self.sample_index, self.A], 2)

        self.A_matrix = tf.scatter_nd(
            tf.cast(self.A_with_sample_index[:,:,:-1], tf.int32),
            self.A_with_sample_index[:,:,-1],
            [self.n_samples, self.n_nodes, self.n_nodes],
            name=None
        )

        self.A_matrix_inv = tf.linalg.inv(self.A_matrix)

        # Solve for X
        self.X = tf.matmul(self.A_matrix_inv, self.B)


















