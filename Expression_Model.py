import numpy as np
import tensorflow as tf
import pandas as pd
from Seq import Seq
from tensorflow.contrib.slim import fully_connected
import pickle

class Expression_Model(object):

    def __init__(self, sequence_length):

        self.sLen_ = sequence_length
        self.w_bind = []
        self.w_bind_initilized = False
        self.w_coop = []
        self.w_coop_initilized = False
        self.sess = tf.Session()

        self.build_graph()
        self.saver = tf.train.Saver()

    def build_graph(self):

        self.seq_placeholder = tf.placeholder(tf.float32, [None, self.sLen_, 4])
        self.TF_conc_placeholder = tf.placeholder(tf.float32, [None, 3])
        self.rho_expr_placeholder = tf.placeholder(tf.float32, [None,1])
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [])
        self.training_placeholder = tf.placeholder(tf.bool, [])
        self.drop_seq_placeholder = tf.placeholder(tf.float32, [])
        self.drop_motifs_placeholder = tf.placeholder(tf.float32, [])
        self.regScale_motifs_placeholder = tf.placeholder(tf.float32, [])
        self.regScale_coops_placeholder = tf.placeholder(tf.float32, [])
        self.regScale_NN_placeholder = tf.placeholder(tf.float32, [])

        self.binding = self.build_binding_features(self.seq_placeholder, self.TF_conc_placeholder, drop_rate_seq = self.drop_seq_placeholder, drop_rate_motifs = self.drop_motifs_placeholder, reg_scale_motifs = self.regScale_motifs_placeholder, training = self.training_placeholder)
        self.binding_combined = self.combine_pooled_binding(self.binding)
        self.coop = self.build_coop_features(self.binding, reg_scale_coops = self.regScale_coops_placeholder)
        self.coop_combined = self.combine_pooled_coop(self.coop)
        self.final_features = self.concat_all_features(self.binding_combined, self.coop_combined)
        self.forward = self.NN(self.final_features, reg_scale_NN = self.regScale_NN_placeholder)
        self.loss = self.calculate_loss(self.rho_expr_placeholder, self.forward)
        self.updateOpt = self.update_optimizer(self.loss, self.learning_rate_placeholder)
        self.init = tf.global_variables_initializer()

    def initilize_w_bind(self, dim_pooled):
        init_w_bind = tf.ones([dim_pooled,3],dtype=tf.float32)
        self.w_bind = tf.Variable(init_w_bind, dtype=tf.float32, name = "binding_weights")
        self.w_bind_initilized = True

    def initilize_w_coop(self, dim_pooled):
        init_w_coop = tf.ones([dim_pooled,6],dtype=tf.float32)
        self.w_coop = tf.Variable(init_w_coop, dtype=tf.float32, name = "coop_weights")
        self.w_coop_initilized = True

    def build_binding_features(self, encoded_seq, TF_conc, drop_rate_seq = 0.0, drop_rate_motifs = 0.15, reg_scale_motifs = 0.005, training = True):
        """
        This function applies 3 convolutional filters corresponding to
        the main 3 regulators of rho (i.e. dorsal, twist, snail).

        Inputs:
            one-hot encoded sequence = tf.tensor of size batch_size * sequence_length * 4
            TF_conc: tf.tensor of size nBatch * 3
        Returns: shorter representation of sequence after convolution and pooling = batch_size * Np * #Filters
        """

        dropout_seq = tf.layers.dropout(encoded_seq, rate = drop_rate_seq, training = training)
        #dropout_seq = encoded_seq
        conved_seq = tf.layers.conv1d(inputs = dropout_seq, activation = tf.nn.relu, use_bias = False, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = reg_scale_motifs), filters = 3, kernel_size = 10, name = "motifs")
        pooled_conved_seq = tf.layers.max_pooling1d(inputs = conved_seq, pool_size = 20, strides = 20)
        pooled_conved_seq = tf.layers.dropout(pooled_conved_seq, rate = drop_rate_motifs, training = training)

        nPooled = pooled_conved_seq.shape[1]
        if not self.w_bind_initilized:
            self.initilize_w_bind(nPooled)

        ### multiply TF TF_concentration:
        conc = tf.tile(TF_conc, [1,nPooled])
        conc = tf.reshape(conc, (-1, nPooled, 3))
        pooled_conved_seq = tf.multiply(pooled_conved_seq, conc)

        return pooled_conved_seq

    def combine_pooled_binding(self, pooled_conved_seq):
        """
        This function linearly combines pooled binding features.

        Input: pooled_conved_seq; output of build_binding_features
        Return: Tensor nbatch * 3 (3 shows number filters)
        """

        final_binding_features = tf.multiply(pooled_conved_seq, self.w_bind)
        #final_binding_features = pooled_conved_seq
        final_binding_features = tf.reduce_sum(final_binding_features, axis = 1)

        return final_binding_features

    def build_coop_features(self, pooled_conved_seq, reg_scale_coops = 0.0001):
        """
        This function applies 6 convolutional filters corresponding to
        the 4 cooperative and 2 repressive interaction between TFs

        Input: pooled_conved_seq; output of build_binding_features
        Returns: Tensor nbatch * Np2 * 6
        """

        c11 = tf.stack((pooled_conved_seq[:,:,0], pooled_conved_seq[:,:,0]), axis = 2)
        c22 = tf.stack((pooled_conved_seq[:,:,1], pooled_conved_seq[:,:,1]), axis = 2)
        c33 = tf.stack((pooled_conved_seq[:,:,2], pooled_conved_seq[:,:,2]), axis = 2)
        c12 = tf.stack((pooled_conved_seq[:,:,0], pooled_conved_seq[:,:,1]), axis = 2)
        c13 = tf.stack((pooled_conved_seq[:,:,0], pooled_conved_seq[:,:,2]), axis = 2)
        c23 = tf.stack((pooled_conved_seq[:,:,1], pooled_conved_seq[:,:,2]), axis = 2)

        conved_c11 = tf.layers.conv1d(inputs = c11, filters = 1, activation = tf.nn.tanh, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = reg_scale_coops), kernel_size = 5, name = "c11")
        conved_c22 = tf.layers.conv1d(inputs = c22, filters = 1, activation = tf.nn.tanh, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = reg_scale_coops), kernel_size = 5, name = "c22")
        conved_c33 = tf.layers.conv1d(inputs = c33, filters = 1, activation = tf.nn.tanh, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = reg_scale_coops), kernel_size = 5, name = "c33")
        conved_c12 = tf.layers.conv1d(inputs = c12, filters = 1, activation = tf.nn.tanh, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = reg_scale_coops), kernel_size = 5, name = "c12")
        conved_c13 = tf.layers.conv1d(inputs = c13, filters = 1, activation = tf.nn.tanh, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = reg_scale_coops), kernel_size = 5, name = "c13")
        conved_c23 = tf.layers.conv1d(inputs = c23, filters = 1, activation = tf.nn.tanh, kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = reg_scale_coops), kernel_size = 5, name = "c23")

        # The below tensor has dimension nbatch * N_reduced_conv * 6 (6 represents the 6 interaction convolution filters)
        conved_c = tf.stack((conved_c11[:,:,0], conved_c22[:,:,0], conved_c33[:,:,0], conved_c12[:,:,0], conved_c13[:,:,0], conved_c23[:,:,0]), axis = 2)
        pooled_conved_c = tf.layers.max_pooling1d(inputs = conved_c, pool_size = 3, strides = 3)
        #pooled_conved_c = tf.nn.softplus(pooled_conved_c)
        #pooled_conved_c = conved_c
        if not self.w_coop_initilized:
            self.initilize_w_coop(pooled_conved_c.shape[1])

        return pooled_conved_c

    def combine_pooled_coop(self, pooled_conved_c):
        """
        This function linearly combines pooled cooperativity (both coop and repression) features.

        Input: pooled_conved_c; output of build_coop_features
        Return: Tensor nbatch * 6 (6 shows number of coop filters)
        """

        final_coop_features = tf.multiply(pooled_conved_c, self.w_coop)
        #final_coop_features = pooled_conved_c
        final_coop_features = tf.reduce_sum(final_coop_features, axis = 1)

        return final_coop_features

    def concat_all_features(self, final_binding_features, final_coop_features):
        """
        This returns the final features tensor of dimension nbatch * 9 that will be passed to the NN
        """
        return tf.concat((final_binding_features, final_coop_features), axis = 1)

    def NN(self, final_features, reg_scale_NN = 0.0001):
        """
        Builds a one layer network of fully connected layers, with 3 hidden nodes,
        and outputs expression.


        Input(9) --> hidden(3) --> expression(1)

        Args:
            final_features (tf.Tensor): The input tensor of dimension (None, 9).
        Returns:
            expression(tf.Tensor): of dimension (None, 1).
        """
        hiddenLayer = fully_connected(final_features, 3, activation_fn = tf.nn.softplus, weights_regularizer = tf.contrib.layers.l2_regularizer(scale = reg_scale_NN))
        output = fully_connected(hiddenLayer, 1, activation_fn = tf.nn.sigmoid, weights_regularizer = tf.contrib.layers.l2_regularizer(scale = reg_scale_NN))

        return output

    def calculate_loss(self, expr_true, expr_predicted):
        """
        Calculates MSE loss between ground_truth and predicition
        Args:
            expr_true(tf.Tensor): Tensor of shape (batch_size,1) containing ground truth
            expr_prediction(tf.Tensor): Tensor of shape (batch_size,1) containing model prediction

        Returns:
            MSE loss (tf.tensor)
        """
        return tf.reduce_mean(tf.squared_difference(expr_true, expr_predicted))

    def update_optimizer(self, loss, learning_rate):
        """
        Updates optimization
        Args:
            loss(tf.Tensor): Tensor of shape () containing the loss function.
            learning_rate(tf.Tensor): Tensor of shape (). Learning rate for
                gradient descent.
        Returns:
            (tf.Operation): Update optimizer tensorflow operation.
        """
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        return optimizer.minimize(loss = loss)

    def save_model(self, name):
        self.saver.save(self.sess, name)

    def get_var_names(self):
        """
        This returns all the varibale names
        """
        ret = [v.name for v in tf.trainable_variables()]
        return ret

    def get_vars(self, name):
        """
        This return the variables of the given variable name
        """
        var = [self.sess.run(v) for v in tf.trainable_variables() if v.name == name]
        return var[0]

    def write_variables(self, output_name):
        var_names = self.get_var_names()
        vars = {}
        for n in var_names:
            vars[n] = self.get_vars(n)

        f = open(output_name,"wb")
        pickle.dump(vars,f)
        f.close()

    def train(self, data, learning_rate, batch_size, num_steps, drop_rate_seq = 0, drop_rate_motifs = 0.15, reg_scale_motifs = 0.005, reg_scale_coops = 0.0001, reg_scale_NN = 0.0001, num_iter_batch = 20, print_performance = True, output_name_Tensorflow = None, output_name_train_err = None, output_name_valid_err = None, output_name_test_err = None, output_name_pars_dict = None):
        """
        Trains the model
        Args:
            data(Seq object)
            learning_rate(float): Learning rate.
            batch_size(int): Batch size used for training.
            num_steps(int): Number of steps to run the updateOpt.
        """

        self.sess.run(self.init)
        #print np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        train_seq, train_TF, train_rho = data.next_batch(all_data = 'train')
        test_seq, test_TF, test_rho = data.next_batch(all_data = 'test')
        valid_seq, valid_TF, valid_rho = data.next_batch(all_data = 'valid')
        train_rho = np.reshape(train_rho, (np.shape(train_rho)[0], 1))
        test_rho = np.reshape(test_rho, (np.shape(test_rho)[0], 1))
        valid_rho = np.reshape(valid_rho, (np.shape(valid_rho)[0], 1))

        err_train = []
        err_test = []
        err_valid = []

        for step in range(0, num_steps):
            batch_seq, batch_TF_conc, batch_true_label = data.next_batch(all_data = None, size = batch_size)
            batch_true_label = np.reshape(batch_true_label, (np.shape(batch_true_label)[0], 1))

            for epoc in range(num_iter_batch):
                self.sess.run(self.updateOpt, feed_dict={self.seq_placeholder: batch_seq, self.TF_conc_placeholder: batch_TF_conc, self.rho_expr_placeholder: batch_true_label, self.learning_rate_placeholder: learning_rate, self.training_placeholder: True, self.drop_seq_placeholder : drop_rate_seq, self.drop_motifs_placeholder: drop_rate_motifs, self.regScale_motifs_placeholder: reg_scale_motifs, self.regScale_coops_placeholder: reg_scale_coops, self.regScale_NN_placeholder: reg_scale_NN})

            if step % 100 == 0:
                curr_err_train = self.sess.run(self.loss, feed_dict={self.seq_placeholder: train_seq, self.TF_conc_placeholder: train_TF, self.rho_expr_placeholder: train_rho, self.training_placeholder: False, self.drop_seq_placeholder : drop_rate_seq, self.drop_motifs_placeholder: drop_rate_motifs, self.regScale_motifs_placeholder: reg_scale_motifs, self.regScale_coops_placeholder: reg_scale_coops, self.regScale_NN_placeholder: reg_scale_NN})
                curr_err_test = self.sess.run(self.loss, feed_dict={self.seq_placeholder: test_seq, self.TF_conc_placeholder: test_TF, self.rho_expr_placeholder: test_rho, self.training_placeholder: False, self.drop_seq_placeholder : drop_rate_seq, self.drop_motifs_placeholder: drop_rate_motifs, self.regScale_motifs_placeholder: reg_scale_motifs, self.regScale_coops_placeholder: reg_scale_coops, self.regScale_NN_placeholder: reg_scale_NN})
                curr_err_valid = self.sess.run(self.loss, feed_dict={self.seq_placeholder: valid_seq, self.TF_conc_placeholder: valid_TF, self.rho_expr_placeholder: valid_rho, self.training_placeholder: False, self.drop_seq_placeholder : drop_rate_seq, self.drop_motifs_placeholder: drop_rate_motifs, self.regScale_motifs_placeholder: reg_scale_motifs, self.regScale_coops_placeholder: reg_scale_coops, self.regScale_NN_placeholder: reg_scale_NN})

                if print_performance:
                    print ("Step: " + str(step))
                    print ("Current train error:")
                    print (curr_err_train)
                    print ("Current validation error:")
                    print (curr_err_valid)
                    print ("Current test error:")
                    print (curr_err_test)


                err_train.append(curr_err_train)
                err_test.append(curr_err_test)
                err_valid.append(curr_err_valid)

        final_err_train = self.sess.run(self.loss, feed_dict={self.seq_placeholder: train_seq, self.TF_conc_placeholder: train_TF, self.rho_expr_placeholder: train_rho, self.training_placeholder: False, self.drop_seq_placeholder : drop_rate_seq, self.drop_motifs_placeholder: drop_rate_motifs, self.regScale_motifs_placeholder: reg_scale_motifs, self.regScale_coops_placeholder: reg_scale_coops, self.regScale_NN_placeholder: reg_scale_NN})
        final_err_test = self.sess.run(self.loss, feed_dict={self.seq_placeholder: test_seq, self.TF_conc_placeholder: test_TF, self.rho_expr_placeholder: test_rho, self.training_placeholder: False, self.drop_seq_placeholder : drop_rate_seq, self.drop_motifs_placeholder: drop_rate_motifs, self.regScale_motifs_placeholder: reg_scale_motifs, self.regScale_coops_placeholder: reg_scale_coops, self.regScale_NN_placeholder: reg_scale_NN})
        final_err_valid = self.sess.run(self.loss, feed_dict={self.seq_placeholder: valid_seq, self.TF_conc_placeholder: valid_TF, self.rho_expr_placeholder: valid_rho, self.training_placeholder: False, self.drop_seq_placeholder : drop_rate_seq, self.drop_motifs_placeholder: drop_rate_motifs, self.regScale_motifs_placeholder: reg_scale_motifs, self.regScale_coops_placeholder: reg_scale_coops, self.regScale_NN_placeholder: reg_scale_NN})
        err_train.append(final_err_train)
        err_test.append(final_err_test)
        err_valid.append(final_err_valid)
        if print_performance:
            print ("Final train error:")
            print (final_err_train)
            print ("Final validation error:")
            print (final_err_valid)
            print ("Final test error:")
            print (final_err_test)


        if output_name_Tensorflow != None:
            self.save_model(output_name_Tensorflow)
        if output_name_pars_dict != None:
            self.write_variables(output_name_pars_dict)
        if output_name_train_err != None:
            df = pd.DataFrame(err_train)
            df.to_csv(output_name_train_err, header = 0, index = 0)
        if output_name_valid_err != None:
            df = pd.DataFrame(err_valid)
            df.to_csv(output_name_valid_err, header = 0, index = 0)
        if output_name_test_err != None:
            df = pd.DataFrame(err_test)
            df.to_csv(output_name_test_err, header = 0, index = 0)

    def predict(self, seq, TF, rho):
        res = self.sess.run(self.forward, feed_dict={self.seq_placeholder: seq, self.TF_conc_placeholder: TF, self.rho_expr_placeholder: rho, self.training_placeholder: False, self.drop_seq_placeholder : 0, self.drop_motifs_placeholder: 0, self.regScale_motifs_placeholder: 0, self.regScale_coops_placeholder: 0, self.regScale_NN_placeholder: 0})
        return res
