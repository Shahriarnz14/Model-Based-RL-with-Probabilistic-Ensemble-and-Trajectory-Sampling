import tensorflow as tf
from keras.layers import Dense, Flatten, Input, Concatenate, Lambda, Activation
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import numpy as np
from util import ZFilter

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400


class PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        self.sess = tf.Session()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        K.set_session(self.sess)

        # Log variance bounds
        self.max_logvar = tf.Variable(-3 * np.ones([1, self.state_dim]), dtype=tf.float32)
        self.min_logvar = tf.Variable(-7 * np.ones([1, self.state_dim]), dtype=tf.float32)

        # Create and initialize the model
        self.learning_rate = learning_rate
        self.models = [self.create_network() for _ in range(self.num_nets)]

        self.trainable_weights = [network.trainable_weights for network in self.models]
        self.inputs = [network.input for network in self.models]
        self.outputs = [self.get_output(network.output) for network in self.models]
        self.targets = [tf.placeholder(dtype=tf.float32,
                                       shape=[None, self.state_dim],
                                       name="training_targets") for _ in range(self.num_nets)]

        self.gauss_loss = [self.compile_loss(output, target, is_rmse=False) for (output, target) in zip(self.outputs, self.targets)]
        self.rmse_loss = [self.compile_loss(output, target, is_rmse=True) for (output, target) in zip(self.outputs, self.targets)]

        self.train_op = [tf.train.AdamOptimizer(self.learning_rate).minimize(loss, var_list=weights)
                         for (loss, weights) in zip(self.gauss_loss, self.trainable_weights)]

        self.sess.run(tf.global_variables_initializer())

        self.log_time_stamp = str(np.datetime64('now'))
        self.log_gauss_loss = []
        self.log_rmse_loss = []

    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
        return mean, logvar

    def create_network(self):
        I = Input(shape=[self.state_dim + self.action_dim], name='input')
        h1 = Dense(HIDDEN1_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(I)
        h2 = Dense(HIDDEN2_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h1)
        h3 = Dense(HIDDEN3_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h2)
        O = Dense(2 * self.state_dim, activation='linear', kernel_regularizer=l2(0.0001))(h3)
        model = Model(input=I, output=O)
        return model

    def train(self, inputs, targets, batch_size=128, epochs=5):
        """
        Arguments:
          inputs: state and action inputs.  Assumes that inputs are standardized.
          targets: resulting states
        """

        gauss_losses = np.zeros((epochs,))
        rmse_losses = np.zeros((epochs,))

        for e in range(epochs):
            indices = np.random.randint(inputs.shape[0], size=[self.num_nets, inputs.shape[0]])

            for transition_idx in range(0, inputs.shape[0], batch_size):
                inputs_batch = [inputs[indices[n, transition_idx:min(inputs.shape[0], batch_size+transition_idx)]] for n in range(self.num_nets)]
                targets_batch = [targets[indices[n, transition_idx:min(inputs.shape[0], batch_size+transition_idx)]]
                                 for n in range(self.num_nets)]

                feed_dictionary = {input_net: state_action for (input_net, state_action) in zip(self.inputs, inputs_batch)}
                feed_dictionary.update({yhat: y for (yhat,y) in zip(self.targets, targets_batch)})

                _, gauss_loss, rmse_loss = self.sess.run([self.train_op, self.gauss_loss, self.rmse_loss], feed_dict=feed_dictionary)

                gauss_losses[e] += np.mean(gauss_loss)
                rmse_losses[e] += np.mean(rmse_loss)
            # end_for
            # print('Epoch: %d\t|\tGauss Loss: %.2f\t|\tRMSE: %.2f' % (e, gauss_losses[e], rmse_losses[e]))
            self.log_gauss_loss.append(gauss_losses[e])
            self.log_rmse_loss.append(rmse_losses[e])
        # end_for



    def compile_loss(self, outputs, targets, is_rmse=False):
        """Helper method for compiling the loss function.
        The loss function is obtained from the log likelihood, assuming that the output
        distribution is Gaussian, with both mean and (diagonal) covariance matrix being determined
        by network outputs.
        Arguments:
            outputs: (tf.Tensor) A tensor representing the input batch
            targets: (tf.Tensor) The desired targets for each input vector in inputs.
            is_rmse: (bool) If False, includes log variance loss.
        Returns: (tf.Tensor) A tensor representing the loss on the input arguments.
        """
        # print(outputs)
        # print(type(outputs))
        # print(outputs.shape)
        # mean, log_var = self.get_output(outputs)
        mean = outputs[0]
        log_var = outputs[1]
        # print(mean)
        # print(log_var)
        inv_var = tf.exp(-log_var)

        if is_rmse:
            total_losses = tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.square(mean - targets), axis=-1), axis=-1))
        else:
            mse_losses = tf.reduce_mean(tf.square(mean - targets) * inv_var, axis=-1)
            # mse_losses = tf.reduce_mean(tf.reduce_sum(tf.square(mean - targets) * inv_var, axis=-1), axis=-1)
            var_losses = tf.reduce_mean(tf.reduce_sum(log_var, axis=-1), axis=-1)
            total_losses = 0.5 * (mse_losses + var_losses)

        return total_losses
