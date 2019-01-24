"""
A graph object is a neural network architecture defined to be trained in the target computing platform.
The computational graph applieds more than one operation to the weights of the neural network architecture.
"""

from typing import Sequence, NamedTuple

import tensorflow as tf

from diagnosenet.layers import Layer
from diagnosenet.losses import Loss
from diagnosenet.optimizers import Optimizer

from diagnosenet.monitor import Metrics

class FullyConnected:
    """
    Implements the back-propagation algorithm...
    Args: A neural network architecture defined by the user.
    Returns: A graph object with trainable parameters;
            that will be assign data in the executors class.
    """
    def __init__(self, input_size: int, output_size: int,
                        layers: Sequence[Layer],
                        loss: Loss,
                        optimizer: Optimizer,
                        dropout: float = 1.0) -> None:

        ## A neural network architecture:
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer
        self.dropout = dropout

        ## Graph object trainable parameters:
        self.mlp_graph = tf.Graph()
        self.X: tf.placeholder
        self.Y: tf.placeholder
        self.keep_prob: tf.placeholder
        self.projection: tf.Tensor
        self.mlp_loss: tf.Tensor
        self.mlp_grad_op: tf.Tensor

        ## metrics
        # self.metrics = Metrics()
        self.accuracy: tf.Tensor



    def stacked(self, input_holder, keep_prob) -> tf.Tensor:
        for i in range(len(self.layers)):
            ## Prevention to use dropout in the projection layer
            if len(self.layers)-1 == i:
                input_holder = self.layers[i].activation(input_holder)
            else:
                input_holder = self.layers[i].dropout_activation(input_holder, keep_prob)
        return input_holder

    def stacked_valid(self, input_holder) -> tf.Tensor:
        for layer in self.layers:
            print("layer: {}".format(layer.__class__.__name__))
            if layer.__class__.__name__ == "Dropout":
                pass
            else:
                input_holder = layer.activation(input_holder)
                print("layer: {}".format(input_holder))
        return input_holder


    def desktop_graph(self) -> tf.Tensor:
        with tf.Graph().as_default() as self.mlp_graph:
            self.X = tf.placeholder(tf.float32, shape=(None, self.input_size), name="Inputs")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="Output")
            self.keep_prob = tf.placeholder(tf.float32)

            self.projection = self.stacked(self.X, self.keep_prob)
            self.mlp_loss = self.loss.desktop_loss(self, self.projection, self.Y)
            self.mlp_grad_op = self.optimizer.desktop_Grad(self.mlp_loss)

            ## Accuracy
            # self.accuracy = Metrics().accuracy(self.Y, self.projection)
            # print("self.accuracy: {}".format(self.accuracy))

            ## # Convert prediction to one hot encoding
            self.soft_projection = tf.nn.softmax(self.projection)
            self.max_projection = tf.argmax(tf.nn.softmax(self.projection), 1)
            self.projection_1hot = tf.one_hot(self.max_projection, depth = int(self.output_size))


    def tower_loss(self, y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        """
        """
        loss = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

        ## Assemble all the losses for the current
        losses = tf.get_collection('losses', loss)

        ## Calculate the total loss for the current tower
        # total_loss = tf.add_n(losses(losses, name='total_loss'))
        total_loss = tf.add_n(losses)

        ##
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        with tf.control_dependencies([loss_averages_op]):
            total_loss = tf.identity(total_loss)
        return total_loss


    def multiGPU_loss(self, y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        # self.y_pred = y_pred
        # self.y_true = y_true
        # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.y_pred, labels = self.y_true))
        # return tf.sqrt(tf.reduce_mean(tf.square(self.y_true - self.y_pred)))

        ############################"
        ###############################"
        # print("y_pred: {}".format(y_pred))
        # print("y_true: {}".format(y_true))

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        cross_entropy_reduce = tf.reduce_mean(cross_entropy)

        tf.add_to_collection('losses', cross_entropy_reduce)

        # return tf.add_n(tf.get_collection('losses'), name='total_loss')
        return cross_entropy_reduce

        # tf.add_to_collection('losses', cross_entropy_mean)
        # return tf.add_n(tf.get_collection('losses'), name='total_loss')


    def multiGPU_graph(self, batch_size) -> tf.Tensor:

        with tf.Graph().as_default() as self.mlp_graph:
            self.num_gpus=2
            self.gpu_batch_size=int((batch_size/self.num_gpus))
            ###########################
            self.tower_grads = []
            self.mlp_losses = []

            self.X = tf.placeholder(tf.float32, shape=(None, self.input_size), name="Inputs")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="Targets")
            self.keep_prob = tf.placeholder(tf.float32)

            for gpu in range(self.num_gpus):
                print("gpu: {}".format(gpu))

                with tf.device('/gpu:%d' % gpu):
                    _X = self.X[(gpu * self.gpu_batch_size):
                                (gpu * self.gpu_batch_size) + (self.gpu_batch_size)]
                    _Y = self.Y[(gpu * self.gpu_batch_size):
                                (gpu * self.gpu_batch_size) + (self.gpu_batch_size)]

                    print("_X: {}, {}".format((gpu * self.gpu_batch_size),
                                            (gpu * self.gpu_batch_size) + (self.gpu_batch_size)))


                    # self.projection = self.stacked(_X)
                    self.projection = self.stacked(_X, self.keep_prob)

                    print("{}".format("+"*20))
                    print("self.projection: {}".format(self.projection))


                    # self.mlp_loss = self.loss.multiGPU_loss(self.projection, _Y)
                    self.mlp_loss = self.multiGPU_loss(self.projection, _Y)
                    self.mlp_grad_op = self.optimizer.desktop_Grad(self.mlp_loss)

                    self.mlp_losses.append(self.mlp_loss)

                    #############################################################
                    #############################################################


                    # self.loss = self.tower_loss(y_pred=self.projection, y_true=_Y)

                    self.tower_grads.append(self.mlp_grad_op)

                    ## Accuracy
                    # self.accuracy = Metrics().accuracy(_Y, self.projection)
                    # print("self.accuracy: {}".format(self.accuracy))

                    ## # Convert prediction to one hot encoding
                    self.soft_projection = tf.nn.softmax(self.projection)
                    self.max_projection = tf.argmax(tf.nn.softmax(self.projection), 1)
                    self.projection_1hot = tf.one_hot(self.max_projection, depth = int(self.output_size))



    # def multiGPU_graph(self) -> tf.Tensor:
    #
    #
    #     with tf.Graph().as_default() as self.mlp_graph:
    #
    #         self.num_gpus=2
    #         self.gpu_batch_size=50
    #
    #         self.X = tf.placeholder(tf.float32, shape=(None, self.input_size), name="Inputs")
    #         self.Y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="Output")
    #
    #         for gpu in range(self.num_gpus):
    #             print("gpu: {}".format(gpu))
    #
    #             with tf.device('/gpu:%d' % gpu):
    #                 _X = self.X[gpu * self.gpu_batch_size:
    #                                 gpu+1 * self.gpu_batch_size]
    #
    #                 _Y = self.Y[gpu * self.gpu_batch_size:
    #                                 gpu+1 * self.gpu_batch_size]
    #
    #
    #                 self.projection = self.stacked(_X)
    #                 self.mlp_loss = self.loss.desktop_loss(self, self.projection, _Y)
    #                 self.mlp_grad_op = self.optimizer.desktop_Grad(self.mlp_loss)
    #
    #                 ## Accuracy
    #                 self.accuracy = Metrics().accuracy(_Y, self.projection)
    #                 # print("self.accuracy: {}".format(self.accuracy))
    #
    #                 ## # Convert prediction to one hot encoding
    #                 self.soft_projection = tf.nn.softmax(self.projection)
    #                 self.max_projection = tf.argmax(tf.nn.softmax(self.projection), 1)
    #                 self.projection_1hot = tf.one_hot(self.max_projection, depth = int(self.output_size))
