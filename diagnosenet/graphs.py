"""
A graph object is a neural network architecture defined to be trained in the target computing platform.
The computational graph applieds more than one operation to the weights of the neural network architecture.
"""

from typing import Sequence, NamedTuple

import tensorflow as tf

from diagnosenet.layers import Layer
from diagnosenet.losses import Loss
from diagnosenet.optimizers import Optimizer
from diagnosenet.metrics import Metrics

class MLP:
    """
    Implements the back-propagation algorithm...
    Args: A neural network architecture defined by the user.
    Returns: A graph object with trainable parameters;
            that will be assign data in the executors class.
    """
    def __init__(self, input_size: int, output_size: int,
                        layers: Sequence[Layer],
                        loss: Loss,
                        optimizer: Optimizer) -> None:

        ## A neural network architecture:
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.loss_fn = loss
        self.optimizer_fn = optimizer

        ## Graph object trainable parameters:
        self.mlp_graph = tf.Graph()
        self.X: tf.placeholder
        self.Y: tf.placeholder
        self.projection: tf.Tensor
        self.mlp_loss: tf.Tensor
        self.mlp_grad_op: tf.Tensor

        ## metrics
        # self.metrics = Metrics()
        self.accuracy: tf.Tensor


    def stacked(self, input_holder) -> tf.Tensor:
        for layer in self.layers:
            input_holder = layer.activation(input_holder)
        return input_holder


    def desktop_graph(self) -> tf.Tensor:
        with tf.Graph().as_default() as self.mlp_graph:
            self.X = tf.placeholder(tf.float32, shape=(None, self.input_size), name="Inputs")
            self.Y = tf.placeholder(tf.float32, shape=(None, self.output_size), name="Output")
            self.projection = self.stacked(self.X)
            self.mlp_loss = self.loss_fn.desktop_loss(self, self.projection, self.Y)
            self.mlp_grad_op = self.optimizer_fn.desktop_Grad(self.mlp_loss)

            self.accuracy = Metrics().accuracy(self.Y, self.projection)
            print("self.accuracy: {}".format(self.accuracy))
