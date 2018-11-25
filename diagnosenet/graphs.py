"""
A graph object is a neural network architecture defined to be trained in the target computing platform.
The computational graph applieds more than one operation to the weights of the neural network architecture.
"""

from typing import Sequence, NamedTuple

import tensorflow as tf 

from diagnosenet.layers import Layer

class MLP:
    """
    Implements the back-propagation algorithm...
    Args:
    Returns:
    """
    def __init__(self, layers: Sequence[Layer], input_size: int, output_size: int) -> None:
        self.layers = layers
        self.input_size = input_size
        self.output_size = output_size
        self.X: tf.placeholder
        self.mlp_graph = tf.Graph()
        
    def stacked(self, input_holder) -> tf.Tensor:
        for layer in self.layers:
            input_holder = layer.activation(input_holder)
        return input_holder
        
    def desktop_graph(self) -> tf.Tensor:
        with tf.Graph().as_default() as self.mlp_graph:
            self.X = tf.placeholder(tf.float32, shape=(None, self.input_size))
            activation = self.stacked(self.X)
        return activation