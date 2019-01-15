"""
Is an instantiation of a layer callable by the user.
Each layer needs to pass its dimension and their inputs to return the ops.
For example a neural network might look like:

Inputs -> Relu -> ReLu -> Linear -> Output
"""

from typing import Dict

import tensorflow as tf

class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, tf.Tensor] = {}

    def activation(self, input_holder) -> tf.Tensor:
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def activation(self, input_holder) -> tf.Tensor:
        self.params["w"] = tf.Variable(tf.random_normal([self.input_size, self.output_size], stddev=0.1), dtype=tf.float32)
        self.params["b"] = tf.Variable(tf.random_normal([self.output_size]), dtype=tf.float32)
        return tf.matmul(input_holder, self.params["w"]) + self.params["b"]

    def dropout_activation(self, input_holder, k_prob) -> tf.Tensor:
        self.params["w"] = tf.Variable(tf.random_normal([self.input_size, self.output_size], stddev=0.1), dtype=tf.float32)
        self.params["b"] = tf.Variable(tf.random_normal([self.output_size]), dtype=tf.float32)
        act_layer = tf.matmul(input_holder, self.params["w"]) + self.params["b"]
        return tf.nn.dropout(act_layer, k_prob)


class Relu(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def activation(self, input_holder) -> tf.Tensor:
        self.params["w"] = tf.Variable(tf.random_normal([self.input_size, self.output_size], stddev=0.1), dtype=tf.float32)
        self.params["b"] = tf.Variable(tf.random_normal([self.output_size]), dtype=tf.float32)
        return tf.nn.relu(tf.matmul(input_holder, self.params["w"]) + self.params["b"])

    def dropout_activation(self, input_holder, k_prob) -> tf.Tensor:
        self.params["w"] = tf.Variable(tf.random_normal([self.input_size, self.output_size], stddev=0.1), dtype=tf.float32)
        self.params["b"] = tf.Variable(tf.random_normal([self.output_size]), dtype=tf.float32)
        act_layer =  tf.nn.relu(tf.matmul(input_holder, self.params["w"]) + self.params["b"])
        return tf.nn.dropout(act_layer, k_prob)


class Softmax(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    def activation(self, input_holder) -> tf.Tensor:
        self.params["w"] = tf.Variable(tf.random_normal([self.input_size, self.output_size], stddev=0.1), dtype=tf.float32)
        self.params["b"] = tf.Variable(tf.random_normal([self.output_size]), dtype=tf.float32)
        return tf.nn.softmax(tf.matmul(input_holder, self.params["w"]) + self.params["b"])

    def dropout_activation(self, input_holder, k_prob) -> tf.Tensor:
        self.params["w"] = tf.Variable(tf.random_normal([self.input_size, self.output_size], stddev=0.1), dtype=tf.float32)
        self.params["b"] = tf.Variable(tf.random_normal([self.output_size]), dtype=tf.float32)
        act_layer =  tf.nn.softmax(tf.matmul(input_holder, self.params["w"]) + self.params["b"])
        return tf.nn.dropout(act_layer, k_prob)
