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


class Conv1D(Layer):
    def __init__(self, input_size: int, input_length: int, dimension: int,
                    filter: tf.Tensor = None,) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_length = input_length
        self.dimension = dimension
        # self.k = tf.constant(kernel, dtype=tf.float32)

    def activation(self, input_holder) -> tf.Tensor:
        # self.filter = tf.zeros([1300, 1, 1])
        self.filter = tf.zeros([1300, 1, 1])
        return tf.nn.conv1d(input_holder, self.filter, stride=2, padding='VALID')


class BatchNormalization(Layer):
    def __init__(self, epsilon: float = 0.001) -> None:
        super().__init__()
        self.epsilon = epsilon

    def activation(self, input_holder) -> tf.Tensor:
        shape = input_holder.get_shape().as_list()
        gamma = tf.get_variable("gamma", shape[-1], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable("beta", shape[-1], initializer=tf.constant_initializer(0.0))
        # avg, var = tf.nn.moments(input_holder, range(len(shape)-1))
        avg, var = tf.nn.moments(input_holder, list(range(len(shape)-1))) #range(len(shape)-1))

        return tf.nn.batch_normalization(input_holder, avg, var, offset=beta, scale=gamma, variance_epsilon=self.epsilon)


class ReluConv(Layer):
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
