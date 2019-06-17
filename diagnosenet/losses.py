"""
A loss function measures the predictions performance to adjust the network parameters.
"""

import tensorflow as tf

class Loss:
    def __init__(self) -> None:
        self.y_pred: tf.Tensor
        self.y_true: tf.Tensor

    def desktop_loss(self, y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError


class CrossEntropy(Loss):
    def __init__(self) -> None:
        super().__init__()

    def desktop_loss(self, y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        self.y_pred = y_pred
        self.y_true = y_true
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.y_pred, labels = self.y_true))


    def multiGPU_loss(self, y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        self.y_pred = y_pred
        self.y_true = y_true
        # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.y_pred, labels = self.y_true))
        return tf.sqrt(tf.reduce_mean(tf.square(self.y_true - self.y_pred)))


class MSE(Loss):
    def __init__(self) -> None:
        super().__init__()

    def desktop_loss(self, y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        self.y_pred = y_pred
        self.y_true = y_true

        return tf.losses.mean_squared_error(labels = self.y_true, predictions = self.y_pred)
