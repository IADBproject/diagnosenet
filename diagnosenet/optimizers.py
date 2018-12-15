"""
We use an optimizer to adjust the parameters of our network based
on the gradients based on the gradients computed during backpropagation
"""

import tensorflow as tf

class Optimizer:
    def __init__(self) -> None:
        self.lr: float

    def desktop_Grad(self, loss) -> tf.Tensor:
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, lr: float = 0.001) -> None:
        super().__init__()
        self.lr = lr

    def desktop_Grad(self, loss) -> tf.Tensor:
        return tf.train.AdamOptimizer(self.lr).minimize(loss)
