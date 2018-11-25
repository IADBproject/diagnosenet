"""
A session executor...
"""

from typing import Sequence

import tensorflow as tf 
import numpy as np

class DesktopExecution:
    """
    Implements the back-propagation algorithm ...
    Args:
        model: is a graph object of the neural network architecture selected
    Returns:
    """
    
    def __init__(self, model) -> None:
        self.model = model
    
    def memoryexecutor(self, batch_x: np.ndarray, batch_y: np.ndarray) -> tf.Tensor:
        activation = self.model.desktop_graph()
        
        with tf.Session(graph=self.model.mlp_graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())

            sess.run(init)
            projection = sess.run(activation, feed_dict={self.model.X: batch_x})
        
        return projection
        

