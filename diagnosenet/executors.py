"""
A session executor...
"""

from typing import Sequence

import tensorflow as tf 
import numpy as np
import logging
logger = logging.getLogger('_DiagnoseNET_')

class DesktopExecution:
    """
    Implements the back-propagation algorithm ...
    Args:
        model: Is a graph object of the neural network architecture selected
    Returns:
    """
    
    def __init__(self, model,  max_epochs: int = 100) -> None:
        self.model = model
        self.max_epochs: int = max_epochs
    
    def memoryexecutor(self, batch_x: np.ndarray, batch_y: np.ndarray) -> tf.Tensor:
        ## Generates a Desktop Graph
        self.model.desktop_graph()
        
        with tf.Session(graph=self.model.mlp_graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())
            sess.run(init)
            
            epoch: int = 0
            list_train_losses: list = []
                    
            while epoch < self.max_epochs:
                projection = sess.run(self.model.projection, feed_dict={self.model.X: batch_x})
                loss, _ = sess.run([self.model.mlp_loss, self.model.mlp_grad_op], 
                                    feed_dict={self.model.X: batch_x, self.model.Y: batch_y})
                
                if epoch % 10 == 0:
                    logger.info("Epoch {} | Training loss: {}".format(epoch, loss))
                epoch = epoch + 1
                                                            
            return projection
        