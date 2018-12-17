"""
Metrics ...
"""

import tensorflow as tf

class Metrics:
    def __init__(self) -> None:
        pass
        
    def accuracy(self, target, projection):
        """
        Computes the percentage of times that predictions matches labels.
        """
        correct_prediction = tf.equal(tf.nn.l2_normalize(projection, 1),
                                         tf.nn.l2_normalize(target, 1))
                                                
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy
        
        