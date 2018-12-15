"""
A session executor...
"""

from typing import Sequence

import tensorflow as tf
import numpy as np

from diagnosenet.datamanager import Dataset, Batching

import logging
logger = logging.getLogger('_DiagnoseNET_')

class DesktopExecution:
    """
    Implements the back-propagation algorithm ...
    Args:
        model: Is a graph object of the neural network architecture selected
    Returns:
    """

    def __init__(self, model, max_epochs: int = 5, datamanager: Dataset = None) -> None:
        self.model = model
        self.max_epochs = max_epochs
        self.data = datamanager
        print("++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("datamanager Type: {}".format(type(self.data)))
        print("self.max_epochs: {}".format(self.max_epochs))

    def memoryexecutor(self, inputs: np.ndarray, targets: np.ndarray) -> tf.Tensor:

        ################### Datamanager
        try:
            ## Using MultiTask for split the dataset
            self.data.set_data_file(inputs, targets)
            if 'MultiTask' in str(type(self.data)):
                train, valid, test = self.data.memory_one_target()
            elif 'Batching' in str(type(self.data)):
                train_batches, valid_batches, test_batches = self.data.memory_batching()
            else:
                self.data.dataset_split()
                # train_batches = self.data.train
                # valid_batches = self.data.valid
                # test_batches = self.data.test
        except AttributeError:
            self.data = Batching(valid_size=0.1, test_size=0)
            self.data.set_data_file(inputs, targets)
            self.data.dataset_split()
            # train_batches = self.data.train
            # valid_batches = self.data.valid
            # test_batches = self.data.test


        ##################### Build Graph
        ## Generates a Desktop Graph
        self.model.desktop_graph()

        with tf.Session(graph=self.model.mlp_graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())
            sess.run(init)

            epoch: int = 0
            list_train_losses: list = []
            while epoch < self.max_epochs:
                for i in range(len(train.inputs)):
                    projection = sess.run(self.model.projection, feed_dict={self.model.X: train.inputs[i]})
                    train_loss, _ = sess.run([self.model.mlp_loss, self.model.mlp_grad_op],
                                    feed_dict={self.model.X: train.inputs[i], self.model.Y: train.targets[i]})

                for i in range(len(valid.inputs)):
                    valid_loss = sess.run(self.model.mlp_loss,
                                    feed_dict={self.model.X: valid.inputs[i], self.model.Y: valid.targets[i]})

                # if epoch % 10 == 0:
                logger.info("Epoch {} | Train loss: {} |  Valid loss: {}".format(epoch, train_loss valid_loss))
                epoch = epoch + 1

            return projection
