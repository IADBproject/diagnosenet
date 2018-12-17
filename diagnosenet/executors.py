"""
A session executor...
"""

from typing import Sequence, NamedTuple

import tensorflow as tf
import numpy as np
import time

from diagnosenet.datamanager import Dataset, Batching
from diagnosenet.io_functions import IO_Functions

import logging
logger = logging.getLogger('_DiagnoseNET_')

Batch = NamedTuple("Batch", [("inputs", np.ndarray), ("targets", np.ndarray)])
BatchPath = NamedTuple("BatchPath", [("input_files", list), ("target_files", list)])

class DesktopExecution:
    """
    Implements the back-propagation algorithm ...
    Args:
        model: Is a graph object of the neural network architecture selected
    Returns:
    """

    def __init__(self, model, max_epochs: int = 10, datamanager: Dataset = None) -> None:
        self.model = model
        self.max_epochs = max_epochs
        self.data = datamanager

    def set_dataset_memory(self, inputs: np.ndarray, targets: np.ndarray) -> Batch:
        """
        Uses datamanager classes for splitting, batching the dataset and target selection
        """
        try:
            self.data.set_data_file(inputs, targets)
            if 'MultiTask' in str(type(self.data)):
                train, valid, test = self.data.memory_one_target()
            elif 'Batching' in str(type(self.data)):
                train, valid, test = self.data.memory_batching()
            else:
                self.data = Batching(batch_size=inputs.shape[0], valid_size=0.1, test_size=0)
                self.data.set_data_file(inputs, targets)
                train, valid, test = self.data.memory_batching()
        except AttributeError:
            if 'numpy' in str(type(inputs)):
                batch_size=inputs.shape[0]
            elif 'list' in str(type(inputs)):
                batch_size=len(inputs)
            else:
                raise AttributeError("set_data_file(inputs, targets) requires: numpy, pandas or list ")
            self.data = Batching(batch_size=batch_size, valid_size=0.1, test_size=0)
            self.data.set_data_file(inputs, targets)
            train, valid, test = self.data.memory_batching()

        return train, valid, test

    def training_memory(self, inputs: np.ndarray, targets: np.ndarray) -> tf.Tensor:
        """
        Training the deep neural network exploit the memory on desktop machine
        """
        ## Set dataset on memory
        train, valid, test = self.set_dataset_memory(inputs, targets)
        ## Generates a Desktop Graph
        self.model.desktop_graph()

        with tf.Session(graph=self.model.mlp_graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())
            sess.run(init)

            epoch: int = 0
            list_train_losses: list = []

            while epoch < self.max_epochs:
                epoch_start = time.time()
                for i in range(len(train.inputs)):
                    projection = sess.run(self.model.projection, feed_dict={self.model.X: train.inputs[i]})
                    train_loss, _ = sess.run([self.model.mlp_loss, self.model.mlp_grad_op],
                                    feed_dict={self.model.X: train.inputs[i], self.model.Y: train.targets[i]})

                    train_acc = sess.run(self.model.accuracy,
                                    feed_dict={self.model.X: train.inputs[i], self.model.Y: train.targets[i]})

                for i in range(len(valid.inputs)):
                    valid_loss = sess.run(self.model.mlp_loss,
                                    feed_dict={self.model.X: valid.inputs[i], self.model.Y: valid.targets[i]})

                    valid_acc = sess.run(self.model.accuracy,
                                    feed_dict={self.model.X: valid.inputs[i], self.model.Y: valid.targets[i]})

                epoch_elapsed = (time.time() - epoch_start)
                logger.info("Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(epoch,
                                                        train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                epoch = epoch + 1

            for i in range(len(test.inputs)):
                test_loss = sess.run(self.model.mlp_loss,
                                feed_dict={self.model.X: test.inputs[i], self.model.Y: test.targets[i]})

                logger.info("Test Batch: {} | Test Loss: {}".format(i, test_loss))

            return projection


    def set_dataset_disk(self,  dataset_name: str, dataset_path: str,
                        inputs_name: str, targets_name: str) -> BatchPath:
        """
        Uses datamanager classes for splitting, batching the dataset and target selection
        """
        try:
            self.data.set_data_path(dataset_name=dataset_name,
                               dataset_path=dataset_path,
                               inputs_name=inputs_name,
                               targets_name=targets_name)
            if 'MultiTask' in str(type(self.data)):
                train, valid, test = self.data.disk_one_target()
            elif 'Batching' in str(type(self.data)):
                train, valid, test = self.data.disk_batching()
            else:
                raise AttributeError("training_disk() requires a datamanager class type, gives: {}".format(str(type(self.data))))
        except AttributeError:
                raise AttributeError("training_disk() requires a datamanager class type, gives: {}".format(str(type(self.data))))
        return train, valid, test

    def training_disk(self, dataset_name: str, dataset_path: str,
                        inputs_name: str, targets_name: str) -> tf.Tensor:
        """
        Training the deep neural network exploit the memory on desktop machine
        """
        ## Set dataset on memory
        train, valid, test = self.set_dataset_disk(dataset_name, dataset_path,
                                                    inputs_name, targets_name)
        ## Generates a Desktop Graph
        self.model.desktop_graph()

        with tf.Session(graph=self.model.mlp_graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())
            sess.run(init)

            epoch: int = 0
            list_train_losses: list = []
            while epoch < self.max_epochs:
                epoch_start = time.time()

                for i in range(len(train.input_files)):
                    train_inputs = IO_Functions()._read_file(train.input_files[i])
                    train_targets = IO_Functions()._read_file(train.target_files[i])
                    ## Convert list in a numpy matrix
                    train_batch= Dataset()
                    train_batch.set_data_file(train_inputs, train_targets)

                    projection = sess.run(self.model.projection, feed_dict={self.model.X: train_batch.inputs})
                    train_loss, _ = sess.run([self.model.mlp_loss, self.model.mlp_grad_op],
                                    feed_dict={self.model.X: train_batch.inputs, self.model.Y: train_batch.targets})
                    train_acc = sess.run(self.model.accuracy,
                                    feed_dict={self.model.X: train_batch.inputs, self.model.Y: train_batch.targets})

                for i in range(len(valid.input_files)):
                    valid_inputs = IO_Functions()._read_file(valid.input_files[i])
                    valid_targets = IO_Functions()._read_file(valid.target_files[i])
                    ## Convert list in a numpy matrix
                    valid_batch= Dataset()
                    valid_batch.set_data_file(valid_inputs, valid_targets)

                    valid_loss = sess.run(self.model.mlp_loss,
                                    feed_dict={self.model.X: valid_batch.inputs, self.model.Y: valid_batch.targets})
                    valid_acc = sess.run(self.model.accuracy,
                                    feed_dict={self.model.X: valid_batch.inputs, self.model.Y: valid_batch.targets})

                epoch_elapsed = (time.time() - epoch_start)
                logger.info("Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(epoch,
                                                        train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                epoch = epoch + 1

            for i in range(len(test.input_files)):
                test_inputs = IO_Functions()._read_file(test.input_files[i])
                test_targets = IO_Functions()._read_file(test.target_files[i])
                ## Convert list in a numpy matrix
                test_batch= Dataset()
                test_batch.set_data_file(test_inputs, test_targets)


                test_loss = sess.run(self.model.mlp_loss,
                                feed_dict={self.model.X: test_batch.inputs, self.model.Y: test_batch.targets})

                logger.info("Test Batch: {} | Test Loss: {}".format(i, test_loss))

            return projection
