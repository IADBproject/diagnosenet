"""
A session executor...
"""

from typing import Sequence, NamedTuple

import tensorflow as tf
import numpy as np
import os, time

from diagnosenet.datamanager import Dataset, Batching
from diagnosenet.io_functions import IO_Functions

## Write metrics
from diagnosenet.metrics import Testbed, Metrics #, enerGyPU
from diagnosenet.energypu import enerGyPU

from sklearn.metrics import f1_score

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

    def __init__(self, model, datamanager: Dataset = None, max_epochs: int = 10) -> None:
        self.model = model
        self.data = datamanager
        self.max_epochs = max_epochs

        ## Testbed and Metrics
        testbed_path: str = 'testbed'
        self.training_track: list = []

        self.egpu = enerGyPU(self.model, self.data, self.__class__.__name__, self.max_epochs)
        self.exp_id = self.egpu.generate_testbed(testbed_path)
        self.testbed_exp = str(testbed_path+"/"+self.exp_id+"/")

        ## Start power recording
        self.egpu.start_power_recording(self.testbed_exp, self.exp_id)

        ## Get GPU availeble and set for processing
        idgpu = self.egpu._get_available_GPU()
        print("idgpu: {}".format(idgpu))
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="3,4"    #idgpu[0]

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
            while epoch < self.max_epochs:
                epoch_start = time.time()
                for i in range(len(train.inputs)):

                    train_loss, _ = sess.run([self.model.mlp_loss, self.model.mlp_grad_op],
                                    feed_dict={self.model.X: train.inputs[i],
                                                self.model.Y: train.targets[i]})

                    train_pred = sess.run(self.model.projection_1hot,
                                    feed_dict={self.model.X: train.inputs[i]})

                    train_acc = f1_score(y_true=train.targets[i].astype(np.float),
                                            y_pred=train_pred, average='micro')

                for i in range(len(valid.inputs)):
                    valid_loss = sess.run(self.model.mlp_loss,
                                    feed_dict={self.model.X: valid.inputs[i],
                                                self.model.Y: valid.targets[i]})

                    valid_pred = sess.run(self.model.projection_1hot,
                                    feed_dict={self.model.X: valid.inputs[i]})

                    valid_acc = f1_score(y_true=valid.targets[i].astype(np.float),
                                            y_pred=valid_pred, average='micro')

                epoch_elapsed = (time.time() - epoch_start)
                logger.info("Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(epoch,
                                                        train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                self.training_track.append((epoch,train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                epoch = epoch + 1


            ## Testing
            if len(test.inputs) != 0:
                test_pred_probas: list = []
                test_pred_1hot: list = []
                test_true_1hot: list = []

                for i in range(len(test.inputs)):
                    tt_pred_probas = sess.run(self.model.soft_projection,
                                                    feed_dict={self.model.X: test.inputs[i]})
                    tt_pred_1hot = sess.run(self.model.projection_1hot,
                                                    feed_dict={self.model.X: test.inputs[i]})

                    test_pred_probas.append(tt_pred_probas)
                    test_pred_1hot.append(tt_pred_1hot)
                    test_true_1hot.append(test.targets[i].astype(np.float))

                self.test_pred_probas = np.vstack(test_pred_probas)
                self.test_pred_1hot = np.vstack(test_pred_1hot)
                self.test_true_1hot = np.vstack(test_true_1hot)

                ## Compute the F1 Score
                self.test_f1_weighted = f1_score(self.test_true_1hot,
                                                    self.test_pred_1hot, average = "weighted")
                self.test_f1_micro = f1_score(self.test_true_1hot,
                                                    self.test_pred_1hot, average = "micro")
                logger.info("-- Test Results --")
                logger.info("F1-Score Weighted: {}".format(self.test_f1_weighted))
                logger.info("F1-Score Micro: {}".format(self.test_f1_micro))

                ## compute_metrics by each label
                self.metrics_values = Metrics().compute_metrics(y_pred=self.test_pred_1hot,
                                                            y_true=self.test_true_1hot)

                return self.test_pred_probas
            return train_pred


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
                    train_batch = Dataset()
                    train_batch.set_data_file(train_inputs, train_targets)

                    train_loss, _ = sess.run([self.model.mlp_loss, self.model.mlp_grad_op],
                                        feed_dict={self.model.X: train_batch.inputs,
                                                    self.model.Y: train_batch.targets})

                    train_pred = sess.run(self.model.projection_1hot,
                                                feed_dict={self.model.X: train_batch.inputs})
                    train_acc = f1_score(y_true=train_batch.targets.astype(np.float),
                                                y_pred=train_pred.astype(np.float), average='micro')

                for i in range(len(valid.input_files)):
                    valid_inputs = IO_Functions()._read_file(valid.input_files[i])
                    valid_targets = IO_Functions()._read_file(valid.target_files[i])
                    ## Convert list in a numpy matrix
                    valid_batch= Dataset()
                    valid_batch.set_data_file(valid_inputs, valid_targets)

                    valid_loss = sess.run(self.model.mlp_loss,
                                        feed_dict={self.model.X: valid_batch.inputs,
                                            self.model.Y: valid_batch.targets})
                    valid_pred = sess.run(self.model.projection_1hot,
                                            feed_dict={self.model.X: valid_batch.inputs})
                    valid_acc = f1_score(y_true=valid_batch.targets.astype(np.float),
                                            y_pred=valid_pred.astype(np.float), average='micro')


                epoch_elapsed = (time.time() - epoch_start)
                logger.info("Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(epoch,
                                                        train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                self.training_track.append((epoch,train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                epoch = epoch + 1


            ## Testing
            if len(test.input_files) != 0:
                test_pred_probas: list = []
                test_pred_1hot: list = []
                test_true_1hot: list = []

                for i in range(len(test.input_files)):
                    test_inputs = IO_Functions()._read_file(test.input_files[i])
                    test_targets = IO_Functions()._read_file(test.target_files[i])
                    ## Convert list in a numpy matrix
                    test_batch = Dataset()
                    test_batch.set_data_file(test_inputs, test_targets)

                    tt_pred_probas = sess.run(self.model.soft_projection,
                                    feed_dict={self.model.X: test_batch.inputs})
                    tt_pred_1hot = sess.run(self.model.projection_1hot,
                                    feed_dict={self.model.X: test_batch.inputs})

                    test_pred_probas.append(tt_pred_probas)
                    test_pred_1hot.append(tt_pred_1hot)
                    test_true_1hot.append(test_batch.targets.astype(np.float))

                self.test_pred_probas = np.vstack(test_pred_probas)
                self.test_pred_1hot = np.vstack(test_pred_1hot)
                self.test_true_1hot = np.vstack(test_true_1hot)

                ## Compute the F1 Score
                self.test_f1_weighted = f1_score(self.test_true_1hot,
                                                    self.test_pred_1hot, average = "weighted")
                self.test_f1_micro = f1_score(self.test_true_1hot,
                                                    self.test_pred_1hot, average = "micro")
                logger.info("-- Test Results --")
                logger.info("F1-Score Weighted: {}".format(self.test_f1_weighted))
                logger.info("F1-Score Micro: {}".format(self.test_f1_micro))

                ## compute_metrics by each label
                self.metrics_values = Metrics().compute_metrics(y_pred=self.test_pred_1hot,
                                                            y_true=self.test_true_1hot)
                return self.test_pred_probas
            return train_pred


    def write_metrics(self, testbed_path: str = 'testbed') -> None:
        """
        Uses Testbed to isolate the training metrics by experiment directory
        """

        ## Generate a Testebed directory
        # tesbed = Testbed(self.model, self.data, self.__class__.__name__, self.max_epochs)
        # self.exp_id = tesbed.generate_testbed(testbed_path)
        # self.testbed_exp = str(testbed_path+"/"+self.exp_id+"/")

        ## Writes the training and validation track
        track_path=str(self.testbed_exp+"/"+self.exp_id+"-training_track.txt")
        IO_Functions()._write_list(self.training_track, track_path)

        ## Writes the Test labels
        true_1h_path=str(self.testbed_exp+"/"+self.exp_id+"-true_1hot.txt")
        np.savetxt(true_1h_path, self.test_true_1hot, delimiter=',', fmt='%d')

        pred_1h_path=str(self.testbed_exp+"/"+self.exp_id+"-pred_1hot.txt")
        np.savetxt(pred_1h_path, self.test_pred_1hot, delimiter=',', fmt='%d')

        pred_probas_path=str(self.testbed_exp+"/"+self.exp_id+"-pred_probas.txt")
        np.savetxt(pred_probas_path, self.test_pred_probas, delimiter=',', fmt='%f')

        ## Writes Summarize Metrics
        metrics_values_path=str(self.testbed_exp+"/"+self.exp_id+"-metrics_values.txt")
        np.savetxt(metrics_values_path, self.metrics_values, delimiter=',', fmt='%d')

        ## End power recording
        self.egpu.end_power_recording()

        logger.info("Tesbed directory: {}".format(self.testbed_exp))




class MultiGPU:
    """
    Implements the back-propagation algorithm ...
    Args:
        model: Is a graph object of the neural network architecture selected
    Returns:
    """

    def __init__(self, model, datamanager: Dataset = None, max_epochs: int = 10) -> None:
        self.model = model
        self.data = datamanager
        self.max_epochs = max_epochs

        ## Testbed and Metrics
        testbed_path: str = 'testbed'
        self.training_track: list = []

        self.egpu = enerGyPU(self.model, self.data, self.__class__.__name__, self.max_epochs)
        self.exp_id = self.egpu.generate_testbed(testbed_path)
        self.testbed_exp = str(testbed_path+"/"+self.exp_id+"/")

        ## Start power recording
        # self.egpu.start_power_recording(self.testbed_exp, self.exp_id)

        ## Get GPU availeble and set for processing
        idgpu = self.egpu._get_available_GPU()
        print("idgpu: {}".format(idgpu))
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="3,4"    #idgpu[0]


        #######################################################################
        ## MultiGPU
        batch_size = 100
        self.num_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        self.gpu_batch_size = batch_size/len(self.num_gpus)
        print("self.gpu: {}".format(self.gpu_batch_size))


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
            while epoch < self.max_epochs:
                epoch_start = time.time()
                for i in range(len(train.inputs)):

                    train_loss, _ = sess.run([self.model.mlp_loss, self.model.mlp_grad_op],
                                    feed_dict={self.model.X: train.inputs[i],
                                                self.model.Y: train.targets[i]})

                    train_pred = sess.run(self.model.projection_1hot,
                                    feed_dict={self.model.X: train.inputs[i]})

                    train_acc = f1_score(y_true=train.targets[i].astype(np.float),
                                            y_pred=train_pred, average='micro')

                for i in range(len(valid.inputs)):
                    valid_loss = sess.run(self.model.mlp_loss,
                                    feed_dict={self.model.X: valid.inputs[i],
                                                self.model.Y: valid.targets[i]})

                    valid_pred = sess.run(self.model.projection_1hot,
                                    feed_dict={self.model.X: valid.inputs[i]})

                    valid_acc = f1_score(y_true=valid.targets[i].astype(np.float),
                                            y_pred=valid_pred, average='micro')

                epoch_elapsed = (time.time() - epoch_start)
                logger.info("Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(epoch,
                                                        train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                self.training_track.append((epoch,train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                epoch = epoch + 1


            ## Testing
            if len(test.inputs) != 0:
                test_pred_probas: list = []
                test_pred_1hot: list = []
                test_true_1hot: list = []

                for i in range(len(test.inputs)):
                    tt_pred_probas = sess.run(self.model.soft_projection,
                                                    feed_dict={self.model.X: test.inputs[i]})
                    tt_pred_1hot = sess.run(self.model.projection_1hot,
                                                    feed_dict={self.model.X: test.inputs[i]})

                    test_pred_probas.append(tt_pred_probas)
                    test_pred_1hot.append(tt_pred_1hot)
                    test_true_1hot.append(test.targets[i].astype(np.float))

                self.test_pred_probas = np.vstack(test_pred_probas)
                self.test_pred_1hot = np.vstack(test_pred_1hot)
                self.test_true_1hot = np.vstack(test_true_1hot)

                ## Compute the F1 Score
                self.test_f1_weighted = f1_score(self.test_true_1hot,
                                                    self.test_pred_1hot, average = "weighted")
                self.test_f1_micro = f1_score(self.test_true_1hot,
                                                    self.test_pred_1hot, average = "micro")
                logger.info("-- Test Results --")
                logger.info("F1-Score Weighted: {}".format(self.test_f1_weighted))
                logger.info("F1-Score Micro: {}".format(self.test_f1_micro))

                ## compute_metrics by each label
                self.metrics_values = Metrics().compute_metrics(y_pred=self.test_pred_1hot,
                                                            y_true=self.test_true_1hot)

                return self.test_pred_probas
            return train_pred

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
                    train_batch = Dataset()
                    train_batch.set_data_file(train_inputs, train_targets)

                    train_loss, _ = sess.run([self.model.mlp_loss, self.model.mlp_grad_op],
                                        feed_dict={self.model.X: train_batch.inputs,
                                                    self.model.Y: train_batch.targets})

                    train_pred = sess.run(self.model.projection_1hot,
                                                feed_dict={self.model.X: train_batch.inputs})
                    train_acc = f1_score(y_true=train_batch.targets.astype(np.float),
                                                y_pred=train_pred.astype(np.float), average='micro')

                for i in range(len(valid.input_files)):
                    valid_inputs = IO_Functions()._read_file(valid.input_files[i])
                    valid_targets = IO_Functions()._read_file(valid.target_files[i])
                    ## Convert list in a numpy matrix
                    valid_batch= Dataset()
                    valid_batch.set_data_file(valid_inputs, valid_targets)

                    valid_loss = sess.run(self.model.mlp_loss,
                                        feed_dict={self.model.X: valid_batch.inputs,
                                            self.model.Y: valid_batch.targets})
                    valid_pred = sess.run(self.model.projection_1hot,
                                            feed_dict={self.model.X: valid_batch.inputs})
                    valid_acc = f1_score(y_true=valid_batch.targets.astype(np.float),
                                            y_pred=valid_pred.astype(np.float), average='micro')


                epoch_elapsed = (time.time() - epoch_start)
                logger.info("Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(epoch,
                                                        train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                self.training_track.append((epoch,train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                epoch = epoch + 1


            ## Testing
            if len(test.input_files) != 0:
                test_pred_probas: list = []
                test_pred_1hot: list = []
                test_true_1hot: list = []

                for i in range(len(test.input_files)):
                    test_inputs = IO_Functions()._read_file(test.input_files[i])
                    test_targets = IO_Functions()._read_file(test.target_files[i])
                    ## Convert list in a numpy matrix
                    test_batch = Dataset()
                    test_batch.set_data_file(test_inputs, test_targets)

                    tt_pred_probas = sess.run(self.model.soft_projection,
                                    feed_dict={self.model.X: test_batch.inputs})
                    tt_pred_1hot = sess.run(self.model.projection_1hot,
                                    feed_dict={self.model.X: test_batch.inputs})

                    test_pred_probas.append(tt_pred_probas)
                    test_pred_1hot.append(tt_pred_1hot)
                    test_true_1hot.append(test_batch.targets.astype(np.float))

                self.test_pred_probas = np.vstack(test_pred_probas)
                self.test_pred_1hot = np.vstack(test_pred_1hot)
                self.test_true_1hot = np.vstack(test_true_1hot)

                ## Compute the F1 Score
                self.test_f1_weighted = f1_score(self.test_true_1hot,
                                                    self.test_pred_1hot, average = "weighted")
                self.test_f1_micro = f1_score(self.test_true_1hot,
                                                    self.test_pred_1hot, average = "micro")
                logger.info("-- Test Results --")
                logger.info("F1-Score Weighted: {}".format(self.test_f1_weighted))
                logger.info("F1-Score Micro: {}".format(self.test_f1_micro))

                ## compute_metrics by each label
                self.metrics_values = Metrics().compute_metrics(y_pred=self.test_pred_1hot,
                                                            y_true=self.test_true_1hot)
                return self.test_pred_probas
            return train_pred

    def write_metrics(self, testbed_path: str = 'testbed') -> None:
        """
        Uses Testbed to isolate the training metrics by experiment directory
        """

        ## Generate a Testebed directory
        # tesbed = Testbed(self.model, self.data, self.__class__.__name__, self.max_epochs)
        # self.exp_id = tesbed.generate_testbed(testbed_path)
        # self.testbed_exp = str(testbed_path+"/"+self.exp_id+"/")

        ## Writes the training and validation track
        track_path=str(self.testbed_exp+"/"+self.exp_id+"-training_track.txt")
        IO_Functions()._write_list(self.training_track, track_path)

        ## Writes the Test labels
        true_1h_path=str(self.testbed_exp+"/"+self.exp_id+"-true_1hot.txt")
        np.savetxt(true_1h_path, self.test_true_1hot, delimiter=',', fmt='%d')

        pred_1h_path=str(self.testbed_exp+"/"+self.exp_id+"-pred_1hot.txt")
        np.savetxt(pred_1h_path, self.test_pred_1hot, delimiter=',', fmt='%d')

        pred_probas_path=str(self.testbed_exp+"/"+self.exp_id+"-pred_probas.txt")
        np.savetxt(pred_probas_path, self.test_pred_probas, delimiter=',', fmt='%f')

        ## Writes Summarize Metrics
        metrics_values_path=str(self.testbed_exp+"/"+self.exp_id+"-metrics_values.txt")
        np.savetxt(metrics_values_path, self.metrics_values, delimiter=',', fmt='%d')

        ## End power recording
        self.egpu.end_power_recording()

        logger.info("Tesbed directory: {}".format(self.testbed_exp))



    def training_multigpu(self, inputs: np.ndarray, targets: np.ndarray) -> tf.Tensor:
        """
        Training the deep neural network exploit the memory on desktop machine
        """
        ## Set dataset on memory
        train, valid, test = self.set_dataset_memory(inputs, targets)
        ## Generates a Desktop Graph
        self.model.multiGPU_graph()
        print("++ execute multiGPU graph ++")

        with tf.Session(graph=self.model.mlp_graph) as sess:
        # with tf.Session(config=tf.ConfigProto(log_device_placement=False),
        #                 graph=self.model.mlp_graph) as sess:

            # init = tf.group(tf.global_variables_initializer(),
            #                     tf.local_variables_initializer())

            init = tf.group(tf.global_variables_initializer())
            sess.run(init)

            epoch: int = 0
            while epoch < self.max_epochs:
                epoch_start = time.time()
                for i in range(len(train.inputs)):

                    train_loss, _ = sess.run([self.model.mlp_loss, self.model.mlp_grad_op],
                                    feed_dict={self.model.X: train.inputs[i],
                                                self.model.Y: train.targets[i]})

                    # train_pred = sess.run(self.model.projection_1hot,
                    #                 feed_dict={self.model.X: train.inputs[i]})

                    # train_acc = f1_score(y_true=train.targets[i].astype(np.float),
                    #                         y_pred=train_pred, average='micro')


                for i in range(len(valid.inputs)):
                    valid_loss = sess.run(self.model.mlp_loss,
                                    feed_dict={self.model.X: valid.inputs[i],
                                                self.model.Y: valid.targets[i]})

                    # valid_pred = sess.run(self.model.projection_1hot,
                    #                 feed_dict={self.model.X: valid.inputs[i]})

                    # valid_acc = f1_score(y_true=valid.targets[i].astype(np.float),
                    #                         y_pred=valid_pred, average='micro')


                epoch_elapsed = (time.time() - epoch_start)
                print("train_loss: {} || valid_loss: {}".format(train_loss, valid_loss))
                # logger.info("Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(epoch,
                #                                         train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                # self.training_track.append((epoch,train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))

                epoch = epoch + 1

            return train_loss
