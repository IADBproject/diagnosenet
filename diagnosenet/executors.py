"""
A session executor...
"""

from typing import Sequence, NamedTuple

import tensorflow as tf
import numpy as np

import os, time, json
import multiprocessing as mp

from diagnosenet.datamanager import Dataset, Batching
from diagnosenet.io_functions import IO_Functions
from diagnosenet.monitor import enerGyPU, Metrics
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

    def __init__(self, model, monitor: enerGyPU = None, datamanager: Dataset = None,
                    max_epochs: int = 10, min_loss: float = 2.0) -> None:
        self.model = model
        self.data = datamanager
        self.max_epochs = max_epochs
        self.min_loss = min_loss
        self.monitor = monitor

        ## Time logs
        self.time_latency: time()
        self.time_dataset: time()
        self.time_training: time()
        self.time_testing: time()
        self.time_metrics: time()

        ## Testbed and Metrics
        self.processing_mode: str
        self.training_track: list = []


    def set_monitor_recording(self) -> None:
        """
        Power and performance monitoring launcher for workload characterization
        """
        latency_start = time.time()
        if self.monitor == None:
            self.monitor = enerGyPU(testbed_path="testbed",
                                write_metrics=True,
                                power_recording=True,
                                platform_recording=True)

        ## Generate ID-experiment and their testebed directory
        self.monitor.generate_testbed(self.monitor.testbed_path,
                                        self.model, self.data,
                                        self.__class__.__name__,
                                        self.max_epochs)

        ## Start power recording
        if self.monitor.power_recording == True: self.monitor.start_power_recording()

        ## Start platform recording
        if self.monitor.platform_recording == True: self.monitor.start_platform_recording(os.getpid())

        ## Get GPU availeble and set for processing
        #self.idgpu = self.monitor._get_available_GPU()
        self.idgpu = "3"
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=self.idgpu[0] #"3,4"

        ## Time recording
        self.time_latency = time.time()-latency_start

    def set_dataset_memory(self, inputs: np.ndarray, targets: np.ndarray) -> Batch:
        """
        Uses datamanager classes for splitting, batching the dataset and target selection
        """
        dataset_start = time.time()
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

        self.time_dataset = time.time()-dataset_start
        return train, valid, test

    def training_memory(self, inputs: np.ndarray, targets: np.ndarray) -> tf.Tensor:
        """
        Training the deep neural network exploit the memory on desktop machine
        """
        ## Set processing_mode flat
        self.processing_mode = "memory_batching"
        ## Set Monitor Recording
        self.set_monitor_recording()
        ## Set dataset on memory
        train, valid, test = self.set_dataset_memory(inputs, targets)

        ### Training Start
        training_start = time.time()
        ## Generates a Desktop Graph
        self.model.desktop_graph()

        with tf.Session(graph=self.model.mlp_graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())
            sess.run(init)

            epoch: int = 0
            epoch_convergence: bin = 0
            while (epoch_convergence == 0):

                epoch_start = time.time()
                for i in range(len(train.inputs)):
                    train_loss, _ = sess.run([self.model.mlp_loss, self.model.mlp_grad_op],
                                    feed_dict={self.model.X: train.inputs[i],
                                                self.model.Y: train.targets[i],
                                                self.model.keep_prob: self.model.dropout})
                    train_pred = sess.run(self.model.projection_1hot,
                                    feed_dict={self.model.X: train.inputs[i],
                                                self.model.keep_prob: self.model.dropout})
                    ## F1_score from Skit-learn metrics
                    train_acc = f1_score(y_true=train.targets[i].astype(np.float),
                                            y_pred=train_pred, average='micro')

                for i in range(len(valid.inputs)):
                    valid_loss = sess.run(self.model.mlp_loss,
                                    feed_dict={self.model.X: valid.inputs[i],
                                                self.model.Y: valid.targets[i],
                                                self.model.keep_prob: 1.0})
                    valid_pred = sess.run(self.model.projection_1hot,
                                    feed_dict={self.model.X: valid.inputs[i],
                                                self.model.keep_prob: 1.0})
                    ## F1_score from Skit-learn metrics
                    valid_acc = f1_score(y_true=valid.targets[i].astype(np.float),
                                            y_pred=valid_pred, average='micro')

                epoch_elapsed = (time.time() - epoch_start)
                logger.info("Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(epoch,
                                                        train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                self.training_track.append((epoch,train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                epoch = epoch + 1

                ## While Convergence conditional
                if valid_loss <= self.min_loss or epoch == self.max_epochs:
                    epoch_convergence = 1
                    self.max_epochs=epoch
                    self.min_loss=valid_loss
                else:
                    epoch_convergence = 0
                ### end While loop
            self.time_training = time.time()-training_start

            ### Testing Starting
            testing_start = time.time()

            if len(test.inputs) != 0:
                test_pred_probas: list = []
                test_pred_1hot: list = []
                test_true_1hot: list = []

                for i in range(len(test.inputs)):
                    tt_pred_probas = sess.run(self.model.soft_projection,
                                                feed_dict={self.model.X: test.inputs[i],
                                                            self.model.keep_prob: 1.0})
                    tt_pred_1hot = sess.run(self.model.projection_1hot,
                                                feed_dict={self.model.X: test.inputs[i],
                                                            self.model.keep_prob: 1.0})

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

                ## Compute_metrics by each label
                self.metrics_values = Metrics().compute_metrics(y_pred=self.test_pred_1hot,
                                                            y_true=self.test_true_1hot)
                self.time_testing = time.time()-testing_start

                ## Write metrics on testbet directory = self.monitor.testbed_exp
                if self.monitor.write_metrics == True: self.write_metrics()

                return self.test_pred_probas
            return train_pred

    def set_dataset_disk(self,  dataset_name: str, dataset_path: str,
                        inputs_name: str, targets_name: str) -> BatchPath:
        """
        Uses datamanager classes for splitting, batching the dataset and target selection
        """
        dataset_start = time.time()
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
        self.time_dataset = time.time()-dataset_start
        return train, valid, test

    def training_disk(self, dataset_name: str, dataset_path: str,
                        inputs_name: str, targets_name: str) -> tf.Tensor:
        """
        Training the deep neural network exploit the memory on desktop machine
        """
        ## Set processing_mode flat
        self.processing_mode = "disk_batching"
        ## Set Monitor Recording
        self.set_monitor_recording()
        ## Set dataset on memory
        train, valid, test = self.set_dataset_disk(dataset_name, dataset_path,
                                                    inputs_name, targets_name)

        ### Training Start
        training_start = time.time()

        ## Generates a Desktop Graph
        self.model.desktop_graph()

        with tf.Session(graph=self.model.mlp_graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())
            sess.run(init)

            epoch: int = 0
            epoch_convergence: bin = 0
            while (epoch_convergence == 0):
                epoch_start = time.time()

                for i in range(len(train.input_files)):
                    train_inputs = IO_Functions()._read_file(train.input_files[i])
                    train_targets = IO_Functions()._read_file(train.target_files[i])
                    ## Convert list in a numpy matrix
                    train_batch = Dataset()
                    train_batch.set_data_file(train_inputs, train_targets)

                    train_loss, _ = sess.run([self.model.mlp_loss, self.model.mlp_grad_op],
                                        feed_dict={self.model.X: train_batch.inputs,
                                                    self.model.Y: train_batch.targets,
                                                    self.model.keep_prob: self.model.dropout})
                    train_pred = sess.run(self.model.projection_1hot,
                                                feed_dict={self.model.X: train_batch.inputs,
                                                self.model.keep_prob: self.model.dropout})
                    ## F1_score from Skit-learn metrics
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
                                                    self.model.Y: valid_batch.targets,
                                                    self.model.keep_prob: 1.0})
                    valid_pred = sess.run(self.model.projection_1hot,
                                        feed_dict={self.model.X: valid_batch.inputs,
                                                    self.model.keep_prob: 1.0})
                    ## F1_score from Skit-learn metrics
                    valid_acc = f1_score(y_true=valid_batch.targets.astype(np.float),
                                            y_pred=valid_pred.astype(np.float), average='micro')


                epoch_elapsed = (time.time() - epoch_start)
                logger.info("Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(epoch,
                                                        train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                self.training_track.append((epoch,train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                epoch = epoch + 1

                ## While Convergence conditional
                if valid_loss <= self.min_loss or epoch == self.max_epochs:
                    epoch_convergence = 1
                    self.max_epochs=epoch
                    self.min_loss=valid_loss
                else:
                    epoch_convergence = 0
                ### end While loop
            self.time_training = time.time()-training_start

            ### Testing Starting
            testing_start = time.time()

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
                                    feed_dict={self.model.X: test_batch.inputs,
                                                self.model.keep_prob: 1.0})
                    tt_pred_1hot = sess.run(self.model.projection_1hot,
                                    feed_dict={self.model.X: test_batch.inputs,
                                                self.model.keep_prob: 1.0})

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
                self.time_testing = time.time()-testing_start

                ## Write metrics on testbet directory = self.monitor.testbed_exp
                if self.monitor.write_metrics == True: self.write_metrics()
                return self.test_pred_probas
            return train_pred

    def write_metrics(self, testbed_path: str = 'testbed') -> None:
        """
        Uses Testbed to isolate the training metrics by experiment directory
        """
        metrics_start = time.time()

        ## Writes the training and validation track
        track_path=str(self.monitor.testbed_exp+"/"+self.monitor.exp_id+"-training_track.txt")
        IO_Functions()._write_list(self.training_track, track_path)

        ## Writes the Test labels
        true_1h_path=str(self.monitor.testbed_exp+"/"+self.monitor.exp_id+"-true_1hot.txt")
        np.savetxt(true_1h_path, self.test_true_1hot, delimiter=',', fmt='%d')

        pred_1h_path=str(self.monitor.testbed_exp+"/"+self.monitor.exp_id+"-pred_1hot.txt")
        np.savetxt(pred_1h_path, self.test_pred_1hot, delimiter=',', fmt='%d')

        pred_probas_path=str(self.monitor.testbed_exp+"/"+self.monitor.exp_id+"-pred_probas.txt")
        np.savetxt(pred_probas_path, self.test_pred_probas, delimiter=',', fmt='%f')

        ## Writes Summarize Metrics
        metrics_values_path=str(self.monitor.testbed_exp+"/"+self.monitor.exp_id+"-metrics_values.txt")
        np.savetxt(metrics_values_path, self.metrics_values, delimiter=',', fmt='%d')

        ### Add elements to json experiment Description architecture
        eda_json = self.monitor.read_eda_json(self.monitor.testbed_exp, self.monitor.exp_id)

        ## Add values to platform_parameters
        eda_json['model_hyperparameters']['max_epochs'] = self.max_epochs

        ## Add dataset shape as number of records (inputs, targets)
        eda_json['dataset_config']['train_records'] = str(self.data.train_shape)
        eda_json['dataset_config']['valid_records'] = str(self.data.valid_shape)
        eda_json['dataset_config']['test_records'] = str(self.data.test_shape)

        ## Add values to platform_parameters
        eda_json['platform_parameters']['processing_mode'] = self.processing_mode
        eda_json['platform_parameters']['gpu_id'] = self.idgpu[0]

        ## Add values to results
        eda_json['results']['f1_score_weigted'] = self.test_f1_weighted
        eda_json['results']['f1_score_micro'] = self.test_f1_micro
        eda_json['results']['loss_validation'] = str(self.min_loss)
        eda_json['results']['time_latency'] = self.time_latency
        eda_json['results']['time_dataset'] = self.time_dataset
        eda_json['results']['time_training'] = self.time_training
        eda_json['results']['time_testing'] = self.time_testing

        ## End time metrics
        self.time_metrics = time.time()-metrics_start
        eda_json['results']['time_metrics'] = self.time_metrics

        ## Serialize the eda json and rewrite the file
        eda_json = json.dumps(eda_json, separators=(',', ': '), indent=2)
        file_path = str(self.monitor.testbed_exp+"/"+self.monitor.exp_id+"-exp_description.json")
        IO_Functions()._write_file(eda_json, file_path)


        ## End computational recording
        self.monitor.end_platform_recording()

        ## End power recording
        self.monitor.end_power_recording()

        logger.info("Tesbed directory: {}".format(self.monitor.testbed_exp))



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


        #######################################################################
        ## MultiGPU
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
        self.num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))


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


    def training_multigpu(self, inputs: np.ndarray, targets: np.ndarray) -> tf.Tensor:
        """
        Training the deep neural network exploit the memory on desktop machine
        """
        ## Set dataset on memory
        train, valid, test = self.set_dataset_memory(inputs, targets)
        ## Generates a Desktop Graph
        self.model.multiGPU_graph(self.data.batch_size, self.num_gpus)
        print("++ Trainig Graph on MultiGPU ++")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4

        with tf.Session(config=config, graph=self.model.mlp_graph) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            epoch: int = 0
            while epoch < self.max_epochs:
                epoch_start = time.time()
                for i in range(len(train.inputs)):

                    ### Temporaly conditional
                    if train.inputs[i].shape[0] < (self.data.batch_size/2):
                        print("+++++++++")
                        print("train exclude -> {}".format(train.inputs[i].shape))
                        print("+++++++++")

                    else:
                        train_pred = sess.run(self.model.output1,
                                    feed_dict={self.model.X: train.inputs[i],
                                            self.model.keep_prob: self.model.dropout})

                        # print("train.inputs {}".format(train.inputs[i].shape))
                        # print("train_targets: {}".format(train.targets[i].shape))
                        print("train_pred: {}".format(train_pred.shape))

                        train_loss = sess.run(self.model.output2,
                                    feed_dict={self.model.X: train.inputs[i],
                                            self.model.Y: train.targets[i],
                                            self.model.keep_prob: self.model.dropout})

                        print("train_loss: {}".format(train_loss))



                        train_grads = sess.run(self.model.train_op,
                                    feed_dict={self.model.X: train.inputs[i],
                                            self.model.Y: train.targets[i],
                                            self.model.keep_prob: self.model.dropout})

                        print("train_grads: {}".format(train_grads))

                epoch_elapsed = (time.time() - epoch_start)
                epoch = epoch + 1



    # def training_multigpu_OLD(self, inputs: np.ndarray, targets: np.ndarray) -> tf.Tensor:
    #     """
    #     Training the deep neural network exploit the memory on desktop machine
    #     """
    #     ## Set dataset on memory
    #     train, valid, test = self.set_dataset_memory(inputs, targets)
    #     ## Generates a Desktop Graph
    #     self.model.multiGPU_graph(self.data.batch_size)
    #     print("++ execute multiGPU graph ++")
    #
    #
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    #
    #     with tf.Session(config=config, graph=self.model.mlp_graph) as sess:
    #     # with tf.Session(config=tf.ConfigProto(log_device_placement=False),
    #     #                 graph=self.model.mlp_graph) as sess:
    #
    #         # init = tf.group(tf.global_variables_initializer(),
    #         #                     tf.local_variables_initializer())
    #
    #         init = tf.group(tf.global_variables_initializer())
    #         # init = tf.global_variables_initializer()
    #         sess.run(init)
    #
    #         epoch: int = 0
    #         while epoch < self.max_epochs:
    #             epoch_start = time.time()
    #             for i in range(len(train.inputs)):
    #
    #
    #                 ### Temporaly conditional
    #                 if train.inputs[i].shape[0] < (self.data.batch_size/2):
    #                     print("+++++++++")
    #                     print("train exclude -> {}".format(train.inputs[i].shape))
    #                     print("+++++++++")
    #
    #                 else:
    #
    #                     ### Normal Gradient
    #                     # train_loss = sess.run(self.model.grad_from_optimizer,
    #
    #                     ### Optimizer
    #                     # train_loss = sess.run(self.model.grads_computation,
    #
    #                     ### Works with one GPU
    #                     # train_loss = sess.run(self.model.train_op,
    #
    #                     train_loss = sess.run(self.model.total_loss,
    #                                     feed_dict={self.model.X: train.inputs[i],
    #                                                 self.model.Y: train.targets[i],
    #                                                 self.model.keep_prob: self.model.dropout})
    #
    #                     print("train_loss: {}".format(train_loss))
    #
    #                     train_pred = sess.run(self.model.projection,
    #                                 feed_dict={self.model.X: train.inputs[i],
    #                                         self.model.keep_prob: self.model.dropout})
    #                     # print("train.inputs {}".format(train.inputs[i].shape))
    #                     # print("train_targets: {}".format(train.targets[i].shape))
    #                     print("train_pred: {}".format(train_pred.shape))
    #                     # print("train_pred: {}".format(train_pred[0]))
    #
    #                     # train_loss, _ = sess.run([self.model.mlp_loss, self.model.mlp_grad_op],
    #                     # train_loss, _ = sess.run([self.model.mlp_losses, self.model.mlp_grad_op],
    #                     #                 feed_dict={self.model.X: train.inputs[i],
    #                     #                             self.model.Y: train.targets[i],
    #                     #                             self.model.keep_prob: self.model.dropout})
    #
    #                     # print("train_loss: {}".format(train_loss))
    #                     # print("train_1: {}".format(train_loss[0]))
    #                     # print("train_2: {}".format(train_loss[1]))
    #
    #
    #                 # train_acc = f1_score(y_true=train.targets[i].astype(np.float),
    #                 #                         y_pred=train_pred, average='micro')
    #
    #             print("{}".format("-"*20))
    #             for i in range(len(valid.inputs)):
    #
    #                 ### Temporaly conditional
    #                 # if train.inputs[i].shape[0] < self.data.batch_size:
    #                 if valid.inputs[i].shape[0] < (self.data.batch_size/2):
    #                     print("+++++++++")
    #                     print("valid exclude -> {}".format(valid.inputs[i].shape))
    #                     print("+++++++++")
    #
    #                 else:
    #                     # valid_loss = sess.run(self.model.mlp_loss,
    #                     #                 feed_dict={self.model.X: valid.inputs[i],
    #                     #                             self.model.Y: valid.targets[i],
    #                     #                             self.model.keep_prob: self.model.dropout})
    #                     #
    #                     #
    #                     # valid_pred = sess.run(self.model.projection,
    #                     #             feed_dict={self.model.X: valid.inputs[i],
    #                     #                      self.model.keep_prob: self.model.dropout})
    #
    #                     # print("valid.inputs[i]: {}".format(valid.inputs[i].shape))
    #                     # print("valid_targets: {}".format(valid.targets[i].shape))
    #                     # print("valid_pred: {}".format(valid_pred.shape))
    #                     # print("valid_pred: {}".format(valid_pred))
    #
    #                     valid_loss = 113374.24
    #                     print("valid_loss: {}".format(valid_loss))
    #
    #
    #                     # valid_acc = f1_score(y_true=valid.targets[i].astype(np.float),
    #                     #                         y_pred=valid_pred, average='micro')
    #
    #
    #             epoch_elapsed = (time.time() - epoch_start)
    #             print("train_loss: {} || valid_loss: {}".format(train_loss, valid_loss))
    #             print("{} \n".format("/"*20))
    #             # logger.info("Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(epoch,
    #             #                                         train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
    #             # self.training_track.append((epoch,train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
    #
    #             epoch = epoch + 1
    #
    #         # return train_loss
