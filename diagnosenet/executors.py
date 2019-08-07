"""
A session executor...
"""

import json
import logging
import os
import time
from typing import NamedTuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

from diagnosenet.datamanager import Dataset, Batching
from diagnosenet.io_functions import IO_Functions
from diagnosenet.monitor import enerGyPU, Metrics

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

    def __init__(self, model, datamanager: Dataset = None, monitor: enerGyPU = None,
                 max_epochs: int = 10, early_stopping: int = 3) -> None:
        self.model = model
        self.data = datamanager
        self.max_epochs = max_epochs
        self.min_valid_loss = float("Infinity")
        self.min_train_loss = float("Infinity")
        self.early_stopping = early_stopping
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

        ## Closing the computing tracks
        ## End computational recording
        self.monitor.end_platform_recording()
        ## End power recording
        self.monitor.end_power_recording()

        if self.monitor == None:
            self.monitor = enerGyPU(testbed_path="testbed",
                                    machine_type="x86",
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
        # self.idgpu = self.monitor._get_available_GPU()
        self.idgpu = "0"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.idgpu[0]  # "3,4"

        ## Time recording
        self.time_latency = time.time() - latency_start

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
                batch_size = inputs.shape[0]
            elif 'list' in str(type(inputs)):
                batch_size = len(inputs)
            else:
                raise AttributeError("set_data_file(inputs, targets) requires: numpy, pandas or list ")
            self.data = Batching(batch_size=batch_size, valid_size=0.1, test_size=0)
            self.data.set_data_file(inputs, targets)
            train, valid, test = self.data.memory_batching()

        self.time_dataset = time.time() - dataset_start
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

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config, graph=self.model.graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            sess.run(init)
            saver = tf.train.Saver()
            not_update = 0
            epoch: int = 0
            epoch_convergence: bin = 0
            while (epoch_convergence == 0):

                epoch_start = time.time()
                for i in range(len(train.inputs)):
                    train_loss, _ = sess.run([self.model.loss, self.model.grad_op],
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
                    valid_loss = sess.run(self.model.loss,
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
                logger.info(
                    "Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(
                        epoch, train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                self.training_track.append(
                    (epoch, train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                epoch = epoch + 1

                ## record minimum valid loss and its weights
                if valid_loss >= self.min_valid_loss and train_loss < self.min_train_loss:
                    not_update +=1

                if valid_loss <= self.min_valid_loss:
                    self.min_valid_loss = valid_loss

                if train_loss <= self.min_train_loss:
                    self.min_train_loss = train_loss

                ## While Stopping conditional
                if not_update >= self.early_stopping or epoch >= self.max_epochs:
                    epoch_convergence = 1
                    self.max_epochs = epoch
                    saver.save(sess,  str(self.monitor.testbed_exp+"/"+self.monitor.exp_id+ "-model.ckpt"))
                    self.convergence_time = time.time()-training_start
                else:
                    epoch_convergence = 0

                self.time_training = time.time()-training_start
                ### end While loop

            ### Testing Starting
            testing_start = time.time()
            checkpoint_path = str(self.monitor.testbed_exp + "./" + self.monitor.exp_id + "-model.ckpt")
            if os.path.isfile(checkpoint_path):
                saver.restore(sess, checkpoint_path)

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
                                                 self.test_pred_1hot, average="weighted")
                self.test_f1_micro = f1_score(self.test_true_1hot,
                                              self.test_pred_1hot, average="micro")
                logger.info("-- Test Results --")
                logger.info("F1-Score Weighted: {}".format(self.test_f1_weighted))
                logger.info("F1-Score Micro: {}".format(self.test_f1_micro))

                ## Compute_metrics by each label
                self.metrics_values = Metrics().compute_metrics(y_pred=self.test_pred_1hot,
                                                                y_true=self.test_true_1hot)
                self.time_testing = time.time() - testing_start

                ## Write metrics on testbet directory = self.monitor.testbed_exp
                if self.monitor.write_metrics == True: self.write_metrics()

                return self.test_pred_probas
            return train_pred

    def set_dataset_disk(self, dataset_name: str, dataset_path: str,
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
                raise AttributeError(
                    "training_disk() requires a datamanager class type, gives: {}".format(str(type(self.data))))
        except AttributeError:
            raise AttributeError(
                "training_disk() requires a datamanager class type, gives: {}".format(str(type(self.data))))
        self.time_dataset = time.time() - dataset_start
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
        train, valid, test = self.set_dataset_disk(dataset_name, dataset_path, inputs_name, targets_name)

        ### Training Start
        training_start = time.time()

        ## Generates a Desktop Graph
        self.model.desktop_graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config, graph=self.model.graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            sess.run(init)
            saver = tf.train.Saver()
            not_update = 0
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

                    train_loss, _ = sess.run([self.model.loss, self.model.grad_op],
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
                    valid_batch = Dataset()
                    valid_batch.set_data_file(valid_inputs, valid_targets)

                    valid_loss = sess.run(self.model.loss,
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
                logger.info(
                    "Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(
                        epoch, train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                self.training_track.append(
                    (epoch, train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                epoch = epoch + 1

                ## Early stopping when the validation loss decreases and train loss increases
                if valid_loss >= self.min_valid_loss and train_loss < self.min_train_loss:
                    not_update +=1

                if valid_loss <= self.min_valid_loss:
                    self.min_valid_loss = valid_loss

                if train_loss <= self.min_train_loss:
                    self.min_train_loss = train_loss

                ## While Stopping conditional
                if not_update >= self.early_stopping or epoch == self.max_epochs:
                    epoch_convergence = 1
                    self.max_epochs = epoch
                    saver.save(sess, str(self.monitor.testbed_exp + "/" + self.monitor.exp_id + "-model.ckpt"))
                    self.convergence_time = time.time() - training_start
                else:
                    epoch_convergence = 0

                self.time_training = time.time()-training_start
                ### end While loop

            ### Testing Starting
            testing_start = time.time()
            checkpoint_path = str(self.monitor.testbed_exp + "./" + self.monitor.exp_id + "-model.ckpt")
            if os.path.isfile(checkpoint_path):
                saver.restore(sess, checkpoint_path)

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
                                                 self.test_pred_1hot, average="weighted")
                self.test_f1_micro = f1_score(self.test_true_1hot,
                                              self.test_pred_1hot, average="micro")
                logger.info("-- Test Results --")
                logger.info("F1-Score Weighted: {}".format(self.test_f1_weighted))
                logger.info("F1-Score Micro: {}".format(self.test_f1_micro))

                ## compute_metrics by each label
                self.metrics_values = Metrics().compute_metrics(y_pred=self.test_pred_1hot,
                                                                y_true=self.test_true_1hot)
                self.time_testing = time.time() - testing_start

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
        track_path = str(self.monitor.testbed_exp + "/" + self.monitor.exp_id + "-training_track.txt")
        IO_Functions()._write_list(self.training_track, track_path)

        ## Writes the Test labels
        true_1h_path = str(self.monitor.testbed_exp + "/" + self.monitor.exp_id + "-true_1hot.txt")
        np.savetxt(true_1h_path, self.test_true_1hot, delimiter=',', fmt='%d')

        pred_1h_path = str(self.monitor.testbed_exp + "/" + self.monitor.exp_id + "-pred_1hot.txt")
        np.savetxt(pred_1h_path, self.test_pred_1hot, delimiter=',', fmt='%d')

        pred_probas_path = str(self.monitor.testbed_exp + "/" + self.monitor.exp_id + "-pred_probas.txt")
        np.savetxt(pred_probas_path, self.test_pred_probas, delimiter=',', fmt='%f')

        ## Writes Summarize Metrics
        metrics_values_path = str(self.monitor.testbed_exp + "/" + self.monitor.exp_id + "-metrics_values.txt")
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
        eda_json['results']['loss_validation'] = str(self.min_valid_loss)
        eda_json['results']['time_latency'] = self.time_latency
        eda_json['results']['time_dataset'] = self.time_dataset
        eda_json['results']['time_training'] = self.time_training
        eda_json['results']['time_testing'] = self.time_testing
        eda_json['results']['time_convergence'] = self.convergence_time

        ## End time metrics
        self.time_metrics = time.time() - metrics_start
        eda_json['results']['time_metrics'] = self.time_metrics

        ## Serialize the eda json and rewrite the file
        eda_json = json.dumps(eda_json, separators=(',', ': '), indent=2)
        file_path = str(self.monitor.testbed_exp + "/" + self.monitor.exp_id + "-exp_description.json")
        IO_Functions()._write_file(eda_json, file_path)

        ## End computational recording
        self.monitor.end_platform_recording()

        ## End power recording
        self.monitor.end_power_recording()

        logger.info("Tesbed directory: {}".format(self.monitor.testbed_exp))


class Distibuted_GRPC:
    """
    Implements the back-propagation algorithm ...
    Args:
        model: Is a graph object of the neural network architecture selected
    Returns:
    """

    def __init__(self, model, monitor: enerGyPU = None, datamanager: Dataset = None,
                    max_epochs: int = 10, early_stopping: int = 3,
                    ip_ps: str = "localhost:2222",
                    ip_workers: str = "localhost:2223") -> None:

        self.model = model
        self.data = datamanager
        self.max_epochs = max_epochs
        self.min_valid_loss = float("Infinity")
        self.min_train_loss = float("Infinity")
        self.early_stopping = early_stopping
        self.monitor = monitor

        ## Use Hosts IPs to set a tensorflow cluster object
        self.set_tf_cluster(ip_ps, ip_workers)

        ## Time logs
        self.time_latency: time()
        self.time_dataset: time()
        self.time_training: time()
        self.time_testing: time()
        self.time_metrics: time()

        ## Testbed and Metrics
        self.processing_mode: str
        self.training_track: list = []

        ## Get GPU availeble and set for processing
        # self.idgpu = self.monitor._get_available_GPU()
        self.idgpu = "0"
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=self.idgpu


    def set_tf_cluster(self, ip_ps, ip_workers) -> tf.Tensor:
        ## splitting the IP hosts
        ip_ps = ip_ps.split(",")
        ip_workers = ip_workers.split(",")
        self.ip_ps = ip_ps
        self.ip_workers = ip_workers
        # print("++++ ip_ workers: {}".format(ip_workers))

        self.num_ps = len(ip_ps)
        self.num_workers = len(ip_workers)

        ## Build a tf_ps collection
        tf_ps = []
        [tf_ps.append(str(ip_ps[i] + ":2222")) for i in range(len(ip_ps))]
        # tf_ps=','.join(tf_ps)
        # print("++ tf_ps: ",tf_ps)

        ## Build a tf_workers collection
        tf_workers = []
        [tf_workers.append(str(ip_workers[i] + ":2222")) for i in range(len(ip_workers))]
        # tf_workers=','.join(tf_workers)
        # print("++ tf_workers: ", tf_workers)

        self.tf_cluster = tf.train.ClusterSpec({"ps": tf_ps,  # ["134.59.132.135:2222"],
                                                "worker": tf_workers})  # ["134.59.132.20:2222"]})
        ## A collection of tf_ps nodes
        # return tf.train.ClusterSpec({"ps": tf_ps, "worker": tf_workers})

    def set_monitor_recording(self) -> None:
        """
        Power and performance monitoring launcher for workload characterization
        """
        latency_start = time.time()

        ## Closing the computing tracks
        ## End computational recording
        self.monitor.end_platform_recording()
        ## End power recording
        self.monitor.end_power_recording()

        if self.monitor == None:
            self.monitor = enerGyPU(testbed_path="testbed",
                                    machine_type="x86",
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

        ## Start bandwith recording
        if self.job_name == "worker": self.monitor.start_bandwidth_recording(self.ip_ps[0])

        ## Start platform recording
        if self.monitor.platform_recording == True: self.monitor.start_platform_recording(os.getpid())

        ## Get GPU availeble and set for processing
        # self.idgpu = self.monitor._get_available_GPU()
        self.idgpu = "0"
        # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"]=self.idgpu[0] #"3,4"

        ## Time recording
        self.time_latency = time.time() - latency_start

    def set_dataset_disk(self, dataset_name: str, dataset_path: str,
                         inputs_name: str, targets_name: str) -> BatchPath:
        """
        Uses datamanager classes for splitting, batching the dataset for distibuted training
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
                print("+++++ job_name: {} || index: {}".format(self.job_name, self.task_index))
                train, valid, test = self.data.distributed_batching(dataset_name, self.job_name, self.task_index)
                # train, valid, test = self.data.distributed_batching(1)

            else:
                raise AttributeError(
                    "training_disk() requires a datamanager class type, gives: {}".format(str(type(self.data))))
        except AttributeError:
            raise AttributeError(
                "training_disk() requires a datamanager class type, gives: {}".format(str(type(self.data))))
        self.time_dataset = time.time() - dataset_start
        return train, valid, test

    def create_done_queue(self, i):
        '''Queue used to signal death for i'th ps shard. Intended to have
        all workers enqueue an item onto it to signal doneness.'''
        print("******* def create_done_queue: /job:ps/task: -> {} ".format(i))
        with tf.device("/job:ps/task:%d" % (i)):
            return tf.FIFOQueue(self.num_workers, tf.int32, shared_name="done_queue" + str(i))

    def create_done_queues(self):
        print("****** def -> create_done_queues")
        return [self.create_done_queue(i) for i in range(self.num_ps)]

    def asynchronous_training(self, dataset_name: str, dataset_path: str,
                              inputs_name: str, targets_name: str,
                              job_name: str = "ps", task_index: int = 0) -> tf.Tensor:
        """
        Training the deep neural network exploit the memory on desktop machine
        """
        ## Set processing_mode flat
        self.processing_mode = "distributed_GRPC_async_processing"

        ## Define the machine rol = input flags
        ## start a server for a specific task
        self.job_name = job_name
        self.task_index = task_index
        self.server = tf.train.Server(self.tf_cluster,
                                      job_name=self.job_name,
                                      task_index=self.task_index)

        ## Set Monitor Recording
        if self.job_name == "worker": self.set_monitor_recording()

        ## Set dataset on memory
        train, valid, test = self.set_dataset_disk(dataset_name, dataset_path, inputs_name, targets_name)

        #print("++++ train: {}".format(train))

        ### Training Start
        training_start = time.time()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        if job_name == "ps":
            sess = tf.Session(self.server.target)
            queue = self.create_done_queue(self.task_index)

            ### Wait intil all workers are done
            for i in range(self.num_workers):
                sess.run(queue.dequeue())
                print("ps %d recieved done %d" % (self.task_index, i))
            print("ps %d: quitting" % (self.task_index))

            ## End computational recording
            self.monitor.end_platform_recording()
            ## End power recording
            self.monitor.end_power_recording()
            ## End bandwidth recording
            # self.monitor.end_bandwidth_recording()

        elif job_name == "worker":
            ## Generates a distributed graph object from graphs
            with tf.Graph().as_default() as distributed_graph:
                self.model.distributed_grpc_graph(self.tf_cluster, self.task_index)

                enq_ops = []
                for q in self.create_done_queues():
                    qop = q.enqueue(1)
                    enq_ops.append(qop)

                ##################################################
                ## Create a distributed session whit training supervisor
                # saver = tf.train.Saver()
                sv = tf.train.Supervisor(is_chief=(self.task_index == 0),
                                         graph=self.model.graph,  # saver=saver,
                                         # checkpoint_basename=str(),
                                         global_step=self.model.global_step,
                                         init_op=self.model.init_op)

                with sv.managed_session(self.server.target) as sess:
                    epoch: int = 0
                    not_update = 0
                    # saver = tf.train.Saver()
                    epoch_convergence: bin = 0
                    # print("**** epoch_convergence: {}".format(epoch_convergence))
                    while epoch_convergence == 0:  # (epoch < self.max_epochs):
                        epoch_start = time.time()

                        for i in range(len(train.input_files)):
                            train_inputs = IO_Functions()._read_file(train.input_files[i])
                            train_targets = IO_Functions()._read_file(train.target_files[i])
                            ## Convert list in a numpy matrix
                            train_batch = Dataset()
                            train_batch.set_data_file(train_inputs, train_targets)

                            train_loss, _ = sess.run([self.model.loss, self.model.grad_op],
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
                            valid_batch = Dataset()
                            valid_batch.set_data_file(valid_inputs, valid_targets)

                            valid_loss = sess.run(self.model.loss, feed_dict={self.model.X: valid_batch.inputs,
                                                                              self.model.Y: valid_batch.targets,
                                                                              self.model.keep_prob: 1.0})

                            valid_pred = sess.run(self.model.projection_1hot,
                                                  feed_dict={self.model.X: valid_batch.inputs,
                                                             self.model.keep_prob: 1.0})
                            ## F1_score from Skit-learn metrics
                            valid_acc = f1_score(y_true=valid_batch.targets.astype(np.float),
                                                 y_pred=valid_pred.astype(np.float), average='micro')

                        epoch_elapsed = (time.time() - epoch_start)

                        logger.info("Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(epoch, train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))
                        self.training_track.append((epoch, train_loss, valid_loss, train_acc, valid_acc, np.round(epoch_elapsed, decimals=4)))

                        epoch = epoch + 1

                        ## Early stopping when the validation loss decreases and train loss increases
                        if valid_loss >= self.min_valid_loss and train_loss < self.min_train_loss:
                            not_update +=1

                        if valid_loss <= self.min_valid_loss:
                            self.min_valid_loss = valid_loss

                        if train_loss <= self.min_train_loss:
                            self.min_train_loss = train_loss

                        ## While Stopping conditional
                        if not_update >= self.early_stopping or epoch == self.max_epochs:
                            self.max_epochs=epoch
                            if self.job_name == 'worker':
                                epoch_convergence = 1
                                self.convergence_time = time.time()-training_start
                            else:
                                epoch_convergence = 0
                                self.convergence_time = time.time()-training_start

                        self.time_training = time.time()-training_start
                        ## end While loop

                    ### Testing Starting
                    testing_start = time.time()
                    # saver.restore(sess,  str(self.monitor.testbed_exp+"./"+self.monitor.exp_id+ "-model.ckpt"))
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
                                                feed_dict={self.model.X: test_batch.inputs, self.model.keep_prob: 1.0})

                        test_pred_probas.append(tt_pred_probas)
                        test_pred_1hot.append(tt_pred_1hot)
                        test_true_1hot.append(test_batch.targets.astype(np.float))

                    self.test_pred_probas = np.vstack(test_pred_probas)
                    self.test_pred_1hot = np.vstack(test_pred_1hot)
                    self.test_true_1hot = np.vstack(test_true_1hot)

                    ## Compute the F1 Score
                    self.test_f1_weighted = f1_score(self.test_true_1hot,
                                                     self.test_pred_1hot, average="weighted")
                    self.test_f1_micro = f1_score(self.test_true_1hot,
                                                  self.test_pred_1hot, average="micro")
                    logger.info("-- Test Results --")
                    logger.info("F1-Score Weighted: {}".format(self.test_f1_weighted))
                    logger.info("F1-Score Micro: {}".format(self.test_f1_micro))

                    ## compute_metrics by each label
                    self.metrics_values = Metrics().compute_metrics(y_pred=self.test_pred_1hot,
                                                                    y_true=self.test_true_1hot)
                    self.time_testing = time.time() - testing_start

                    ## Write metrics on testbet directory = self.monitor.testbed_exp
                    if self.monitor.write_metrics == True: self.write_metrics()

                    ## signal to ps shards that we are done
                    for op in enq_ops:
                        sess.run(op)
                    print('-- Done! --')
                sv.stop()
            sess.close()
            ## End asynchronous_training


    def write_metrics(self) -> None:
        """
        Uses Testbed to isolate the training metrics by experiment directory
        """
        metrics_start = time.time()

        exp_id = str(os.uname()[1] + "-" + self.monitor.exp_id)

        ## Writes the training and validation track
        track_path = str(self.monitor.testbed_exp + "/" + exp_id + "-training_track.txt")
        IO_Functions()._write_list(self.training_track, track_path)

        ## Writes the Test labels
        true_1h_path = str(self.monitor.testbed_exp + "/" + exp_id + "-true_1hot.txt")
        np.savetxt(true_1h_path, self.test_true_1hot, delimiter=',', fmt='%d')

        pred_1h_path = str(self.monitor.testbed_exp + "/" + exp_id + "-pred_1hot.txt")
        np.savetxt(pred_1h_path, self.test_pred_1hot, delimiter=',', fmt='%d')

        pred_probas_path = str(self.monitor.testbed_exp + "/" + exp_id + "-pred_probas.txt")
        np.savetxt(pred_probas_path, self.test_pred_probas, delimiter=',', fmt='%f')

        ## Writes Summarize Metrics
        metrics_values_path = str(self.monitor.testbed_exp + "/" + exp_id + "-metrics_values.txt")
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
        eda_json['platform_parameters']['processing_job_name'] = self.job_name
        eda_json['platform_parameters']['processing_task_index'] = self.task_index
        # eda_json['platform_parameters']['processing_host_name'] = self.myhost

        ## Add values to results
        eda_json['results']['f1_score_weigted'] = self.test_f1_weighted
        eda_json['results']['f1_score_micro'] = self.test_f1_micro
        eda_json['results']['loss_validation'] = str(self.min_valid_loss)
        eda_json['results']['time_latency'] = self.time_latency
        eda_json['results']['time_dataset'] = self.time_dataset
        eda_json['results']['time_training'] = self.time_training
        eda_json['results']['time_convergence'] = self.convergence_time

        ## End time metrics
        self.time_metrics = time.time() - metrics_start
        eda_json['results']['time_metrics'] = self.time_metrics

        ## Serialize the eda json and rewrite the file
        eda_json = json.dumps(eda_json, separators=(',', ': '), indent=2)
        file_path = str(self.monitor.testbed_exp + "/" + exp_id + "-exp_description.json")
        IO_Functions()._write_file(eda_json, file_path)

        ## End computational recording
        self.monitor.end_platform_recording()

        ## End power recording
        self.monitor.end_power_recording()

        ## End bandwidth recording
        self.monitor.end_bandwidth_recording()

        logger.info("Tesbed directory: {}".format(self.monitor.testbed_exp))


class MultiGPU:
    """
    Implements the back-propagation algorithm ...
    Args:
        model: Is a graph object of the neural network architecture selected
    Returns:
    """

    def __init__(self, model, monitor: enerGyPU = None,
                 datamanager: Dataset = None,
                 gpus: int = 2, max_epochs: int = 10) -> None:
        self.model = model
        self.monitor = monitor
        self.data = datamanager
        self.num_gpus = gpus
        self.max_epochs = max_epochs

        ## Testbed and Metrics
        testbed_path: str = 'testbed'
        self.training_track: list = []

        #######################################################################
        ## MultiGPU
        # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"]="5,6"
        # self.num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    def set_monitor_recording(self) -> None:
        """
        Power and performance monitoring launcher for workload characterization
        """
        latency_start = time.time()
        if self.monitor == None:
            self.monitor = enerGyPU(testbed_path="testbed",
                                    write_metrics=False,
                                    power_recording=True,
                                    platform_recording=False)

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
        self.idgpu = self.monitor._get_available_GPU()
        # self.idgpu = "4"
        # print()

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
        self.num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

        ## Time recording
        self.time_latency = time.time() - latency_start

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
                batch_size = inputs.shape[0]
            elif 'list' in str(type(inputs)):
                batch_size = len(inputs)
            else:
                raise AttributeError("set_data_file(inputs, targets) requires: numpy, pandas or list ")
            self.data = Batching(batch_size=batch_size, valid_size=0.1, test_size=0)
            self.data.set_data_file(inputs, targets)
            train, valid, test = self.data.memory_batching()

        return train, valid, test

    # def training_multigpu(self, inputs: np.ndarray, targets: np.ndarray) -> tf.Tensor:
    #     """
    #     Training the deep neural network exploit the memory on desktop machine
    #     """
    #     ## Set dataset on memory
    #     train, valid, test = self.set_dataset_memory(inputs, targets)
    #     ## Generates a Desktop Graph
    #     self.model.multiGPU_graph(self.data.batch_size, self.num_gpus)
    #     print("++ Trainig Graph on MultiGPU ++")
    #
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     config.intra_op_parallelism_threads = 16
    #
    #     # config.gpu_options.per_process_gpu_memory_fraction = 0.4
    #
    #     with tf.Session(config=config, graph=self.model.mlp_graph) as sess:
    #         init = tf.group(tf.global_variables_initializer(),
    #                         tf.local_variables_initializer())
    #         sess.run(init)
    #
    #         epoch: int = 0
    #         while epoch < self.max_epochs:
    #             epoch_start = time.time()
    #             for i in range(len(train.inputs)):
    #
    #                 ### Temporaly conditional
    #                 if train.inputs[i].shape[0] < (self.data.batch_size/2):
    #                     print("+++++++++")
    #                     print("train exclude -> {}".format(train.inputs[i].shape))
    #                     print("+++++++++")
    #
    #                 else:
    #                     train_pred = sess.run(self.model.output1,
    #                                 feed_dict={self.model.X: train.inputs[i],
    #                                         self.model.keep_prob: self.model.dropout})
    #
    #                     # print("train.inputs {}".format(train.inputs[i].shape))
    #                     # print("train_targets: {}".format(train.targets[i].shape))
    #                     print("train_pred: {}".format(train_pred.shape))
    #
    #                     train_loss = sess.run(self.model.output2,
    #                                 feed_dict={self.model.X: train.inputs[i],
    #                                         self.model.Y: train.targets[i],
    #                                         self.model.keep_prob: self.model.dropout})
    #
    #                     print("train_loss: {}".format(train_loss))
    #
    #
    #
    #                     train_grads = sess.run(self.model.train_op,
    #                                 feed_dict={self.model.X: train.inputs[i],
    #                                         self.model.Y: train.targets[i],
    #                                         self.model.keep_prob: self.model.dropout})
    #
    #                     print("train_grads: {}".format(train_grads))
    #
    #             epoch_elapsed = (time.time() - epoch_start)
    #             epoch = epoch + 1

    ###############################################################################"
    ###############################################################################"

    def stacked_multigpu(self, input_holder, keep_prob, reuse) -> tf.Tensor:
        """
        """
        # with tf.variable_scope("BackPropagation", reuse=reuse):
        w1 = tf.Variable(tf.random_normal([14637, 2048], stddev=0.1), dtype=tf.float32)
        b1 = tf.Variable(tf.random_normal([2048]), dtype=tf.float32)
        l1 = tf.nn.relu(tf.matmul(input_holder, w1) + b1)

        w2 = tf.Variable(tf.random_normal([2048, 14], stddev=0.1), dtype=tf.float32)
        b2 = tf.Variable(tf.random_normal([14]), dtype=tf.float32)
        l2 = tf.matmul(l1, w2 + b2)
        return l2

    def multiGPU_loss(self, y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
        """
        """
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        cross_entropy_reduce = tf.reduce_mean(cross_entropy)

        return cross_entropy_reduce

    def average_gradients(self, tower_grads):
        """
        Merge the grads computations done by each GPU tower
        """
        ### First Print
        print("\n \n")
        # print("tower_grads: {}".format(tower_grads))
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            ## Second print
            # print("grad_and_vars: {}".format(grad_and_vars))
            grads = []
            for g, _ in grad_and_vars:
                ## Third Print
                print("+ Grad by Tower: {}".format(g))
                if g is None:
                    pass
                else:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(g, 0)

                    # Append on a 'tower' dimension which we will average over below.
                    grads.append(expanded_g)

            #### JAGH DEbug
            #         grads.append(g)
            # return grads

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def training_multigpu(self, inputs: np.ndarray, targets: np.ndarray) -> tf.Tensor:
        """
        Training the deep neural network exploit the memory on desktop machine
        """
        ## Set processing_mode flat
        self.processing_mode = "multiGPU"
        ## Set Monitor Recording
        self.set_monitor_recording()

        ## Set dataset on memory
        train, valid, test = self.set_dataset_memory(inputs, targets)
        ## Generates a Desktop Graph

        ######################################################################
        self.gpu_batch_size = int((self.data.batch_size / self.num_gpus))

        # with tf.Graph().as_default() as self.mlp_graph:
        with tf.device('/cpu:0'):
            ###########################
            self.total_projection = []
            self.total_losses = []
            self.total_grads = []

            self.X = tf.placeholder(tf.float32, shape=(None, 14637), name="Inputs")
            self.Y = tf.placeholder(tf.float32, shape=(None, 14), name="Targets")
            self.keep_prob = tf.placeholder(tf.float32)

            self.adam_op = tf.train.AdamOptimizer(learning_rate=0.001)
            reuse_vars = False

            ################
            for gpu in range(self.num_gpus):
                with tf.device('/gpu:%d' % gpu):
                    # tf.variable_scope.reuse_variables()
                    # Split data between GPUs
                    self._X = self.X[(gpu * self.gpu_batch_size):
                                     (gpu * self.gpu_batch_size) + (self.gpu_batch_size)]
                    self._Y = self.Y[(gpu * self.gpu_batch_size):
                                     (gpu * self.gpu_batch_size) + (self.gpu_batch_size)]

                    ## Projection by Tower Model operations
                    if gpu == 0:
                        with tf.variable_scope("BackPropagation", reuse=True):
                            self.projection = self.stacked_multigpu(self._X, self.keep_prob, reuse_vars)

                    else:
                        with tf.variable_scope("BackPropagation", reuse=True):
                            self.projection = self.stacked_multigpu(self._X, self.keep_prob, reuse_vars)
                    self.total_projection.append(self.projection)

                    # ## Loss by Tower Model operations
                    self.loss = self.multiGPU_loss(self.projection, self._Y)
                    self.total_losses.append(self.loss)

                    ## Grads by Tower Model operations
                    self.grads_computation = self.adam_op.compute_gradients(self.loss)
                    # reuse_vars = True
                    self.total_grads.append(self.grads_computation)

                    print("{}".format("+" * 20))
                    print("+ GPU: {}".format(gpu))
                    print("+ Split_X: {}, {}".format((gpu * self.gpu_batch_size),
                                                     (gpu * self.gpu_batch_size) + (self.gpu_batch_size)))
                    print("+ Tower_Projection: {}".format(self.projection.name))
                    print("{}".format("+" * 20))

            with tf.device('/cpu:0'):
                self.output1 = tf.concat(self.total_projection, axis=0)
                self.output2 = self.total_losses
                self.output3 = self.average_gradients(self.total_grads)
                self.train_op = tf.group(self.adam_op.apply_gradients(self.output3))

        #################################################################""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 16

        # config.gpu_options.per_process_gpu_memory_fraction = 0.4

        with tf.Session(config=config) as sess:
            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            sess.run(init)

            epoch: int = 0
            while epoch < self.max_epochs:
                epoch_start = time.time()
                for i in range(len(train.inputs)):

                    ### Temporaly conditional
                    if train.inputs[i].shape[0] < (self.data.batch_size / 2):
                        print("+++++++++")
                        print("train exclude -> {}".format(train.inputs[i].shape))
                        print("+++++++++")

                    else:
                        train_pred = sess.run(self.output1,
                                              feed_dict={self.X: train.inputs[i],
                                                         self.keep_prob: self.model.dropout})

                        # print("train.inputs {}".format(train.inputs[i].shape))
                        # print("train_targets: {}".format(train.targets[i].shape))
                        print("train_pred: {}".format(train_pred.shape))

                        train_loss = sess.run(self.output2,
                                              feed_dict={self.X: train.inputs[i],
                                                         self.Y: train.targets[i],
                                                         self.keep_prob: self.model.dropout})

                        print("train_loss: {}".format(train_loss))

                        train_grads = sess.run(self.train_op,
                                               feed_dict={self.X: train.inputs[i],
                                                          self.Y: train.targets[i],
                                                          self.keep_prob: self.model.dropout})

                        print("train_grads: {}".format(train_grads))

                epoch_elapsed = (time.time() - epoch_start)
                epoch = epoch + 1

            ## Print the sandbox
            logger.info("Tesbed directory: {}".format(self.monitor.testbed_exp))

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


from mpi4py import MPI
class Distibuted_MPI:

    def __init__(self, model, monitor: enerGyPU = None, datamanager: Dataset = None,
                 max_epochs: int = 10, early_stopping: int = 3) -> None:
        self.model = model
        self.data = datamanager
        self.max_epochs = max_epochs
        self.min_valid_loss = float("Infinity")
        self.min_train_loss = float("Infinity")
        self.early_stopping = early_stopping
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

        ## MPI parameters
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.myhost = MPI.Get_processor_name()
        self.status = MPI.Status()

        ## Distributed batching role definition
        if self.rank == 0:
            self.job_name = 'ps'
            self.task_index = self.rank
        else:
            self.job_name = 'worker'
            self.task_index = self.rank

    def set_monitor_recording(self) -> None:
        """
        Power and performance monitoring launcher for workload characterization
        """
        latency_start = time.time()

        ## Closing the computing tracks
        ## End computational recording
        self.monitor.end_platform_recording()
        ## End power recording
        self.monitor.end_power_recording()

        if self.monitor == None:
            self.monitor = enerGyPU(testbed_path="testbed",
                                    machine_type="x86",
                                    write_metrics=True,
                                    power_recording=True,
                                    platform_recording=True)

        platform_name = (str(self.__class__.__name__) + "-" + str(self.size) + "-" + str(self.rank))
        ## Generate ID-experiment and their testebed directory
        self.monitor.generate_testbed(self.monitor.testbed_path,
                                      self.model, self.data,
                                      platform_name,
                                      self.max_epochs)

        ## Start power recording
        if self.monitor.power_recording == True: self.monitor.start_power_recording()

        ## Start platform recording
        if self.monitor.platform_recording == True: self.monitor.start_platform_recording(os.getpid())

        ## Get GPU availeble and set for processing
        self.idgpu = "0"
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = self.idgpu[0]

        ## Time recording
        self.time_latency = time.time() - latency_start

    def set_dataset_disk(self, dataset_name: str, dataset_path: str,
                         inputs_name: str, targets_name: str) -> BatchPath:
        """
        Uses datamanager classes for splitting, batching the dataset for distibuted training
        """
        dataset_start = time.time()

        print("+++ Type: {}".format(type(self.data)))
        try:
            self.data.set_data_path(dataset_name=dataset_name,
                                    dataset_path=dataset_path,
                                    inputs_name=inputs_name,
                                    targets_name=targets_name)
            if 'MultiTask' in str(type(self.data)):
                train, valid, test = self.data.disk_one_target()
            elif 'Batching' in str(type(self.data)):
                if self.rank != 0:
                    train, valid, test = self.data.distributed_batching(dataset_name, self.job_name,
                                                                        self.task_index - 1)
                else:
                    train, valid, test = self.data.distributed_batching(dataset_name, self.job_name,
                                                                        self.task_index)
            else:
                raise AttributeError(
                    "training_disk() requires a datamanager class type, gives: {}".format(str(type(self.data))))
        except AttributeError:
            raise AttributeError(
                "training_disk() requires a datamanager class type, gives: {}".format(str(type(self.data))))
        self.time_dataset = time.time() - dataset_start
        return train, valid, test

    def asynchronous_training(self, dataset_name: str, dataset_path: str,
                              inputs_name: str, targets_name: str, weighting: int = 1) -> tf.Tensor:

        self.processing_mode = "distributed_MPI_async_processing"
        ## Set Monitor Recording
        self.set_monitor_recording()
        self.best_model_weights = None
        ## Set distributed dataset
        ## Master split and batch the dataset by worker
        if self.rank == 0:
            train, valid, test = self.set_dataset_disk(dataset_name, dataset_path,
                                                       inputs_name, targets_name)
            worker_batching = 1
        else:
            worker_batching = None
        worker_batching = self.comm.bcast(worker_batching, root=0)
        ## Each worker read their dataset proportion
        if worker_batching == 1:
            train, valid, test = self.set_dataset_disk(dataset_name, dataset_path, inputs_name, targets_name)

        training_start = time.time()
        self.model.distributed_mpi_graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config, graph=self.model.graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            sess.run(init)
            epoch: int = 0
            not_update = 0
            master_queue = []
            saver = tf.train.Saver()
            epoch_convergence: bin = 0
            model_weights = None
            while (epoch_convergence == 0):
                epoch_start = time.time()
                acc, train_loss, val_acc, valid_loss = 0, 0, 0, 0
                if self.rank != 0:
                    for i in range(len(train.input_files)):
                        train_inputs = IO_Functions()._read_file(train.input_files[i])
                        train_targets = IO_Functions()._read_file(train.target_files[i])
                        ## Convert list in a numpy matrix
                        train_batch = Dataset()
                        train_batch.set_data_file(train_inputs, train_targets)
                        if i == (len(train.input_files) - 1):
                            grads, train_loss, train_pred = sess.run(
                                [self.model._grad_op, self.model.loss, self.model.projection_1hot],
                                feed_dict={self.model.X: train_batch.inputs,
                                           self.model.Y: train_batch.targets,
                                           self.model.keep_prob: self.model.dropout})
                        else:
                            _, train_loss, train_pred = sess.run(
                                [self.model.sub_grad_op, self.model.loss, self.model.projection_1hot],
                                feed_dict={self.model.X: train_batch.inputs,
                                           self.model.Y: train_batch.targets,
                                           self.model.keep_prob: self.model.dropout})

                        ## F1_score from Skit-learn metrics
                        train_acc = f1_score(y_true=train_batch.targets.astype(np.float),
                                             y_pred=train_pred.astype(np.float), average='micro')
                        acc += train_acc / len(train.input_files)
                        train_loss += train_loss / len(train.input_files)
                if self.rank == 0:
                    weight_collection = []
                    weight_recv, epoch_recv = self.comm.recv(source=MPI.ANY_SOURCE,
                                                             status=self.status, tag=0)
                    # if this is the first epoch, we don't have previous weights
                    if epoch == 0:
                        model_weights = weight_recv
                    # add previous weights to the collection
                    weight_collection.append(model_weights)
                    weight_collection.append(weight_recv)
                    if epoch_recv != 0:
                        master_queue.append(epoch_recv)
                    if len(master_queue) == self.size - 1:
                        epoch_convergence = 1
                        self.convergence_time = time.time() - training_start
                    # compute the weighted average of the gradient
                    average_weights = [np.average(np.stack([g[i] for g in weight_collection], axis=0),
                                                  axis=0,
                                                  weights=[weighting / (weighting + 1), 1 / (weighting + 1)]) for i in
                                       range(len(weight_collection[0]))]
                    # send it to the source of the last reception
                    source = self.status.Get_source()
                    self.comm.send(average_weights, dest=source, tag=1)
                else:
                    if epoch == self.max_epochs or not_update >= self.early_stopping:
                        epoch_convergence = 1
                        self.max_epochs = epoch
                    self.comm.send([grads, epoch_convergence], dest=0, tag=0)
                    _weights = self.comm.recv(source=0, tag=1)
                    feed_dict = {}
                    self.model._gradients = _weights
                    for i, placeholder in enumerate(self.model._grad_placeholders):
                        feed_dict[placeholder] = self.model._gradients[i]

                    sess.run(self.model._train_op, feed_dict=feed_dict)
                    model_weights = feed_dict

                if self.rank != 0:
                    for i in range(len(valid.input_files)):
                        valid_inputs = IO_Functions()._read_file(valid.input_files[i])
                        valid_targets = IO_Functions()._read_file(valid.target_files[i])
                        ## Convert list in a numpy matrix
                        valid_batch = Dataset()
                        valid_batch.set_data_file(valid_inputs, valid_targets)

                        valid_loss = sess.run(self.model.loss,
                                              feed_dict={self.model.X: valid_batch.inputs,
                                                         self.model.Y: valid_batch.targets,
                                                         self.model.keep_prob: 1.0})
                        valid_pred = sess.run(self.model.projection_1hot,
                                              feed_dict={self.model.X: valid_batch.inputs,
                                                         self.model.keep_prob: 1.0})
                        ## F1_score from Skit-learn metrics
                        valid_acc = f1_score(y_true=valid_batch.targets.astype(np.float),
                                             y_pred=valid_pred.astype(np.float), average='micro')
                        val_acc += valid_acc / len(valid.input_files)
                        valid_loss += valid_loss / len(valid.input_files)

                    epoch_elapsed = (time.time() - epoch_start)
                    logger.info("Worker {} | Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(self.rank, epoch, train_loss, valid_loss, acc, val_acc, np.round(epoch_elapsed, decimals=4)))
                    self.training_track.append((epoch, train_loss, valid_loss, acc, val_acc, np.round(epoch_elapsed, decimals=4)))

                epoch = epoch + 1
                ## Early stopping when the validation loss decreases and train loss increases
                if self.rank != 0:
                    if valid_loss >= self.min_valid_loss and train_loss < self.min_train_loss:
                        not_update += 1
                    if valid_loss <= self.min_valid_loss:
                        self.best_model_weights = model_weights
                        self.min_valid_loss = valid_loss
                        self.convergence_time = time.time() - training_start
                    if train_loss <= self.min_train_loss:
                        self.min_train_loss = train_loss

                self.time_training = time.time() - training_start
                ## end While loop


            # make workers that finished traininig wait for others to finish
            self.comm.Barrier()
            print(self.rank, "finishes training ...")

            ### Testing Starting
            testing_start = time.time()

            if self.rank != 0:
                test_pred_probas: list = []
                test_pred_1hot: list = []
                test_true_1hot: list = []
                sess.run(self.model._train_op, feed_dict=self.best_model_weights)

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

            if self.rank == 0:
                test_true_1hot, test_pred_probas, test_pred_1hot = [], [], []
                for i in range(1, self.size):
                    tmp1, tmp2, tmp3 = self.comm.recv(source=i, tag=4)
                    test_pred_probas.append(np.vstack(tmp1))
                    test_pred_1hot.append(np.vstack(tmp2))
                    test_true_1hot.append(np.vstack(tmp3))
            else:
                self.comm.send([test_pred_probas, test_pred_1hot, test_true_1hot], dest=0, tag=4)

            self.test_pred_probas = np.vstack(test_pred_probas)
            self.test_pred_1hot = np.vstack(test_pred_1hot)
            self.test_true_1hot = np.vstack(test_true_1hot)

            ## Compute the F1 Score
            self.test_f1_weighted = f1_score(self.test_true_1hot,
                                             self.test_pred_1hot, average="weighted")
            self.test_f1_micro = f1_score(self.test_true_1hot,
                                          self.test_pred_1hot, average="micro")
            if self.rank == 0:
                logger.info("-- Test Results --")
                logger.info("F1-Score Weighted: {}".format(self.test_f1_weighted))
                logger.info("F1-Score Micro: {}".format(self.test_f1_micro))

            ## Compute_metrics by each label
            self.metrics_values = Metrics().compute_metrics(y_pred=self.test_pred_1hot,
                                                            y_true=self.test_true_1hot)
            self.time_testing = time.time() - testing_start

            ## Write metrics on testbet directory = self.monitor.testbed_exp
            if self.monitor.write_metrics == True: self.write_metrics()

            sess.close()
            return self.test_pred_probas

    def synchronous_training(self, dataset_name: str, dataset_path: str,
                             inputs_name: str, targets_name: str) -> tf.Tensor:

        self.processing_mode = "distributed_MPI_sync_processing"
        ## Set Monitor Recording
        self.set_monitor_recording()
        ## Set dataset on disk
        train, valid, test = self.set_dataset_disk(dataset_name, dataset_path,
                                                   inputs_name, targets_name)

        training_start = time.time()
        self.model.distributed_mpi_graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config, graph=self.model.graph) as sess:
            init = tf.group(tf.global_variables_initializer(),
                            tf.local_variables_initializer())
            sess.run(init)
            epoch: int = 0
            not_update = 0
            saver = tf.train.Saver()
            epoch_convergence: bin = 0
            while (epoch_convergence == 0):
                epoch_start = time.time()
                acc, train_loss, val_acc, valid_loss = 0, 0, 0, 0
                update_flag = False

                if self.rank != 0:
                    for i in range(len(train.input_files)):

                        train_inputs = IO_Functions()._read_file(train.input_files[i])
                        train_targets = IO_Functions()._read_file(train.target_files[i])
                        ## Convert list in a numpy matrix
                        train_batch = Dataset()
                        train_batch.set_data_file(train_inputs, train_targets)

                        if i == (len(train.input_files) - 1):
                            grads, train_loss, train_pred = sess.run(
                                [self.model._grad_op, self.model.loss, self.model.projection_1hot],
                                feed_dict={self.model.X: train_batch.inputs,
                                           self.model.Y: train_batch.targets,
                                           self.model.keep_prob: self.model.dropout})
                        else:
                            _, train_loss, train_pred = sess.run(
                                [self.model.sub_grad_op, self.model.loss, self.model.projection_1hot],
                                feed_dict={self.model.X: train_batch.inputs,
                                           self.model.Y: train_batch.targets,
                                           self.model.keep_prob: self.model.dropout})

                        ## F1_score from Skit-learn metrics
                        train_acc = f1_score(y_true=train_batch.targets.astype(np.float),
                                             y_pred=train_pred.astype(np.float), average='micro')
                        acc += train_acc / len(train.input_files)
                        train_loss += train_loss / len(train.input_files)

                if self.rank == 0:
                    weight_collection = []
                    for i in range(1, self.size):
                        weight_recv, acc_recv, loss_recv = self.comm.recv()
                        weight_collection.append(weight_recv)
                        acc += acc_recv / (self.size - 1)
                        train_loss += loss_recv / (self.size - 1)
                    average_weights = [np.stack([g[i] for g in weight_collection], axis=0).mean(axis=0) for i in
                                       range(len(weight_collection[0]))]
                    for i in range(1, self.size):
                        self.comm.send(average_weights, dest=i)
                else:
                    self.comm.send([grads, acc, train_loss], dest=0)
                    _weights = self.comm.recv(source=0)
                    feed_dict = {}
                    self.model._gradients = _weights
                    for i, placeholder in enumerate(self.model._grad_placeholders):
                        feed_dict[placeholder] = self.model._gradients[i]
                    sess.run(self.model._train_op, feed_dict=feed_dict)
                    update_weight = feed_dict

                if self.rank != 0:
                    for i in range(len(valid.input_files)):
                        valid_inputs = IO_Functions()._read_file(valid.input_files[i])
                        valid_targets = IO_Functions()._read_file(valid.target_files[i])
                        ## Convert list in a numpy matrix
                        valid_batch = Dataset()
                        valid_batch.set_data_file(valid_inputs, valid_targets)

                        valid_loss = sess.run(self.model.loss,
                                              feed_dict={self.model.X: valid_batch.inputs,
                                                         self.model.Y: valid_batch.targets,
                                                         self.model.keep_prob: 1.0})
                        valid_pred = sess.run(self.model.projection_1hot,
                                              feed_dict={self.model.X: valid_batch.inputs,
                                                         self.model.keep_prob: 1.0})
                        ## F1_score from Skit-learn metrics
                        valid_acc = f1_score(y_true=valid_batch.targets.astype(np.float),
                                             y_pred=valid_pred.astype(np.float), average='micro')
                        val_acc += valid_acc / len(valid.input_files)
                        valid_loss += valid_loss / len(valid.input_files)

                epoch_elapsed = (time.time() - epoch_start)
                if self.rank == 0:
                    for i in range(1, self.size):
                        val_acc_recv, val_loss_recv = self.comm.recv()
                        val_acc += val_acc_recv / (self.size - 1)
                        valid_loss += val_loss_recv / (self.size - 1)
                    logger.info("Epoch {} | Train loss: {} |  Valid loss: {} | Train Acc: {} | Valid Acc: {} | Epoch_Time: {}".format(epoch, loss, valid_loss, acc, val_acc, np.round(epoch_elapsed, decimals=4)))
                else:
                    self.comm.send([val_acc, valid_loss], dest=0)
                self.training_track.append(
                    (epoch, train_loss, valid_loss, acc, val_acc, np.round(epoch_elapsed, decimals=4)))

                epoch = epoch + 1

                ## Early stopping when the validation loss decreases and train loss increases
                if valid_loss >= self.min_valid_loss and train_loss < self.min_train_loss:
                    not_update += 1
                if valid_loss <= self.min_valid_loss:
                    self.min_valid_loss = valid_loss
                    self.convergence_time = time.time() - training_start
                    update_flag = True
                if train_loss <= self.min_train_loss:
                    self.min_train_loss = train_loss
                # else:
                #     not_update += 1

                ## While Stopping conditional
                if not_update >= self.early_stopping or epoch == self.max_epochs:
                    self.max_epochs = epoch
                    if self.rank == 0:
                        epoch_convergence = 1
                else:
                    epoch_convergence = 0

                if self.rank == 0:
                    for i in range(1, self.size):
                        self.comm.send([epoch_convergence, update_flag], dest=i)
                else:
                    epoch_convergence, update_flag = self.comm.recv(source=0)
                    if update_flag == True:
                        self.best_model_weights = update_weight

                self.time_training = time.time() - training_start
                ### end While loop

            ### Testing Starting
            testing_start = time.time()

            if self.rank != 0:
                test_pred_probas: list = []
                test_pred_1hot: list = []
                test_true_1hot: list = []
                sess.run(self.model._train_op, feed_dict=self.best_model_weights)

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

            if self.rank == 0:
                test_true_1hot, test_pred_probas, test_pred_1hot = [], [], []
                for i in range(1, self.size):
                    tmp1, tmp2, tmp3 = self.comm.recv(source=i)
                    test_pred_probas.append(np.vstack(tmp1))
                    test_pred_1hot.append(np.vstack(tmp2))
                    test_true_1hot.append(np.vstack(tmp3))
            else:
                self.comm.send([test_pred_probas, test_pred_1hot, test_true_1hot], dest=0)

            self.test_pred_probas = np.vstack(test_pred_probas)
            self.test_pred_1hot = np.vstack(test_pred_1hot)
            self.test_true_1hot = np.vstack(test_true_1hot)

            ## Compute the F1 Score
            self.test_f1_weighted = f1_score(self.test_true_1hot,
                                             self.test_pred_1hot, average="weighted")
            self.test_f1_micro = f1_score(self.test_true_1hot,
                                          self.test_pred_1hot, average="micro")
            if self.rank == 0:
                logger.info("-- Test Results --")
                logger.info("F1-Score Weighted: {}".format(self.test_f1_weighted))
                logger.info("F1-Score Micro: {}".format(self.test_f1_micro))

            ## Compute_metrics by each label
            self.metrics_values = Metrics().compute_metrics(y_pred=self.test_pred_1hot,
                                                            y_true=self.test_true_1hot)
            self.time_testing = time.time() - testing_start

            ## Write metrics on testbet directory = self.monitor.testbed_exp
            if self.monitor.write_metrics == True: self.write_metrics()

            sess.close()

            return self.test_pred_probas

    def write_metrics(self) -> None:
        """
        Uses Testbed to isolate the training metrics by experiment directory
        """
        metrics_start = time.time()

        ## Writes the training and validation track
        track_path = str(self.monitor.testbed_exp + "/" + self.monitor.exp_id + "-training_track.txt")
        IO_Functions()._write_list(self.training_track, track_path)

        ## Writes the Test labels
        true_1h_path = str(self.monitor.testbed_exp + "/" + self.monitor.exp_id + "-true_1hot.txt")
        np.savetxt(true_1h_path, self.test_true_1hot, delimiter=',', fmt='%d')

        pred_1h_path = str(self.monitor.testbed_exp + "/" + self.monitor.exp_id + "-pred_1hot.txt")
        np.savetxt(pred_1h_path, self.test_pred_1hot, delimiter=',', fmt='%d')

        pred_probas_path = str(self.monitor.testbed_exp + "/" + self.monitor.exp_id + "-pred_probas.txt")
        np.savetxt(pred_probas_path, self.test_pred_probas, delimiter=',', fmt='%f')

        ## Writes Summarize Metrics
        metrics_values_path = str(self.monitor.testbed_exp + "/" + self.monitor.exp_id + "-metrics_values.txt")
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
        eda_json['platform_parameters']['processing_size'] = self.size
        eda_json['platform_parameters']['processing_rank'] = self.rank
        eda_json['platform_parameters']['processing_host_name'] = self.myhost

        ## Add values to results
        eda_json['results']['f1_score_weigted'] = self.test_f1_weighted
        eda_json['results']['f1_score_micro'] = self.test_f1_micro
        eda_json['results']['loss_validation'] = str(self.min_valid_loss)
        eda_json['results']['time_latency'] = self.time_latency
        eda_json['results']['time_dataset'] = self.time_dataset
        eda_json['results']['time_training'] = self.time_training
        eda_json['results']['time_convergence'] = self.convergence_time

        ## End time metrics
        self.time_metrics = time.time() - metrics_start
        eda_json['results']['time_metrics'] = self.time_metrics

        ## Serialize the eda json and rewrite the file
        eda_json = json.dumps(eda_json, separators=(',', ': '), indent=2)
        file_path = str(self.monitor.testbed_exp + "/" + self.monitor.exp_id + "-exp_description.json")
        IO_Functions()._write_file(eda_json, file_path)

        ## End computational recording
        self.monitor.end_platform_recording()

        ## End power recording
        self.monitor.end_power_recording()

        logger.info("Tesbed directory: {}".format(self.monitor.testbed_exp))
