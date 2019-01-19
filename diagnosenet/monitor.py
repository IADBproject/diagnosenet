"""
Integrated performance monitor for workload characterization
"""

from time import gmtime, strftime
import hashlib
import json

import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support

import subprocess as sp
import psutil, datetime, os

from diagnosenet.io_functions import IO_Functions

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

    def auc_roc(self, y_pred, y_true):
        """
        Compute AUC | Note that the y_pred feeded to auc_roc is one hot encoded
        Get the indexes of the maximum values in each row and
        y_pred is the output of softmax function
        """

        ## roc_curve need values as float
        y_true = y_true.astype(np.float)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        n_classes = y_true.shape[1]

        for i in range(n_classes):
            fpr[i], tpr[i] , thresholds = roc_curve(y_true[:,i], y_pred[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        #Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], thresholds = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        return roc_auc

    def compute_metrics(self, y_true, y_pred):
        """
        This function return a summarize matrix with:
        true_positive, false_positive, false_negative, precision, recall and F1_score.
        """

        y_true = y_true.astype(np.float)
        y_pred = y_pred.astype(np.float)
        y_true =  np.argmax(y_true, axis = 1)
        y_pred = np.argmax(y_pred, axis = 1)

        ## Get labels from y_true
        labels, counts = np.unique(y_true, return_counts = True)

        ## Compute confusion_matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels)

        ## Compute True Positive
        true_positive = np.diag(conf_matrix)

        ## Compute False Negative and False Positive
        false_negative = []
        false_positive = []
        for i in range(len(conf_matrix)):
            false_positive.append(int(sum(conf_matrix[:,i]) - conf_matrix[i,i]))
            false_negative.append(int(sum(conf_matrix[i,:]) - conf_matrix[i,i]))

        ## Compute True negative
        # true_negative = []
        # for i in range(len(conf_matrix)):
        #     temp = np.delete(conf_matrix, i, 0)
        #     temp = np.delete(temp, i, 1)
        #     true_negative.append(int(sum(sum(temp))))

        ## Compute metrics per class
        precision, recall, F1_score, support = precision_recall_fscore_support(y_true,
                                                        y_pred, average = None)

        for i in range(len(labels)):
            precision[i] =  round(precision[i], 2)
            recall[i] =  round(recall[i], 2)
            F1_score[i] = round(F1_score[i], 2)


        label_occurrences = np.where(support !=0)
        occs = label_occurrences[0]
        metrics_values = np.vstack((labels, true_positive, false_negative,
                                    false_positive, precision[occs],
                                    recall[occs], F1_score[occs], support[occs]))
        metrics_values = np.transpose(metrics_values)
        metrics_values = pd.DataFrame(metrics_values, columns = ["Labels", "TP", "FN", "FP",
                                    "Precision", "Recall", "F1 Score", "Records by Labels"])
        print("{}".format(metrics_values))

        return metrics_values


class Testbed(Metrics):
    """
    Build an experiment directory to isolate the training metrics files
    """
    def __init__(self, testbed_path, write_metrics) -> None:
        super().__init__()
        self.testbed_path = testbed_path
        self.write_metrics = write_metrics
        self.exp_id: str
        self.testbed_exp: str

    def _hashing_(self) -> float.hex:
        """
        Generates a SHA256 hash object and return hexadecimal digits
        pip install hashids
        """
        # hashids = Hashids(salt="diagnosenet")
        # exp_id = hashids.encode(exp_id)
        # hashids = hashids.decode(exp_id)

        datetime = strftime("%Y%m%d%H%M%S", gmtime())
        exp_id=str(self.data.dataset_name)+"-"+str(self.model.__class__.__name__)+"-"+str(self.platform_name)+"-"+str(datetime)
        hash_id = hashlib.sha256(exp_id.encode('utf-8')).hexdigest()

        return hash_id

    def _set_eda_json(self):
        """
        Experiment description architecture:
        build a document that consists of a header and body in JSON format
        """
        act_layer = []
        dim_layer = []
        for layer in self.model.layers:
            act_layer.append(layer.__class__.__name__)
            dim_layer.append((layer.input_size, layer.output_size))

        target_range = []
        target_range.append(str(self.data.target_start))
        target_range.append(str(self.data.target_end))

        exp_serialized = {
            "exp_id": self.exp_id,
            "dnn_type": str(self.model.__class__.__name__),
            "model_hyperparameters": {
                    "activation_layer": act_layer,
                    "dimension_layer": dim_layer,
                    "optimizer": str(self.model.optimizer.__class__.__name__),
                    "learning_rate": self.model.optimizer.lr,
                    "loss": str(self.model.loss),
                    "dropout": self.model.dropout,
                    "max_epochs": self.max_epochs,
                    },
            "dataset_config":{
                    "dataset_name": str(self.data.dataset_name),
                    "valid_rate": str(self.data.valid_size),
                    "test_rate": str(self.data.test_size),
                    "batch_size": str(self.data.batch_size),
                    "target_range": target_range,
                    "target_name": str(self.data.target_name),
                    },
            "platform_parameters": {
                    "platform": self.platform_name,
                    "hostname": os.uname()[1],
                    "processing_mode": None,
                    "gpu_id": None,
                    },
            "results": {
                    "f1_score_weigted": None,
                    "f1_score_micro": None,
                    "loss_validation": None,
                    "time_training": None,
                    "time_dataset": None,
                    "time_testing": None,
                    "time_metrics": None,
                    "time_latency": None,
                    },}

        exp_description = json.dumps(exp_serialized, separators=(',', ': '), indent=2)
        return exp_description

    def generate_testbed(self, testbed_path, model, data,
                                            platform_name, max_epochs) -> None:
        """
        Build an experiment directory to isolate the training metrics files
        and return experiment id
        """
        self.testbed = testbed_path
        self.model = model
        self.data = data
        self.platform_name = platform_name
        self.max_epochs = max_epochs

        ## Define a experiment id
        self.exp_id=self._hashing_()

        ## Build a experiment testbed directory
        IO_Functions()._mkdir_(self.testbed)

        self.testbed_exp = str(self.testbed+"/"+self.exp_id+"/")
        IO_Functions()._mkdir_(self.testbed_exp)

        ## Write the experiment description in json format
        exp_description = self._set_eda_json()

        file_path = str(self.testbed_exp+"/"+self.exp_id+"-exp_description.json")
        IO_Functions()._write_file(exp_description, file_path)

    def read_eda_json(self, testbed_exp, exp_id) -> None:
        """
        Read Json experiment Description architecture File
        """
        with open(testbed_exp+"/"+exp_id+"-exp_description.json") as eda_json:
            eda_json = json.load(eda_json)
        return eda_json


class enerGyPU(Testbed):
    """
    This module deploys an energy monitor to collect the energy consumption metrics
    while the DNN model is executed on the target platform.
    """
    def __init__(self, testbed_path, write_metrics: bool = True,
                                power_recording: bool = True,
                                platform_recording: bool = True) -> None:
        super().__init__(testbed_path,write_metrics)
        self.power_recording = power_recording
        self.platform_recording = platform_recording
        self.idgpu_available: list = []

    def _get_available_GPU(self) -> list:
        """
        Returns a list of ID GPUs available in the computational platform
        """
        ## Identification of GPUs
        idgpu = []
        cmd_nvidia_smi = "nvidia-smi | grep '0000' | awk '{if($7 ~ '0000') print $7; else if($8 ~ '0000') print $8}'"
        idgpu.append(sp.Popen(cmd_nvidia_smi, stdout=sp.PIPE, shell=True).stdout.readlines())

        ## Get ID GPUs available
        if not idgpu:
            print("++ no NVIDIA GPU detected ++")
        else:
            for i in range(len(idgpu)):
                ## Convert byte sp.Popen output to string
                gpu_tmp = str(idgpu[0][i].decode("utf-8").rstrip('\n'))
                ## Uses the "nvidia-smi pmon" command-line for monitoring the availeble GPU
                cmd_nvidia_pmon = "nvidia-smi pmon -i "+gpu_tmp+" -c 1 | awk '{if(NR == 3) print $1}'"
                tmp_idgpu = sp.Popen(cmd_nvidia_pmon, stdout=sp.PIPE, shell=True).stdout.readline()
                self.idgpu_available.append(tmp_idgpu.decode("utf-8").rstrip('\n'))

        return self.idgpu_available

    def start_power_recording(self) -> None:
        """
        Launches a subprocess for recording the global GPU factors
        to power consumption measures.
        """
        sp.run(["enerGyPU/dataCapture/enerGyPU_record.sh", self.testbed_exp, self.exp_id])

    def end_power_recording(self) -> None:
        """
        Kill the subprocess enerGyPU_record.
        """
        sp.call(["killall", "-9", "nvidia-smi"])

    def start_platform_recording(self, pid) -> None:
        """
        Subprocess recording for memory and cpu usage while the models are training
        This function uses the library psutil-5.4.8
        """
        self.proc_platform = sp.Popen(["python3.6", "enerGyPU/dataCapture/platform_record.py",
                                str(pid), self.testbed_exp, self.exp_id])

    def end_platform_recording(self) -> None:
        """
        Send signal to kill platform recording process
        """
        self.proc_platform.kill()
