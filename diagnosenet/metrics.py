"""
Metrics ...
"""

from time import gmtime, strftime
from hashids import Hashids
import json

import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_recall_fscore_support

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
    def __init__(self, model, data, platform_name, max_epochs) -> None:
        super().__init__()
        self.model = model
        self.data = data
        self.platform_name = platform_name
        self.max_epochs = max_epochs

    def _hashing_(self) -> float.hex:
        """
        Generates a SHA256 hash object and return hexadecimal digits
        pip install hashids
        """
        date = strftime("%Y%m%d", gmtime())
        time = strftime("%H%M%S", gmtime())

        hashids = Hashids(salt="diagnosenet")
        exp_id = hashids.encode(int(date), int(time))

        # datetime = hashids.decode(exp_id)
        # exp_id = hashlib.sha256(exp_idn.encode('utf-8')).hexdigest()

        return exp_id

    def eda_json(self):
        """
        Experiment description architecture:
        build a document that consists of a header and body in JSON format
        """
        act_layer = []
        dim_layer = []
        for layer in self.model.layers:
            act_layer.append(layer.__class__.__name__)
            dim_layer.append((layer.input_size, layer.output_size))

        exp_serialized = {
            "exp_id": self.exp_id,
            "dnn_type": str(self.model.__class__.__name__),
            "model_hyperparameters": {
                    "activation_layer": act_layer,
                    "dimension_layer": dim_layer,
                    "optimizer": str(self.model.optimizer.__class__.__name__),
                    "loss": str(self.model.loss),
                    "max_epochs": self.max_epochs
                    },
            "dataset_config":{
                    "dataset_name": str(self.data.dataset_name),
                    "batch_size": str(self.data.batch_size),
                    "target_name": str(self.data.target_name),
                    },
            "platform_parameters": {
                    "platform": self.platform_name,
                    "processing_mode": None,
                    "gpu_id": None,
                    },
            "results": {
                    "f1_score_weigted": None,
                    "f1_score_micro": None,
                    "time_training": None,
                    "time_dataset": None,
                    "time_testing": None,
                    "time_metrics": None,
                    "time_latency": None,
                    },}

        exp_description = json.dumps(exp_serialized, separators=(',', ': '), indent=2)
        return exp_description

    def _get_eda_json(self, testbed_exp, exp_id) -> None:
        """
        Read Json experiment Description architecture File
        """
        with open(testbed_exp+"/"+exp_id+"-exp_description.json") as eda_json:
            eda_json = json.load(eda_json)
        return eda_json

    def generate_testbed(self, testbed: str = 'testbed') -> None:
        """
        Build an experiment directory to isolate the training metrics files
        """
        self.testbed = testbed

        ## Define a experiment id
        datetime = strftime("%Y%m%d%H%M%S", gmtime())
        self.exp_id=str(self.data.dataset_name)+"-"+str(self.model.__class__.__name__)+"-"+str(self.platform_name)+"-"+str(datetime)

        ## Build a experiment testbed directory
        IO_Functions()._mkdir_(self.testbed)

        self.testbed_exp = str(self.testbed+"/"+self.exp_id+"/")
        IO_Functions()._mkdir_(self.testbed_exp)

        ## Write the experiment description in json format
        exp_description = self.eda_json()

        file_path = str(self.testbed_exp+"/"+self.exp_id+"-exp_description.json")
        IO_Functions()._write_file(exp_description, file_path)

        return self.exp_id
