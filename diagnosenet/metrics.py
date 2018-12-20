"""
Metrics ...
"""

from time import gmtime, strftime
from hashids import Hashids
import json

import tensorflow as tf

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

        # exp_id = hashlib.sha256(exp_description.encode('utf-8')).hexdigest()
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
                    },
            "dataset_config":{
                    "dataset_name": str(self.data.dataset_name),
                    "batch_size": str(self.data.batch_size),
                    "target_name": str(self.data.target_name),
                    },
            "platform_parameters": {
                    "platform": self.platform_name,
                    "max_epochs": self.max_epochs
                    } }

        exp_description = json.dumps(exp_serialized, separators=(',', ': '))
        return exp_description


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
