"""
Data manager its adjusted for training a medical diagnostic workflow inside the hospitals.
It provides an isolation to fine-tuning different neural network hyper-parameters.
"""

from typing import NamedTuple

import glob, os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from diagnosenet.loaders import Loaders

import logging
logger = logging.getLogger('_DiagnoseNET_')

class Dataset:
    """
    A desktop manager is provided for memory or disk training.
    """
    def __init__(self) -> None:
        self.inputs: np.ndarray
        self.targets: np.ndarray

        self.dataset_name: str
        self.dataset_path: str
        self.inputs_name: str
        self.labels_name: str

    def set_data_file(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        Set data file
        """
        try:
            ## Convert a pandas dataframe (df) to a numpy ndarray
            self.inputs = inputs.values
            self.targets = targets.values
        except AttributeError:
            if 'numpy' in str(type(inputs)):
                self.inputs = inputs
                self.targets = targets
            elif 'list' in str(type(inputs)):
                ## Convert a list to a numpy ndarray
                self.inputs = pd.DataFrame(inputs)
                self.inputs = np.asarray(self.inputs[0].str.split(',').tolist())
                self.targets = pd.DataFrame(targets)
                self.targets = np.asarray(self.targets[0].str.split(',').tolist())
            else:
                raise AttributeError("set_data_file(inputs, targets) requires: numpy, pandas or list ")

    def set_data_path(self, dataset_name: str, dataset_path: str,
                        inputs_name: str, targets_name: str) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.inputs_name = inputs_name
        self.targets_name = targets_name
        self.sandbox = str(dataset_path+"sandbox-"+self.dataset_name)
        self.inputs_path = glob.glob(self.sandbox+"/1_Mining-Stage/binary_representation/"+self.inputs_name+"*")
        self.targets_path = glob.glob(self.sandbox+"/1_Mining-Stage/binary_representation/"+self.targets_name+"-*")

        if os.path.exists(self.inputs_path[0]):
            self.inputs = Loaders()._read_file(self.inputs_path[0])
        else:
            raise NameError("inputs_path not localized: {}".format(self.inputs_path))

        if os.path.isfile(self.targets_path[0]):
            self.targets = Loaders()._read_file(self.targets_path[0])
        else:
            raise NameError("targets_path not localized: {}".format(self.targets_path))

    def dataset_split(self, valid: float = None, test: float = None) -> None:
        """
        split dataset in Training, Valid and test,
        and split into mini-batches give a size factor.
        """
        try:
            prop1=valid+test
            prop2=(test*1)/(valid+test)
        except TypeError:
            logger.warning('!!! Splitting proportions are missing !!!')
            valid = 0.05; test = 0.10
            logger.warning('!!! Set valid={} and test={} by the fault !!!'.format(valid, test))
            prop1=valid+test
            prop2=(test*1)/(valid+test)

        X_train, X_temp, y_train, y_temp = train_test_split(self.inputs, self.targets, test_size=prop1)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp,y_temp, test_size=prop2)

        logger.info('---------------------------------------------------------')
        logger.info('++ Dataset Split ++')
        logger.info('-- Train records:  {} | {} --'.format(len(X_train), len(y_train)))
        logger.info('-- Valid records:  {} | {} --'.format(len(X_valid), len(y_valid)))
        logger.info('-- Test records:   {} | {} --'.format(len(X_test), len(y_test)))


class Batching(Dataset):
    """
    Write micro-batches by a size split factor
    for each part of the data, Training, Valid, Test
    """
    
    def __init__(self):
        super().__init__()

    def memory_batching(self, batch_size: int = 1000, shuffle: bool = True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

        # ## Conver matrix list to np.array
        self.X_train = pd.DataFrame(self.X_train)
        self.X_train = self.X_train[0].str.split(',').tolist()
        self.y_train = pd.DataFrame(self.y_train)
        self.y_train = self.y_train[0].str.split(',').tolist()

        train_starts = np.arange(0, len(self.X_train), self.batch_size)
        if self.shuffle:
            np.random.shuffle(train_starts)

        for start in train_starts:
            end = start + self.batch_size
            batch_inputs = np.asarray(self.X_train)[start:end]
            batch_targets = np.asarray(self.y_train)[start:end]
            yield DataSplit(batch_inputs, batch_targets)

    def disk_batching(self, batch_size: int) -> None:
        self.batch_size = batch_size

        ## Defining split directories
        self.split_path = str(self.sandbox+"/2_Split_Point-"+str(batch_size))
        train_path = str(self.split_path+"/data_training/")
        valid_path = str(self.split_path+"/data_valid/")
        test_path = str(self.split_path+"/data_test/")

        ## Build split directories
        self.loader._mkdir_(self.split_path)
        self.loader._mkdir_(train_path)
        self.loader._mkdir_(valid_path)
        self.loader._mkdir_(test_path)

        ## Writing records in batches
        self.loader._write_batches(train_path, DataSplit(self.X_train, self.y_train),
                                    self.batch_size, self.dataset_name)

        self.loader._write_batches(valid_path, DataSplit(self.X_valid, self.y_valid),
                                    self.batch_size, self.dataset_name)

        self.loader._write_batches(test_path, DataSplit(self.X_test, self.y_test),
                                    self.batch_size, self.dataset_name)

        logger.info('-- Split path: {} --'.format(self.split_path))
