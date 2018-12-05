"""
I/O functions
Module that cotains a loader functions to be use over diagnosenet modules.
"""

from typing import NamedTuple

import os, pickle
import numpy as np
import pandas as pd

DataSplit = NamedTuple("DataSplit", [("X", list), ("y", list)])

class Loaders:
    """
    """

    def __init__(self) -> None:
        self.dataset_name: str
        self.dataset_path: str
        self.inputs: str
        self.output: str
        self.testbed_path: str
        self.sandbox: str

    def _read_file(self, file_path) -> None:
        items_corpus = []
        f = open(file_path, 'r')
        for line in f:
            items_corpus.append(line)
        f.close()
        data = pd.DataFrame(items_corpus)
        return np.asarray(data[0].str.split(',').tolist())

    def _mkdir_(self, directory) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _write_batches(self, path: str, data: DataSplit,
                        batch_size: int, dataset_name: str,) -> None:
        X, y = data
        filename = 1
        for i in range(len(data[0])):
            if i % batch_size == 0:
                X_file = str(path+"X-"+dataset_name+"-"+str(filename)+'.txt')
                y_file = str(path+"y-"+dataset_name+"-"+str(filename)+'.txt')

                if os.path.isfile(X_file) == False:
                    open(X_file, 'w+').writelines(X[i:i+batch_size])
                else:
                    pass
                    # print("Can't overwrite the file: {} ".format(X_file))
                if os.path.isfile(y_file) == False:
                    open(y_file, 'w+').writelines(y[i:i+batch_size])
                else:
                    pass
                filename += 1
