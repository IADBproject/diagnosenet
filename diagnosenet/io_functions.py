"""
I/O functions
Module that cotains a loader functions to be use over diagnosenet modules.
"""

import collections

import os, pickle
import numpy as np
import pandas as pd

DataSplit = collections.namedtuple('DataSplit', 'name inputs targets')

class IO_Functions:
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
        fnumber = 1
        for i in range(data.inputs.shape[0]):
            if i % batch_size == 0:
                X_fname = str(path+"X-"+dataset_name+"-"+str(fnumber)+'.txt')
                y_fname = str(path+"y-"+dataset_name+"-"+str(fnumber)+'.txt')
                if os.path.exists(X_fname) == False:
                    with open(X_fname, 'a+') as x_fn:
                        np.savetxt(x_fn, data.inputs[i:i+batch_size], fmt='%s', delimiter=',', newline='\n')
                    x_fn.close()
                else:
                    pass
                if os.path.exists(y_fname) == False:
                    with open(y_fname, 'a+') as y_fn:
                        np.savetxt(y_fn, data.targets[i:i+batch_size], fmt='%s', delimiter=',', newline='\n')
                    y_fn.close()
                else:
                    pass
                fnumber += 1
