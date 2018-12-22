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
    Module that cotains a loader functions to be use over diagnosenet modules.
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
        return items_corpus

    def _write_file(self, data, file_path) -> None:
        with open(file_path, 'w+') as f:
            f.write(data)
        f.close()

    def _write_list(self, data, file_path) -> None:
        with open(file_path, 'w') as f:
            f.write('\n'.join('%s, %s, %s, %s, %s, %s' % x for x in data))
            # f.write('\n'.join('{}, {}, {}, {}, {} %s' % x for x in data))

    def _mkdir_(self, directory) -> None:
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _write_batches(self, path: str, data: DataSplit,
                        batch_size: int, dataset_name: str,) -> None:
        fnumber = 1
        for i in range(len(data.inputs)):
            if i % batch_size == 0:
                X_fname = str(path+"X-"+dataset_name+"-"+str(fnumber)+'.txt')
                y_fname = str(path+"y-"+dataset_name+"-"+str(fnumber)+'.txt')
                fnumber += 1

                if os.path.exists(X_fname) == False:
                    open(X_fname, 'w+').writelines(data.inputs[i:i+batch_size])
                else:
                    pass

                if os.path.exists(y_fname) == False:
                    open(y_fname, 'w+').writelines( data.targets[i:i+batch_size])
                else:
                    pass
