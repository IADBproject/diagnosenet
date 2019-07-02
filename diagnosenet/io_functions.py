"""
I/O functions
Module that cotains a loader functions to be use over diagnosenet modules.
"""

import collections

import os, pickle, math
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

    def _write_batches_worker(self, path: str, data: DataSplit, devices_number: int,
                        batch_size: int, dataset_name: str,) -> None:
        """
        IO_Function to baching the datse over the number of working for NFS.
        """

        def batching_woker(mod_index, X_worker_name, y_worker_name, batch_inputs, batch_targets):
            batch_file_num = 1
            for j in range(len(batch_inputs)):
                ## batch_range_end guarantees that the last batch by worker
                ## dont write more lines than worker_samples
                if batch_file_num == math.ceil(worker_samples/float(batch_size)):
                    batch_range_end = samples_index
                elif batch_file_num < math.ceil(worker_samples/float(batch_size)):
                    batch_range_end = mod_index+j+batch_size

                # print("Batch file: {} || i: {} || worker_count: {}".format(batch_file_num, mod_index+j, batch_range_end))
                ## Write batches by worker
                if j % batch_size == 0:
                    ## Added the batch index into the path name
                    X_fname = str(X_worker_name+"-"+str(batch_file_num)+'.txt')
                    y_fname = str(y_worker_name+"-"+str(batch_file_num)+'.txt')
                    batch_file_num += 1

                    if os.path.exists(X_fname) == False:
                        open(X_fname, 'w+').writelines(data.inputs[mod_index+j:batch_range_end])
                    else:
                        pass
                    if os.path.exists(y_fname) == False:
                        open(y_fname, 'w+').writelines(data.targets[mod_index+j:batch_range_end])
                    else:
                        pass

        ## samples sweep by worker
        worker_samples = math.ceil(len(data.inputs)/int(devices_number))
        worker_file_num = 0
        samples_index = 0
        for i in range(len(data.inputs)):

            if i % worker_samples == 0:
                worker_file_num += 1
                samples_index = worker_file_num * worker_samples

                ## Defined the worker index into the path name
                X_worker_name = str(path+"X-"+dataset_name+"-"+str(worker_file_num))
                y_worker_name = str(path+"y-"+dataset_name+"-"+str(worker_file_num))
                # print("Worker mod: {} || Worker num: {} || Samples index: {}".format(i, worker_file_num, samples_index))

                ## Guarantees that the number of files is created
                ## as number of workers
                if worker_file_num == int(devices_number):
                    batching_woker(i, X_worker_name, y_worker_name,
                                            data.inputs[i:samples_index],
                                            data.targets[i:samples_index])

                elif worker_file_num < int(devices_number):
                    batching_woker(i, X_worker_name, y_worker_name,
                                            data.inputs[i:i+samples_index],
                                            data.targets[i:i+samples_index])






        #########################################################################
        ### working
        # worker_samples = math.ceil(len(data.inputs)/int(devices_number))
        # worker_num = 0
        # samples_index = 0
        # for i in range(len(data.inputs)):
        #
        #     if i % worker_samples == 0:
        #         worker_num += 1
        #         samples_index = worker_num * worker_samples
        #         batch_file_num = 1
        #
        #         ## Defined the worker index into the path name
        #         X_worker_name = str(path+"X-"+dataset_name+"-"+str(worker_num))
        #         y_worker_name = str(path+"y-"+dataset_name+"-"+str(worker_num))
        #
        #         print("Worker mod: {} || Worker num: {} || Sample count: {}".format(i, worker_num, samples_index))
        #
        #
        #     if batch_file_num == math.ceil(worker_samples/float(batch_size)):
        #         if i % batch_size == 0 or i % worker_samples == 0:
        #
        #             ## Added the batch index into the path name
        #             X_fname = str(X_worker_name+"-"+str(batch_file_num)+'.txt')
        #             y_fname = str(y_worker_name+"-"+str(batch_file_num)+'.txt')
        #             batch_file_num += 1
        #
        #             print("Last Samples_index: {}".format(int(samples_index)))
        #             print("Last Batch file: {} || i: {} || worker_count: {}".format(batch_file_num, i, worker_samples))
        #
        #             if os.path.exists(X_fname) == False:
        #                 open(X_fname, 'w+').writelines(data.inputs[i:samples_index])
        #             else:
        #                 pass
        #
        #             if os.path.exists(y_fname) == False:
        #                     open(y_fname, 'w+').writelines(data.targets[i:samples_index])
        #             else:
        #                 pass
        #
        #     elif batch_file_num < math.ceil(worker_samples/float(batch_size)):
        #
        #         if i % batch_size == 0 or i % worker_samples == 0:  #or i % worker_samples == batch_size:
        #             print("samples_index: {}".format(int(samples_index)))
        #             print("batch_file_num: {} || i: {} || worker_count: {}".format(batch_file_num, i, i+batch_size))
        #             # print("Data shape: {} | {}".format(i, i+batch_size))
        #             X_fname = str(X_worker_name+"-"+str(batch_file_num)+'.txt')
        #             y_fname = str(y_worker_name+"-"+str(batch_file_num)+'.txt')
        #             batch_file_num += 1
        #
        #             if os.path.exists(X_fname) == False:
        #                 open(X_fname, 'w+').writelines(data.inputs[i:i+batch_size])
        #             else:
        #                 pass
        #
        #             if os.path.exists(y_fname) == False:
        #                 open(y_fname, 'w+').writelines( data.targets[i:i+batch_size])
        #             else:
        #                 pass




                # sample_count = worker_num*len(data.inputs)
            #
            #     if worker_num == int(devices_number):
            #         print("Data shape: {} | {}".format(i, len(data.inputs)))
            #         fnumber = 1
            #
            #     elif worker_num < int(devices_number):
            #         print("Data shape: {} | {}".format(i, i+new_sdw_factor))
            #         fnumber = 1
            #     worker_num += 1

                # if worker_num == int(devices_number):
                #     print("Data shape: {} | {}".format(i, len(data.inputs)))
                #
                # elif worker_num <= int(devices_number):
                #     print("Data shape: {} | {}".format(i, new_sdw_factor))
                #     # print("++ sample_count: {}".format(sample_count))

                # if i <= (sample_count - batch_size):
