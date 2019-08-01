"""
Device replica....
"""

from __future__ import print_function
import os, sys, socket, time

## Makes diagnosenet library visible in samples folder
import platform

ON_ASTRO = platform.node().startswith("astro")

import sys
if ON_ASTRO:
    file_path = "/home/mpiuser/cloud/0/diagnosenet/"
else:
    file_path = "/home/mpiuser/cloud/diagnosenet/"

#file_path = "../../"
sys.path.append(file_path)

from diagnosenet.io_functions import IO_Functions
from diagnosenet.datamanager import MultiTask, Batching
from diagnosenet.layers import Relu, Linear
from diagnosenet.losses import CrossEntropy
from diagnosenet.optimizers import Adam
from diagnosenet.graphs import SequentialGraph
from diagnosenet.executors import Distibuted_GRPC
from diagnosenet.resourcemanager import ResourceManager
from diagnosenet.monitor import enerGyPU



def main(argv):
    ## Formating the argv from reseource manager
    for i in range(len(argv)):
        argv[i]=str(argv[i]).replace('[','').replace(']','').replace(',','')


    ## Formatiing the workers IP to build
    ## a tensorflow cluster in executor class
    temp_workers = []
    for i in range(len(argv)-3):
        temp_workers.append(argv[i+3])
    temp_workers = ','.join(temp_workers)


    ## PMSI-ICU Dataset shapes
    X_shape = 14637 #10833	#14637
    y_shape = 381
    Y1_shape = 14
    Y2_shape = 239
    Y3_shape = 5

    ## 1) Define stacked layers funtion activation, number of layers and their neurons
    stacked_layers_1 = [Relu(X_shape, 1024),
                        Relu(1024, 1024),
                        Linear(1024, Y1_shape)]

    stacked_layers_2 = [Relu(X_shape, 512),
                        Relu(512, 512),
                        Relu(512, 512),
                        Relu(512, 512),
                        Linear(512, Y1_shape)]

    stacked_layers_3 = [Relu(X_shape, 256),
                        Relu(256, 256),
                        Relu(256, 256),
                        Relu(256, 256),
                        Relu(256, 256),
                        Relu(256, 256),
                        Relu(256, 256),
                        Relu(256, 256),
                        Linear(256, Y1_shape)]


    ## 2) Select the neural network architecture and pass the hyper-parameters
    mlp_model = SequentialGraph(input_size=X_shape, output_size=Y1_shape,
                            layers=stacked_layers_2,
                            loss=CrossEntropy,
                            optimizer=Adam(lr=0.001),
                            dropout=0.8)

    ## 3) Dataset configurations for splitting, batching and target selection
    data_config_1 = Batching(dataset_name="MCP-PMSI",
                            valid_size=0.05, test_size=0.10,
                            devices_number=2,
                            batch_size=150)

    ## 4) Select the computational platform and pass the DNN and Dataset configurations
    if ON_ASTRO:
        testbed_path = "/home/mpiuser/cloud/0/diagnosenet/samples/0_sequentialgraph/testbed"
    else:
        testbed_path = "/home/mpiuser/cloud/diagnosenet/samples/0_sequentialgraph/testbed"
    platform = Distibuted_GRPC(model=mlp_model,
                             datamanager=data_config_1,
                             monitor=enerGyPU(testbed_path=testbed_path,
                                              machine_type="arm", file_path=file_path),
                             max_epochs=2, min_loss=0.0002,
                             ip_ps=argv[2], ip_workers=temp_workers)

    ## 5) Uses the platform modes for training in an efficient way
    if ON_ASTRO:
        dataset_path = "/home/mpiuser/cloud/0/diagnosenet/samples/0_sequentialgraph/dataset/"
    else:
        dataset_path = "/home/mpiuser/cloud/diagnosenet/samples/0_sequentialgraph/dataset/"
    platform.asynchronous_training(dataset_name="MCP-PMSI",
                                 dataset_path=dataset_path,
                                 inputs_name="patients_features.txt",
                                 targets_name="medical_targets_Y14.txt",
                                 job_name=argv[0],
                                 task_index=int(argv[1]))


if __name__ == '__main__':
    main(sys.argv[1:])
