"""
Device replica....
"""

from __future__ import print_function
import os, sys, socket, time

## Makes diagnosenet library visible in samples folder
import sys
file_path = "/home/mpiuser/cloud/0/diagnosenet/"
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
    X_shape = 14637
    y_shape = 381
    Y1_shape = 14
    Y2_shape = 239
    Y3_shape = 5

    ## 1) Define the stacked layers as the number of layers and their neurons
    layers = [Relu(X_shape, 2048),
            Relu(2048, 2048),
            Relu(2048, 1024),
            Relu(1024, 1024),
            Linear(1024, y_shape)]

    ## 2) Select the neural network architecture and pass the hyper-parameters
    mlp_model = SequentialGraph(input_size=X_shape, output_size=y_shape,
                            layers=layers,
                            loss=CrossEntropy,
                            optimizer=Adam(lr=0.001),
                            dropout=0.8)

    ## 3) Dataset configurations for splitting, batching and target selection
    data_config_1 = Batching(dataset_name="MCP-PMSI",
                        valid_size=0.05, test_size=0.10,
                        devices_number=2,
                        batch_size=100)

    ## 4) Select the computational platform and pass the DNN and Dataset configurations
    platform = Distibuted_GRPC(model=mlp_model,
                             datamanager=data_config_1,
                             monitor=enerGyPU(testbed_path="/home/mpiuser/cloud/0/diagnosenet/samples/0_sequentialgraph/testbed",
                                              machine_type="arm", file_path=file_path),
                             max_epochs=10,
                             ip_ps=argv[2],
                             ip_workers=temp_workers)	#argv[1])

    ## 5) Uses the platform modes for training in an efficient way
    platform.asynchronous_training(dataset_name="MCP-PMSI",
                                 dataset_path="/home/mpiuser/cloud/0/diagnosenet/samples/0_sequentialgraph/dataset/",
                                 inputs_name="patients_features.txt",
                                 targets_name="medical_targets.txt",
                                 job_name=argv[0],
                                 task_index=int(argv[1]))


if __name__ == '__main__':
    main(sys.argv[1:])
