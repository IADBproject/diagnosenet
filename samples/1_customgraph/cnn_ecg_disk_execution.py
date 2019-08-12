

import time
execution_start = time.time()

## Makes diagnosenet library visible in samples folder
import sys
#file_path="../../"
file_path = "/home/mpiuser/cloud/diagnosenet/"
sys.path.append(file_path)

from diagnosenet.datamanager import MultiTask, Batching
from diagnosenet.layers import Relu, Linear
from diagnosenet.losses import CrossEntropy
from diagnosenet.optimizers import Adam
from diagnosenet.graphs import CustomGraph
from diagnosenet.executors import DesktopExecution
from diagnosenet.monitor import enerGyPU
import numpy as np
import pandas as pd




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


    ## 1) Define stacked layers funtion activation, number of layers and their neurons
    layer_1=[Linear(1300, 4)]


    ## 2) Select the neural network architecture and pass the hyper-parameters
    cnn_model = CustomGraph(input_size_1=1300,input_size_2=1, output_size=4,
                        loss=CrossEntropy,
                        optimizer=Adam(lr=0.0001),layers=layer_1)

    ## 3) Dataset configurations for splitting, batching and target selection
    data_config = Batching(dataset_name="ECG",
                        valid_size=0.1, test_size=0.1,
                        batch_size=10, devices_number=1)

    ## 4) Select the computational platform and pass the DNN and Dataset configurations
    projection = Distibuted_GRPC(model=cnn_model,
                        datamanager=data_config,
                        monitor=enerGyPU(testbed_path="/home/mpiuser/cloud/diagnosenet/samples/1_customgraph/testbed",
                                              machine_type="arm", file_path=file_path),
                        max_epochs=2, min_loss=0.002,
                        ip_ps=argv[2], ip_workers=temp_workers)

    ## 5) Uses the platform modes for training in an efficient way
    projection.training_disk(dataset_name="ECG",
                        dataset_path="/home/mpiuser/cloud/diagnosenet/samples/1_customgraph/dataset/",
                        inputs_name="xdata.npy",
                        targets_name="undim-ydata.npy",
                        job_name=argv[0],
                        task_index=int(argv[1]))

    print("Execution Time: {}".format((time.time()-execution_start)))

if __name__ == '__main__':
    main(sys.argv[1:])

