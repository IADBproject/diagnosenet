"""
Medical Care Purpose Classification for PMSI-ICU Dataset
User example for training DiagnoseNET exploit the memory desktop machine
"""

import time
execution_start = time.time()

from diagnosenet.io_functions import IO_Functions
from diagnosenet.datamanager import MultiTask
from diagnosenet.layers import Relu, Linear
from diagnosenet.losses import CrossEntropy
from diagnosenet.optimizers import Adam
from diagnosenet.graphs import FullyConnected
from diagnosenet.executors import DesktopExecution
from diagnosenet.monitor import enerGyPU


### Read the PMSI-Dataset using Pickle from diagnosenet.io_functions
#path = "/data_B/datasets/drg-PACA/healthData/sandbox-FULL-W1_x1_x2_x3_x4_x5_x7_x8_Y1/1_Mining-Stage/binary_representation/"
#X = IO_Functions()._read_file(path+"BPPR-FULL-W1_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")
#y = IO_Functions()._read_file(path+"labels_Y1-FULL-W1_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")


path = "/data_B/datasets/drg-PACA/healthData/sandbox-SENSE-CUSTOM_x1_x2_x3_x4_x5_x7_x8_Y1/1_Mining-Stage/binary_representation/"
X = IO_Functions()._read_file(path+"BPPR-SENSE-CUSTOM_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")
y = IO_Functions()._read_file(path+"labels_Y1-SENSE-CUSTOM_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")





## 1) Define the stacked layers as the number of layers and their neurons
layers_1 = [Relu(14637, 2048),
            Relu(1024, 1024),
            Relu(1024, 1024),
            Relu(1024, 1024),
            Linear(1024, 14)] #MT1
            #Linear(2048, 5)] #MT3
            #Linear(2048, 239)] #MT2

## 2) Select the neural network architecture and pass the hyper-parameters
mlp_model_1 = FullyConnected(input_size=14637, output_size=14,   #239,
                layers=layers_1,
                loss=CrossEntropy,
                optimizer=Adam(lr=0.001),
                dropout=0.8)

## Added a second model for a network architecture search
layers_2 = [Relu(10833, 1024),
            Relu(1024, 1024),
            Relu(1024, 1024),
            Relu(1024, 1024),
            Linear(1024, 14)]

mlp_model_2 = FullyConnected(input_size=10833, output_size=14,
                layers=layers_2,
                loss=CrossEntropy,
                optimizer=Adam(lr=0.001),
                dropout=0.8)


## 3) Dataset configurations for splitting, batching and target selection
data_config = MultiTask(dataset_name="SENSE-CUSTOM_x1_x2_x3_x4_x5_x7_x8_Y1",
                        valid_size=0.05, test_size=0.15,
                        batch_size=100,	#3072,	#100,
                        target_name='Y11',
                        target_start=0, target_end=14)
#                        target_start=239, target_end=244)
                        #target_start=14, target_end=253) 

## 4) Select the computational platform and pass the DNN and Dataset configurations
platform = DesktopExecution(model=mlp_model_1,
                            datamanager=data_config,
                            monitor=enerGyPU(testbed_path="/data/jagh/green_learning/testbed"),
                            max_epochs=40,
                            min_loss=0.02)

## 5) Uses the platform modes for training in an efficient way
platform.training_memory(X, y)


print("Execution Time: {}".format((time.time()-execution_start)))
