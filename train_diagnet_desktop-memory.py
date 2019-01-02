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

### Read the PMSI-Dataset using Pickle from diagnosenet.io_functions
path = "healthData/sandbox-W1-TEST_x1_x2_x3_x4_x5_x7_x8_Y1/1_Mining-Stage/binary_representation/"
X = IO_Functions()._read_file(path+"BPPR-W1-TEST_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")
y = IO_Functions()._read_file(path+"labels_Y1-W1-TEST_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")

## 1) Define the stacked layers as the number of layers and their neurons
layers = [Relu(14637, 2048),
            Relu(2048, 2048),
            Relu(2048, 2048),
            Relu(2048, 2048),
            Linear(2048, 14)]

## 2) Select the neural network architecture and pass the hyper-parameters
mlp_model = FullyConnected(input_size=14637, output_size=14,   #381,
                layers=layers,
                loss=CrossEntropy,
                optimizer=Adam(lr=0.01))

## 3) Dataset configurations for splitting, batching and target selection
data_config = MultiTask(valid_size=0.05, test_size=0.10,
                        batch_size=100,
                        target_name='Y11',
                        target_start=0, target_end=14)

## 4) Select the computational platform and pass the DNN and Dataset configurations
platform = DesktopExecution(model=mlp_model,
                            datamanager=data_config,
                            max_epochs=10)

## 5) Uses the platform modes for training in an efficient way
platform.training_memory(X, y)
platform.write_metrics()



print("Execution Time: {}".format((time.time()-execution_start)))
