"""
Medical Care Purpose Classification for PMSI-ICU Dataset
User example for training DiagnoseNET exploit the memory desktop machine
"""

import time
from diagnosenet.io_functions import IO_Functions
from diagnosenet.datamanager import MultiTask, Batching, Splitting

from diagnosenet.layers import Relu, Softmax
from diagnosenet.losses import CrossEntropy
from diagnosenet.optimizers import Adam
from diagnosenet.graphs import MLP
from diagnosenet.executors import DesktopExecution


start = time.time()

### Read the PMSI-Dataset using Pickle
path = "healthData/sandbox-W1-TEST_x1_x2_x3_x4_x5_x7_x8_Y1/1_Mining-Stage/binary_representation/"
X = IO_Functions()._read_file(path+"BPPR-W1-TEST_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")
y = IO_Functions()._read_file(path+"labels_Y1-W1-TEST_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")


## 1) Define the stacked layers as the number of layers and their neurons
staked_layers = [Relu(14637, 500),
                Relu(500, 500),
                Relu(500, 100),
                Relu(100, 100),
                Softmax(100, 14)]

## 2) Select the neural network architecture and pass the hyper-parameters
mlp_model = MLP(input_size=14637, output_size=14,   #381,
                layers=staked_layers,
                loss=CrossEntropy,
                optimizer=Adam(lr=0.01))

## 3) Config the Data Manager for data processing in an efficient way
dataset_conf = MultiTask(batch_size=100,
                        valid_size=0.05, test_size=0.10,
                        target_name='Y11',
                        target_start=0, target_end=14)

## 4) Select the processing machine mode
projection = DesktopExecution(model=mlp_model,
                max_epochs=10,
                datamanager=dataset_conf).memoryexecutor(X, y)


elapsed = (time.time() - start)
print("Time.time: {}".format(elapsed))
