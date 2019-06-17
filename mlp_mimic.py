"""
Medical Care Purpose Classification for PMSI-ICU Dataset
User example for training DiagnoseNET exploit the memory desktop machine
"""

import time
execution_start = time.time()

from diagnosenet.io_functions import IO_Functions
from diagnosenet.datamanager import MultiTask
from diagnosenet.layers import Relu, Linear
from diagnosenet.losses import MSE
from diagnosenet.optimizers import Adam
from diagnosenet.graphs import FullyConnected
from diagnosenet.executors import DesktopExecution
from diagnosenet.monitor import enerGyPU

### Read the PMSI-Dataset using Pickle from diagnosenet.io_functions
## Path for Octopus Machine: Full Representation

path = "healthData/sandbox-W1-TEST_x1_x2_x3_x4_x5_x7_x8_Y1/1_Mining-Stage/binary_representation/"
X = IO_Functions()._read_file(path + 'BPPR-pre-trained-2019.txt')
y = IO_Functions()._read_file(path + 'labels_Y3-pre-trained-2019.txt')

## 1) Define the stacked layers as the number of layers and their neurons
input_size = 38
output_size = 5

# layers_1 = [Relu(input_size, 2048),
#             Relu(2048, 2048),
#             Relu(2048, 1024),
#             Relu(1024, 1024),
#             Linear(1024, output_size)]

layers_1 = [Relu(input_size, 100),
            Relu(100, 100),
            Relu(100, 50),
            Relu(50, 50),
            Linear(50, output_size)]

## 2) Select the neural network architecture and pass the hyper-parameters
# mlp_model_1 = FullyConnected(input_size=input_size, output_size=output_size,   #239,
#                 layers=layers_1,
#                 loss=CrossEntropy,
#                 optimizer=Adam(lr=0.001),
#                 dropout=0.8)
mlp_model_1 = FullyConnected(input_size=input_size, output_size=output_size,   #239,
                layers=layers_1,
                loss=MSE,
                optimizer=Adam(lr=0.001),
                dropout=0.8)

## 3) Dataset configurations for splitting, batching and target selection
dataset_name = 'MIMIC_x1_x2_x3_Y3'
data_config = MultiTask(dataset_name=dataset_name,
                        valid_size=0.3, test_size=0.2,
                        batch_size=3,	#3072,	#100,
                        target_name='Y3',
                        target_start=0, target_end=output_size) #MT1

## 4) Select the computational platform and pass the DNN and Dataset configurations
platform = DesktopExecution(model=mlp_model_1,
                            datamanager=data_config,
                            monitor=enerGyPU(testbed_path="enerGyPU/testbed"),
                            max_epochs=10,
                            min_loss=0.02)

## 5) Uses the platform modes for training in an efficient way
platform.training_memory(X, y)

print("Execution Time: {}".format((time.time()-execution_start)))
