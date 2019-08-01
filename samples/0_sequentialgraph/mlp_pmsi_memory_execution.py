"""
Medical Care Purpose Classification for PMSI-ICU Dataset
User example for training DiagnoseNET exploit the memory desktop machine
"""

import time
execution_start = time.time()

## Makes diagnosenet library visible in samples folder
import sys
workspace_path="../../"
sys.path.append(workspace_path)

from diagnosenet.io_functions import IO_Functions
from diagnosenet.datamanager import MultiTask
from diagnosenet.layers import Relu, Linear
from diagnosenet.losses import CrossEntropy
from diagnosenet.optimizers import Adam
from diagnosenet.graphs import SequentialGraph
from diagnosenet.executors import DesktopExecution
from diagnosenet.monitor import enerGyPU


## PMSI-ICU Dataset shapes
X_shape = 14637
Y1_shape = 14
Y2_shape = 239
Y3_shape = 5

## 1) Define the stacked layers as the number of layers and their neurons
layers_1 = [Relu(X_shape, 512),
            Relu(512, 512),
            Relu(512, 512),
            Relu(512, 512),
            Linear(512, Y1_shape)]

## 2) Select the neural network architecture and pass the hyper-parameters
mlp_model_1 = SequentialGraph(input_size=X_shape, output_size=Y1_shape,
                layers=layers_1,
                loss=CrossEntropy,
                optimizer=Adam(lr=0.001),
                dropout=0.8)

## 3) Dataset configurations for splitting, batching and target selection
data_config = MultiTask(dataset_name="MCP-PMSI",
                        valid_size=0.05, test_size=0.15,
                        batch_size=250,
                        target_name='Y11',
                        target_start=0, target_end=14)

## 4) Select the computational platform and pass the DNN and Dataset configurations
platform = DesktopExecution(model=mlp_model_1,
                            datamanager=data_config,
                            monitor=enerGyPU(machine_type="arm", file_path=workspace_path),
                            max_epochs=10, early_stopping=5)

### Read the PMSI-Dataset using Pickle from diagnosenet.io_functions
X = IO_Functions()._read_file("dataset/patients_features.txt")
y = IO_Functions()._read_file("dataset/medical_targets_Y14.txt")

## 5) Uses the platform modes for training in an efficient way
platform.training_memory(X, y)

print("Execution Time: {}".format((time.time()-execution_start)))



####################################################
## Path for Octopus Machine: Full Representation
#path = "/data_B/datasets/drg-PACA/healthData/sandbox-FULL-W1_x1_x2_x3_x4_x5_x7_x8_Y1/1_Mining-Stage/binary_representation/"
#X = IO_Functions()._read_file(path+"BPPR-FULL-W1_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")
#y = IO_Functions()._read_file(path+"labels_Y1-FULL-W1_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")

## Path for Octopus Machine: SENSE-CUSTOM Representation
#path = "/data_B/datasets/drg-PACA/healthData/sandbox-SENSE-CUSTOM_x1_x2_x3_x4_x5_x7_x8_Y1/1_Mining-Stage/binary_representation/"
#X = IO_Functions()._read_file(path+"BPPR-SENSE-CUSTOM_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")
#y = IO_Functions()._read_file(path+"labels_Y1-SENSE-CUSTOM_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")
