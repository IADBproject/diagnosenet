"""
Medical Care Purpose Classification for PMSI-ICU Dataset
Developing example for training DiagnoseNET  on multiGPU machine
"""

import time
execution_start = time.time()

## Makes diagnosenet library visible in samples folder
import sys
file_path="../../"
sys.path.append(file_path)

from diagnosenet.io_functions import IO_Functions
from diagnosenet.datamanager import MultiTask
from diagnosenet.layers import Relu, Linear
from diagnosenet.losses import CrossEntropy
from diagnosenet.optimizers import Adam
from diagnosenet.graphs import SequentialGraph
from diagnosenet.executors import MultiGPU


## PMSI-ICU Dataset shapes
X_shape = 14637
Y1_shape = 14
Y2_shape = 239
Y3_shape = 5

## 1) Define the stacked layers as the number of layers and their neurons
layers = [Relu(X_shape, 2048),
            Relu(2048, 2048),
            Relu(2048, 2048),
            Relu(2048, 2048),
            Linear(2048, y_shape)]

## 2) Select the neural network architecture and pass the hyper-parameters
mlp_model = SequentialGraph(input_size=X_shape, output_size=y_shape,
                layers=layers,
                loss=CrossEntropy(),
                optimizer=Adam(lr=0.001),
                dropout=0.8)

## 3) Dataset configurations for splitting, batching and target selection
data_config = MultiTask(dataset_name="W1-TEST_x1_x2_x3_x4_x5_x7_x8_Y1",
                        valid_size=0.10, test_size=0.10,
                        batch_size=200,
                        target_name='Y11',
                        target_start=0, target_end=14)

## 4) Select the computational platform and pass the DNN and Dataset configurations
platform = MultiGPU(model=mlp_model,
                    datamanager=data_config,
                    monitor=enerGyPU(file_path=file_path),
                    gpus=2,
                    max_epochs=2)

### Read the PMSI-Dataset using Pickle from diagnosenet.io_functions
X = IO_Functions()._read_file("dataset/patients_features.txt")
y = IO_Functions()._read_file("dataset/medical_targets.txt")

## 5) Uses the platform modes for training in an efficient way
platform.training_multigpu(X, y)
# platform.write_metrics()

print("Execution Time: {}".format((time.time()-execution_start)))


###########################################################
# path = "/data_B/datasets/drg-PACA/healthData/sandbox-FULL-W1_x1_x2_x3_x4_x5_x7_x8_Y1/1_Mining-Stage/binary_representation/"
# X = IO_Functions()._read_file(path+"BPPR-FULL-W1_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")
# y = IO_Functions()._read_file(path+"labels_Y1-FULL-W1_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")
