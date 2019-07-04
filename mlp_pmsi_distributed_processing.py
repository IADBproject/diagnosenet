"""
Medical Care Purpose Classification for PMSI-ICU Dataset
User example for training DiagnoseNET on distributed platform
"""

import time
execution_start = time.time()

from diagnosenet.datamanager import MultiTask, Batching
from diagnosenet.layers import Relu, Linear
from diagnosenet.losses import CrossEntropy
from diagnosenet.optimizers import Adam
from diagnosenet.graphs import FullyConnected
from diagnosenet.executors import Distibuted_GRPC
from diagnosenet.monitor import enerGyPU


input_size=14637
output_size=381

## 1) Define the stacked layers as the number of layers and their neurons
layers = [Relu(input_size, 2048),
            Relu(2048, 2048),
            Relu(2048, 2048),
            Relu(2048, 2048),
            Linear(2048, output_size)]

## 2) Select the neural network architecture and pass the hyper-parameters
mlp_model = FullyConnected(input_size=input_size, output_size=output_size,
                            layers=layers,
                            loss=CrossEntropy,
                            optimizer=Adam(lr=0.001),
                            dropout=0.8)

## 3) Dataset configurations for splitting, batching and target selection
data_config_1 = Batching(dataset_name="W1-TEST_x1_x2_x3_x4_x5_x7_x8_Y1",
                        valid_size=0.05, test_size=0.10,
                        devices_number=2,
                        batch_size=200)

## 4) Select the computational platform and pass the DNN and Dataset configurations
platform = Distibuted_GRPC(model=mlp_model,
                            datamanager=data_config_1,
                            monitor=None,
                            max_epochs=2,
                            min_loss=2.0,
                            ip_ps="134.59.132.135:2222",
                            ip_workers="134.59.132.20:2222,134.59.132.21:2222")

## 5) Uses the platform modes for training in an efficient way
platform.synchronous_training(dataset_name="W1-TEST_x1_x2_x3_x4_x5_x7_x8_Y1",
                                dataset_path="healthData/",
                                inputs_name="BPPR",
                                targets_name="labels_Y1",
                                num_ps=1,
                                num_workers=2)

print("Execution Time: {}".format((time.time()-execution_start)))
