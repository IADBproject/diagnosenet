"""
Medical Care Purpose Classification for PMSI-ICU Dataset
User example for training DiagnoseNET on distributed platform
"""

import time
execution_start = time.time()

## Makes diagnosenet library visible in samples folder
import sys
sys.path.append('../../')

from diagnosenet.datamanager import MultiTask, Batching
from diagnosenet.layers import Relu, Linear
from diagnosenet.losses import CrossEntropy
from diagnosenet.optimizers import Adam
from diagnosenet.graphs import SequentialGraph
from diagnosenet.executors import Distibuted_GRPC
from diagnosenet.monitor import enerGyPU


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
                        batch_size=500)

## 4) Select the computational platform and pass the DNN and Dataset configurations
platform = Distibuted_GRPC(model=mlp_model,
                            datamanager=data_config_1,
                            monitor=enerGyPU(machine_type="arm"),
                            max_epochs=10,
                            min_loss=2.0,
                            ip_ps="134.59.132.135:2222",
                            ip_workers="134.59.132.20:2222")	#,134.59.132.21:2222")

## 5) Uses the platform modes for training in an efficient way
platform.synchronous_training(dataset_name="MCP-PMSI",
                                dataset_path="dataset/",
                                inputs_name="patients_features.txt",
                                targets_name="medical_targets.txt",
                                num_ps=1,
                                num_workers=1)

print("Execution Time: {}".format((time.time()-execution_start)))
