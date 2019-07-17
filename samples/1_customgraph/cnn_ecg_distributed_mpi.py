import time
execution_start = time.time()

## Makes diagnosenet library visible in samples folder
import sys
file_path="../../"
sys.path.append(file_path)

from diagnosenet.datamanager import MultiTask, Batching
from diagnosenet.layers import Relu, Linear
from diagnosenet.losses import CrossEntropy
from diagnosenet.optimizers import Adam
from diagnosenet.graphs import CustomGraph
from diagnosenet.executors import Distibuted_MPI
from diagnosenet.monitor import enerGyPU


layer_1=[Linear(1300, 4)]

data_config = Batching(dataset_name="ECG",
                        valid_size=0.1, test_size=0.1,
                        batch_size=10,devices_number=2)

model = CustomGraph(input_size_1=1300,input_size_2=1, output_size=4,
                        loss=CrossEntropy,
                        optimizer=Adam(lr=0.0001),layers=layer_1)

projection = Distibuted_MPI(model,datamanager=data_config,monitor=enerGyPU(machine_type="arm",file_path=file_path), max_epochs=2, min_loss=0.5)

projection.synchronous_training(dataset_name="ECG",
                        dataset_path="dataset/",
                        inputs_name="xdata.npy",
                        targets_name="undim-ydata.npy")

print("Execution Time: {}".format((time.time()-execution_start)))


