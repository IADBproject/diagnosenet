import time
execution_start = time.time()

## Makes diagnosenet library visible in samples folder
import sys
workspace_path="../../"
sys.path.append(workspace_path)

from diagnosenet.losses import CrossEntropy
from diagnosenet.optimizers import Adam
from diagnosenet.graphs import CustomGraph
from diagnosenet.executors import DesktopExecution
from diagnosenet.datamanager import Batching
from diagnosenet.monitor import enerGyPU
from diagnosenet.layers import Linear
import numpy as np
import pandas as pd

#load data 
file_dir = "dataset/"
inputs = np.load(file_dir+'xdata-big.npy')
targets = np.load(file_dir+'ydata-big.npy')
targets=pd.get_dummies(targets).values

layer_1=[Linear(1300, 4)]

model = CustomGraph(input_size_1=1300,input_size_2=1, output_size=4,
                        loss=CrossEntropy,
                        optimizer=Adam(lr=0.0001),layers=layer_1)

data_config = Batching(dataset_name="ECG",
                        valid_size=0.1, test_size=0.1,
                        batch_size=1000)

projection = DesktopExecution(model,
                        datamanager=data_config,
                        monitor=enerGyPU(machine_type="x86", file_path=workspace_path),
                        max_epochs=20, early_stopping=5)

projection.training_memory(inputs, targets)

print("Execution Time: {}".format((time.time()-execution_start)))
