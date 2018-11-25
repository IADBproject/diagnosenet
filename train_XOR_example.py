"""
Example to use the DiagnoseNET Framework
for training a MLP on a desktop machine
"""

import numpy as np

from diagnosenet.layers import Relu, Linear
from diagnosenet.graphs import MLP
from diagnosenet.executors import DesktopExecution


#####################################################################
## A XOR example is defined as:
inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
input_size = 2; output_size = 2


#####################################################################
## A simple feedforward network is implemented to solve a XOR example
## 1) Define the stacked layers
layers = [Relu(input_size, 5),
            Relu(5, 2),
            Linear(2, output_size)]

## 2) Select the neural network graph
model = MLP(layers, input_size, output_size)

## 3) Select the execution machine mode
projection = DesktopExecution(model).memoryexecutor(inputs, targets)

print("Projection: {}".format(projection))
