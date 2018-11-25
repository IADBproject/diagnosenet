"""
Example to use DiagnoseNET Framework
for training a feedforward network on a desktop machine.

A XOR example is defined by:    Entries      its  Output
                            [0, 0] or [1, 1]  ->   [1, 0]
                            [0, 1] or [1, 0]  ->   [0, 1]
"""

import numpy as np

from diagnosenet.layers import Relu, Softmax
from diagnosenet.losses import CrossEntropy
from diagnosenet.optimizers import Adam
from diagnosenet.graphs import MLP
from diagnosenet.executors import DesktopExecution


#####################################################################
## A simple feedforward network is implemented to solve a XOR example

inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1],
                   [0, 0], [1, 0], [0, 1], [1, 1]])
targets = np.array([[1, 0], [0, 1], [0, 1], [1, 0],
                    [1, 0], [0, 1], [0, 1], [1, 0]])

## 1) Define the stacked layers as the number of layers and their neurons
staked_layers = [Relu(2, 32),
                Relu(32, 32),
                Softmax(32, 2)]

## 2) Select the neural network architecture and pass the hyper-parameters
model = MLP(input_size=2, output_size=2,
            layers=staked_layers,
            loss=CrossEntropy,
            optimizer=Adam(lr=0.01))

## 3) Select the execution machine mode
projection = DesktopExecution(model, max_epochs=100).memoryexecutor(inputs, targets)

print("Projection: {}".format(projection))
