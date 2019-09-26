<p align="center"><img width="70%" src="docs/img/diagnosenet-logo.png" /></p>

--------------------------------------------------------------------------------
**DiagnoseNET** is designed as a modular framework that enables the application-workflow management and the expressivity to build and finetune deep neural networks, while its runtime abstracts the distributed orchestration of portability and scalability from a GPU workstation to multi-nodes computational platforms. It is oriented to propose a green intelligence medical workflow for deploying medical diagnostic tools with minimal infrastructure requirements and low power consumption. The first application built was to automate the [unsupervised patient phenotype representation workflow](https://link.springer.com/chapter/10.1007/978-3-030-16205-4_1) trained on a mini-cluster of Nvidia Jetson TX2. This workflow was divided into three stages:
1. The first stage mining electronic health records for patient feature extraction and serialised each patient record in a clinical document architecture schema to create a binary patient representation.
2. The second stage embedding the patient’s binary matrix via an unsupervised learning to obtain a new latent space and identify the patient’s phenotypic representations.
3. The last stage focuses on supervised learning using the patient's features (binary or latent representation) as an input for machine learning algorithms or as an initialiser for deep neural networks.


## Installation ##
DiagnoseNET is building on Ubuntu 16.04, with CUDA 8.0 support, cuDNN v6 for Python 3.6.

The main dependencies install:
```bash
pip3 install numpy==1.15.4, scipy==1.1.0, pandas==0.23.4, scikit-learn==0.20.1, psutil==5.6.3
pip3 install tensorflow-gpu==1.3.0
## Warning: When using Tensorflow-gpu in version 1.4.1,
## the module 'tensorflow.python.framework.fast\_tensor\_util' does not match with the runtime Python 3.6.

```

To install the current release, clone it including submodules:
```bash
git clone --recurse-submodules https://github.com/IADBproject/diagnosenet.git
```

See a good practice guide to build a [mini-cluster Nvidia Jetson TK1 & TX2](https://diagnosenet.github.io/getstarted/) from scratch.


## Cross-platform Library ##
DiagnoseNET is extending the TensorFLow library to characterize and automate the deep learning workflows on distributed platforms to improve the ratio between accuracy and energy-efficient performance.
It is designed into independent and interchangeable modules to exploit the computational resources on two levels of parallel and distributed processing. The first level management and synchronize the data parallelism for mini-batch learning between the workers, while the second level adjust the task granularity (model dimension and batch partition) according to the computational platform characteristics (memory capacity, number of CPUs, GPUs, the GPU micro-architecture, clocks frequency and among of others), as shown in the next schema.

<p align="center"><img width="100%" src="docs/img/diagnosenet_cross-platform_library.png" /></p>

The cross-platform library contains a task-based programming interface module for building the DNN model graphs, in which the developers design and parameterize a pre-build neural network family as fully-connected, stacked encoder-decoder and among others. The second module is called platform execution modes to select the computational platform for training the DNN model graph and hides the complexity posed by the heterogeneity in the computing platforms. Another module integrates a data and resource manager for training the DNN model graph, over CPU-GPU desktop machines, on multi-GPU nodes or in the embedded computation cluster of Jetson TX2. And the last module integrated an energy-monitoring tool called [enerGyPU](https://github.com/jagh/enerGyPU) for workload characterization, which collects the energy consumption metrics while the DNN model is executed on the target platform.


## Get Started with DiagnoseNET ##
Let’s start with an example to build a feed-forward neural network to predict medical care purpose of hospitalized patients and training it on a traditional CPU-GPU machine.
* The first and the second step consist in builds a fully-connected model graph. In the simplest way, the developer set the type of each layer, their neurons numbers and the number of layers building a stacked network and followed by a linear output on top.  After is to select the neural network family graph 'FullyConnected' and set the hyperparameters.
```bash
from diagnosenet.layers import Relu, Linear
from diagnosenet.graphs import FullyConnected
from diagnosenet.losses import CrossEntropy
from diagnosenet.optimizers import Adam

## 1) Set stacked layers as type, depth and width:
layers_1 = [Relu(14637, 2048),
            Relu(1024, 1024),
            Relu(1024, 1024),
            Relu(1024, 1024),
            Linear(1024, 14)]

## 2) Model and hyperparameters setting:
mlp_model_1 = FullyConnected(input_size=14637, output_size=14,
                layers=layers_1,
                loss=CrossEntropy,
                optimizer=Adam(lr=0.001),
                dropout=0.8)
```

* The third step allows: managing a multitask target; automatically split the dataset in training, validation and test; as well as partitioning each set according to defined batch size.
```bash
from diagnosenet.datamanager import MultiTask

## 3) Set splitting, batching and target for the dataset:
data_config = MultiTask(dataset_name="SENSE-CUSTOM_x1_x2_x3_x4_x5_x7_x8_Y1",
                        valid_size=0.05, test_size=0.15,
                        batch_size=3072,
                        target_name='Y11',
                        target_start=0, target_end=14)
```

* In the last two steps, we select the computational platform 'DesktopExecution' and pass the DNN model and dataset configuration. According to the machine-memory capacity, if the full-dataset plus the model can be allocated in memory is to select 'training_memory' execution modes or in another wise is selected 'training_disk'.
```bash
from diagnosenet.executors import DesktopExecution
from diagnosenet.monitor import enerGyPU
from diagnosenet.io_functions import IO_Functions

## 4) Select the computational platform and pass the DNN and Dataset configurations:
platform = DesktopExecution(model=mlp_model_1,
                            datamanager=data_config,
                            monitor=enerGyPU(testbed_path="/data/jagh/green_learning/testbed"),
                            max_epochs=40,
                            min_loss=0.02)

## 5) Select the training platform modes:
## Dataset Load
path = "healthData/sandbox-SENSE-CUSTOM_x1_x2_x3_x4_x5_x7_x8_Y1/1_Mining-Stage/binary_representation/"
X = IO_Functions._read_file(path+"BPPR-SENSE-CUSTOM_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")
y = IO_Functions._read_file(path+"labels_Y1-SENSE-CUSTOM_x1_x2_x3_x4_x5_x7_x8_Y1-2008.txt")

## Training Start
platform.training_memory(X, y)
```
The source code is available here for [diagnosenet_desktop-memory](https://github.com/IADBproject/diagnosenet/blob/master/diagnosenet_desktop-memory.py) or [diagnosenet_desktop-disk](https://github.com/IADBproject/diagnosenet/blob/master/diagnosenet_desktop-disk.py).

--------------------------------------------------------------------------------
