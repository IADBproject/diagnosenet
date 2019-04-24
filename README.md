<p align="center"><img width="70%" src="docs/img/diagnosenet-logo.png" /></p>

--------------------------------------------------------------------------------
**DiagnoseNET** is an open source framework for tailoring deep neural networks into different computational architectures from CPU-GPU implementation to multi-GPU and multi-nodes with an efficient ratio between accuracy and energy consumption. It is oriented to propose a green intelligence medical workflow for deploying medical diagnostic tools inside the hospitals with minimal infrastructure requirements and low power consumption.

The first application built was to automate the [unsupervised patient phenotype representation workflow](https://link.springer.com/chapter/10.1007/978-3-030-16205-4_1) trained on a mini-cluster of Nvidia Jetson TX2. This workflow was divided into three stages: 
1. The first stage mining electronic health records for patient feature extraction and serialised each patient record in a clinical document architecture schema to create a binary patient representation.
2. The second stage embedding the patient’s binary matrix via an unsupervised learning to obtain a new latent space and identify the patient’s phenotypic representations. 
3. The last stage focuses on supervised learning using the patient's features (binary or latent representation) as an input for machine learning algorithms or as an initialiser for deep neural networks.


## Installation ##
DiagnoseNET is building on Ubuntu 16.04, with CUDA 8.0 support, cuDNN v6 for Python 3.6.

The main dependencies install:
```bash
pip3 install numpy==1.15.4, scipy==1.1.0, pandas==0.23.4, scikit-learn==0.20.1
pip3 install tensorflow-gpu==1.3.0
## Warning: When using Tensorflow-gpu in version 1.4.1, 
## the module 'tensorflow.python.framework.fast\_tensor\_util' does not match with the runtime Python 3.6.
```

To install the current release:
```bash
git clone https://github.com/IADBproject/diagnosenet.git
```

See a good practice guide to build a [mini-cluster Nvidia Jetson TK1 & TX2](https://diagnosenet.github.io/getstarted/) from scratch.


## Cross-platform Library ##
DiagnoseNET is extending the TensorFLow library to characterize and automate the deep learning workflows on distributed platforms to improve the ratio between accuracy and energy-efficient performance.
It is designed into independent and interchangeable modules to exploit the computational resources on two levels of parallel and distributed processing. The first level management and synchronize the data parallelism for mini-batch learning between the workers, while the second level adjust the task granularity (model dimension and batch partition) according to the computational platform characteristics (memory capacity, number of CPUs, GPUs, the GPU micro-architecture, clocks frequency and among of others), as shown in the next schema.

<p align="center"><img width="100%" src="docs/img/diagnosenet_cross-platform_library.png" /></p>

The cross-platform library contains a task-based programming interface module for building the DNN model graphs, in which the developers design and parameterize a pre-build neural network family as fully-connected, stacked encoder-decoder and among others. The second module is called platform execution modes to select the computational platform and exploit the training process according to the capabilities of the machine. Another module integrates a data and resource manager for training the DNN model graph, over CPU-GPU desktop machines, on multi-GPU nodes or in the embedded computation cluster of Jetson TX2. And the last module integrated an energy-monitoring tool called enerGyPU for workload characterization, which collects the energy consumption metrics while the DNN model is executed on the target platform.

