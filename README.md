<p align="center"><img width="80%" src="docs/img/diagnosenet-logo.png" /></p>

--------------------------------------------------------------------------------
DiagnoseNET is a development application-framework for tailoring deep neural networks to computational architectures with an efficient balance between accuracy and energy consumption. The first application built is for deploying medical diagnostic tools inside the hospitals. In which, a feedforward network was trained on mini-cluster Jetson TX2, delivering the performance of an HPC platform in embedded modules with minimal infrastructure requirements and low power consumption.

DiagnoseNET is extending TensorFLow library to characterize the deep learning tasks and improve the balance between accuracy and energy-efficient performance.
The Package provides three high-level features:

1. A medical diagnostic workflow for deploying inside the hospitals: This workflow is divided into three stages: The first stage mining electronic health records to build a binary matrix of patients clinical descriptors. The second stage embedding the patient’s binary matrix via an unsupervised learning to obtain a new latent space and identify the patient’s phenotypic representations. The last stage focuses on supervised learning using the latent representation of patients as an input for a machine learning algorithms or as an initialiser for deep neural networks.

2. A multi-platform training DNN model: This module integrates a data and resource manager for training the DNN model, over: CPU-GPU desktop machines, on multi-GPU nodes or in the embedded computation cluster of Jetson TX2.

3. An energy-monitoring tool for workload characterization: This module deploys an energy monitor to collect the energy consumption metrics while the DNN model is executed on the target platform for analyzing the balance between accuracy and energy consumption.
