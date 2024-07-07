# Class-Attention-An-Improved-Profied-Attack-for-SM4-ALgorithm
Welcome to the repository for the paper “Class-Attention: An Improved Profiled Attack for SM4 Algorithm.” This work explores the application of a Transformer structure with a Class Token in the context of side-channel attacks. It lays the foundation for cross-device attacks using multimodal aligned features for side-channel attacks. We have also programmed firmware incorporating the SM4 algorithm mask and collected power traces.

Repository Overview

This repository is divided into two main parts:

Firmware Part

	•	Jupyter notebook files and Makefile used for power trace collection with ChipWhisperer.
	•	Firmware files with the masked SM4 algorithm for various platforms.
	•	Example power traces of the SM4 algorithm with masked key encryption.

Software Part

	•	Python files for the network structure.
	•	Python files used for measuring the signal-to-noise ratio (SNR).
	•	Python files used for implementing random delay.

The model code provided in the paper may not be directly transferable due to the differences in deep learning frameworks and versions used by different readers (the paper uses TensorFlow2, which many readers may no longer use). Below, the author provides a reference implementation in PyTorch:

In the model provided in the paper, to keep the model lightweight, the embedding layer is not used during multi-head attention computation. However, readers can add it when reproducing the model. To replicate the model structure in your own environment, follow these steps:

	1.	Ensure you have the code for the one-dimensional local connection layer (LC) for power traces.
 #############################################
import torch
import torch.nn as nn

class LocalConnection1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(LocalConnection1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)
##############################################	
	2.	Ensure you have the code for a standard Transformer, or design your own self-attention model.
	3.	Have the code for the ClassToken.
 ##############################################
 class ClassToken(nn.Module):
    def __init__(self, token_dim):
        super(ClassToken, self).__init__()
        self.token = nn.Parameter(torch.zeros(1, 1, token_dim))

    def forward(self, batch_size):
        return self.token.expand(batch_size, -1, -1)
##############################################
	4.	After the power traces are processed by the LC, concatenate them with the ClassToken and input them into the self-attention model for computation.
	5.	Use the output token corresponding to the ClassToken as the variable for classification.
