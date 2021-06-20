# Multi-scale-PDEs
This codes are implemented on tensorflow 1.14.0

This repository contains the code for the following paper:

# A multi-scale DNN algorithm for nonlinear elliptic equations with multiple scales  
created by Xi-An Li, Zhi-Qin John, Xu and Lei Zhang

[[Paper]](https://arxiv.org/pdf/2009.14597.pdf)

Abstract: Algorithms based on deep neural networks (DNNs) have attracted increasing attention from the scientific computing community. DNN based algorithms are easy to implement, natural for nonlinear problems, and have shown great potential to overcome the curse of dimensionality. In this work, we utilize the multi-scale DNN-based algorithm (MscaleDNN) proposed by Liu, Cai and Xu (2020) to solve multi-scale elliptic problems with possible nonlinearity, for example, the p-Laplacian problem. We improve the MscaleDNN algorithm by a smooth and localized activation function. Several numerical examples of multi-scale elliptic problems with separable or non-separable scales in low-dimensional and high-dimensional Euclidean spaces are used to demonstrate the effectiveness and accuracy of the MscaleDNN numerical scheme.


# Ideas
This work is used to solve a class of multi-scale pde problems by means of the technique of DNN with the idea of fourier basis.

# Abstract
Non-linear partial different equations (PDEs) with multiple scale are quite troublesome issue in the fields of science and engineering. In this work, we probe into solving this type of PDEs by means of a multi-scale deep neural networks (MscaleDNN) algorithm. Such MscaleDNN based algorithm is attractive due to its potentiality for dealing with well multi-scale problems. Inspired by Fourier expansion and decomposition, two local and smooth activation functions, i.e., sine and cosine functions, are introduced to enhance the performance of MscaleDNN. In addition, the MscaleDNN architecture employs lopsidedly activation function for different neural layers. By introducing some numerical examples of $p$-Laplacian problems with different scales for independent variables in low-dimensional and high-dimensional Euclidean spaces, we demonstrate that the improved MscaleDNN algorithm is feasible and can attain favorable accuracy to multi-scale problems for both low-frequency and high-frequency oscillation cases in regular domain.

# Noting:
The codes can be implemented in tensorflow--1.14 or 1.15 under the interpreter python3.6 or python3.7

The matlab codes in 2D辅助matlab代码/p=2 are useful for E1,E2,E3 and E4.

The matlab codes in 2D辅助matlab代码/p=3Forier_scale are useful for E5.

The matlab codes in 2D辅助matlab代码/p=3Subspace are useful for E6.

We have also submitted the new codes for solving multi-scale PDEs on Tensorflow 2.X, one can redirect to https://github.com/Blue-Giant/MscaleDNN-tf2
