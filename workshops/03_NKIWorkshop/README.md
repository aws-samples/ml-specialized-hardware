# Building Custom Accelerator Kernels with AWS Neuron Kernel Interface (NKI)

## Introduction

The Neuron Kernel Interface (NKI) is a domain-specific language and runtime that allows developers to write custom kernels optimized for AWS Neuron devices (Trainium/Inferentia). NKI enables direct access to the hardware's compute and memory resources, giving you fine-grained control over how your workloads execute on Neuron accelerators.

With NKI, you can create highly optimized implementations of custom operators, extend ML frameworks with new functionality, and maximize performance for your unique machine learning workloads. This interface bridges the gap between high-level ML frameworks and the underlying Neuron hardware, providing a programming environment similar to CUDA but specifically designed for AWS Neuron devices.

NKI allows developers to harness the full computational power of AWS Neuron devices by writing kernels that explicitly manage computation and memory operations. This level of control is essential for specialized workloads that require custom optimization beyond what's possible with standard operators provided by ML frameworks.
Scenarios and Use Cases for NKI

## NKI Programming Model

NKI follows a three-phase programming model that gives developers explicit control over the execution of their kernels:

1. Load - Move data from device memory (HBM) to on-chip memory (SBUF)
   * Explicitly define which data to bring into fast on-chip memory
   * Control memory access patterns to optimize bandwidth utilization\
   * Apply data transformations during loading if needed

2. Compute - Perform operations using on-chip memory
   * Execute arithmetic operations on data in on-chip memory
   * Leverage vector and matrix operations for efficient computation
   * Utilize specialized hardware units for operations like matrix multiplication

3. Store - Move results from on-chip memory back to device memory
   * Control when and how results are written back to device memory
   * Optimize for memory bandwidth by storing results efficiently
   * Apply masks or conditions to selectively update memory

This programming model is based on the architecture of Neuron devices, which feature a large HBM (High Bandwidth Memory) for storing model weights and activations, and a smaller but faster on-chip memory (SBUF) for active computations. By explicitly managing data movement between these memory tiers, developers can optimize for both performance and energy efficiency.

## This Workshop

This hands-on workshop will teach you how to build, optimize, and integrate custom kernels for AWS Neuron devices using NKI. You'll learn the fundamentals of kernel development, how to integrate kernels with PyTorch, and how to analyze performance with Neuron Profile.

Duration: Approximately 90 minutes
Workshop Outline

1. Environment Setup
   * Configuring your Trn1 or Inf2 instance
   * Installing required packages
   * Verifying your setup

2. Implementing Your First NKI Kernel
   * Understanding the NKI programming model
   * Writing a simple tensor addition kernel
   * Running kernels in baremetal mode and with PyTorch

3. Integrating Prebuilt Kernels
   * Using optimized kernels from the neuronxcc.nki.kernels namespace
   * Comparing custom Flash Attention implementation with standard attention
   * Understanding performance benefits of optimized kernels

4. Creating Custom Operators
   * Inserting NKI kernels as custom operators in PyTorch
   * Implementing forward and backward passes for training
   * Supporting autograd for custom operators

5. Performance Analysis with Neuron Profile
   * Installing and using Neuron Profile
   * Capturing execution traces
   * Analyzing kernel performance metrics
   * Identifying optimization opportunities


Each section builds upon the previous one, gradually introducing more advanced concepts while providing hands-on experience with real code examples.
Prerequisites

In the first notebook, you'll set up your environment and verify that NKI is properly installed and configured. Then you'll implement your first kernel - a simple tensor addition operation - and learn how to run it both in baremetal mode and through PyTorch. This will establish the foundation for the more advanced topics covered in subsequent notebooks.

Let's begin with the environment setup and your first NKI kernel!
