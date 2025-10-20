# Reducing ML Inference Costs with AWS Inferentia

This workshop demonstrates how to reduce inference costs using AWS Inferentia2 with OpenAI's Whisper model for automatic speech recognition. You'll learn different deployment approaches and compare their cost-performance characteristics.

## Notebooks Overview

### [01_Whisper_gpu.ipynb](01_Whisper_gpu.ipynb)
Baseline deployment on ml.g5.xlarge. Establishes cost and performance benchmarks for comparison.

### [02_Whisper_optimum_neuron.ipynb](02_Whisper_optimum_neuron.ipynb)
Deploy Whisper on AWS Inferentia2 using [Hugging Face Optimum Neuron](https://huggingface.co/docs/optimum-neuron/index). This is the **recommended approach** for supported models - compilation is automated and straightforward.

**Key steps:**
- Compile model on ml.trn1.2xlarge (~15-20 min)
- Deploy on ml.inf2.xlarge
- Option to use pre-compiled model from S3

### [03_Whisper_neuron_manual_port.ipynb](03_Whisper_neuron_manual_port.ipynb)
Manual model porting to Neuron by splitting Whisper into encoder, decoder, and projection components. Use this approach when:
- Model is not yet supported by Optimum Neuron
- You need fine-grained control over compilation
- You want to understand the underlying compilation process

**Key steps:**
- Manual component tracing with torch_neuronx
- Custom inference handlers
- Upload compiled artifacts and inference code to S3
- Compilation time ~18 min

## Prerequisites

- AWS account with SageMaker access
- IAM role with SageMaker permissions
- Access to ml.g5.xlarge, ml.trn1.2xlarge, and ml.inf2.xlarge instances

## Cost Analysis

Each notebook calculates the cost per second of audio transcribed. This allows you to evaluate the cost-performance benefits of deploying on AWS Inferentia2 (ml.inf2.xlarge) compared to the baseline.

## Getting Started

1. Start with notebook 01 to establish the baseline
2. Run notebook 02 for the easiest Inferentia deployment
3. Explore notebook 03 if you need manual control or want to learn advanced techniques

## Resources

- [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/)
- [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/)
- [Optimum Neuron Documentation](https://huggingface.co/docs/optimum-neuron/index)
- [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/)
