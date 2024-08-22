# Llama3-8B Deployment on AWS Inferentia 2 with Amazon EKS and vLLM

This repository contains the necessary files and configurations to deploy the Llama3-8B model on AWS Inferentia 2 instances using Amazon EKS (Elastic Kubernetes Service) and vLLM.

## Files in this Directory

1. `Dockerfile`: Defines the container image for running vLLM with Llama3-8B on Inferentia 2.

2. `cluster-config.yaml`: Configuration file for creating the Amazon EKS cluster.

3. `deployment.yaml`: Kubernetes deployment configuration for the Llama3-8B model.

4. `nodegroup-config.yaml`: Configuration for the Inferentia 2 node group in the EKS cluster.

## Overview

This project demonstrates how to:

- Set up an Amazon EKS cluster
- Configure Inferentia 2 node groups
- Build and push a custom Docker image for vLLM
- Deploy Llama3-8B model using vLLM on Inferentia 2 instances
- Configure Kubernetes probes for health checking
- Scale the deployment

## Prerequisites

- AWS CLI
- eksctl
- kubectl
- docker

## Getting Started

1. Create the EKS cluster using the `cluster-config.yaml` file.
2. Set up the Inferentia 2 node group using the `nodegroup-config.yaml` file.
3. Build and push the Docker image using the provided `Dockerfile`.
4. Deploy the Llama3-8B model using the `deployment.yaml` file.

## Important Notes

- The deployment uses 8 Neuron cores per replica for optimal performance.
- Initial startup time for model compilation is around 25 minutes.
- Proper monitoring and scaling strategies are crucial for production use.

For detailed instructions and explanations, please refer to the accompanying blog post.

## Authors

- Dmitri Laptev - Senior GenAI Solutions Architect at AWS
- Maurits de Groot - Solutions Architect at AWS
- Ziwen Ning - Software Development Engineer at AWS
- Jianying Lang - Principal Solutions Architect at AWS Worldwide Specialist Organization (WWSO)
