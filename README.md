# Applied AI/ML Specialized Hardware

Specialized Hardware is a ML (Machine Learning) model accelerator for Inference or Training, like [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/), [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/), [SIMD accel in CPUs](https://en.wikipedia.org/wiki/SIMD) and GPUs. In this repo you'll find reference implementations of different use cases (applications) for Computer Vision, Natural Language Processing, etc. that make use of hardware acceleration to reduce model execution latency and increase throughput.

**Use cases** are represented by questions which can be answered by the reference implementation linked to it.

If you're looking for technical samples that show how to run specific models on Trainium (trn1) and Inferentia (inf1 & inf2), go to [AWS Neuron Samples](https://github.com/aws-neuron/aws-neuron-samples)

## Tutorials/Reference implementations
|Use Case|Description|
|-|-|
|[How to track people in video files?](tutorials/ObjectTrackingSageMakerGStreamer/)|CV/ML Pipeline to process video files in batch with SageMaker+Inferentia, GStreamer and Yolov7+ByteTrack|
|[How to measure the similarity between two sentences?](tutorials/EmbeddingsFromTextWithBert/)|Compute the semantic similarity of two or more sentences by extracting their embeddings with SageMaker+Inferentia and HF Bert Case|
|[How to create a mechanism to answer questions from a FAQ?](tutorials/QuestionAnsweringMachine/)|Fine tune a T5-ssm model (on SageMaker & Trainium) to build a Q&A mechanism, more powerful than a classic chatbot, to answer questions from a FAQ, sent by your customers|

## Contributing
If you have a question related to a business challenge that must be answered by an accelerated AI/ML solution, like the content in this repo, then you can contribute. You can just open an issue with your question or if you have the skills, implement a solution (tutorial, workshop, etc.) using Jupyter notebooks (for SageMaker Studio or Notebook Instances) and create a pull request. We appreciate your help.

Please refer to the [CONTRIBUTING](CONTRIBUTING.md) document for further details on contributing to this repository.
