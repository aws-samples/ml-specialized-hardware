# How to reduce costs and improve performance of your Machine Learning (ML) workloads?

In this repo you'll learn how to use [AWS Trainium](https://aws.amazon.com/machine-learning/trainium/) and [AWS Inferentia](https://aws.amazon.com/machine-learning/inferentia/) with [Amazon SageMaker](https://aws.amazon.com/sagemaker/) and [Hugging Face Optimum Neuron](https://huggingface.co/docs/optimum-neuron/index), to optimize your ML workloads! Here you find workshops, tutorials, blog post content, etc. you can use to learn and inspire your own solution.


The content you find here is focused on particular use cases. If you're looking for standalone model samples for inference and training, please check this other repo: https://github.com/aws-neuron/aws-neuron-samples. 

### Workshops

|Title||
|:-|:-|
|[Fine-tune and deploy LLM from Hugging Face on AWS Trainium and AWS Inferentia](workshops/01_FineTuneSpamClassifier)|Learn how to create a spam classifier that can be easily integrated to your own application|
|[Adapting LLMs for domain-aware applications with AWS Trainium post-training](workshops/02_DomainAdaptation)|Learn how to adapt a pre-trained model to your own business needs and add a conversational interface your customers can interact with|
|[Building Custom Accelerator Kernels with AWS Neuron Kernel Interface (NKI)](workshops/03_NKIWorkshop)|Learn how to use the Neuron Kernel Interface (NKI) to write kernels for Neuron accelerators|


These workshops are supported by **AWS Workshop Studio**

### Tutorials

|Description|
|:-|
|[inf1 - Extract embeddings from raw text](tutorials/01_EmbeddingsFromTextWithBert)|
|[inf1 - Track objects in video streaming using CV](tutorials/02_ObjectTrackingSageMakerGStreamer)|
|[inf1 - Create a closed question Q&A model](tutorials/03_QuestionAnsweringMachine)|
|[inf2 - Generate images using SD](tutorials/04_ImageGenerationWithStableDiffusion)|
|[inf1 - Answer questions given a context](tutorials/05_FastQuestionAnsweringWithBertQA)|
|[trn1 - Fine-tune a LLM using distributed training](tutorials/06_FinetuneLLMs)|
|[inf2 - Deploy a LLM to HF TGI](tutorials/07_DeployToInferentiaWithTGI)|
|[inf2 - Porting BART for Multi-Genre Natural Language Inference](tutorials/08_TextClassificationWithNaturalLanguageInference)|

### Blog posts content
|Description|
|:-|
|[Llama3-8B Deployment on AWS Inferentia 2 with Amazon EKS and vLLM](blogs/01_LLama3-8B_Inferentia_EKS_vLLM/)|

## Contributing
If you have questions, comments, suggestions, etc. please feel free to cut tickets in this repo.

Also, please refer to the [CONTRIBUTING](CONTRIBUTING.md) document for further details on contributing to this repository.
