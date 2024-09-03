FROM public.ecr.aws/neuron/pytorch-inference-neuronx:2.1.2-neuronx-py310-sdk2.19.1-ubuntu20.04

# Clone the vllm repository
RUN git clone https://github.com/vllm-project/vllm.git

# Set the working directory
WORKDIR /vllm
RUN git checkout v0.5.0

# Set the environment variable
ENV VLLM_TARGET_DEVICE=neuron

# Install the dependencies
RUN pip install -U -r requirements-neuron.txt
RUN pip install -e .

# Modify the arg_utils.py file
RUN sed -i "/parser.add_argument('--block-size',/ {N;N;N;N;N;s/\[8, 16, 32\]/[8, 16, 32, 128, 256, 512, 1024, 2048, 4096, 8192]/}" vllm/engine/arg_utils.py

# Install ray
RUN pip install ray
RUN pip install pynvml

# Set the entry point
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]