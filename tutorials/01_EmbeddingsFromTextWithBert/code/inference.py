# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
import os
os.environ['NEURON_RT_NUM_CORES'] = '4'
import json
import torch
import torch.neuron
from typing import List
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

def compute_embeddings(features, sentences):
    attention_mask = sentences['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(features.size()).float()
    masked_embeddings = features * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    
    return (summed / summed_mask).numpy()

def model_fn(model_dir):
    # load tokenizer and neuron model from model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = torch.jit.load(os.path.join(model_dir, "model_neuron.pt"))
    model_config = AutoConfig.from_pretrained(model_dir)

    return model, tokenizer, model_config

def predict_fn(data, model_tokenizer_model_config):
    # destruct model and tokenizer
    model, tokenizer, model_config = model_tokenizer_model_config
    encoded_input = tokenizer.batch_encode_plus(
        data['inputs'],
        return_tensors="pt",
        max_length=model_config.traced_sequence_length,
        padding="max_length",
        truncation=True,
    )
    # convert for neuron model
    sentences_inputs = encoded_input['input_ids'], encoded_input['attention_mask'], encoded_input['token_type_ids']

    with torch.no_grad():
        model_output = model(*sentences_inputs)[0]

    # Perform pooling & return numpy
    return compute_embeddings(model_output, encoded_input)
