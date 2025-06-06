import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn import functional as F
import numpy as np

activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()

    return hook


def load_model_and_tokenizer(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def register_hooks(model):
    for i, block in enumerate(model.transformer.h):
        block.register_forward_hook(get_activation(f'layer_{i}'))


def prepare_prompt(system_prompt, user_prompt):
    return f"{system_prompt}\n{user_prompt}"


def encode_prompt(tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs


def run_inference(model, tokenizer, prompt):
    inputs = encode_prompt(tokenizer, prompt)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs['input_ids'])
    return outputs