import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.nn import functional as F
import numpy as np

def get_activation(activations_dict, name):
    def hook(module, input, output):
        # Unpack if output is a tuple (common in GPT-2 blocks)
        hidden_states = output[0] if isinstance(output, tuple) else output
        activations_dict[name] = hidden_states.detach()
    return hook



def load_model_and_tokenizer(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return model, tokenizer


def register_hooks(model, activations_dict):
    for i, block in enumerate(model.transformer.h):
        # Remove existing hooks first (if any)
        block._forward_hooks.clear()
        block.register_forward_hook(get_activation(activations_dict, f'layer_{i}'))
