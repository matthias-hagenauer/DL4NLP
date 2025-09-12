import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# this needs rework.. turns our llama attention is not used the way we overwrite it here. 
# I will have a look another day.

###############################################
# Quantization helpers
###############################################
def quantize(x, bits=8):
    qmin, qmax = -2**(bits-1), 2**(bits-1)-1
    scale = x.abs().max() / qmax
    x_q = (x / scale).round().clamp(qmin, qmax).to(torch.int8)
    return x_q, scale

def dequantize(x_q, scale):
    return (x_q.float() * scale)

class QuantizedAttention(nn.Module):
    def __init__(self, attn_module):
        super().__init__()
        self.attn = attn_module  # original attention

    def forward(self, hidden_states, past_key_value=None, **kwargs):
        # standard projections
        k = self.attn.k_proj(hidden_states)
        v = self.attn.v_proj(hidden_states)

        # quantize before storing
        k_q, k_scale = quantize(k, bits=8)
        v_q, v_scale = quantize(v, bits=8)

        # store quantized
        new_past_key_value = ((k_q, k_scale), (v_q, v_scale))

        # if past exists, dequantize before using
        if past_key_value is not None:
            (k_q_prev, k_scale_prev), (v_q_prev, v_scale_prev) = past_key_value
            k_prev = dequantize(k_q_prev, k_scale_prev)
            v_prev = dequantize(v_q_prev, v_scale_prev)
            k = torch.cat([k_prev, k], dim=1)
            v = torch.cat([v_prev, v], dim=1)

        # continue with standard attention using dequantized k,v
        return self.attn.attention(hidden_states, k, v), new_past_key_value

# How to update attention block: model.model.layers[i].self_attn = QuantizedAttention(model.model.layers[i].self_attn)
