import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from QuantizedAttention import QuantizedAttention

###############################################
# Main
###############################################
if __name__ == "__main__":
    model_name = "haoranxu/ALMA-7B"

    print(">>> Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(">>> Patching attention layers with QuantizedAttention...")
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = QuantizedAttention(layer.self_attn)

    # Simple test run
    prompt = "The Snellius supercomputer is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    
    print(">>> Output:")
    print(tokenizer.decode(output[0], skip_special_tokens=True))
