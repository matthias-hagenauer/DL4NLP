import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="google/gemma-3-12b-it",
    token=os.environ["HUGGINGFACE_HUB_TOKEN"],
)

prompt = "Write a haiku about HPC clusters."
out = client.text_generation(
    prompt,
    max_new_tokens=128,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.05,
)
print(out)
