# Install vllm
# pip install vllm

from vllm import LLM
from transformers import AutoTokenizer

# Load the Hugging Face tokenizer
tokenizer = AutoTokenizer.from_pretrained("Unbabel/TowerInstruct-Mistral-7B-v0.2")

# Initialize the vLLM model
llm = LLM("Unbabel/TowerInstruct-Mistral-7B-v0.2", tensor_parallel_size=1, dtype="bfloat16")

# Prepare your messages
messages = [
    {"role": "user", "content": "Translate the following text from Portuguese into English.\nPortuguese: Um grupo de investigadores lançou um novo modelo para tarefas relacionadas com tradução.\nEnglish:"},
]

# Apply the chat template (same as transformers)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generate output with vLLM
responses = llm.generate([prompt])

# vLLM returns an iterator of response objects
for r in responses:
    print(r)
