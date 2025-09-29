import sentinel_metric
from sentinel_metric import download_model, load_from_checkpoint
 
# Load model directly
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("translation", model="Prosho/sentinel-src-24")

data = [
    {"src": "Please sign the form."},
    {"src": "He spilled the beans, then backpedaled—talk about mixed signals!"}
]

output = pipe(data)
