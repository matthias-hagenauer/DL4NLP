import sentinel_metric
from sentinel_metric import download_model, load_from_checkpoint
 
# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("Prosho/sentinel-src-24", torch_dtype="auto")

data = [
    {"src": "Please sign the form."},
    {"src": "He spilled the beans, then backpedaled—talk about mixed signals!"}
]

output = model.predict(data, batch_size=8, gpus=1)
