from sentinel_metric import download_model, load_from_checkpoint

model_path = download_model("Prosho/sentinel-src-24")
model = load_from_checkpoint(model_path)

data = [
    {"src": "Please sign the form."},
    {"src": "He spilled the beans, then backpedaled—talk about mixed signals!"}
]

output = model.predict(data, batch_size=8, gpus=1)
