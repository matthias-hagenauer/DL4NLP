from transformers import AutoTokenizer, AutoModelForSequenceClassification

"""
I tried downloading the model checkpoints locally as doing it via huggingface did not work. 
However, that did not work either grmpf
"""

# Path to the local model folder
local_model_path = r"C:\Users\15813274\OneDrive - UvA\Documents\tmp\sentinel-src-da-for-wmt25"


# Load tokenizer and model from local folder
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForSequenceClassification.from_pretrained(local_model_path)


# tried loading it like this but did not work
#tokenizer = AutoTokenizer.from_pretrained("zouharvi/Sentinel-src-25")
#model = AutoModelForSequenceClassification.from_pretrained("zouharvi/Sentinel-src-25")


text = "This is a test sentence for difficulty estimation."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)  # Higher values = more difficult to translate
