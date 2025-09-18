# Install dependencies first:
# pip install transformers accelerate bitsandbytes evaluate sacrebleu

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import evaluate
import sacrebleu
from sacrebleu.metrics import CHRF as SacreCHRF

# -------------------
# 1. Load model + tokenizer (quantized 4-bit with bitsandbytes)
# -------------------
model_id = "Unbabel/TowerInstruct-Mistral-7B-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# This is quantizing the model to 4 bits
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # loads in in 4 bits 
    bnb_4bit_compute_dtype="float16", # however the calculations will be done using float 16
    bnb_4bit_use_double_quant=True, # to quantizies the model we need to save meta data, this meta data can be quantized as well, that is why it says double quant
    bnb_4bit_quant_type="nf4", # chatGPT says: This specifies the type of 4-bit quantization to use. nf4 (Normal Float 4) is a data type specifically designed for weights that are typically initialized from a normal distribution. It's an efficient quantization scheme that provides better performance and accuracy compared to standard 4-bit integer quantization for these types of models.
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config
)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# -------------------
# 2. Example dataset
# -------------------
dataset = [
    {"input": "Translate to French: Hello, how are you?", 
     "reference": "Bonjour, comment ça va ?"},
    {"input": "Translate to French: I love machine learning.", 
     "reference": "J'adore l'apprentissage automatique."},
]

# -------------------
# 3. Generate predictions
# -------------------
predictions = []
references = []

for example in dataset:
    input_text = example["input"]
    reference_text = example["reference"]

    # Generate output
    output = generator(
        input_text,
        max_new_tokens=64,
        do_sample=False
    )[0]["generated_text"]

    # Strip input prompt from output (model usually echoes the input)
    pred = output.replace(input_text, "").strip()

    predictions.append(pred)
    references.append(reference_text)

# -------------------
# 4. Compute BLEU (evaluate + sacrebleu)
# -------------------
bleu = evaluate.load("bleu")
results = bleu.compute(predictions=predictions, references=references)
print("BLEU score (evaluate):", results["bleu"])

bleu_score = sacrebleu.corpus_bleu(predictions, [references])
print("BLEU score (sacrebleu):", bleu_score.score)

# -------------------
# 5. Compute chrF (evaluate + sacrebleu)
# -------------------
# evaluate.chrf expects references as List[List[str]]
chrf_metric = evaluate.load("chrf")
chrf_results = chrf_metric.compute(predictions=predictions, references=[[r] for r in references])
print("chrF (evaluate):", chrf_results["score"])

# sacrebleu CHRF (same inputs as corpus_bleu)
sacre_chrf = SacreCHRF(word_order=0)  # default chrF++
chrf_score = sacre_chrf.corpus_score(predictions, [references])
print("chrF (sacrebleu):", chrf_score.score)
