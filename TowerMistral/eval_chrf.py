import evaluate
from transformers import pipeline

try:
    from comet import download_model, load_from_checkpoint
except Exception as e:
    raise RuntimeError(
        "Could not import COMET. Install it with: pip install 'unbabel-comet>=2.2.6'. "
        "Also ensure 'comet-ml' is not shadowing 'comet'."
    ) from e


MODEL_ID = "Unbabel/TowerInstruct-Mistral-7B-v0.2"

# from huggingface.co/Unbabel/TowerInstruct-Mistral-7B-v0.2
SOURCES = [
    "Um grupo de investigadores lançou um novo modelo para tarefas relacionadas com tradução.",
]
REFERENCES = [
    "A group of researchers has launched a new model for translation-related tasks.",
]

def format_prompt(tokenizer, pt_text: str) -> str:
    messages = [{
        "role": "user",
        "content": f"Translate the following text from Portuguese into English.\nPortuguese: {pt_text}\nEnglish:",
    }]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def translate(texts, model_id=MODEL_ID, max_new_tokens=128):
    pipe = pipeline("text-generation", model=model_id, device_map="auto")
    tokenizer = pipe.tokenizer

    prompts = [format_prompt(tokenizer, t) for t in texts]
    outputs = pipe(prompts, max_new_tokens=max_new_tokens, do_sample=False)

    preds = []
    for out in outputs:
        full = out[0]["generated_text"]
        if "English:" in full:
            full = full.split("English:", 1)[-1]
        preds.append(full.strip().split("\n")[0].strip())
    return preds

def chrF_score(preds, refs):
    chrf = evaluate.load("chrf")
    return chrf.compute(predictions=preds, references=[[r] for r in refs])

def comet22_score(srcs, mts, refs, model_name="Unbabel/wmt22-comet-da", gpus=1, batch_size=32):
    if not (len(srcs) == len(mts) == len(refs)):
        raise ValueError("Lengths of sources, predictions, and references must match.")
    ckpt_path = download_model(model_name)   # downloads the model if not cached
    model = load_from_checkpoint(ckpt_path)
    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, refs)]
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return {
        "system_score": float(out.system_score),
        "segment_scores": [float(x) for x in out.scores],
    }
def main():
    preds = translate(SOURCES)
    print("Predictions:")
    for s, p in zip(SOURCES, preds):
        print(f"- PT: {s}\n  EN_pred: {p}\n")

    chrf = chrF_score(preds, REFERENCES)
    print("chrF:", chrf)

    comet22 = comet22_score(SOURCES, preds, REFERENCES, gpus=1, batch_size=8)
    print("COMET-22:", {"system": comet22["system_score"], "segments": comet22["segment_scores"]})


if __name__ == "__main__":
    main()