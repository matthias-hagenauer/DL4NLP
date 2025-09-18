import evaluate
from transformers import pipeline
import re

import argparse
import json
from typing import List, Tuple, Dict

try:
    from comet import download_model, load_from_checkpoint
except Exception as e:
    raise RuntimeError(
        "Could not import COMET. Install it with: pip install 'unbabel-comet>=2.2.6'. "
        "Also ensure 'comet-ml' is not shadowing 'comet'."
    ) from e


MODEL_ID = "Unbabel/TowerInstruct-Mistral-7B-v0.2"

LANG_NAME = {
    "en": "English",
    "pt": "Portuguese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
    "cs": "Czech",
    "pl": "Polish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
}

def load_jsonl_pairs(path: str) -> List[Dict]:
    """Load JSON lines with fields: langs (e.g. en-cs), src, tgt (reference)."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Basic sanity checks
            if "langs" not in rec or "src" not in rec or "tgt" not in rec:
                continue
            # Parse langs like "en-cs" → ("en","cs")
            try:
                src_code, tgt_code = rec["langs"].lower().split("-")
            except Exception:
                # Fallback: if split fails, skip this record
                continue
            pairs.append(
                {
                    "src_lang": src_code,
                    "tgt_lang": tgt_code,
                    "src": rec["src"],
                    "tgt": rec["tgt"],
                    # keep optional metadata if present
                    "meta": {k: v for k, v in rec.items() if k not in {"langs", "src", "tgt"}},
                }
            )
    if not pairs:
        raise ValueError(f"No valid records found in {path}")
    return pairs


def lang_name(code: str) -> str:
    """Map a language code to a human-readable name for prompts."""
    return LANG_NAME.get(code.lower(), code)


def format_prompt(tokenizer, src_text: str, src_lang: str, tgt_lang: str) -> str:
    """Create a chat-style prompt telling the model exactly which way to translate."""
    messages = [
        {
            "role": "user",
            "content": (
                f"Translate the following text from {lang_name(src_lang)} into {lang_name(tgt_lang)}.\n"
                f"{lang_name(src_lang)}: {src_text}\n"
                f"{lang_name(tgt_lang)}:"
            ),
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def translate_batch(items, model_id=MODEL_ID, max_new_tokens=256):
    pipe = pipeline("text-generation", model=model_id, device_map="auto")
    tokenizer = pipe.tokenizer

    prompts = [format_prompt(tokenizer, it["src"], it["src_lang"], it["tgt_lang"]) for it in items]
    # Ask pipeline not to return the prompt to simplify extraction (some backends ignore this; we still guard below)
    outputs = pipe(prompts, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)

    preds = []
    for prompt, out, it in zip(prompts, outputs, items):
        gen = out[0]["generated_text"]

        # If the backend ignored return_full_text=False, remove the prompt prefix.
        if gen.startswith(prompt):
            cont = gen[len(prompt):]
        else:
            cont = gen

        # If the model echoed the "TargetLang:" label, strip it once.
        target_label = f"{lang_name(it['tgt_lang'])}:"
        if target_label in cont:
            cont = cont.split(target_label, 1)[-1]

        # Trim common stop tokens and artifacts
        cont = re.split(r"(?:</s>|<\|endoftext\|>|Assistant:)", cont)[0]

        # Final clean-up—keep the whole continuation (no newline truncation!)
        pred = cont.strip().strip('"').strip("“”").strip()
        preds.append(pred)
    return preds


def chrF_score(preds: List[str], refs: List[str]) -> Dict:
    chrf = evaluate.load("chrf")
    return chrf.compute(predictions=preds, references=[[r] for r in refs])


def comet22_score(
    srcs: List[str],
    mts: List[str],
    refs: List[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    gpus: int = 1,
    batch_size: int = 32,
) -> Dict:
    if not (len(srcs) == len(mts) == len(refs)):
        raise ValueError("Lengths of sources, predictions, and references must match.")
    ckpt_path = download_model(model_name)  # downloads if not cached
    model = load_from_checkpoint(ckpt_path)
    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, refs)]
    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    return {
        "system_score": float(out.system_score),
        "segment_scores": [float(x) for x in out.scores],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to JSONL with fields: langs, src, tgt")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--comet_gpus", type=int, default=1)
    parser.add_argument("--comet_batch", type=int, default=8)
    args = parser.parse_args()

    items = load_jsonl_pairs(args.data)
    preds = translate_batch(items, max_new_tokens=args.max_new_tokens)

    print("Predictions:")
    for it, pred in zip(items, preds):
        print(
            f"- [{it['src_lang']}→{it['tgt_lang']}] "
            f"src: {it['src'][:120]}{'...' if len(it['src'])>120 else ''}\n"
            f"  pred: {pred}\n"
        )

    refs = [it["tgt"] for it in items]
    srcs = [it["src"] for it in items]

    chrf = chrF_score(preds, refs)
    print("chrF:", chrf)

    comet22 = comet22_score(srcs, preds, refs, gpus=args.comet_gpus, batch_size=args.comet_batch)
    print("COMET-22:", {"system": comet22["system_score"], "segments": comet22["segment_scores"]})


if __name__ == "__main__":
    main()