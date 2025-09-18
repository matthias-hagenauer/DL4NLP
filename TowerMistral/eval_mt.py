import evaluate
from transformers import pipeline

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


def translate_batch(
    items: List[Dict],
    model_id: str = MODEL_ID,
    max_new_tokens: int = 256,
) -> List[str]:
    """Translate a mixed-language batch using instruction prompts."""
    pipe = pipeline("text-generation", model=model_id, device_map="auto")
    tokenizer = pipe.tokenizer

    prompts = [format_prompt(tokenizer, it["src"], it["src_lang"], it["tgt_lang"]) for it in items]
    outputs = pipe(prompts, max_new_tokens=max_new_tokens, do_sample=False)

    preds = []
    for out in outputs:
        full = out[0]["generated_text"]
        # Heuristic: strip everything up to the final "{TargetLang}:"
        # so we keep only the model’s continuation in the target language.
        # e.g., "... English:" → take the tail.
        # If label not found, just trim.
        tails = []
        for it in LANG_NAME.values():
            marker = f"{it}:"
            if marker in full:
                tails.append(full.split(marker, 1)[-1])
        if tails:
            full = tails[-1]
        preds.append(full.strip().split("\n")[0].strip())
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