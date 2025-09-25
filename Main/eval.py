from __future__ import annotations
from typing import Any, Dict, List


def chrf_segment_scores(preds: List[str], refs: List[str]) -> List[float]:
    import evaluate  
    metric = evaluate.load("chrf")
    if len(preds) != len(refs):
        raise ValueError("preds and refs must have the same length")
    return [
        metric.compute(predictions=[p], references=[[r]])["score"]
        for p, r in zip(preds, refs)
    ]


def bleu_corpus(preds: List[str], refs: List[str]) -> Dict[str, Any]:
    import evaluate  
    if len(preds) != len(refs):
        raise ValueError("preds and refs must have the same length")
    bleu = evaluate.load("bleu")
    return bleu.compute(predictions=preds, references=[[r] for r in refs])


def sentence_bleu_scores(preds: List[str], refs: List[str]) -> List[float]:
    if len(preds) != len(refs):
        raise ValueError("preds and refs must have the same length")
    try:
        from sacrebleu.metrics import BLEU
    except Exception as e:
        raise RuntimeError(
            "Install sacrebleu for sentence BLEU: pip install sacrebleu"
        ) from e
    bleu = BLEU(effective_order=True)
    return [float(bleu.sentence_score(hyp, [ref]).score) for hyp, ref in zip(preds, refs)]


def comet22_scores(srcs: List[str], mts: List[str], refs: List[str], *, gpus: int = 1, batch_size: int = 8) -> Dict[str, Any]:
    if not (len(srcs) == len(mts) == len(refs)):
        raise ValueError("srcs, mts, refs must have the same length")

    try:
        from comet import download_model, load_from_checkpoint
    except Exception as e:
        raise RuntimeError(
            "Install COMET: pip install 'unbabel-comet>=2.2.6' "
            "and make sure 'comet-ml' isn't shadowing 'comet'."
        ) from e

    ckpt_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(ckpt_path)

    data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(srcs, mts, refs)]
    out = model.predict(data, batch_size=batch_size, gpus=gpus)

    return {
        "system_score": float(out.system_score),
        "segment_scores": [float(x) for x in out.scores],
    }
