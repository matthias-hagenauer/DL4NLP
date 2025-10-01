def chrf_segment_scores(preds, refs):
    """
    Returns a list of chrF scores, one per (pred, ref) pair.
    """
    if len(preds) != len(refs):
        raise ValueError("preds and refs must have the same length")
    try:
        import evaluate  
    except Exception as e:
        raise RuntimeError("Please install the 'evaluate' package: pip install evaluate") from e

    metric = evaluate.load("chrf")
    scores = []
    for p, r in zip(preds, refs):
        result = metric.compute(predictions=[p], references=[[r]])
        scores.append(float(result.get("score", 0.0)))

    return scores


def bleu_corpus(preds, refs):
    """
    Returns a dict with corpus BLEU (and extra fields from evaluate).
    """
    if len(preds) != len(refs):
        raise ValueError("preds and refs must have the same length")
    try:
        import evaluate  
    except Exception as e:
        raise RuntimeError("Please install the 'evaluate' package: pip install evaluate") from e

    bleu = evaluate.load("bleu")

    return bleu.compute(predictions=preds, references=[[r] for r in refs])


def comet22_scores(srcs, mts, refs, gpus=1, batch_size=8):
    """
    Returns:
      {
        "system_score": float,
        "segment_scores": [float, ...]
      }
    Uses Unbabel/wmt22-comet-da.
    """
    if not (len(srcs) == len(mts) == len(refs)):
        raise ValueError("srcs, mts, refs must have the same length")

    try:
        from comet import download_model, load_from_checkpoint  
    except Exception as e:
        raise RuntimeError(
            "Install COMET: pip install 'unbabel-comet>=2.2.6' "
            "and ensure the 'comet' package isn't shadowed by 'comet-ml'."
        ) from e

    ckpt_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(ckpt_path)

    data = []
    for s, m, r in zip(srcs, mts, refs):
        data.append({"src": s, "mt": m, "ref": r})

    out = model.predict(data, batch_size=batch_size, gpus=gpus)
    seg = [float(x) for x in out.scores] if getattr(out, "scores", None) is not None else []

    return {
        "system_score": float(getattr(out, "system_score", 0.0)),
        "segment_scores": seg,
    }
