import re
from transformers import pipeline

DEFAULT_MODEL_ID = "Unbabel/TowerInstruct-Mistral-7B-v0.2"

LANG_NAME = {
    "en": "English",  "pt": "Portuguese", "es": "Spanish", "fr": "French",
    "de": "German",   "it": "Italian",    "nl": "Dutch",   "cs": "Czech",
    "pl": "Polish",   "ru": "Russian",    "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean",   "ar": "Arabic",
}

def _lang_name(code):
    if not isinstance(code, str):
        return str(code)
    return LANG_NAME.get(code.lower(), code)

def _build_prompt(tokenizer, src_text, src_lang, tgt_lang):
    """Return a single-string prompt. If tokenizer has chat template, use it."""
    content = (
        "Translate the following text from {src} into {tgt}.\n"
        "{src}: {text}\n"
        "{tgt}:"
    ).format(src=_lang_name(src_lang), tgt=_lang_name(tgt_lang), text=src_text)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": content}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass
    return content

def _collect_eos_ids(tokenizer):
    """Include regular EOS and chat end-of-message if present."""
    eos_ids = []
    if getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids.append(tokenizer.eos_token_id)
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is not None and im_end_id != -1:
            eos_ids.append(im_end_id)
    except Exception:
        pass
    eos_ids = list(dict.fromkeys(eos_ids))
    return eos_ids

class HFPipelineModel:
    """Simple HF pipeline wrapper (no quantization)."""

    def __init__(self, model_id=None, device_map="auto"):
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.device_map = device_map

        self.pipe = pipeline(
            "text-generation",
            model=self.model_id,
            device_map=self.device_map,
        )
        self.tokenizer = self.pipe.tokenizer
        self._eos_ids = _collect_eos_ids(self.tokenizer)
        # Use eos as pad to silence warnings/ensure batching works
        self._pad_id = getattr(self.tokenizer, "eos_token_id", None)

    def translate_batch(self, items, **gen_kwargs):
        """
        items: list of dicts with keys: 'src', 'src_lang', 'tgt_lang'
        returns: list of translated strings (one per input)
        """
        # Deterministic, one hypothesis per input
        params = {
            "max_new_tokens": 128,
            "do_sample": False,
            "num_return_sequences": 1,
            "return_full_text": False,
            "temperature": 0.0,  # redundant with do_sample=False, but explicit
        }
        # Stop at assistant turn end
        if self._eos_ids:
            params["eos_token_id"] = self._eos_ids if len(self._eos_ids) > 1 else self._eos_ids[0]
        if self._pad_id is not None:
            params["pad_token_id"] = self._pad_id

        params.update(gen_kwargs or {})

        # Build prompts
        prompts = []
        for it in items:
            src_text = it.get("src", "")
            src_lang = it.get("src_lang", "en")
            tgt_lang = it.get("tgt_lang", "en")
            prompts.append(_build_prompt(self.tokenizer, src_text, src_lang, tgt_lang))

        # Generate
        outputs = self.pipe(prompts, **params)  # list[list[dict]]

        # Post-process → exactly one line per input
        preds = []
        for prompt, out, it in zip(prompts, outputs, items):
            text = ""
            if out and isinstance(out, list) and isinstance(out[0], dict):
                text = out[0].get("generated_text", "") or ""

            cont = text[len(prompt):] if text.startswith(prompt) else text

            # Strip a single echoed "TargetLang:" if present
            tgt_label = f"{_lang_name(it.get('tgt_lang', ''))}:"
            if tgt_label in cont:
                cont = cont.split(tgt_label, 1)[-1]

            # Hard stops: end-of-message markers or fake new turns
            cont = re.split(
                r"(?:<\|im_end\|>|</s>|<\|endoftext\|>|\n<\|im_start\|>user|\n\s*User:|\n\s*Assistant:)",
                cont,
                maxsplit=1,
            )[0]
            # Also cut if the source-language label reappears
            src_label = f"{_lang_name(it.get('src_lang', ''))}:"
            cont = cont.split(src_label, 1)[0]

            # Keep only the first non-empty line
            first_line = next((ln for ln in cont.strip().splitlines() if ln.strip()), "")
            pred = first_line.strip().strip('"').strip("“”").strip()
            preds.append(pred)

        return preds

def build_model(model_id=None, device_map="auto"):
    return HFPipelineModel(model_id=model_id, device_map=device_map)