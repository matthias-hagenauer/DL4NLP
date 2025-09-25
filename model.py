import re

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

    # Use chat template if available (HF chat models)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": content}]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fall back to plain content if anything goes wrong
            pass
    return content


class HFPipelineModel:
    """
    Quantization options:
      - quant='none' → standard weights (device_map respected)
      - quant='8bit' → load_in_8bit=True (requires bitsandbytes)
      - quant='4bit' → load_in_4bit=True with NF4 (requires bitsandbytes)
    """

    def __init__(self, model_id=None, quant="none", device_map="auto"):
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.quant = (quant or "none").lower()
        self.device_map = device_map

        try:
            from transformers import (
                pipeline,
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except Exception as e:
            raise RuntimeError(
                "Please install 'transformers' (and 'bitsandbytes' for 8/4-bit)."
            ) from e

        # Build pipeline according to quantization mode
        if self.quant == "none":
            self.pipe = pipeline(
                "text-generation",
                model=self.model_id,
                device_map=self.device_map,
            )
            self.tokenizer = self.pipe.tokenizer

        elif self.quant in ("8bit", "4bit"):
            # Explicit tokenizer + model load
            tok = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            load_kwargs = {"device_map": self.device_map}

            if self.quant == "8bit":
                load_kwargs["load_in_8bit"] = True
            else:
                # 4-bit with NF4
                try:
                    qconf = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=None,  # let HF choose
                    )
                except Exception as e:
                    raise RuntimeError(
                        "bitsandbytes not available or misconfigured for 4-bit."
                    ) from e
                load_kwargs["load_in_4bit"] = True
                load_kwargs["quantization_config"] = qconf

            model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)

            # Build the pipeline using the loaded model/tokenizer
            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tok,
                device_map=self.device_map,
            )
            self.tokenizer = tok

        else:
            raise ValueError("Unknown quant option. Use 'none', '8bit', or '4bit'.")

    def translate_batch(self, items, **gen_kwargs):
        """
        items: list of dicts with keys: 'src', 'src_lang', 'tgt_lang'
        returns: list of translated strings
        """
        # Safe defaults; deterministic unless you set do_sample=True explicitly
        params = {
            "max_new_tokens": 256,
            "do_sample": False,
            "return_full_text": False,
        }
        params.update(gen_kwargs or {})

        # Build prompts
        prompts = []
        for it in items:
            src_text = it.get("src", "")
            src_lang = it.get("src_lang", "en")
            tgt_lang = it.get("tgt_lang", "en")
            prompts.append(_build_prompt(self.tokenizer, src_text, src_lang, tgt_lang))

        # Generate
        outputs = self.pipe(prompts, **params)  # list of lists of dicts

        # Post-process generations
        preds = []
        for prompt, out, it in zip(prompts, outputs, items):
            # Robustly get text
            text = ""
            if out and isinstance(out, list) and isinstance(out[0], dict):
                text = out[0].get("generated_text", "") or ""

            # If backend ignored return_full_text=False, remove prompt prefix
            if text.startswith(prompt):
                cont = text[len(prompt):]
            else:
                cont = text

            # Remove a single echoed "TargetLang:" label if present
            target_label = _lang_name(it.get("tgt_lang", ""))
            label_str = f"{target_label}:"
            if label_str in cont:
                cont = cont.split(label_str, 1)[-1]

            # Trim common end tokens/artifacts
            cont = re.split(r"(?:</s>|<\|endoftext\|>|Assistant:)", cont)[0]

            # Basic stripping of quotes/whitespace
            cont = cont.strip().strip('"').strip("“”").strip()
            preds.append(cont)

        return preds


def build_model(model_id=None, quant="none", device_map="auto"):
    return HFPipelineModel(model_id=model_id, quant=quant, device_map=device_map)
