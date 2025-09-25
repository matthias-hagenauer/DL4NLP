from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional, Sequence
import re

try:
    from transformers import (
        pipeline,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedTokenizerBase,
        Pipeline,
    )
except Exception as e:
    raise RuntimeError(
        "transformers (and bitsandbytes for 8/4-bit) must be installed."
    ) from e

DEFAULT_MODEL_ID: str = "Unbabel/TowerInstruct-Mistral-7B-v0.2"

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

def _lang_name(code: str) -> str:
    return LANG_NAME.get(code.lower(), code)

def _build_prompt(tokenizer: Optional[PreTrainedTokenizerBase], src_text: str, src_lang: str, tgt_lang: str) -> str:
    content = (
        f"Translate the following text from {_lang_name(src_lang)} into {_lang_name(tgt_lang)}.\n"
        f"{_lang_name(src_lang)}: {src_text}\n"
        f"{_lang_name(tgt_lang)}:"
    )
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(  
            messages, tokenize=False, add_generation_prompt=True
        )
    return content

class HFPipelineModel:
    """Quantization:
      - quant='none' → FP16/BF16 weights (
      - quant='8bit' → load_in_8bit=True
      - quant='4bit' → load_in_4bit=True with NF4 (BitsAndBytesConfig)
    """

    pipe: Pipeline
    tokenizer: PreTrainedTokenizerBase

    def __init__(self, model_id: Optional[str] = None, quant: str = "none", device_map: str = "auto") -> None:
        model_id = model_id or DEFAULT_MODEL_ID
        q = (quant or "none").lower()

        if q == "none":
            self.pipe = pipeline("text-generation", model=model_id, device_map=device_map)
            self.tokenizer = self.pipe.tokenizer

        elif q in ("8bit", "4bit"):
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            load_kwargs: Dict[str, Any] = {"device_map": device_map}
            if q == "8bit":
                load_kwargs["load_in_8bit"] = True
            else:
                load_kwargs["load_in_4bit"] = True
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit = True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=None
                )
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            self.pipe = pipeline("text-generation", model=model, tokenizer=tok, device_map=device_map)  # type: ignore[assignment]
            self.tokenizer = tok
        else:
            raise ValueError("Unknown quant option. Use one of: 'none', '8bit', '4bit'.")
        
    def translate_batch(self, items: Sequence[Dict[str, str]], **gen_kwargs: Any) -> List[str]:
        defaults: Dict[str, Any] = {"max_new_tokens": 256, "do_sample": False, "return_full_text": False}
        params: Dict[str, Any] = {**defaults, **gen_kwargs}

        prompts: List[str] = [
            _build_prompt(self.tokenizer, it["src"], it["src_lang"], it["tgt_lang"]) for it in items
        ]
        outputs: List[List[Dict[str, Any]]] = self.pipe(prompts, **params)  # type: ignore[call-arg]

        preds: List[str] = []
        for prompt, out, it in zip(prompts, outputs, items):
            gen: str = out[0]["generated_text"]

            # If backend ignored return_full_text=False, remove prompt prefix.
            cont: str = gen[len(prompt):] if gen.startswith(prompt) else gen

            # Remove a single echoed "TargetLang:" label if present.
            target_label: str = f"{_lang_name(it['tgt_lang'])}:"
            if target_label in cont:
                cont = cont.split(target_label, 1)[-1]

            # Trim common end tokens/artifacts.
            cont = re.split(r"(?:</s>|<\|endoftext\|>|Assistant:)", cont)[0]

            preds.append(cont.strip().strip('"').strip("“”").strip())

        return preds
    
def build_model(model_id: Optional[str] = None, quant: str = "none", device_map: str = "auto") -> HFPipelineModel:
    return HFPipelineModel(model_id=model_id, quant=quant, device_map=device_map)

