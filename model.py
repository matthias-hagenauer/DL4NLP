import re
from transformers import pipeline
import os

# Baseline HF models
DEFAULT_MODEL_ID = "Unbabel/TowerInstruct-Mistral-7B-v0.2"
GEMMA3_MODEL_ID  = "google/gemma-3-4b-it"

# Local + HF GGUF presets
GGUF_DEFAULTS = {
    # --- TowerInstruct-Mistral (TM) ---
    "TM_2bit": (
        "models/gguf/2bit/TowerInstruct-Mistral-7B-v0.2-Q2_K.gguf",
        "tensorblock/TowerInstruct-Mistral-7B-v0.2-GGUF",
        "TowerInstruct-Mistral-7B-v0.2-Q2_K.gguf",
    ),
    "TM_4bit": (
        "models/gguf/4bit/TowerInstruct-Mistral-7B-v0.2-Q4_K_M.gguf",
        "tensorblock/TowerInstruct-Mistral-7B-v0.2-GGUF",
        "TowerInstruct-Mistral-7B-v0.2-Q4_K_M.gguf",
    ),
    "TM_8bit": (
        "models/gguf/8bit/TowerInstruct-Mistral-7B-v0.2-Q8_0.gguf",
        "tensorblock/TowerInstruct-Mistral-7B-v0.2-GGUF",
        "TowerInstruct-Mistral-7B-v0.2-Q8_0.gguf",
    ),

    # --- Gemma 3 12B Instruct (G3) ---
    "G3_2bit": (
        "models/gguf/2bit/gemma-3-4b-it-Q2_K.gguf",
        "tensorblock/gemma-3-4b-it-GGUF",
        "gemma-3-4b-it-Q2_K.gguf",
    ),
    "G3_4bit": (
        "models/gguf/4bit/gemma-3-4b-it-Q4_K_M.gguf",
        "tensorblock/gemma-3-4b-it-GGUF",
        "gemma-3-4b-it-Q4_K_M.gguf",
    ),
    "G3_8bit": (
        "models/gguf/8bit/gemma-3-4b-it-Q8_0.gguf",
        "tensorblock/gemma-3-4b-it-GGUF",
        "gemma-3-4b-it-Q8_0.gguf",
    ),
}

# System prompt for strict translation-only behavior in Gemini
STRICT_TRANSLATION_SYSTEM = (
    "You are a translation engine. Translate the user-provided text from the "
    "source language into the target language. Return ONLY the translation in "
    "the target language — no explanations, no alternatives, no lists, no markdown."
)

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
    """
    Build either a chat-formatted prompt (preferred) or a plain string fallback.
    Includes a strict system message to avoid enumerations/options.
    """
    src_name = _lang_name(src_lang)
    tgt_name = _lang_name(tgt_lang)

    user_content = (
        f"Translate the following text from {src_name} into {tgt_name}.\n\n"
        f"Source ({src_name}): {src_text}\n"
        f"Target ({tgt_name}):"
    )

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": STRICT_TRANSLATION_SYSTEM},
            {"role": "user",   "content": user_content},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            pass  # fall back to plain string

    return f"{STRICT_TRANSLATION_SYSTEM}\n\n{user_content}"


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


class GGUFModel:
    """
    Loads a local .gguf file or pulls from Hugging Face via llama.cpp.
    Works on CPU; can use GPU if llama-cpp-python was installed with CUDA.
    """
    def __init__(self, repo_id=None, filename=None, model_path=None,
                 n_ctx=4096, n_gpu_layers=0, **kwargs):
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise RuntimeError("Please `pip install llama-cpp-python` first.") from e

        if model_path:
            self.llm = Llama(model_path=model_path, n_ctx=n_ctx,
                             n_gpu_layers=n_gpu_layers, **kwargs)
        else:
            # pulls weights from HF the first time and caches them
            self.llm = Llama.from_pretrained(
                repo_id=repo_id, filename=filename,
                n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, **kwargs
            )

        # No tokenizer object here; we pass None to _build_prompt
        self.tokenizer = None

        # Detect whether the model has a chat template
        self.has_chat = hasattr(self.llm, "chat_format") and (self.llm.chat_format is not None)


    def translate_batch(self, items, **gen_kwargs):
        max_new_tokens = gen_kwargs.get("max_new_tokens", 256)
        do_sample      = gen_kwargs.get("do_sample", False)
        temperature    = (gen_kwargs.get("temperature", 0.7) if do_sample else 0.0)

        preds = []
        for it in items:
            try:
                src = it.get("src", "")
                src_name = _lang_name(it.get("src_lang", "en"))
                tgt_name = _lang_name(it.get("tgt_lang", "en"))
                user_content = (
                    f"Translate the following text from {src_name} into {tgt_name}.\n\n"
                    f"Source ({src_name}): {src}\n"
                    f"Target ({tgt_name}):"
                )

                if self.has_chat:
                    out = self.llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": STRICT_TRANSLATION_SYSTEM},
                            {"role": "user",   "content": user_content},
                        ],
                        temperature=0.0,
                        max_tokens=max_new_tokens,
                    )
                    text = (out or {}).get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    prompt = f"{STRICT_TRANSLATION_SYSTEM}\n\n{user_content}"
                    out = self.llm.create_completion(
                        prompt=prompt,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        stop=[
                            "<|im_end|>",
                            "\nEnglish:", "English:",
                            "\nGerman:",  "German:",
                            "\nUser:", "User:",
                            "\nAssistant:", "Assistant:",
                            "\n\n</s>", "</s>", "<|endoftext|>",
                            "Option 1", "Option 2", "**Option", "Here are a few options",
                        ],
                    )
                    text = (out or {}).get("choices", [{}])[0].get("text", "")

                # Post-process
                target_label = f"{tgt_name}:"
                if target_label in text:
                    text = text.split(target_label, 1)[-1]
                text = re.sub(r"^Here (?:is|are)\b.*?:\s*", "", text, flags=re.IGNORECASE)
                text = re.sub(r"^\*\*Option.*?\*\*:\s*", "", text, flags=re.IGNORECASE)
                text = re.split(r"(?:</s>|<\|endoftext\|>|Assistant:)", text)[0]
                pred = text.strip().strip('"').strip("“”").strip()
                preds.append(pred)
            except Exception:
                # Ensure output length matches inputs even if a call fails
                preds.append("")
        return preds


class HFPipelineModel:
    """Simple HF pipeline wrapper (no quantization)."""

    def __init__(self, model_id=None, device_map="auto"):
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.device_map = device_map

        token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")

        pipe_kwargs = dict(
            task="text-generation",
            model=self.model_id, 
            device_map=self.device_map,
        )

        # Only pass token if we actually have one.
        if token:
            pipe_kwargs["token"] = token

        self.pipe = pipeline(**pipe_kwargs)
        self.tokenizer = self.pipe.tokenizer
        self._eos_ids = _collect_eos_ids(self.tokenizer)
        # Use eos as pad to silence warnings/ensure batching works
        self._pad_id = getattr(self.tokenizer, "eos_token_id", None)

    def translate_batch(self, items, **gen_kwargs):
        params = {
            "max_new_tokens": 128,
            "do_sample": False,
            "num_return_sequences": 1,
            "return_full_text": False,
            "temperature": 0.0,
        }
        if self._eos_ids:
            params["eos_token_id"] = self._eos_ids if len(self._eos_ids) > 1 else self._eos_ids[0]
        if self._pad_id is not None:
            params["pad_token_id"] = self._pad_id
        params.update(gen_kwargs or {})

        # Prompts
        prompts = []
        for it in items:
            prompts.append(_build_prompt(self.tokenizer, it.get("src",""), it.get("src_lang","en"), it.get("tgt_lang","en")))

        # Generate (ALWAYS return a list the same length as items)
        try:
            outputs = self.pipe(prompts, **params)  # usually list[list[dict]]
        except Exception:
            return [""] * len(items)

        preds = []
        for prompt, out, it in zip(prompts, outputs if isinstance(outputs, list) else [outputs], items):
            try:
                text = ""
                if isinstance(out, list) and out and isinstance(out[0], dict):
                    text = out[0].get("generated_text", "") or ""
                elif isinstance(out, dict):
                    text = out.get("generated_text", "") or ""

                cont = text[len(prompt):] if text.startswith(prompt) else text

                tgt_label = f"{_lang_name(it.get('tgt_lang',''))}:"
                if tgt_label in cont:
                    cont = cont.split(tgt_label, 1)[-1]

                cont = re.sub(r"^Here (?:is|are)\b.*?:\s*", "", cont, flags=re.IGNORECASE)
                cont = re.sub(r"^\*\*Option.*?\*\*:\s*", "", cont, flags=re.IGNORECASE)

                cont = re.split(
                    r"(?:<\|im_end\|>|</s>|<\|endoftext\|>|\n<\|im_start\|>user|\n\s*User:|\n\s*Assistant:)",
                    cont,
                    maxsplit=1,
                )[0]

                paragraphs = [p.strip() for p in cont.strip().split("\n\n") if p.strip()]
                pred = (paragraphs[0] if paragraphs else "").strip().strip('"').strip("“”").strip()
                preds.append(pred)
            except Exception:
                preds.append("")
        return preds
        

def build_model(model_id=None, device_map="auto",
                gguf_repo=None, gguf_file=None, gguf_path=None,
                n_ctx=4096, n_gpu_layers=0):
    """
    Single entry point:
      - model_id == "TM"                 -> HF baseline (Unbabel/TowerInstruct-Mistral-7B-v0.2)
      - model_id == "G3"                 -> HF baseline (google/gemma-3-4b-it)
      - model_id in {"TM_2bit","TM_4bit","TM_8bit","G3_2bit","G3_4bit","G3_8bit"}
                                         -> GGUF quant under models/gguf/<bit>/..., fallback to HF repo if missing
      - otherwise                        -> treat as Hugging Face model id
    Optional overrides: gguf_path, gguf_repo, gguf_file
    """
    mid = model_id or "TM"

    # Friendly HF baselines
    if mid == "TM":
        return HFPipelineModel(model_id=DEFAULT_MODEL_ID, device_map=device_map)
    if mid == "G3":
        return HFPipelineModel(model_id=GEMMA3_MODEL_ID, device_map=device_map)

    # Friendly GGUF variants
    if mid in GGUF_DEFAULTS:
        local_path_default, repo_default, file_default = GGUF_DEFAULTS[mid]
        # allow explicit overrides
        use_path = gguf_path or (local_path_default if os.path.exists(local_path_default) else None)
        repo     = gguf_repo or repo_default
        fname    = gguf_file or file_default

        if use_path:
            return GGUFModel(model_path=use_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)
        else:
            return GGUFModel(repo_id=repo, filename=fname, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers)

    # Any other id -> assume HF hub id
    return HFPipelineModel(model_id=mid, device_map=device_map)