import re
from transformers import pipeline
import os

# Baseline HF models
DEFAULT_MODEL_ID = "Unbabel/TowerInstruct-Mistral-7B-v0.2"
GEMMA3_MODEL_ID  = "google/gemma-3-12b-it"

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
        "models/gguf/2bit/gemma-3-12b-it-Q2_K.gguf",
        "tensorblock/gemma-3-12b-it-GGUF",
        "gemma-3-12b-it-Q2_K.gguf",
    ),
    "G3_4bit": (
        "models/gguf/4bit/gemma-3-12b-it-Q4_K_M.gguf",
        "tensorblock/gemma-3-12b-it-GGUF",
        "gemma-3-12b-it-Q4_K_M.gguf",
    ),
    "G3_8bit": (
        "models/gguf/8bit/gemma-3-12b-it-Q8_0.gguf",
        "tensorblock/gemma-3-12b-it-GGUF",
        "gemma-3-12b-it-Q8_0.gguf",
    ),
}

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
        """Generate translations for a batch of items."""
        max_new_tokens = gen_kwargs.get("max_new_tokens", 256)
        do_sample      = gen_kwargs.get("do_sample", False)
        temperature    = (gen_kwargs.get("temperature", 0.7) if do_sample else 0.0)

        preds = []
        for it in items:
            content = _build_prompt(None, it.get("src",""), it.get("src_lang","en"), it.get("tgt_lang","en"))

            if self.has_chat:
                # Use the model's chat template (recommended when available)
                out = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                )
                text = out["choices"][0]["message"]["content"]
            else:
                # Fallback: completion mode with stronger stops
                out = self.llm.create_completion(
                    prompt=content,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    stop=[
                        "<|im_end|>",
                        "\nEnglish:", "English:",
                        "\nGerman:",  "German:",
                        "\n\n", "</s>", "<|endoftext|>", "Assistant:",
                    ],
                )
                text = out["choices"][0]["text"]

            # Clean up output
            target_label = _lang_name(it.get("tgt_lang",""))
            label_str = f"{target_label}:"
            if label_str in text:
                text = text.split(label_str, 1)[-1]
            text = re.split(r"(?:</s>|<\|endoftext\|>|Assistant:)", text)[0]
            preds.append(text.strip().strip('"').strip("“”").strip())

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

def build_model(model_id=None, device_map="auto",
                gguf_repo=None, gguf_file=None, gguf_path=None,
                n_ctx=4096, n_gpu_layers=0):
    """
    Single entry point:
      - model_id == "TM"                 -> HF baseline (Unbabel/TowerInstruct-Mistral-7B-v0.2)
      - model_id == "G3"                 -> HF baseline (google/gemma-3-12b-it)
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
