import os
import re
import sys
import json
import math
import random
import torch
from typing import List, Dict, Optional
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

import pronouncing  # CMUdict-based

# Allow running this module either as ``python -m Models.Llama`` or as a
# top-level script; either way we need to import the sibling ``Data``
# package that ships the phoneme noise augmenter.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from Data.phoneme_noise import (  # noqa: E402
    PhonemeNoiseAugmenter,
    build_augmenter_from_config,
)
from Data.phoneme_utils import (  # noqa: E402
    phoneme_line_to_tokens,
    strip_stress,
    text_to_arpabet_words,
    text_to_phoneme_line,
)
from Models.ctc_decode import (  # noqa: E402
    CTCHypothesis,
    format_topk_prompt,
)

# ---------- Phoneme utilities (see Data/phoneme_utils.py for the helpers) ----------
# ``strip_stress``, ``text_to_arpabet_words``, ``text_to_phoneme_line``
# and ``phoneme_line_to_tokens`` are re-exported from Data.phoneme_utils
# via the import above so existing imports of these names from
# ``Models.Llama`` continue to work.


# ---------- Dataset building (phonemes -> text) ----------

TAGS = [
    "<S2S>", "<PHONEMES>", "</PHONEMES>", "<TEXT>",
    "<PHONEME_HYPOTHESES>", "</PHONEME_HYPOTHESES>",
]


def _format_prompt(phon_line: str) -> str:
    """Format a prompt from a (possibly noisy) phoneme line."""
    return (
        f"{TAGS[0]}\n"
        f"{TAGS[1]}\n{phon_line}\n{TAGS[2]}\n"
        f"{TAGS[3]}\n"
    )


def build_example(
    text: str,
    augmenter: Optional[PhonemeNoiseAugmenter] = None,
) -> Optional[Dict[str, str]]:
    """
    Builds one training pair where the INPUT is phonemes and the TARGET is original text.
    We’ll create:
      - prompt: everything up to (and including) the <TEXT> line
      - target: the original text (model should generate this)

    If ``augmenter`` is provided the phoneme sequence is perturbed before
    being written into the prompt. The *target* is always the clean
    text, so the LLM learns to map noisy phonemes back to clean words.
    The returned dict also contains ``clean_phonemes`` and
    ``noisy_phonemes`` for downstream debug logging.
    """
    text = (text or "").strip()
    if not text:
        return None

    clean_line = text_to_phoneme_line(text)

    if augmenter is not None:
        noisy_tokens = augmenter(phoneme_line_to_tokens(clean_line))
        noisy_line = " ".join(noisy_tokens)
    else:
        noisy_line = clean_line

    prompt = _format_prompt(noisy_line)
    target = text
    full = prompt + target
    return {
        "prompt": prompt,
        "target": target,
        "full": full,
        "clean_phonemes": clean_line,
        "noisy_phonemes": noisy_line,
    }


def _synthetic_scores(n_phonemes: int, k: int) -> List[float]:
    """Generate plausible synthetic CTC log-prob scores for k hypotheses.

    Real CTC scores are sums of per-frame log-probs, so they scale with
    sequence length. We simulate this: top-1 gets a high-confidence score,
    each subsequent rank gets a progressively worse (more negative) score
    with some jitter so the model learns a range of score gaps.

    The absolute values are in a range typical for a 3B LM decoding
    ~N phonemes worth of frames at moderate confidence.
    """
    base = -random.uniform(0.4, 0.8) * n_phonemes
    scores = [base]
    for rank in range(1, k):
        drop = random.uniform(1.5, 5.0) * (rank ** 0.7) * max(1, n_phonemes / 8)
        scores.append(base - drop + random.gauss(0, 0.3))
    return scores


def build_topk_example(
    text: str,
    augmenter: PhonemeNoiseAugmenter,
    top_k: int = 5,
    include_scores: bool = True,
) -> Optional[Dict[str, str]]:
    """Build a top-k training example with synthetic CTC hypotheses.

    The top-1 hypothesis is always the clean phoneme sequence (the
    "true" decoding). The remaining k-1 hypotheses are independently
    augmented noisy variants. Synthetic log-prob scores are attached
    so the LLM learns to weight hypotheses by confidence.

    The target is still the original clean text, so the model learns to
    reconstruct the correct sentence even when the top-k beam contains
    errors.
    """
    text = (text or "").strip()
    if not text:
        return None

    clean_line = text_to_phoneme_line(text)
    clean_tokens = phoneme_line_to_tokens(clean_line)
    if not clean_tokens:
        return None

    scores = _synthetic_scores(len(clean_tokens), top_k)

    hyps: List[CTCHypothesis] = [
        CTCHypothesis(phonemes=list(clean_tokens), score=scores[0])
    ]
    for rank in range(1, top_k):
        noisy = augmenter(list(clean_tokens))
        hyps.append(CTCHypothesis(phonemes=noisy, score=scores[rank]))

    prompt = format_topk_prompt(hyps, include_scores=include_scores)
    return {
        "prompt": prompt,
        "target": text,
        "full": prompt + text,
        "clean_phonemes": clean_line,
        "noisy_phonemes": hyps[1].as_string() if len(hyps) > 1 else clean_line,
    }


def prepare_split(
    split: str,
    augmenter: Optional[PhonemeNoiseAugmenter] = None,
    include_clean: bool = True,
    top_k: int = 1,
    topk_include_scores: bool = True,
):
    """Build the phonemes->text dataset for a split.

    Parameters
    ----------
    split:
        The wikitext split name.
    augmenter:
        Optional :class:`PhonemeNoiseAugmenter`. When provided, each
        example is augmented.
    include_clean:
        When ``augmenter`` is provided, also emit the clean phoneme
        version of every example. Ignored when ``augmenter`` is ``None``.
    top_k:
        When > 1 and ``augmenter`` is provided, also emit a top-k
        hypothesis example for every sentence. This teaches the LLM the
        ``<PHONEME_HYPOTHESES>`` prompt format with confidence scores,
        matching the format produced by :func:`Models.ctc_decode.topk_beam_decode`
        at inference time.
    topk_include_scores:
        Whether to attach log-prob scores to each top-k hypothesis line.
    """
    base = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    base = base.filter(lambda ex: isinstance(ex.get("text", ""), str) and len(ex["text"].strip()) > 0)

    def mapper(batch):
        outs: List[Dict[str, str]] = []
        for t in batch["text"]:
            if augmenter is None:
                ex = build_example(t, augmenter=None)
                if ex is not None:
                    outs.append(ex)
            else:
                if include_clean:
                    clean_ex = build_example(t, augmenter=None)
                    if clean_ex is not None:
                        outs.append(clean_ex)
                noisy_ex = build_example(t, augmenter=augmenter)
                if noisy_ex is not None:
                    outs.append(noisy_ex)
                if top_k > 1:
                    topk_ex = build_topk_example(
                        t,
                        augmenter=augmenter,
                        top_k=top_k,
                        include_scores=topk_include_scores,
                    )
                    if topk_ex is not None:
                        outs.append(topk_ex)

        if not outs:
            return {
                "prompt": [], "target": [], "full": [],
                "clean_phonemes": [], "noisy_phonemes": [],
            }
        return {
            "prompt":          [o["prompt"]          for o in outs],
            "target":          [o["target"]          for o in outs],
            "full":            [o["full"]            for o in outs],
            "clean_phonemes":  [o["clean_phonemes"]  for o in outs],
            "noisy_phonemes":  [o["noisy_phonemes"]  for o in outs],
        }

    ds = base.map(
        mapper,
        batched=True,
        remove_columns=base.column_names,
        num_proc=int(os.environ.get("LLAMA_DATA_NUM_PROC", "2")),
    )
    return ds


def debug_print_clean_noisy(ds, n: int = 5, tag: str = "") -> None:
    """Print ``n`` clean/noisy phoneme pairs from a raw dataset."""
    n = min(n, len(ds))
    header = f"--- clean/noisy phoneme samples{f' [{tag}]' if tag else ''} ---"
    print(header)
    for i in range(n):
        ex = ds[i]
        clean = ex.get("clean_phonemes", "<missing>")
        noisy = ex.get("noisy_phonemes", "<missing>")
        target = ex.get("target", "<missing>")
        print(f"[{i:03d}] TEXT : {target}")
        print(f"      CLEAN: {clean}")
        print(f"      NOISY: {noisy}")
    print("-" * len(header))


# ---------- Tokenization with prefix-masked labels ----------

def make_tokenize_fn(tokenizer, max_length: int = 512, min_target_tokens: int = 4):
    """
    Ensures every example has at least `min_target_tokens` supervised tokens.
    We encode prompt and target separately, then truncate the prompt to leave room.
    """
    assert tokenizer.pad_token_id is not None, "pad_token_id must be set"

    def _tok(batch):
        input_ids_batch, attn_batch, labels_batch = [], [], []

        for prompt, target in zip(batch["prompt"], batch["target"]):
            # Encode WITHOUT adding extra special tokens
            p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            t = tokenizer(target, add_special_tokens=False)["input_ids"]

            # Skip pathological cases
            if len(t) == 0:
                continue

            # Reserve space for target
            max_prompt_len = max_length - min_target_tokens
            if max_prompt_len <= 0:
                continue

            # Truncate prompt to leave room
            if len(p) > max_prompt_len:
                p = p[:max_prompt_len]

            # Fit as much target as possible, but keep at least min_target_tokens
            space = max_length - len(p)
            if space < min_target_tokens:
                # Even after truncating prompt, no room left → skip example
                continue
            t = t[:space]

            ids = p + t
            attn = [1] * len(ids)
            labs = ([-100] * len(p)) + t[:]  # supervise only target

            # NOTE: no padding here — the collator pads to the max length in
            # each batch. This avoids spending compute on [PAD] tokens for
            # short wikitext lines and is a big throughput win when combined
            # with ``group_by_length=True``.
            input_ids_batch.append(ids)
            attn_batch.append(attn)
            labels_batch.append(labs)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attn_batch,
            "labels": labels_batch,
        }
    return _tok

def debug_supervision(ds, name):
    import numpy as np
    import random
    n = min(2000, len(ds))
    cnt = 0
    for i in range(n):
        labs = ds[i]["labels"]
        if any(l != -100 for l in labs):
            cnt += 1
    print(f"[{name}] examples with at least 1 supervised token: {cnt}/{n}")

# ---------- Custom collator (keep labels, just pad if needed) ----------

class CausalLMDataCollator(DataCollatorWithPadding):
    """
    Uses tokenizer padding for inputs. Expects 'labels' already provided;
    pads labels with -100 to match input length.
    """
    def __call__(self, features):
        labels = [f["labels"] for f in features]
        for f in features:
            f.pop("labels")
        batch = super().__call__(features)

        max_len = batch["input_ids"].shape[1]
        padded = []
        for lab in labels:
            if len(lab) < max_len:
                lab = lab + [-100] * (max_len - len(lab))
            else:
                lab = lab[:max_len]
            padded.append(lab)
        batch["labels"] = torch.tensor(padded, dtype=torch.long)
        return batch


# ---------- Main training with LoRA ----------

def _load_noise_cfg(path: Optional[str]) -> Optional[dict]:
    """Load a noisy-LLM JSON config; return None when missing."""
    if not path:
        return None
    with open(path, "r") as f:
        return json.load(f)


def _configure_sdpa_backend() -> None:
    """Disable the cuDNN SDPA backend on CUDA builds where it's unstable.

    The cuDNN frontend that ships with recent PyTorch (e.g. 2.5+/cu128)
    sometimes fails to build an execution plan for LLaMA's attention
    shapes in fp16 and raises::

        RuntimeError: cuDNN Frontend error: [cudnn_frontend] Error:
        No valid execution plans built.

    Disabling the cuDNN SDPA path makes PyTorch fall back to flash /
    mem-efficient / math attention, which all work reliably on Colab.
    Set ``LLAMA_ENABLE_CUDNN_SDP=1`` to opt back in.
    """
    if not torch.cuda.is_available():
        return
    if os.environ.get("LLAMA_ENABLE_CUDNN_SDP") == "1":
        return
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)


def main(noise_cfg_path: Optional[str] = None, output_dir: str = "./results_lora"):
    """Train the phoneme-to-text LoRA head.

    Parameters
    ----------
    noise_cfg_path:
        Optional path to a JSON noisy-LLM config. When provided (and the
        config has ``enabled=true``) the training set is augmented with
        noisy phoneme inputs as well as the clean ones.
    output_dir:
        Where to dump TrainingArguments logs/checkpoints.
    """
    _configure_sdpa_backend()
    noise_cfg = _load_noise_cfg(noise_cfg_path)
    augmenter = build_augmenter_from_config(noise_cfg)
    include_clean = True if noise_cfg is None else bool(noise_cfg.get("include_clean", True))
    debug_samples = 0 if noise_cfg is None else int(noise_cfg.get("debug_samples", 5))
    top_k = 1 if noise_cfg is None else int(noise_cfg.get("top_k", 1))
    topk_include_scores = True if noise_cfg is None else bool(noise_cfg.get("topk_include_scores", True))

    if augmenter is None:
        print("Noise augmentation: DISABLED (clean phoneme training).")
    else:
        print(
            "Noise augmentation: ENABLED "
            f"(sub={augmenter.substitute_prob}, ins={augmenter.insert_prob}, "
            f"del={augmenter.delete_prob}, include_clean={include_clean}, "
            f"top_k={top_k}, topk_include_scores={topk_include_scores})"
        )

    if top_k > 1 and augmenter is None:
        print("WARNING: top_k > 1 requires an augmenter (noise config with enabled=true). "
              "Falling back to top_k=1.")
        top_k = 1

    # 1) Build dataset
    print("Building phonemes → text dataset...")
    train_ds = prepare_split(
        "train",
        augmenter=augmenter,
        include_clean=include_clean,
        top_k=top_k,
        topk_include_scores=topk_include_scores,
    )
    val_ds   = prepare_split("validation", augmenter=None)

    if debug_samples > 0:
        debug_print_clean_noisy(train_ds, n=debug_samples, tag="train")
        debug_print_clean_noisy(val_ds,   n=min(debug_samples, 3), tag="val")

    # 2) Tokenizer & special tokens
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # Add tags and ensure a pad token
    special = {"additional_special_tokens": TAGS}
    added = tokenizer.add_special_tokens(special)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # LLaMA convention

    # 3) Tokenize with prompt-masked labels
    print("Tokenizing dataset with masked labels...")
    tok_fn = make_tokenize_fn(tokenizer, max_length=512)
    _num_proc = int(os.environ.get("LLAMA_DATA_NUM_PROC", "2"))
    tokenized_train = train_ds.map(
        tok_fn,
        batched=True,
        remove_columns=train_ds.column_names,
        num_proc=_num_proc,
    )
    tokenized_val = val_ds.map(
        tok_fn,
        batched=True,
        remove_columns=val_ds.column_names,
        num_proc=_num_proc,
    )

    # 4) Load base model, resize embeddings, then wrap with LoRA
    print("Loading base model...")
    attn_impl = os.environ.get("LLAMA_ATTN_IMPL", "sdpa")
    print(f"  attn_implementation = {attn_impl}")

    # Prefer bf16 on Ampere+ (A100/H100) – same speed as fp16 but more
    # numerically stable for LoRA. Fall back to fp16 elsewhere (e.g. T4).
    _use_bf16 = bool(
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    )
    _model_dtype = torch.bfloat16 if _use_bf16 else torch.float16
    print(f"  model dtype = {_model_dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=_model_dtype,
        attn_implementation=attn_impl,
    )

    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # --- LoRA config (typical for LLaMA) ---
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",      # attention
            "gate_proj", "up_proj", "down_proj"          # MLP
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # sanity check

    # 5) Collator & args
    collator = CausalLMDataCollator(tokenizer=tokenizer)

    print("Setting up training arguments...")
    # Throughput knobs (override via env vars without code changes):
    #   LLAMA_BATCH_SIZE       (default 8)  – per-device batch size
    #   LLAMA_GRAD_ACCUM       (default 1)  – gradient accumulation steps
    #   LLAMA_NUM_EPOCHS       (default 2)  – training epochs
    #   LLAMA_DATALOADER_WORKERS (default 4)
    _bs = int(os.environ.get("LLAMA_BATCH_SIZE", "8"))
    _ga = int(os.environ.get("LLAMA_GRAD_ACCUM", "1"))
    _ep = float(os.environ.get("LLAMA_NUM_EPOCHS", "2"))
    _dl_workers = int(os.environ.get("LLAMA_DATALOADER_WORKERS", "4"))

    # The boolean ``group_by_length`` arg was renamed to
    # ``train_sampling_strategy="group_by_length"`` in recent transformers
    # releases (>=4.57 / main). Detect what the installed version
    # supports so the code runs on both old and new transformers.
    import dataclasses as _dc
    _ta_field_names = {f.name for f in _dc.fields(TrainingArguments)}
    _ta_kwargs = dict(
        output_dir=output_dir,
        eval_strategy="epoch",
        per_device_train_batch_size=_bs,
        per_device_eval_batch_size=_bs,
        num_train_epochs=_ep,
        learning_rate=2e-4,          # a bit higher is common for (Q)LoRA
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        save_total_limit=1,
        bf16=_use_bf16,
        fp16=not _use_bf16,
        gradient_accumulation_steps=_ga,
        warmup_ratio=0.03,
        dataloader_num_workers=_dl_workers,
        report_to="none",
    )
    if "tf32" in _ta_field_names:
        _ta_kwargs["tf32"] = torch.cuda.is_available()
    if "train_sampling_strategy" in _ta_field_names:
        _ta_kwargs["train_sampling_strategy"] = "group_by_length"
    elif "group_by_length" in _ta_field_names:
        _ta_kwargs["group_by_length"] = True

    training_args = TrainingArguments(**_ta_kwargs)

    vocab = model.get_input_embeddings().num_embeddings
    print("tokenizer/model vocab:", len(tokenizer), vocab)
    debug_supervision(tokenized_train, "train")
    debug_supervision(tokenized_val, "val")

    # 6) Train
    print("Setting up Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    # 7) Save adapters + tokenizer alongside the checkpoints in output_dir
    # so they land on Drive (or whatever persistent storage output_dir points
    # to) rather than in Colab's ephemeral /content/ filesystem.
    adapters_dir = os.path.join(output_dir, "lora_adapters")
    merged_dir   = os.path.join(output_dir, "lora_merged")

    print(f"Saving the LoRA adapters to {adapters_dir} ...")
    trainer.save_model(adapters_dir)
    tokenizer.save_pretrained(adapters_dir)

    # --- Optional: export a merged full model (fp16, larger) ---
    print(f"Merging LoRA into base weights and saving to {merged_dir} ...")
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    # 8) Evaluate
    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    import argparse as _ap

    _parser = _ap.ArgumentParser(description="Train phoneme→text LLM (optionally with noisy phonemes).")
    _parser.add_argument(
        "--noise_config",
        type=str,
        default=None,
        help="Path to a JSON noisy-LLM config (see configs/noisy_llm.json).",
    )
    _parser.add_argument(
        "--output_dir",
        type=str,
        default="./results_lora",
        help="Where to write training outputs.",
    )
    _args = _parser.parse_args()
    main(noise_cfg_path=_args.noise_config, output_dir=_args.output_dir)
