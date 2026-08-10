"""Zero-shot probe on nuscenes-qa-mini — the Milestone D go/no-go.

ONE question: does the 224x224 image resolution cause a floor effect?

docs/memory.md §6 records that `CAM_FRONT` ships at 224x224 (nuScenes native is
1600x900), and that questions ask about distant pedestrians and vehicle status.
If the image cannot support the question, every benchmark row collapses toward
the majority-class baseline and the QLoRA-vs-DoRA-vs-LoRA comparison becomes
unrankable. That failure is silent -- you get a full results table of noise.

  zero-shot MEANINGFULLY ABOVE majority baseline -> resolution survivable, commit
  zero-shot AT or NEAR baseline                  -> floor effect, switch datasets

Deliberately NOT the frozen eval protocol (Milestone E). This is a probe. But the
scoring here is the skeleton that becomes src/eval.py, so the extraction rule and
the strict/lenient split are built properly rather than thrown away.

Usage:
    python scripts/zeroshot_probe.py --n 5      # sanity check the prompt first
    python scripts/zeroshot_probe.py --n 100    # the actual probe
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

MODEL_ID = "nvidia/Cosmos-Reason2-2B"
REPO_ID = "KevinNotSmile/nuscenes-qa-mini"
SHARD = "day-validation/data-00000-of-00016.arrow"  # eval split, never trained on

GIB = 1024**3
OUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "zeroshot_probe.json"

# Answers are one or two words from a closed 29-class vocabulary. Say so, or a
# reasoning-tuned model emits a paragraph and exact-match scores zero for reasons
# that have nothing to do with whether it understood the image.
PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Answer with ONLY the answer itself - a single word or short phrase. "
    "No explanation, no reasoning, no punctuation."
)


def vram(label: str) -> tuple[float, float]:
    """Resident + phase peak, then RESET the peak counter.

    The reset is not optional: max_memory_allocated() is a high-water mark that
    never decreases, so without it every later phase reports the earliest spike
    and all phase deltas read zero. See docs/memory.md §8 (Milestone B).
    """
    current = torch.cuda.memory_allocated() / GIB
    peak = torch.cuda.max_memory_allocated() / GIB
    print(f"  [VRAM] {label:<30} resident={current:5.2f} GiB   phase peak={peak:5.2f} GiB")
    torch.cuda.reset_peak_memory_stats()
    return current, peak


def normalize(text: str) -> str:
    """Extraction rule. Kept deliberately simple and written down BEFORE seeing
    any score, so it cannot be tuned to flatter the result."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip().lower()
    text = text.strip(string.punctuation + string.whitespace)
    return re.sub(r"\s+", " ", text)


def score(pred_raw: str, gold: str, vocab: set[str]) -> tuple[bool, bool, str]:
    """Return (strict, lenient, extracted).

    strict  -- the normalized output IS the gold answer. What we would report.
    lenient -- the gold answer appears somewhere in the output. Diagnostic only:
               a big strict/lenient gap means the model knows the answer but will
               not follow the format, which is a PROMPT problem, not a floor effect.
               Conflating the two is how you misdiagnose a dataset.
    """
    pred = normalize(pred_raw)
    gold = normalize(gold)
    if pred == gold:
        return True, True, pred
    lenient = bool(re.search(rf"\b{re.escape(gold)}\b", pred))
    return False, lenient, pred


def load_examples(n: int):
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=REPO_ID, filename=SHARD, repo_type="dataset")
    ds = Dataset.from_file(path)
    vocab = {normalize(a) for a in ds["answer"]}
    print(f"shard rows = {len(ds)}   distinct answers = {len(vocab)}")
    return ds.select(range(min(n, len(ds)))), vocab


def build_model():
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # fp16 not bf16 — match Kaggle's T4
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    ip = processor.image_processor
    patch_area = (ip.patch_size * ip.merge_size) ** 2
    # 224x224 = 50,176 px -> ~49 vision tokens. No cap needed; the images are already
    # tiny. Set min_pixels low so the processor does not UPSCALE them, which would
    # invent detail that is not there and make the probe measure the wrong thing.
    ip.min_pixels = 4 * patch_area
    ip.max_pixels = 1024 * patch_area
    print(f"patch={ip.patch_size} merge={ip.merge_size} -> {patch_area} px/token")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        dtype=torch.float16,
        device_map="cuda:0",
    )
    model.eval()
    return model, processor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--verbose", action="store_true", help="print every prediction")
    args = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()
    print(f"\nModel  : {MODEL_ID}")
    print(f"Device : {torch.cuda.get_device_name(0)}")
    print(f"Split  : {SHARD}  (validation — never trained on)\n")

    examples, vocab = load_examples(args.n)
    golds = [normalize(a) for a in examples["answer"]]
    counts = Counter(golds)
    majority_answer, majority_n = counts.most_common(1)[0]
    majority = 100 * majority_n / len(golds)
    print(f"probe subset       = {len(golds)} examples")
    print(f"majority baseline  = {majority:.1f}%  (always answer {majority_answer!r})\n")

    model, processor = build_model()
    vram("after load")

    strict_hits = lenient_hits = in_vocab = 0
    records = []
    t_start = time.perf_counter()

    for i, row in enumerate(examples):
        image = Image.fromarray(np.array(row["CAM_FRONT"], dtype=np.uint8))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_TEMPLATE.format(question=row["question"])},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
        n_prompt = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy — reruns must be identical (Milestone E)
            )
        raw = processor.decode(out[0][n_prompt:], skip_special_tokens=True).strip()

        s, l, pred = score(raw, row["answer"], vocab)
        strict_hits += s
        lenient_hits += l
        in_vocab += pred in vocab
        records.append({
            "question": row["question"],
            "gold": row["answer"],
            "pred": pred,
            "raw": raw,
            "strict": s,
            "lenient": l,
        })

        if args.verbose or i < 5:
            mark = "OK " if s else ("~  " if l else "X  ")
            print(f"{mark}[{i:3d}] Q: {row['question'][:64]}")
            print(f"         gold={row['answer']!r}  pred={pred[:60]!r}")
        elif i % 10 == 0:
            print(f"  ... {i}/{len(examples)}  strict={100*strict_hits/max(i,1):.1f}%")

    elapsed = time.perf_counter() - t_start
    n = len(records)
    strict = 100 * strict_hits / n
    lenient = 100 * lenient_hits / n

    peak = torch.cuda.max_memory_allocated() / GIB
    print("\n" + "=" * 72)
    print(f"zero-shot strict  : {strict:5.1f}%   <- the number that decides")
    print(f"zero-shot lenient : {lenient:5.1f}%   (gold appears anywhere in output)")
    print(f"majority baseline : {majority:5.1f}%   (always {majority_answer!r})")
    print(f"in-vocabulary     : {100*in_vocab/n:5.1f}%   (format compliance)")
    print(f"delta over baseline: {strict - majority:+5.1f} pp")
    print("=" * 72)
    print(f"\n{n} examples in {elapsed:.0f}s  ({elapsed/n:.1f}s each)   peak VRAM {peak:.2f} GiB")

    print("\nVERDICT")
    if strict - majority >= 10:
        print("  Zero-shot clears the baseline comfortably. The 224x224 resolution")
        print("  leaves usable signal -> dataset DISCRIMINATES. Commit and go to E.")
    elif strict - majority >= 3:
        print("  Marginal. Some signal, but thin headroom for ranking 5 methods.")
        print("  Judgement call — widen the probe before committing.")
    else:
        print("  At or below baseline -> FLOOR EFFECT. The benchmark cannot rank")
        print("  methods. Do NOT build on this. Escape route in memory.md §6.")
        if lenient - strict >= 10:
            print("\n  CAUTION: large strict/lenient gap. Some of this is the model")
            print("  refusing the output format, not blindness. Fix the prompt and")
            print("  re-probe before condemning the dataset.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "model": MODEL_ID, "dataset": REPO_ID, "split": SHARD,
        "n": n, "max_new_tokens": args.max_new_tokens,
        "strict": strict, "lenient": lenient, "majority_baseline": majority,
        "majority_answer": majority_answer, "in_vocab_pct": 100 * in_vocab / n,
        "seconds": elapsed, "peak_vram_gib": peak,
        "records": records,
    }, indent=2))
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
