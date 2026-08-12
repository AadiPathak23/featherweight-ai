"""The frozen eval harness — Milestone E.

Implements exactly `docs/eval-protocol.md`. Every row of the benchmark table is
produced by this file: zero-shot, prompt-tuning, LoRA, QLoRA, DoRA. If a number
appears in the paper and did not come out of here, it does not belong there.

Ported from scripts/zeroshot_probe.py, which was built as this file's skeleton.
What changed and why:

  - Rows come from the FROZEN MANIFEST (results/eval_split_manifest.jsonl), not a
    live Arrow shard. The split is now a committed artifact, not a slice decided
    at call time.
  - The answer vocabulary comes from results/answer_vocab.json, derived from
    day-TRAIN. The probe derived it from the eval split's own answers, which made
    format compliance a moving target and leaked split information into a reported
    metric. See docs/eval-protocol.md §5.
  - Optional --adapter, so Weeks 3-4 reuse this harness untouched. A benchmark
    whose rows are scored by different code is not a benchmark.
  - --compare runs McNemar on two results files. Methods are compared PAIRED, on
    the same examples, not by eyeballing overlapping confidence intervals.
  - The probe's go/no-go verdict block is gone. Milestone D is closed.

Usage:
    python -m src.eval --shard0-only     # regression gate: must reproduce 35.7 / 22.9
    python -m src.eval                   # the real thing: all 1,117 rows
    python -m src.eval --compare results/a.json results/b.json   # no GPU needed
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# See scripts/build_eval_split.py: on Windows a non-ASCII progress message raises
# UnicodeEncodeError when stdout is a pipe. A 20-minute eval must not die in a print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MODEL_ID = "nvidia/Cosmos-Reason2-2B"

GIB = 1024**3
REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "results" / "eval_split_manifest.jsonl"
VOCAB_PATH = REPO_ROOT / "results" / "answer_vocab.json"
IMG_DIR = REPO_ROOT / "outputs" / "eval_split"

# FROZEN — docs/eval-protocol.md §2. Byte-exact. Changing this invalidates every
# run scored before the change.
PROMPT_TEMPLATE = (
    "{question}\n\n"
    "Answer with ONLY the answer itself - a single word or short phrase. "
    "No explanation, no reasoning, no punctuation."
)
MAX_NEW_TOKENS = 48


# --------------------------------------------------------------------------- #
# Scoring — frozen. Identical to scripts/zeroshot_probe.py and
# scripts/build_eval_split.py. See docs/eval-protocol.md §3: these three copies
# move together or the benchmark table silently shifts.
# --------------------------------------------------------------------------- #

def normalize(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip().lower()
    text = text.strip(string.punctuation + string.whitespace)
    return re.sub(r"\s+", " ", text)


def score(pred_raw: str, gold: str) -> tuple[bool, bool, str]:
    """(strict, lenient, extracted).

    strict  -- the normalized output IS the gold answer. The reported metric.
    lenient -- gold appears anywhere in the output. DIAGNOSTIC ONLY: a large
               strict/lenient gap is a prompt-format problem, no gap is a
               perception problem. One number cannot distinguish them.
    """
    pred = normalize(pred_raw)
    gold = normalize(gold)
    if pred == gold:
        return True, True, pred
    return False, bool(re.search(rf"\b{re.escape(gold)}\b", pred)), pred


# --------------------------------------------------------------------------- #
# Statistics — dependency-free on purpose. scipy is not in requirements-local.txt
# and adding a dep to compute two closed-form numbers is not worth the install
# surface on a Kaggle image we pin.
# --------------------------------------------------------------------------- #

def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval, in percent.

    Wilson rather than the normal approximation because the normal interval
    misbehaves near 0% and 100% -- and open-ended accuracy sits at 11.4%, close
    enough to the boundary for that to matter.
    """
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return 100 * (centre - half), 100 * (centre + half)


def cluster_icc(groups: dict[str, list[int]]) -> tuple[float, float, float]:
    """Intra-cluster correlation of per-example correctness, grouped by scene.

    Returns (icc, mean_cluster_size, design_effect).

    WHY THIS EXISTS. The eval split has 1,117 questions but only ~270 distinct
    scenes -- 4.14 questions per scene. Questions about the same image are not
    independent: if the model cannot see the far-away pedestrian, it misses every
    question about that pedestrian at once. Treating 1,117 correlated answers as
    1,117 independent samples reports a tighter confidence interval than the data
    supports, which is the exact failure mode Milestone E exists to prevent.

    Measured here rather than assumed, using the one-way ANOVA estimator, and
    converted to a design effect: deff = 1 + (n0 - 1) * ICC, effective n = N/deff.

    ICC = 0 -> scenes carry no shared difficulty, effective n is the real n.
    ICC = 1 -> every question in a scene stands or falls together; the effective
               sample size is the number of SCENES, not the number of questions.
    """
    sizes = [len(v) for v in groups.values()]
    k, total = len(sizes), sum(sizes)
    if k < 2 or total <= k:
        return 0.0, float(total or 0), 1.0

    grand = sum(sum(v) for v in groups.values()) / total
    msb = sum(len(v) * (sum(v) / len(v) - grand) ** 2 for v in groups.values()) / (k - 1)
    msw_num = sum(sum((y - sum(v) / len(v)) ** 2 for y in v) for v in groups.values())
    msw = msw_num / (total - k) if total > k else 0.0

    # n0 is the *effective* cluster size; it equals the mean only when clusters are
    # equal-sized, and they are not here (1 to several questions per scene).
    n0 = (total - sum(s * s for s in sizes) / total) / (k - 1)
    denom = msb + (n0 - 1) * msw
    icc = 0.0 if denom <= 0 else max(0.0, min(1.0, (msb - msw) / denom))
    return icc, total / k, 1 + (n0 - 1) * icc


def mcnemar(b: int, c: int) -> tuple[float, str]:
    """Paired test on discordant pairs. b = A right/B wrong, c = A wrong/B right.

    Concordant pairs carry no information about which method is better, so they
    are excluded by construction -- that is exactly why the paired test has more
    power than comparing two independent confidence intervals, and why n=1,117
    is enough to rank methods that differ by a couple of points.
    """
    n = b + c
    if n == 0:
        return 1.0, "no discordant pairs — the two runs agree on every example"
    if n < 25:
        # Exact binomial. The chi-square approximation is unreliable this small.
        p = 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / (2**n)
        return min(p, 1.0), f"exact binomial (b={b}, c={c}, n={n})"
    chi2 = (abs(b - c) - 1) ** 2 / n  # continuity-corrected
    p = math.erfc(math.sqrt(chi2 / 2))  # chi-square, 1 df
    return p, f"chi-square w/ continuity correction (b={b}, c={c}, chi2={chi2:.3f})"


# --------------------------------------------------------------------------- #
# Provenance + instrumentation
# --------------------------------------------------------------------------- #

def git_sha() -> str:
    """Which code produced this number. Two runs that disagree could disagree over
    the method or over the harness; without this you cannot tell which."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:  # noqa: BLE001 — provenance capture must never kill a run
        return "unknown"


def vram(label: str) -> tuple[float, float]:
    """Resident + phase peak, then RESET the peak counter.

    The reset is not optional. max_memory_allocated() is a high-water mark that
    never decreases, so without it every later phase reports the earliest spike
    and all phase deltas read zero -- silently, with plausible numbers. Peak VRAM
    is a benchmark column; getting this wrong makes every row report the same
    load-time spike. Origin: scripts/infer_local.py, memory.md §8.
    """
    import torch

    current = torch.cuda.memory_allocated() / GIB
    peak = torch.cuda.max_memory_allocated() / GIB
    print(f"  [VRAM] {label:<28} resident={current:5.2f} GiB   phase peak={peak:5.2f} GiB")
    torch.cuda.reset_peak_memory_stats()
    return current, peak


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_split(shard0_only: bool, limit: int | None) -> tuple[list[dict], bool]:
    """Returns (rows, is_full_split)."""
    if not MANIFEST.exists():
        raise SystemExit(f"missing {MANIFEST}\nRun: python scripts/build_eval_split.py")
    rows = [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]
    full = True

    if shard0_only:
        # The Milestone D regression gate: the same 140 rows the probe scored, in
        # the same order. Must reproduce 35.7 / 22.9 exactly.
        rows = [r for r in rows if r["shard"].endswith("data-00000-of-00016.arrow")]
        full = False
    if limit is not None:
        rows = rows[:limit]
        full = False
    return rows, full


def load_vocab() -> set[str]:
    if not VOCAB_PATH.exists():
        raise SystemExit(f"missing {VOCAB_PATH}\nRun: python scripts/build_eval_split.py")
    data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    if not data.get("saturated"):
        # Not fatal, but the closed-vocabulary assumption underpins exact-match
        # scoring (docs/eval-protocol.md §5). Never let this pass unremarked.
        print("  ⚠️  answer vocabulary did NOT saturate — closed-set assumption is in doubt")
    return set(data["classes"])


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_model(adapter: str | None):
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # fp16 not bf16 — the T4 has no bf16 (§5)
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    ip = processor.image_processor
    patch_area = (ip.patch_size * ip.merge_size) ** 2
    # Derived at runtime, never hardcoded: memory.md §3 records that plan.md's
    # 256*28*28 was Qwen2-VL geometry and Qwen3-VL differs.
    # min_pixels is set LOW deliberately -- the source images are 224x224 and the
    # default floor would UPSCALE them, inventing detail that is not in the data.
    ip.min_pixels = 4 * patch_area
    ip.max_pixels = 1024 * patch_area

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        dtype=torch.float16,
        device_map="cuda:0",
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        print(f"  adapter loaded: {adapter}")
    model.eval()
    return model, processor, patch_area


# --------------------------------------------------------------------------- #
# Compare mode
# --------------------------------------------------------------------------- #

def run_compare(path_a: Path, path_b: Path) -> int:
    a = json.loads(path_a.read_text(encoding="utf-8"))
    b = json.loads(path_b.read_text(encoding="utf-8"))
    ra, rb = a["records"], b["records"]

    key = lambda r: (r.get("image") or r["question"], r["gold"])  # noqa: E731
    map_b = {key(r): r for r in rb}
    paired = [(x, map_b[key(x)]) for x in ra if key(x) in map_b]
    if len(paired) != len(ra) or len(paired) != len(rb):
        print(f"⚠️  only {len(paired)} of {len(ra)}/{len(rb)} examples matched — "
              "these runs are not on the same split; the comparison is not valid")

    only_a = sum(1 for x, y in paired if x["strict"] and not y["strict"])
    only_b = sum(1 for x, y in paired if y["strict"] and not x["strict"])
    both = sum(1 for x, y in paired if x["strict"] and y["strict"])
    neither = len(paired) - both - only_a - only_b
    p, method = mcnemar(only_a, only_b)

    print("\n" + "=" * 72)
    print(f"A = {path_a.name}   strict {a['metrics']['strict']:.1f}%")
    print(f"B = {path_b.name}   strict {b['metrics']['strict']:.1f}%")
    print("-" * 72)
    print(f"  both correct        {both:5d}")
    print(f"  A only (b)          {only_a:5d}   <- discordant")
    print(f"  B only (c)          {only_b:5d}   <- discordant")
    print(f"  neither             {neither:5d}")
    print(f"\n  McNemar p = {p:.4g}   [{method}]")
    print(f"  {'SIGNIFICANT at 0.05' if p < 0.05 else 'NOT significant at 0.05'}")
    print("=" * 72)
    print("\nOnly the discordant pairs carry information about which method is better;")
    print("the concordant ones cancel. That is why this is more powerful than")
    print("comparing the two accuracies' confidence intervals.")
    return 0


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None, help="path to a PEFT adapter")
    ap.add_argument("--shard0-only", action="store_true",
                    help="regression gate: the 140 rows the Milestone D probe used")
    ap.add_argument("--limit", type=int, default=None, help="first N rows (iteration only)")
    ap.add_argument("--out", type=str, default=None, help="results filename")
    ap.add_argument("--run-name", type=str, default="eval_zeroshot")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                    help="paired McNemar between two results files; no GPU needed")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.compare:
        return run_compare(Path(args.compare[0]), Path(args.compare[1]))

    import numpy as np  # noqa: F401  (kept for parity with the probe's import set)
    import torch
    from PIL import Image

    rows, full_split = load_split(args.shard0_only, args.limit)
    vocab = load_vocab()

    golds = [normalize(r["answer"]) for r in rows]
    counts = Counter(golds)
    majority_answer, majority_n = counts.most_common(1)[0]
    majority = 100 * majority_n / len(golds)

    torch.cuda.reset_peak_memory_stats()
    print(f"\nModel   : {MODEL_ID}" + (f"  + adapter {args.adapter}" if args.adapter else ""))
    print(f"Device  : {torch.cuda.get_device_name(0)}")
    print(f"Split   : {len(rows)} rows   full_split={full_split}")
    print(f"Vocab   : {len(vocab)} classes (from day-train)")
    print(f"Baseline: {majority:.1f}%  (always {majority_answer!r})\n")

    model, processor, patch_area = build_model(args.adapter)
    vram("after load")

    strict_hits = lenient_hits = in_vocab = 0
    records = []
    t_start = time.perf_counter()

    for i, row in enumerate(rows):
        image = Image.open(IMG_DIR / row["image"])
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
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        raw = processor.decode(out[0][n_prompt:], skip_special_tokens=True).strip()

        s, l, pred = score(raw, row["answer"])
        strict_hits += s
        lenient_hits += l
        in_vocab += pred in vocab
        records.append({
            "image": row["image"], "token": row["token"],  # token = scene, for clustering
            "question": row["question"], "gold": row["answer"],
            "pred": pred, "raw": raw, "strict": s, "lenient": l,
        })

        if args.verbose:
            print(f"{'OK ' if s else 'X  '}[{i:4d}] gold={row['answer']!r} pred={pred[:50]!r}")
        elif i % 50 == 0:
            print(f"  ... {i}/{len(rows)}  strict={100*strict_hits/max(i,1):.1f}%")

    elapsed = time.perf_counter() - t_start
    n = len(records)
    strict = 100 * strict_hits / n
    lenient = 100 * lenient_hits / n
    lo, hi = wilson_ci(strict_hits, n)
    peak = torch.cuda.max_memory_allocated() / GIB

    # Cluster-adjusted interval. Questions sharing a scene are not independent, so
    # the naive interval above is optimistic. See cluster_icc().
    by_scene: dict[str, list[int]] = {}
    for r in records:
        by_scene.setdefault(r["token"], []).append(int(r["strict"]))
    icc, mean_cluster, deff = cluster_icc(by_scene)
    n_eff = n / deff
    clo, chi = wilson_ci(round(strict_hits / deff), max(round(n_eff), 1))

    binary = [x for x in records if normalize(x["gold"]) in ("yes", "no")]
    open_ended = [x for x in records if normalize(x["gold"]) not in ("yes", "no")]
    b_hits = sum(x["strict"] for x in binary)
    o_hits = sum(x["strict"] for x in open_ended)

    print("\n" + "=" * 72)
    print(f"strict exact-match : {strict:5.1f}%   95% CI [{lo:.1f}, {hi:.1f}]   <- reported")
    print(f"  cluster-adjusted : {strict:5.1f}%   95% CI [{clo:.1f}, {chi:.1f}]   "
          f"(ICC={icc:.3f}, {len(by_scene)} scenes, deff={deff:.2f}, n_eff={n_eff:.0f})")
    print(f"lenient            : {lenient:5.1f}%   (diagnostic only)")
    print(f"majority baseline  : {majority:5.1f}%   (always {majority_answer!r})")
    print(f"delta over baseline: {strict - majority:+5.1f} pp")
    print(f"format compliance  : {100*in_vocab/n:5.1f}%   (in frozen vocabulary)")
    print("-" * 72)
    print(f"binary (yes/no)    : {100*b_hits/max(len(binary),1):5.1f}%   n={len(binary):4d}  (chance 50%)")
    print(f"open-ended         : {100*o_hits/max(len(open_ended),1):5.1f}%   n={len(open_ended):4d}")
    print("=" * 72)
    print(f"\n{n} examples in {elapsed:.0f}s ({elapsed/n:.2f}s each)   peak VRAM {peak:.2f} GiB")
    if not full_split:
        print("\n⚠️  NOT the full split — this is an iteration number, not a benchmark row.")

    out_name = args.out or (f"{args.run_name}_shard0.json" if args.shard0_only
                            else f"{args.run_name}.json")
    out_path = REPO_ROOT / "results" / out_name
    out_path.write_text(json.dumps({
        "run": args.run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "protocol": "docs/eval-protocol.md (frozen 2026-08-11)",
        "config": {
            "model": MODEL_ID,
            "adapter": args.adapter,
            "dataset": "KevinNotSmile/nuscenes-qa-mini",
            "split": "day-validation shards 0-7 (frozen manifest)",
            "full_split": full_split,
            "n": n,
            "quantization": "4-bit nf4 + double quant, compute fp16",
            "decoding": "greedy (do_sample=False)",
            "max_new_tokens": MAX_NEW_TOKENS,
            "prompt_template": PROMPT_TEMPLATE,
            "vocab_source": "results/answer_vocab.json (day-train)",
            "vocab_size": len(vocab),
            "px_per_vision_token": patch_area,
            "device": torch.cuda.get_device_name(0),
        },
        "metrics": {
            "n": n,
            "strict": strict,
            "strict_ci95_low": lo,
            "strict_ci95_high": hi,
            # Questions sharing a scene are correlated; the naive interval above is
            # optimistic. These are the numbers to quote in the paper.
            "n_scenes": len(by_scene),
            "mean_questions_per_scene": mean_cluster,
            "scene_icc": icc,
            "design_effect": deff,
            "n_effective": n_eff,
            "strict_ci95_low_clustered": clo,
            "strict_ci95_high_clustered": chi,
            "lenient": lenient,
            "majority_baseline": majority,
            "majority_answer": majority_answer,
            "delta_over_baseline_pp": strict - majority,
            "in_vocab_pct": 100 * in_vocab / n,
            "binary_n": len(binary),
            "binary_acc": 100 * b_hits / max(len(binary), 1),
            "open_ended_n": len(open_ended),
            "open_ended_acc": 100 * o_hits / max(len(open_ended), 1),
        },
        "timings": {"seconds_total": elapsed, "seconds_per_example": elapsed / n},
        "peak_vram_gib": peak,
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
