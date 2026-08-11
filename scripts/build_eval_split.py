"""Build and freeze the Milestone E eval split.

Produces two committed artifacts and one gitignored cache:

  results/eval_split_manifest.jsonl  TRACKED  - the frozen split itself
  results/answer_vocab.json          TRACKED  - the closed answer vocabulary
  outputs/eval_split/                IGNORED  - the PNG pixels (regenerable)

Why the pixels live outside git and the manifest inside it: the split's *identity*
is what must be frozen and reviewable, not 50 MB of binary. Each manifest row
carries the sha256 of its PNG, so the cache can be rebuilt anywhere and verified
byte-for-byte against the committed record.

--- Two design decisions worth reading before changing anything -----------------

1. WHY 1,120 ROWS (day-validation shards 0-7).
   The Milestone D probe used 140 rows. The 95% CI on a 35.7% score at n=140 is
   +/-7.9 pp, and LoRA / QLoRA / DoRA typically land within a few points of each
   other -- so that split would have reported five statistically identical numbers
   and the benchmark would have failed silently at the one thing it exists to do.
   n=1,120 gives +/-2.8 pp unpaired, and much tighter paired (see docs/eval-protocol.md
   on McNemar). Chosen by the confidence interval needed to rank methods, not by
   convenience.

2. WHY PNG AND NOT JPEG.
   docs/memory.md §6 measured JPEG q92 at 15.5 KB/row vs 344 KB/row as stored --
   tempting. But JPEG is lossy, so the eval protocol would be defined on re-encoded
   pixels and every future result would carry an "unless it was the encoding"
   caveat. PNG is lossless: ~45 KB/row, ~50 MB total, and the pixels are provably
   the dataset's own.

   The array -> image conversion here is deliberately IDENTICAL to
   scripts/zeroshot_probe.py (`np.array(row["CAM_FRONT"], dtype=np.uint8)`), because
   src/eval.py must reproduce the probe's 35.7% exactly on shard 0 as a regression
   gate. A "harmless" change to this line would break that check.

Usage:
    python scripts/build_eval_split.py            # build everything
    python scripts/build_eval_split.py --verify   # re-check sha256s, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

# Windows defaults stdout to cp1252 when it is a pipe, and a non-ASCII character in
# a PROGRESS MESSAGE then raises UnicodeEncodeError. That killed one 20-minute build
# after all the real work had already succeeded. A logging call must never be able to
# destroy the run it is reporting on.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ID = "KevinNotSmile/nuscenes-qa-mini"

# The frozen eval split. day-validation, shards 0-7 inclusive -> 8 x 140 = 1,120 rows.
# Never trained on. Shards 8-15 are deliberately left out and unused: if the eval
# split ever needs widening, expanding into untouched shards is honest, whereas
# re-picking rows from a pool we have already scored is not.
EVAL_SHARDS = [f"day-validation/data-{i:05d}-of-00016.arrow" for i in range(8)]

# MEASURED, not assumed. Shards 0-4 hold 140 rows but shards 5-7 hold 139, so the
# split is 1,117 rows and not the 8 x 140 = 1,120 this was planned as. Recorded here
# because every confidence interval in docs/eval-protocol.md is derived from it.
EXPECTED_ROWS = 1117

# The answer vocabulary comes from day-TRAIN, never from the eval split. See
# docs/eval-protocol.md: a vocabulary derived from the eval answers makes format
# compliance a moving target and leaks split-specific information into a reported
# metric. Download train shards until the class set stops growing.
VOCAB_SHARD_TMPL = "day-train/data-{:05d}-of-00016.arrow"
VOCAB_MAX_SHARDS = 6          # hard cap on the download
VOCAB_SATURATION_STREAK = 2   # stop after N consecutive shards add no new class

REPO_ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = REPO_ROOT / "outputs" / "eval_split"
MANIFEST = REPO_ROOT / "results" / "eval_split_manifest.jsonl"
VOCAB_PATH = REPO_ROOT / "results" / "answer_vocab.json"

MB = 1024**2


def line(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


def normalize(text: str) -> str:
    """MUST stay identical to src/eval.py and scripts/zeroshot_probe.py.

    Duplicated rather than imported because this script has to run standalone on a
    fresh Kaggle clone before src/ is on the path. If you change one, change all
    three -- docs/eval-protocol.md records this rule.
    """
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip().lower()
    text = text.strip(string.punctuation + string.whitespace)
    return re.sub(r"\s+", " ", text)


def fetch(filename: str, attempts: int = 5) -> Path:
    """Download one shard, retrying transient network failures.

    ~3.2 GB over a home connection is long enough that a DNS blip mid-run is a
    when, not an if -- it already cost one build here. hf_hub_download resumes
    from its cache, so a retry re-requests only what is missing.
    """
    import time as _time

    from huggingface_hub import hf_hub_download

    for attempt in range(1, attempts + 1):
        try:
            path = Path(hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset"))
            print(f"  {filename}  ({path.stat().st_size / MB:.0f} MB)")
            return path
        except Exception as exc:  # noqa: BLE001 — any network failure is worth retrying
            if attempt == attempts:
                raise
            wait = 5 * 2 ** (attempt - 1)  # 5s, 10s, 20s, 40s
            print(f"  ⚠️  {filename} failed ({type(exc).__name__}), retry {attempt}/{attempts - 1} in {wait}s")
            _time.sleep(wait)
    raise AssertionError("unreachable")


def read_shard(path: Path):
    from datasets import Dataset

    # from_file memory-maps a single shard standalone; load_from_disk would demand
    # all 16. Same call the probe uses.
    return Dataset.from_file(str(path))


def build_vocab() -> dict:
    """Derive the closed answer vocabulary from day-train, with a saturation check.

    nuScenes-QA (AAAI 2024) reports 29 answer classes. We do not take that on faith
    -- we measure until the set stops growing, and record whether it actually
    saturated. If it never saturates the 'closed vocabulary' assumption behind
    exact-match scoring is wrong, and that is worth knowing loudly.
    """
    line("answer vocabulary (from day-train)")
    vocab: set[str] = set()
    counts: Counter = Counter()
    streak = 0
    used = []

    for i in range(VOCAB_MAX_SHARDS):
        shard = VOCAB_SHARD_TMPL.format(i)
        ds = read_shard(fetch(shard))
        before = len(vocab)
        answers = [normalize(a) for a in ds["answer"]]
        vocab.update(answers)
        counts.update(answers)
        used.append(shard)
        new = len(vocab) - before
        print(f"    rows={len(ds):4d}  new classes=+{new:2d}  total={len(vocab)}")

        streak = streak + 1 if new == 0 else 0
        if streak >= VOCAB_SATURATION_STREAK:
            print(f"  saturated: {VOCAB_SATURATION_STREAK} consecutive shards added nothing")
            break
    else:
        print(f"  ⚠️  hit the {VOCAB_MAX_SHARDS}-shard cap WITHOUT saturating")

    saturated = streak >= VOCAB_SATURATION_STREAK
    print(f"\n  vocabulary size = {len(vocab)}   saturated = {saturated}")
    print(f"  most common     = {counts.most_common(8)}")

    return {
        "source": "day-train",
        "shards_used": used,
        "saturated": saturated,
        "n_classes": len(vocab),
        "classes": sorted(vocab),
        "train_frequencies": dict(counts.most_common()),
    }


def build_split(verify_only: bool) -> list[dict]:
    line("eval split (day-validation shards 0-7)")
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    mismatches = 0

    for shard in EVAL_SHARDS:
        ds = read_shard(fetch(shard))
        # Parse the shard number off the filename, not by splitting on "-" -- the
        # split name itself contains one ("day-validation").
        shard_idx = int(re.search(r"data-(\d+)-of-", shard).group(1))

        # Columnar reads are cheap. Decoding CAM_FRONT is not: each image arrives as
        # 150k nested Python ints, and that conversion -- not the download -- is what
        # makes this script slow. So pull the text columns in one go and touch the
        # pixels only for rows whose PNG is missing, which makes a resumed run after
        # a dropped connection nearly free.
        tokens, questions, answers = ds["token"], ds["question"], ds["answer"]
        decoded = 0

        for i in range(len(ds)):
            name = f"{shard_idx:02d}_{i:04d}.png"
            path = IMG_DIR / name

            if not verify_only and not path.exists():
                # IDENTICAL to zeroshot_probe.py -- the regression gate depends on it.
                arr = np.array(ds[i]["CAM_FRONT"], dtype=np.uint8)
                Image.fromarray(arr).save(path, format="PNG")
                decoded += 1

            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({
                "image": name,
                "shard": shard,
                "row_index": i,
                "token": tokens[i],
                "question": questions[i],
                "answer": answers[i],
                "sha256": digest,
            })

        print(f"    {shard}  ->  {len(ds)} rows ({decoded} newly encoded)")

    if verify_only:
        old = [json.loads(x) for x in MANIFEST.read_text().splitlines() if x.strip()]
        by_name = {r["image"]: r for r in old}
        for r in rows:
            prev = by_name.get(r["image"])
            if prev is None or prev["sha256"] != r["sha256"]:
                mismatches += 1
        print(f"\n  sha256 mismatches vs committed manifest: {mismatches}")
        if mismatches:
            raise SystemExit("VERIFY FAILED — the cached pixels do not match the frozen split")

    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="re-check sha256s, write nothing")
    args = ap.parse_args()

    rows = build_split(args.verify)

    line("summary")
    tokens = {r["token"] for r in rows}
    answers = Counter(normalize(r["answer"]) for r in rows)
    majority_answer, majority_n = answers.most_common(1)[0]
    binary = sum(v for k, v in answers.items() if k in ("yes", "no"))

    print(f"  rows                  = {len(rows)}")
    print(f"  distinct scenes       = {len(tokens)}  ({len(rows)/len(tokens):.2f} questions/scene)")
    print(f"  distinct answers      = {len(answers)}")
    print(f"  majority baseline     = {100*majority_n/len(rows):.1f}%  (always {majority_answer!r})")
    print(f"  binary (yes/no) share = {100*binary/len(rows):.1f}%")

    if len(rows) != EXPECTED_ROWS:
        # Not fatal, but every CI figure is derived from this count. Say so loudly.
        print(f"  WARNING: expected {EXPECTED_ROWS} rows, got {len(rows)}"
              " -- update docs/eval-protocol.md")

    if args.verify:
        print("\nVERIFY OK — cached pixels match the committed manifest")
        return 0

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {MANIFEST}  ({MANIFEST.stat().st_size / 1024:.0f} KB)")

    vocab = build_vocab()
    VOCAB_PATH.write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    print(f"wrote {VOCAB_PATH}")

    # How many eval answers are absent from the train-derived vocabulary? If this is
    # not ~0, exact-match scoring against a closed set is on shakier ground than
    # docs/memory.md §6 assumes, and the protocol needs to say so.
    unseen = {a for a in answers} - set(vocab["classes"])
    print(f"\n  eval answers not present in the train vocabulary: {len(unseen)}  {sorted(unseen)[:10]}")
    print(f"  images cached in {IMG_DIR} ({sum(f.stat().st_size for f in IMG_DIR.glob('*.png')) / MB:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
