"""Build and freeze the Week 3 training pool.

Mirrors scripts/build_eval_split.py deliberately -- same download path, same
uint8 -> PNG conversion, same manifest-plus-sha256 shape:

  results/train_pool_manifest.jsonl  TRACKED  - the pool's identity
  outputs/train_pool/                IGNORED  - the PNG pixels (regenerable)

It reuses fetch() and read_shard() from build_eval_split.py rather than
reimplementing them. Two data paths that are supposed to produce identical pixels
should not be two pieces of code -- Milestone E's regression gate exists because
the pixel conversion is load-bearing, and a "harmless" divergence here would show
up as an accuracy difference nobody could attribute.

--- Three decisions worth reading before changing anything -------------------

1. WHY THE POOL IS "ALL OF DAY", TRAIN AND VALIDATION TOGETHER.
   The first version of this script used day-train shards 0-3 and the eval split
   was day-validation shards 0-7. THE LEAK CHECK BELOW KILLED THAT PLAN on its
   first run, 2026-08-12, and the reason is worth stating in full because it is
   a property of the dataset, not a mistake in the plan:

     day-train and day-validation are NOT disjoint sets of images. They are two
     sets of QUESTIONS about the same keyframes. Of the 241 images in day-train
     shards 0-3, 235 also appear in the day eval split -- and all 235 are
     BYTE-IDENTICAL by sha256. Only 6 images (10 rows) of day-train sit outside
     it. Meanwhile 0 of 560 (image, question, answer) triples are shared, so the
     answers do not leak; only the pixels do.

   So the dataset's own train/validation labels do not mean what they say, and
   there is no clean training data inside day-train. The benchmark moved to
   night-validation instead (day and night are different drives: measured
   intersection exactly 0 images). Once eval is night, the ENTIRE day domain --
   both of its so-called splits -- is legitimate training data, and the
   train/validation labels inside it are simply irrelevant.

2. WHY THE LEAK CHECK IS A HARD FAILURE, NOT A WARNING.
   It was written expecting to print "0" and be forgotten. It found a 97.5%
   overlap on its first run. Had it been a warning, it would have printed above
   a successful build and been scrolled past, and every W3/W4 row would have
   been trained on its own eval images -- with an accuracy number that looked
   completely normal. That is the whole argument: the failures worth
   instrumenting are the ones that do not announce themselves.

   It stays in place now that eval is night, because "day and night are disjoint"
   is exactly the kind of structural assumption this dataset has already
   violated once.

3. WHY PNG, AND WHY THE SAME CONVERSION LINE.
   Same reasoning as the eval split: JPEG is lossy, so training pixels would be
   re-encoded and every result would carry an "unless it was the encoding"
   caveat. The `np.array(row["CAM_FRONT"], dtype=np.uint8)` line is copied
   verbatim from build_eval_split.py on purpose -- train and eval must see the
   same pixel pipeline or the comparison is between two things at once.

Usage:
    python scripts/build_train_pool.py            # build + leak check
    python scripts/build_train_pool.py --verify   # re-check sha256s, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

# Windows defaults stdout to cp1252 when it is a pipe; a non-ASCII progress message
# then raises UnicodeEncodeError and kills the run AFTER the expensive work has
# succeeded. That has already happened once here (memory.md §8).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Import the download + shard reader from the eval builder. Inserted on the path so
# this works both as `python scripts/build_train_pool.py` and as an import from src/.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from build_eval_split import fetch, normalize, read_shard  # noqa: E402

# The day domain. Both of the dataset's day splits, because with eval on night the
# distinction between them carries no information -- they are questions about one
# shared pool of ~276 keyframes. These 13 shards are the ones already in the HF cache
# from Milestone E and today's investigation, so the pool costs no new download;
# day-train 5-15 and day-validation 8-15 are available if it ever needs widening.
TRAIN_SHARDS = ([f"day-train/data-{i:05d}-of-00016.arrow" for i in range(5)]
                + [f"day-validation/data-{i:05d}-of-00016.arrow" for i in range(8)])

IMG_DIR = REPO_ROOT / "outputs" / "train_pool"
MANIFEST = REPO_ROOT / "results" / "train_pool_manifest.jsonl"
# The CURRENT benchmark split. Must stay in step with src/eval.py's SPLITS["night"].
EVAL_MANIFEST = REPO_ROOT / "results" / "eval_split_night_manifest.jsonl"

MB = 1024**2


def line(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


def build_pool(verify_only: bool) -> list[dict]:
    line(f"training pool (the day domain — {len(TRAIN_SHARDS)} shards)")
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for shard in TRAIN_SHARDS:
        ds = read_shard(fetch(shard))
        # Parse the shard index off the filename rather than splitting on "-":
        # the split name itself contains one ("day-train").
        shard_idx = int(re.search(r"data-(\d+)-of-", shard).group(1))
        # The source split must be part of the filename. day-train shard 0 and
        # day-validation shard 0 both index from 0, so a name built from the shard
        # index alone would have them overwrite each other's PNGs -- silently, and
        # with a manifest that still listed both.
        prefix = "dt" if shard.startswith("day-train") else "dv"

        # Columnar reads are cheap; decoding CAM_FRONT is not -- each image arrives
        # as ~150k nested Python ints, and that conversion, not the download, is what
        # makes this slow. Pull the text columns once and touch pixels only for rows
        # whose PNG is missing, so a resumed run after a dropped connection is nearly
        # free.
        tokens, questions, answers = ds["token"], ds["question"], ds["answer"]
        decoded = 0

        for i in range(len(ds)):
            name = f"{prefix}{shard_idx:02d}_{i:04d}.png"
            path = IMG_DIR / name

            if not verify_only and not path.exists():
                # IDENTICAL to build_eval_split.py. Train and eval must share one
                # pixel pipeline.
                arr = np.array(ds[i]["CAM_FRONT"], dtype=np.uint8)
                Image.fromarray(arr).save(path, format="PNG")
                decoded += 1

            if not path.exists():
                raise SystemExit(f"missing {path} — run without --verify first")

            rows.append({
                "image": name,
                "shard": shard,
                "row_index": i,
                "token": tokens[i],
                "question": questions[i],
                "answer": answers[i],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })

        print(f"    {shard}  ->  {len(ds)} rows ({decoded} newly encoded)")

    return rows


def leak_check(rows: list[dict]) -> int:
    """Hard gate: no IMAGE may appear in both the training pool and the eval split.

    Returns the number of shared images; the caller exits non-zero if it is not 0.
    Reports overlap at two levels, because they fail completely differently:

      image (token) overlap  -> the model is scored on pixels it trained on.
                                Catastrophic, and invisible in the accuracy number.
      question overlap       -> the same question TEXT about a different image.
                                Expected and harmless: nuScenes-QA questions are
                                generated from templates, so strings recur by
                                design. Reported only so a reader does not mistake
                                it for the first kind.

    `token` is a KEYFRAME id, not a scene id. That distinction is not pedantic --
    calling it a scene is what made "day-train vs day-validation" sound like a
    split over situations when it is a split over questions.
    """
    line("train/eval leak check")
    if not EVAL_MANIFEST.exists():
        raise SystemExit(f"missing {EVAL_MANIFEST}\n"
                         "Run: python scripts/build_eval_split.py")

    eval_rows = [json.loads(x) for x in EVAL_MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]
    train_imgs = {r["token"] for r in rows}
    eval_imgs = {r["token"] for r in eval_rows}
    shared_imgs = train_imgs & eval_imgs

    train_q = {r["question"] for r in rows}
    eval_q = {r["question"] for r in eval_rows}
    shared_q = train_q & eval_q

    print(f"  train images          = {len(train_imgs)}")
    print(f"  eval images           = {len(eval_imgs)}")
    print(f"  SHARED IMAGES         = {len(shared_imgs)}   <- must be 0")
    print(f"  shared question texts = {len(shared_q)} of {len(train_q)} distinct "
          f"({100 * len(shared_q) / max(len(train_q), 1):.1f}%)  <- expected: templated questions")

    if shared_imgs:
        print(f"\n  examples: {sorted(shared_imgs)[:5]}")
        print("  An image in both sets means the model is scored on pixels it trained on.")
        print("  Benchmark row 2 and every W4 row built on it would be inflated, and the")
        print("  accuracy number would look FINE. This is why the check is a hard failure.")
    return len(shared_imgs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="re-check sha256s, write nothing")
    args = ap.parse_args()

    rows = build_pool(args.verify)

    if args.verify:
        if not MANIFEST.exists():
            raise SystemExit(f"missing {MANIFEST} — nothing to verify against")
        old = [json.loads(x) for x in MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]
        by_name = {r["image"]: r for r in old}
        mismatches = sum(1 for r in rows
                         if by_name.get(r["image"], {}).get("sha256") != r["sha256"])
        print(f"\n  sha256 mismatches vs committed manifest: {mismatches}")
        if mismatches or len(rows) != len(old):
            raise SystemExit(f"VERIFY FAILED — {mismatches} mismatched, "
                             f"{len(rows)} rows on disk vs {len(old)} committed")
        print("VERIFY OK — cached pixels match the committed manifest")
        return 0

    shared = leak_check(rows)

    line("summary")
    imgs = {r["token"] for r in rows}
    answers = Counter(normalize(r["answer"]) for r in rows)
    majority_answer, majority_n = answers.most_common(1)[0]
    binary = sum(v for k, v in answers.items() if k in ("yes", "no"))

    print(f"  rows                  = {len(rows)}")
    print(f"  distinct images       = {len(imgs)}  ({len(rows)/len(imgs):.2f} questions/image)")
    print(f"  distinct answers      = {len(answers)}")
    print(f"  majority class        = {100*majority_n/len(rows):.1f}%  (always {majority_answer!r})")
    print(f"  binary (yes/no) share = {100*binary/len(rows):.1f}%")
    # Not a metric -- a distribution-shift check. The adapter is trained on day and
    # scored on night, so the two answer distributions are no longer the same data
    # by construction. If they diverge sharply the adapter learns a prior that does
    # not match what it is scored against, and that shows up as a puzzling result
    # with no obvious cause. Night: 24.9% majority ('yes'), 46.0% binary.
    print("  (eval split is NIGHT: 24.9% majority ('yes'), 46.0% binary)")

    if shared:
        raise SystemExit(f"\nLEAK CHECK FAILED — {shared} images appear in both the "
                         "training pool and the frozen eval split. Refusing to write "
                         "the manifest; a benchmark built on this would be invalid.")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {MANIFEST}  ({MANIFEST.stat().st_size / 1024:.0f} KB)")
    print(f"images cached in {IMG_DIR} "
          f"({sum(f.stat().st_size for f in IMG_DIR.glob('*.png')) / MB:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
