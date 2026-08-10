"""Dataset inspection — Milestone D, step 4.

Downloads ONE Arrow shard of `KevinNotSmile/nuscenes-qa-mini` (~479 MB) and
answers the questions that decide whether we commit to it:

  1. How many rows, and what does a row actually contain?
  2. What do the questions/answers look like? (docs/memory.md §6 flags this as a
     largely *perception* benchmark — this is where we confirm or refute that.)
  3. How big is a row if we keep CAM_FRONT only and drop LiDAR + 5 views?
     memory.md §6 records that reduction as an ESTIMATE. This measures it.

Deliberately does NOT pull all 19.8 GB. One shard is enough to decide, and the
full download only makes sense after the pick is final.

Writes sample images to outputs/dataset_peek/ so they can be eyeballed.
"""

from __future__ import annotations

import io
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ID = "KevinNotSmile/nuscenes-qa-mini"
# day-validation: the eval split is what we will actually score on, so inspect that.
SHARD = "day-validation/data-00000-of-00016.arrow"
N_EXAMPLES = 20
N_IMAGES = 6

MB = 1024**2
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "dataset_peek"


def line(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


def download() -> Path:
    from huggingface_hub import hf_hub_download

    line("download")
    print(f"repo  = {REPO_ID}  (dataset)")
    print(f"shard = {SHARD}")
    print("one shard only — the full dataset is 19.8 GB and we do not need it yet\n")

    path = Path(
        hf_hub_download(repo_id=REPO_ID, filename=SHARD, repo_type="dataset")
    )
    print(f"\ncached at {path}")
    print(f"size      = {path.stat().st_size / MB:.1f} MB")
    return path


def load(path: Path):
    from datasets import Dataset

    line("load")
    # from_file reads a single shard standalone; load_from_disk would demand all 16.
    ds = Dataset.from_file(str(path))
    print(f"rows in this shard = {len(ds)}")
    print(f"columns            = {ds.column_names}")
    print("\nfeatures:")
    for name, feat in ds.features.items():
        print(f"  {name:20s} {feat}")
    return ds


def show_examples(ds) -> None:
    line(f"first {N_EXAMPLES} question/answer pairs")
    n = min(N_EXAMPLES, len(ds))
    qa = ds.select(range(n)).remove_columns(
        [c for c in ds.column_names if c not in ("question", "answer")]
    )
    for i, row in enumerate(qa):
        print(f"{i:3d}. Q: {row['question']}")
        print(f"     A: {row['answer']}")


def answer_stats(ds) -> None:
    line("answer distribution (this shard)")
    answers = ds.select(range(len(ds)))["answer"]
    counts = Counter(answers)
    print(f"distinct answers = {len(counts)}   (paper claims 29 classes overall)")
    total = len(answers)
    for ans, c in counts.most_common(30):
        print(f"  {c:5d}  {100 * c / total:5.1f}%  {ans}")

    # A closed answer space is what makes exact-match eval possible (Milestone E).
    top = counts.most_common(1)[0]
    print(f"\nmajority-class baseline = {100 * top[1] / total:.1f}%  (answer: {top[0]!r})")
    print("any finetuning result must beat this to mean anything.")


def to_pil(val):
    """Images are stored as nested int64 lists (H, W, 3), NOT as encoded JPEG/PNG.

    That is why the repo is 19.8 GB: 8 bytes per channel value that needs 1.
    """
    import numpy as np
    from PIL import Image

    return Image.fromarray(np.array(val, dtype=np.uint8))


def measure_row_size(ds) -> None:
    """The 19.8 GB is 6 cameras + LiDAR at int64. Measure what is actually needed."""
    line("storage — what the 19.8 GB is actually made of")

    import numpy as np

    table = ds.data.table
    for name in table.schema.names:
        print(f"  {name:20s} {table.column(name).nbytes / MB:8.2f} MB")
    print(f"  {'TABLE TOTAL':20s} {table.nbytes / MB:8.2f} MB   ({len(ds)} rows)")

    row = ds[0]
    front = np.array(row["CAM_FRONT"])
    print(f"\n  CAM_FRONT shape = {front.shape}, dtype = {front.dtype}")
    print(f"  RESOLUTION      = {front.shape[1]}x{front.shape[0]}")
    print("  ^ pre-resized for CNN-era models. nuScenes native is 1600x900.")

    img = to_pil(row["CAM_FRONT"])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    jpeg = buf.tell()
    stored = table.column("CAM_FRONT").nbytes / len(ds)
    print(f"\n  CAM_FRONT as stored (int64) = {stored / 1024:8.1f} KB/row")
    print(f"  CAM_FRONT as JPEG q92       = {jpeg / 1024:8.1f} KB/row")
    print(f"  waste factor                = {stored / jpeg:8.1f}x")

    rows_total = 5776  # 16+16 day shards + 5+5 night shards x ~140 rows (measured)
    print(f"\n  => front-camera-only JPEG, whole dataset ~= {rows_total * jpeg / 1024**2:.0f} MB")
    print("     vs 19.8 GB as published. The size was never the real constraint.")


def dump_images(ds) -> None:
    line(f"writing {N_IMAGES} CAM_FRONT images")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(min(N_IMAGES, len(ds))):
        row = ds[i]
        dest = OUT_DIR / f"{i:02d}.jpg"
        to_pil(row["CAM_FRONT"]).save(dest, quality=92)
        print(f"  {dest.name}  <- {row['question']}  [{row['answer']}]")
    print(f"\nopen them: {OUT_DIR}")
    print("JUDGE THE RESOLUTION. Can *you* answer these questions from these images?")
    print("If you cannot, a floor effect will flatten every row of the benchmark.")


def main() -> int:
    try:
        path = download()
        ds = load(path)
        show_examples(ds)
        answer_stats(ds)
        measure_row_size(ds)
        dump_images(ds)
    except Exception as exc:  # noqa: BLE001 - surface the real error, this is a probe
        print(f"\nFAIL: {type(exc).__name__}: {exc}")
        return 1

    line("done")
    print("Next: judge the QA pairs. Are they perception or reasoning?")
    print("Record the verdict in docs/memory.md §6 and close Milestone D.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
