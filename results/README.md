# results/

Tracked run records. One JSON per run, committed back — `plan.md` §4.

**Why this is tracked and `outputs/` is not:** `outputs/` is scratch (images, dumps,
anything large or regenerable). These files are the evidence behind every number in
the benchmark table, and Week 4 compares methods by diffing them. A metric with no
committed record is a metric you cannot defend later.

## Schema

Every run file carries the same top-level shape, so runs stay diffable:

| Key | Purpose |
|---|---|
| `run` | which harness produced this |
| `timestamp_utc` | when |
| `git_sha` | **which code.** `-dirty` suffix = uncommitted changes were present |
| `config` | model, dataset, split, quantization, decoding params, prompt template, device |
| `metrics` | the numbers, including the majority-class baseline they must be read against |
| `timings` | wall-clock — a benchmark column in its own right |
| `peak_vram_gib` | measured with the `reset_peak_memory_stats()` discipline (see `memory.md` §8) |
| `records` | per-example predictions, so results can be re-analysed without re-running |

## Rules

- **Always store the baseline next to the metric.** 35.7% means nothing until you
  know the majority-class answer scores 22.9%.
- **Always store `git_sha`.** Two runs that differ could differ because of the
  method or because of the harness; without the SHA you cannot tell which.
- **Disaggregate.** Blended accuracy hid that binary questions score 67.2% while
  open-ended ones collapse to 11.4%.
- **Always store which split.** `config.split` is `night` or `day`. Since 2026-08-12
  (D11) these are two different benchmarks — a night row and a day row must never be
  put in the same column. `--compare` will warn when two runs do not pair.
- Small JSON only. No weights, images, or dataset shards.

## Files

| File | What it establishes |
|---|---|
| `zeroshot_probe.json` | Milestone D go/no-go — 224×224 resolution does not cause a floor effect. Zero-shot 35.7% vs 22.9% baseline on 140 rows. **Superseded as a benchmark row** by `eval_zeroshot.json`; kept as the record of the dataset decision. |
| `eval_split_manifest.jsonl` | The **retired** day eval split: **1,117** rows of `day-validation` shards 0–7 — image filename, source shard + row index, `token`, question, answer, and the `sha256` of each PNG. Pixels in the gitignored `outputs/eval_split/`; identity here. **Kept**: it defines the split that produced `eval_zeroshot.json` and it is the only split the `--shard0-only` regression gate runs against. Rebuild with `--split day`. |
| `answer_vocab.json` | **The frozen answer vocabulary**, derived from `day-train` with a saturation check — never from the eval split. Used for format-compliance scoring on every run. See `docs/eval-protocol.md` §5. |
| `eval_zeroshot_shard0.json` | Regression gate for the Milestone E port: `src/eval.py` restricted to the probe's original 140 rows. Exists to prove the refactor is faithful. `full_split: false` — **not a benchmark row.** |
| `eval_zeroshot.json` | Zero-shot Cosmos-Reason2-2B, 4-bit NF4, on the full **1,117**-row **day** split. 35.1% strict vs a 26.3% baseline. ⚠️ **Still valid, no longer the benchmark row** — the split was retired 2026-08-12 (D11). Zero-shot, so the image leak does not touch it; it is simply a day number and cannot be compared to a night one. |
| `eval_split_night_manifest.jsonl` | **THE CURRENT BENCHMARK SPLIT.** **659** rows of `night-validation` shards 0–4 over **115** images. Same schema as the day manifest. Verify with `python scripts/build_eval_split.py --verify`. |
| `eval_zeroshot_night.json` | **Benchmark row 1 (night)** — zero-shot on the night split. The go/no-go for D11: pre-registered as ≥10 pp over the 24.9% majority baseline to commit. |
| `train_pool_manifest.jsonl` | **The training pool** — the whole `day` domain (both of its splits, which share images and so carry no train/val distinction). Written only if the image-leak check against the night split passes. |
| `train_batch_probe.json` | Peak VRAM vs physical batch size, measured until OOM. Two points separate the fixed cost (weights + fp32 upcast + optimizer state) from the part that scales (retained activations); one point cannot. |
| `train_qlora.json` | **The QLoRA training run** (local, 3060): 60 min budget → **2,428 optimizer steps**, 1 epoch over 1,817 rows, loss 2.096 → 0.485, 7 scaler skips, scale settled at 1024, peak 3.60 GiB, tripwire silent. |
| `train_tripwire_check.json` | **Proof that `src/train.py` consults the tripwire**, not just that the tripwire works. `--lr 5.0` halts at step 6 on 5 consecutive skips, naming `visual.patch_embed.proj.lora_A`. A clean run here is a failure. |
| `eval_qlora_night.json` | **Benchmark row 2** — QLoRA, day-trained, scored on night. **47.2% strict** vs a **31.9% per-type prior** (+15.3 pp); format compliance **100%**; binary 66.0%, open-ended 31.2%. |
| `stability_baseline.json` | **Milestone F** — a healthy fp16 LoRA run completes with no false alarm from the tripwire. |
| `stability_sabotage.json` | **Milestone F success criterion** — an inflated LR diverges and the tripwire halts it with a named cause. A *clean* sabotage run is a failure. |

## Comparing two runs

```
python -m src.eval --compare results/eval_zeroshot.json results/eval_qlora.json
```

Runs **McNemar** on the paired per-example outcomes rather than comparing the two
accuracies' confidence intervals. Both runs score the same examples, so per-item
difficulty cancels and only the disagreements carry information — which is what makes
1,117 rows enough to separate methods a couple of points apart. See
`docs/eval-protocol.md` §8.
