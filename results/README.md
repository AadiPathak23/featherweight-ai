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
- Small JSON only. No weights, images, or dataset shards.

## Files

| File | What it establishes |
|---|---|
| `zeroshot_probe.json` | Milestone D go/no-go — 224×224 resolution does not cause a floor effect. Zero-shot 35.7% vs 22.9% baseline on 140 rows. **Superseded as a benchmark row** by `eval_zeroshot.json`; kept as the record of the dataset decision. |
| `eval_split_manifest.jsonl` | **The frozen eval split.** 1,120 rows of `day-validation` shards 0–7 — image filename, source shard + row index, `token`, question, answer, and the `sha256` of each PNG. Pixels live in the gitignored `outputs/eval_split/`; identity lives here. Verify with `python scripts/build_eval_split.py --verify`. |
| `answer_vocab.json` | **The frozen answer vocabulary**, derived from `day-train` with a saturation check — never from the eval split. Used for format-compliance scoring on every run. See `docs/eval-protocol.md` §5. |
| `eval_zeroshot_shard0.json` | Regression gate for the Milestone E port: `src/eval.py` restricted to the probe's original 140 rows. Exists to prove the refactor is faithful. `full_split: false` — **not a benchmark row.** |
| `eval_zeroshot.json` | **Benchmark row 1** — zero-shot Cosmos-Reason2-2B, 4-bit NF4, on the full frozen 1,120-row split under `docs/eval-protocol.md`. |

## Comparing two runs

```
python -m src.eval --compare results/eval_zeroshot.json results/eval_qlora.json
```

Runs **McNemar** on the paired per-example outcomes rather than comparing the two
accuracies' confidence intervals. Both runs score the same examples, so per-item
difficulty cancels and only the disagreements carry information — which is what makes
1,120 rows enough to separate methods a couple of points apart. See
`docs/eval-protocol.md` §8.
