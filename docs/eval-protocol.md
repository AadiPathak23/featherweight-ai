# Eval protocol — FROZEN

> **Status: frozen 2026-08-11 (Milestone E).** This document defines how accuracy is
> computed for every row of the benchmark table. It was written **before any adapter
> existed**, which is the entire point: choosing a metric after seeing results is how
> benchmarks become dishonest (`plan.md` open question #6).
>
> **Changing anything here after this date is a documented decision**, recorded in
> `memory.md` §7 with a date and a reason — not an edit. Any run that predates such a
> change is invalidated and must be re-run, because a benchmark whose rows were scored
> under different rules is not a benchmark.

Companions: [`memory.md`](./memory.md) (measured facts — authoritative), [`plan.md`](./plan.md)
(roadmap), [`../results/README.md`](../results/README.md) (results schema).

---

## 1. The protocol

| Item | Frozen value |
|---|---|
| **Model** | `nvidia/Cosmos-Reason2-2B` (D3, D10 — same base model for every run row) |
| **Quantization** | 4-bit NF4 + double quant, `bnb_4bit_compute_dtype=torch.float16` |
| **Vision tokens** | `min_pixels = 4 × patch_area`, `max_pixels = 1024 × patch_area`, derived at runtime from `processor.image_processor` — **never hardcoded** (`memory.md` §3: the `256*28*28` figure is Qwen2-VL geometry; Qwen3-VL is patch 16 × merge 2 = 1024 px/token) |
| **Prompt template** | see §2 — byte-exact |
| **Decoding** | greedy, `do_sample=False` |
| **`max_new_tokens`** | 48 |
| **Extraction rule** | `normalize()` — see §3 |
| **Primary metric** | strict exact-match on the normalized answer |
| **Always reported with it** | lenient, format compliance (in-vocab %), majority-class baseline, and binary vs open-ended **disaggregated** |
| **Eval split** | `day-validation` shards 0–7, all **1,117** rows, frozen by manifest (§4) |
| **Answer vocabulary** | derived from `day-train`, frozen in `results/answer_vocab.json` (§5) |
| **Robustness set** | `night-validation` — **held out. Never used for ranking.** Reported separately in W4, if at all |

### Why `min_pixels` is set low

The source images are 224×224 (`memory.md` §6). Left to its defaults the processor would
**upscale** them, inventing detail that is not in the data and making every score measure
the interpolator as much as the model. The floor is set deliberately low to prevent that.

---

## 2. Prompt template

Byte-exact, `{question}` substituted verbatim from the dataset:

```
{question}

Answer with ONLY the answer itself - a single word or short phrase. No explanation, no reasoning, no punctuation.
```

The instruction sentence is not decoration. Cosmos-Reason2 is reasoning-tuned and emits a
paragraph by default; without it, exact-match scores near zero for reasons that have
nothing to do with whether the model understood the image. Zero-shot format compliance was
still only **72.1%** *with* it (`memory.md` §6).

---

## 3. Extraction rule

```python
def normalize(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip().lower()
    text = text.strip(string.punctuation + string.whitespace)
    return re.sub(r"\s+", " ", text)
```

Applied to **both** prediction and gold before comparison. Written before any score was
seen, so it cannot have been tuned to flatter a result.

**This function is duplicated in three places** — `src/eval.py`, `scripts/zeroshot_probe.py`
and `scripts/build_eval_split.py` — rather than imported, because the split builder must run
standalone on a fresh Kaggle clone. **If you change one, change all three.** A silent
divergence here would move every number in the benchmark table.

### Strict and lenient are both reported, always

- **strict** — the normalized output *is* the gold answer. This is the reported metric.
- **lenient** — the gold answer appears anywhere in the output. **Diagnostic only.**

Never collapse these to one number. In the Milestone D probe they came out *identical*
(35.7% both), and that identity is what proved the low open-ended score was genuine
blindness rather than the model refusing the output format. A large strict/lenient gap
means a prompt problem; no gap means a perception problem. One number cannot tell them
apart.

---

## 4. The eval split

`day-validation` shards 0–7 → **1,117 rows**, frozen by `results/eval_split_manifest.jsonl`
(one line per row: image filename, source shard, row index, `token`, question, answer,
`sha256` of the PNG bytes).

Pixels live in `outputs/eval_split/` (gitignored, regenerable, **85 MB** measured) as **lossless PNG**.
The manifest is tracked. The split's *identity* belongs in git; 50 MB of binary does not.
`python scripts/build_eval_split.py --verify` re-checks every sha256 against the committed
manifest.

**1,117, not 1,120.** Shards 0–4 hold 140 rows each but shards 5–7 hold 139. Measured, not
assumed — `scripts/build_eval_split.py` asserts the count for exactly this reason.

**Why not 140.** The Milestone D probe's n=140 carries a 95% CI of **±7.9 pp** on a 35.7%
score. LoRA, QLoRA and DoRA typically land within a few points of each other, so that split
would have produced five statistically indistinguishable numbers while looking like a
result. The split size was chosen from the confidence interval needed to rank the methods,
not from convenience.

**Why not all 16 shards.** Shards 8–15 are deliberately untouched. If the split ever needs
widening, expanding into shards nobody has scored is honest; re-drawing rows from a pool
already measured is not.

### ⚠️ Scene correlation — this section's own conclusion was falsified. See the correction below.

> **CORRECTION, 2026-08-11 (same day, after the full-split run).** Everything from here to
> the end of this subsection was written *before* the harness measured the full split. The
> measurement came back **ICC = 0.000, design effect 1.00, n_eff = 1,117** — scene
> correlation is **absent** in the zero-shot model, and the naive interval stands unadjusted
> (`memory.md` §9, `results/eval_zeroshot.json`). The text below is kept, not deleted,
> because the sequence is the lesson — see *What actually happened* at the end.
>
> **No protocol rule changes and no run is invalidated.** The requirement to report both
> intervals stays exactly as frozen, and `src/eval.py` still records `scene_icc` per run.
> Only a prediction about what the number would be was wrong.

An earlier draft of this document, written from shard 0 alone, said scene correlation was
mild and effective n ≈ nominal n. **That was wrong**, and it is recorded here rather than
quietly deleted, because the mistake is instructive: a sample too small to show a structure
will report that the structure is absent.

| | shard 0 (n=140) | full split (n=1,117) |
|---|---|---|
| Distinct scenes | 105 | **270** |
| Questions per scene | 1.33 | **4.14** |

Scenes **span shards** — the same `token` recurs in shards 3, 5 and 7 — so within any one
shard the clustering is invisible.

Questions about the same image are not independent: if the model cannot resolve a distant
pedestrian, it misses every question about that pedestrian at once. Treating 1,117
correlated answers as 1,117 independent samples reports a tighter interval than the data
supports.

**Therefore every run reports two intervals.** `src/eval.py` measures the intra-cluster
correlation of per-example correctness (one-way ANOVA estimator, grouped by `token`),
converts it to a design effect `deff = 1 + (n₀ − 1)·ICC`, and reports a cluster-adjusted
CI alongside the naive one. Stored per run: `n_scenes`, `scene_icc`, `design_effect`,
`n_effective`, `strict_ci95_*_clustered`. **The clustered interval is the one to quote in
the paper.** Per-example `records` carry `token`, so any past run can be re-clustered
without re-running it.

*Measured on shard 0 at the regression gate: ICC = 0.259. Its deff was only 1.09 because
shard-0 clusters are small (1.33); on the full split, with 4.14 questions per scene, the
same ICC produces a substantially larger design effect.*

#### What actually happened — and why this section is worth keeping in full

The sentence directly above is a **prediction**, and the full split falsified it:

| | shard 0 (n=140) | full split (n=1,117) |
|---|---|---|
| ICC | 0.259 | **0.000** |
| Design effect | 1.09 | **1.00** |
| n_effective | 129 | **1,117** |

Shard 0's ICC of 0.259 was an **artifact of near-singleton clusters**. With 1.33 questions
per scene, most "clusters" hold a single question, and the one-way ANOVA estimator has
almost no within-cluster variance to work with, so it attributes ordinary between-question
variance to the scene. More data did not sharpen the estimate — it **removed a structure
that was never there**.

So this document contains three successive claims about one quantity: *mild* (draft, from
shard 0, no measurement) → *not mild* (this section, from shard 0's ICC) → *absent*
(measured, full split). The first was right, for the wrong reason. **The correction was
more wrong than the thing it corrected**, and it was more confidently written, because it
came with a number attached.

The rule worth carrying into W4: **a small sample can manufacture a structure as easily as
it can hide one**, and an estimator quoted without the n and the cluster sizes it was
computed from is not yet evidence. The machinery stays regardless — adapters may induce
scene correlation the base model does not (`plan.md` open question #10), and that is a
question this file is now instrumented to answer rather than guess at.

**Never evaluate on anything touched during training.** `day-train` is for training,
`day-validation` shards 0–7 for scoring, `night-validation` for robustness only.

---

## 5. The answer vocabulary is external to the eval split

Frozen in `results/answer_vocab.json`, derived from **`day-train`** answers by downloading
shards until the class set stops growing (saturation check recorded in the file), then
used unchanged for every split and every run.

**Why this matters.** `scripts/zeroshot_probe.py` built its vocabulary from the eval split's
own answers. That is wrong in two ways: format compliance becomes a moving target that
shifts whenever the split changes, and a reported metric ends up carrying information
derived from the answers being scored. Shard 0 alone contains 24 distinct answers against
the paper's 29 — proof that the split cannot define the label space.

The answer space being **closed** is what makes exact-match viable at all and retires the
`plan.md` risk *"reasoning traces defeat exact-match"* without paying for a judge. If the
saturation check ever fails, that assumption is in doubt and this document must be revisited.

### Measured 2026-08-11

The vocabulary **saturated at 29 classes** — exactly the count nuScenes-QA (AAAI 2024)
reports, arrived at independently, from 5 `day-train` shards (+24, +4, +1, +0, +0).

Two long-tail mismatches, both recorded rather than papered over:

- **`not standing`** appears as a gold answer on **1 of 1,117 eval rows (0.09%)** and is not
  in the train vocabulary. That row can score `strict` correct while counting as
  out-of-vocabulary. Accepted — patching the vocabulary with an eval-only answer would
  reintroduce exactly the leak this section exists to prevent.
- **`trailer`** is in the train vocabulary but never appears as an eval gold. Harmless.

The two 29-class sets therefore share 28 classes. This is expected long-tail behaviour in a
closed vocabulary, not a defect.

### Format compliance moved 72.1% → 75.7% **on the same 140 rows**, and that is correct

The Milestone D probe scored in-vocabulary against the eval split's own **24** answers; the
frozen protocol scores against the train-derived **29**. A larger legal set means more
predictions count as well-formed. **Both numbers are right for their own definition** — they
are not comparable, and only the 75.7% figure is on-protocol. Recorded because an
unexplained 3.6 pp move in a benchmark column is exactly the kind of thing that should never
be waved through.

> **Clarification, 2026-08-11.** Both figures above are **n=140** — the probe and the
> shard-0 regression gate, scored on identical rows so that only the vocabulary differs.
> That is what makes the 3.6 pp attributable to the vocabulary change and nothing else.
>
> **Neither is the benchmark column.** Format compliance on the frozen 1,117-row split is
> **81.3%** (`results/eval_zeroshot.json`, `in_vocab_pct`). Quote that one.
>
> Recorded because this section was misread within a day of being frozen — as a claim that
> the benchmark column was 75.7%, contradicting the committed run. It was never wrong; it
> was **missing its n**, which was enough to make a correct sentence unusable. §9 of
> `memory.md` states the rule this violates: *a percentage quoted without its n is not yet a
> result*. It applies to the document that states it.

---

## 6. Metrics

| Metric | Definition |
|---|---|
| `strict` | normalized prediction == normalized gold. **The reported number.** |
| `lenient` | gold appears anywhere in the normalized prediction. Diagnostic. |
| `in_vocab_pct` | prediction ∈ frozen answer vocabulary. **Format compliance.** |
| `majority_baseline` | accuracy of always answering the most frequent class |
| `binary_acc` / `binary_n` | rows whose gold is `yes`/`no`. Chance = 50%. |
| `open_ended_acc` / `open_ended_n` | everything else |

**Disaggregation is mandatory, not optional.** The probe's blended 35.7% concealed binary
at **67.2%** against open-ended at **11.4%** — 224×224 supports presence judgements but not
identity or counting. A single averaged number would have hidden the most important
property of this dataset.

**Format compliance is its own column for an honest-reporting reason.** 39 of the probe's 90
errors were the model answering sensibly in the wrong words (`bike`→`bicycle`, `zero`→`0`).
A LoRA adapter learns a 29-word answer vocabulary in a few hundred steps, so a large part of
any 35.7% → ~70% jump will be **vocabulary alignment, not improved perception**. Reporting
it as a perception win would be dishonest; carrying in-vocab % separately keeps the two
effects visible and separable (`memory.md` §6, open question #8).

**Always store the baseline next to the metric.** 35.7% means nothing until you know that
always answering `yes` scores 22.9%.

---

## 7. Determinism

Greedy decoding makes a run reproducible **on the same device**. Two runs of the Milestone D
probe reproduced 35.7 / 22.9 / 72.1 exactly.

**Determinism is per-device, not absolute.** Floating-point differences between the 3060
(sm_86) and a Kaggle T4 (sm_75) can flip an argmax on a near-tie. Therefore:

- Every results file records `config.device` (already in the schema).
- The two-run identity check is required **on the same device**.
- A cross-device comparison is flagged in the write-up, never silently assumed equal.

---

## 8. Comparing methods — paired, not by overlapping CIs

Every results file stores per-example `records`. Method comparison therefore uses
**McNemar's test on the paired per-example outcomes**, not a comparison of independent
confidence intervals.

This is what makes n=1,117 sufficient. Two methods evaluated on the *same* 1,117 examples
share all per-item difficulty, so the comparison only has to explain the examples where
they *disagree* — far more statistical power than treating the two accuracies as
independent samples. Comparing overlapping intervals would throw that away and declare real
differences insignificant.

Reported per comparison: the discordant counts (`b`, `c`), the McNemar statistic, and the
p-value.

**Pairing helps, but not unconditionally.** Worked example, both cases exactly 2 pp apart
(60.0% vs 58.0%) at n=1,117:

| | discordant pairs | McNemar *p* | verdict |
|---|---|---|---|
| methods usually agree | 58 (b=40, c=18) | **0.006** | separable |
| methods disagree more | 142 (b=82, c=60) | 0.078 | not separable |

What decides it is **how much the two methods disagree per item**, not the size of the
accuracy gap. Two LoRA variants on the same base and data should be highly correlated, so a
small gap is likely detectable — but that is a bet, and the test is what settles it.

⚠️ **Known limitation.** McNemar assumes independent pairs, and §4 establishes that these
questions cluster by scene. Its p-values are therefore optimistic by roughly the same design
effect. For any W4 comparison that lands near p ≈ 0.05, a scene-level cluster bootstrap is
required before the difference is claimed.

---

## 9. Results file

One JSON per run in `results/`, schema in [`../results/README.md`](../results/README.md):
`run`, `timestamp_utc`, `git_sha` (with `-dirty` when uncommitted changes were present),
`config`, `metrics`, `timings`, `peak_vram_gib`, `records`.

`peak_vram_gib` is measured with the `reset_peak_memory_stats()` discipline from
`scripts/infer_local.py`. Without the reset, `max_memory_allocated()` reports the earliest
spike for every later phase and all deltas read zero — silently, with plausible-looking
numbers (`memory.md` §8).

A run over a subset (`--limit`) records `config.full_split: false`. **A subsampled number
is never a benchmark row.**
