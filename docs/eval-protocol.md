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

## 🚨 AMENDMENT 1 — the eval split moved `day` → `night` *(2026-08-12, decision D11)*

**This is the first change to a frozen rule, and it is a change, not a correction.** Read
this before anything below it; where the two disagree, this section wins.

### What was found

Week 3's training-pool builder ran a train/eval image-overlap check that was written
expecting to print `0`. It printed **235**.

| Measured 2026-08-12 | |
|---|---|
| Images in `day-train` shards 0–3 | 241 |
| …that also appear in the day eval split | **235** |
| …whose PNG bytes are **identical** (sha256) | **235 of 235** |
| `day-train` images *outside* the day eval split | **6 images, 10 rows** |
| Shared `(image, question, answer)` triples | **0 of 560** |
| Union of images across 13 day shards (1,817 rows) | **276** |

**`day-train` and `day-validation` are not two sets of images. They are two sets of
questions about the same ~276 keyframes.** The answers do not leak; the pixels leak
almost completely. Any adapter trained on `day-train` would have been scored on images
it had trained on, and §6's accuracy number would have looked entirely normal.

### What changed

| | Before | After |
|---|---|---|
| **Eval split** | `day-validation` shards 0–7, 1,117 rows | **`night-validation` shards 0–4, 659 rows over 115 images** |
| **Training domain** | `day-train` | **all of `day`** — both its splits, since the distinction carries no information |
| **Manifest** | `results/eval_split_manifest.jsonl` | **`results/eval_split_night_manifest.jsonl`** |
| **Pixels** | `outputs/eval_split/` | **`outputs/eval_split_night/`** |
| **Answer vocabulary** | 29 classes from `day-train` | **unchanged** — see below |
| **Everything in §2, §3, §6, §7, §8** | | **unchanged** |

Night is disjoint from day **by measurement**: `day ∩ night-validation = 0` images and
`day ∩ night-train = 0` images, over the 276 day images and 115 night images checked.
Day and night are different drives, so this is structural rather than incidental — but
it was measured, because this dataset's own split labels have already been shown not to
mean what they say.

### What survives, and what does not

- **Benchmark row 1 (35.1% strict on day) stands.** It is zero-shot: nothing was
  trained, so nothing leaked. It remains a valid measurement **of the day split** and is
  simply **not comparable** to a night row. `results/eval_zeroshot.json` is kept.
- **The shard-0 regression gate stands**, unchanged, on the day split.
  `src/eval.py --shard0-only` forces `--split day` for exactly this reason.
- **The answer vocabulary needed no change.** It is derived from `day-train`, and
  `night-validation` contains **0 answers outside it** (0 classes, 0 rows). This is why
  the move cost nothing in scoring machinery — §5's insistence that the vocabulary be
  external to the eval split is what made the eval split replaceable.
- **No finetuned row is invalidated, because none existed yet.** The check fired before
  the first training step, which is the only reason this is an amendment and not a
  retraction.

### Three consequences that must not be lost

1. **`token` is a KEYFRAME id, not a scene id.** §4 below calls these "scenes" throughout
   and reports "270 distinct scenes, 4.14 questions/scene". They are 270 *frames*. A
   nuScenes drive contributes many near-identical frames roughly half a second apart, so
   the real number of independent situations is far smaller. **This mislabel is what hid
   the leak**: "270 distinct scenes" sounds like 270 independent situations, and
   "day-train vs day-validation" sounds like a split over them.
2. **§4's ICC = 0.000 was measured under that mislabel, on the retired split.** It says
   nothing about night. Night is **5.73 questions per image over only 115 images** — far
   more clustered than day's 4.14 over 270. The clustered interval must be re-measured,
   and this time it may not be free.
3. **Near-duplicate frames remain, and this is a disclosed limitation.** Splitting
   day/night removes image overlap entirely. It does **not** make the eval split 115
   independent situations — consecutive keyframes from one night drive are nearly the
   same picture. The design effect in §4 is the instrument that quantifies it; quote the
   clustered interval.

### Alternative rejected

Re-partitioning the day domain by image (or by recovered drive) was the other candidate.
It would have kept the in-domain claim, but it required re-freezing the split *and*
re-measuring row 1 anyway, and it could not remove the near-duplicate-frame problem
either. Night is disjoint by construction and cost one download. The price is that the
benchmark is now a **day → night domain-shift** evaluation, which must be stated as the
claim it is — not quietly reported as in-domain accuracy.

⚠️ **Pre-registered before the night zero-shot was run** (Milestone D's discipline): if
zero-shot strict beats the night majority baseline by ≥10 pp the split is committed;
3–10 pp is marginal; <3 pp is a floor effect and night cannot rank methods.

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
| **Eval split** | ⚠️ **AMENDED 2026-08-12 — `night-validation` shards 0–4, all 659 rows** (was `day-validation` shards 0–7, 1,117 rows). See Amendment 1 |
| **Answer vocabulary** | derived from `day-train`, frozen in `results/answer_vocab.json` (§5) — **unchanged by the amendment**; night has 0 out-of-vocabulary answers |
| **Training domain** | ⚠️ **AMENDED 2026-08-12 — all of `day`**, both of its splits. They share their images, so the dataset's train/validation labels carry no information |

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

> ⚠️ **SUPERSEDED BY AMENDMENT 1 (2026-08-12).** This section describes the **retired day
> split**. It is kept in full because it is still the definition of the split that
> produced benchmark row 1 and still governs the `--shard0-only` regression gate — and
> because the word "scene" throughout it means **keyframe**, which is the mislabel that
> hid the leak. The current benchmark split is `night-validation` shards 0–4: 659 rows,
> 115 images, 5.73 questions/image, majority baseline 24.9% (`yes`), 46.0% binary,
> frozen by `results/eval_split_night_manifest.jsonl`, pixels in
> `outputs/eval_split_night/` (19 MB PNG).

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

**Never evaluate on anything touched during training.** ⚠️ **Amended 2026-08-12:** the
rule is unchanged but the assignment is inverted — **all of `day` is for training,
`night-validation` is for scoring, `night-train` is the reserve.** The old assignment
violated this rule while appearing to satisfy it, because it trusted the dataset's split
names instead of checking the images.

⚠️ **`night-train` is a weak reserve, and that is measured, not assumed.** It shares
**113 of its 116 images** with `night-validation`. Expanding the eval split into it would
add 659 more *questions* about the same pictures — tightening the interval on
question-level accuracy while adding almost no new visual situations. Day-validation
shards 8–15 are the honest place to widen the *training* pool; there is no clean way to
widen the *eval* pool inside this dataset.

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

### 🚨 AMENDMENT 2 (2026-08-12) — the majority-class baseline is too weak. Quote the per-type prior.

`majority_baseline` answers `yes` to **every** row, including *"what colour is the truck"*.
No real system behaves that way, so it is a straw man — and the gap to it was being read as
evidence of perception.

The honest reference is the **per-question-type prior**: answer the most common answer *of
each question type*. Question type is readable straight off the question text (*"are there
any…"* vs *"what / how many…"*), so this strategy needs **no image, no training and no
understanding**.

| Zero-shot, 4-bit NF4 | day (n=1,117) | night (n=659) |
|---|---|---|
| strict | 35.1% | 31.9% |
| majority baseline | 26.3% | 24.9% |
| delta over it | +8.8 pp | +7.0 pp |
| **per-type prior** | **33.4%** | **31.9%** |
| **delta over the prior** | **+1.7 pp** | **+0.0 pp** |

On night the model scores **210/659 — exactly what the prior scores**. On night *binary*
questions it scores **51.2%** against a **54.1%** constant: worse than the straw man.

**New required metrics**, computed by `src/eval.py` on every run and stored per row:
`prior_baseline`, `prior_baseline_parts`, `delta_over_prior_pp`, `binary_best_constant`,
`open_ended_best_constant`. **`delta_over_prior_pp` is the number to quote.**

⚠️ No past run is invalidated — this adds a baseline, it does not change scoring. But
`memory.md` §9's *"+8.8 pp over baseline"* overstates demonstrated perception by roughly
7 pp, and any write-up must use the prior instead. **A baseline no real system would adopt
is not a baseline.**

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
