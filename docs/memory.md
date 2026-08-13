# featherweight-ai — Memory / Context Log

> **Re-read this file first** to regain context in a new session. Companions: [`plan.md`](./plan.md) (roadmap; Week 2 = Milestones D–F) and [`learning-log.md`](./learning-log.md) (Aadi's own-words notes).
>
> 🧭 **How we work — read before doing anything.** Aadi is deliberately working above his current level and wants to *learn*, not receive finished code. Per milestone: **frame** briefly → **Aadi writes a prediction before anything runs** → **build in small pieces**, explaining each non-obvious decision where it appears → **reconcile** prediction vs reality, digging hardest where he was wrong → **Aadi logs it in his own words**. Label each thing **Tier 1** (learn deeply: LoRA/DoRA math, quantization, fp16 stability, experiment design), **Tier 2** (know the shape: HF/PEFT APIs), or **Tier 3** (plumbing: venv, git, Kaggle UI). No black boxes — every flag and magic number gets a reason. Say plainly what is measured fact vs. estimate vs. untested bet. Where he can reason it out, ask and wait. One milestone per session; depth over throughput.
> Everything below traces to a command output or URL captured on the stated date — nothing asserted from recall.
> Last updated: 2026-08-13 (session close — Week 3 half built; **D11: the eval split moved day → night after the leak check fired**; the night zero-shot landed exactly on a no-perception baseline, and **QLoRA then beat it by +15.3 pp — the dataset is viable**)

---

## 1. START HERE — session resume

### 🔖 Where we left off — end of session 2026-08-13

**Week 3's loop is built, run and green — but only after a reason nobody predicted stopped it first: the dataset's own train/validation split shares its images. The eval split moved `day` → `night` (D11) before a single training step ran. Then the night zero-shot showed the model scoring *exactly* at a baseline that needs no perception at all.**

#### ✅ THE DECISIVE EXPERIMENT RAN, AND THE ANSWER IS YES *(2026-08-13)*

**QLoRA on night: 47.2% strict against a 31.9% no-perception prior — +15.3 pp, McNemar p = 6.5e-09.** The dataset can rank methods, and the gain is perception rather than wording: format compliance hit **100%**, and the prior baseline is already 100% format-compliant by construction, so the +15.3 pp is *net of* the vocabulary effect. Full record in **§11**.

Open-ended moved most — **15.4% → 31.2%** (from +2.5 pp to +18.3 pp above its trivial constant) — which is the opposite of what Milestone D's "224x224 collapses open-ended" would have predicted.

#### ▶️ NEXT SESSION — Kaggle, and the two sealed predictions

Everything local is green. What is left is the T4:

```bash
# on Kaggle, via notebooks/kaggle_qlora_train.ipynb
python -m src.train --probe-batch --probe-max 32       # prediction #2 lives or dies here
python -m src.train --budget-minutes 60 --batch-size <from probe> --run-name qlora_t4
python -m src.eval --split night --adapter outputs/adapters/qlora_t4 --run-name eval_qlora_t4
```

- ⚠️ **`requirements-kaggle.txt` has still never run.** It is the one genuinely untested artifact and the first thing that can fail.
- ⚠️ **The local row's wall-clock and VRAM columns are 3060 numbers.** W4's matched-budget comparison must be **all-T4** — this run's 2,428 steps are not what a T4 hour buys.
- **W4 can start immediately after**: LoRA (fp16) and DoRA rows under the same 60 min budget, then McNemar between them. The whole comparison machinery is now proven end to end.

#### 🚨 Finding 1 — the dataset's train/validation split leaks images (D11)

The Week 3 pool builder's leak check was written expecting to print `0`. It printed **235**. Of 241 images in `day-train` shards 0–3, **235 are also in the frozen day eval split, byte-identical by sha256**; only 6 images (10 rows) sit outside it. But **0 of 560 `(image, question, answer)` triples are shared** — the answers do not leak, the pixels do.

`day-train` and `day-validation` are two sets of **questions about the same ~276 keyframes**. Full record + the fix in **§7 D11** and `eval-protocol.md` **Amendment 1**.

- **Eval split is now `night-validation` shards 0–4** — 659 rows, 115 images, `day ∩ night = 0` images (measured, both directions).
- **Training pool is the whole `day` domain**, both of its splits, since the distinction carries no information.
- ⚠️ **`night-train` is disqualified as training data** — it shares **113 of its 116 images** with the eval split. Measuring that is what makes the current setup safe.
- **Benchmark row 1 (35.1% on day) survives** — zero-shot, so nothing leaked. It is a *day* number and must never sit in the same column as a night one.
- **The answer vocabulary needed no change**: night has **0** out-of-vocabulary answers. Milestone E's insistence that the vocabulary come from `day-train` rather than the eval split is exactly what made the split replaceable for free.

#### 🚨 Finding 2 — the reported "delta over baseline" was measured against a straw man

Night zero-shot: **31.9% strict**, 24.9% global majority → +7.0 pp, which reads like signal. It is not.

| | DAY (n=1,117) | NIGHT (n=659) |
|---|---|---|
| Zero-shot strict | 35.1% | **31.9%** |
| Global majority (`always yes`) | 26.3% | 24.9% |
| Delta — *what §9 reports* | +8.8 pp | +7.0 pp |
| **Per-question-type prior** (`yes` to yes/no, `car` to the rest) | **33.4%** | **31.9%** |
| **Delta over the prior** | **+1.7 pp** | **+0.0 pp** |
| Binary | 59.2% vs 56.1% trivial | **51.2% vs 54.1% trivial → −3.0 pp** |
| Open-ended | 13.8% vs 13.3% trivial | 15.4% vs 12.9% trivial |

**On night, zero-shot Cosmos-Reason2 is exactly as accurate as answering `yes` to every yes/no question and `car` to everything else — 210/659 either way.** On binary questions it is *worse* than the trivial strategy.

The global majority baseline answers `yes` to *"what colour is the truck"*. No real system does that. Question type is readable straight off the question text, so routing by type needs **no image, no training, no understanding** — and beats the global baseline by ~7 pp on both splits. **`src/eval.py` now computes `prior_baseline` and `delta_over_prior_pp` on every run.** Quote that delta.

✅ **This did NOT condemn the dataset, and the reasoning for suspending judgement was right.** Zero-shot is a **lower bound**: format compliance was only 80.6% and the model had never seen the task. Finetuning unlocked exactly the perception the zero-shot number could not show — QLoRA reached **47.2%, +15.3 pp over the prior** (§11). **A model scoring at a no-perception baseline is evidence about the model, not yet about the images.**

#### ⏸️ One prediction still genuinely sealed — capture BEFORE running

Prompts for all four are in `learning-log.md`, rewritten 2026-08-13 with the measured figures filled in where a prediction was overtaken:

- **#2, batch size on a 14.6 GiB T4** — `--probe-batch` was deliberately skipped, because it only informs the *Kaggle* batch size. Measured input: **3.60 GiB** at batch 1 (1.47 GiB weights + 0.59 GiB fp32 upcast), 4,411,392 trainable params, sequences only **~96 tokens** (≈49 vision + ~40 text — much shorter than §3's 247, because `max_pixels` caps 224×224 inputs). **A T4 hour is not 2,428 steps**; the 3060's throughput does not transfer.
- ⚠️ **#3 (QLoRA day → night) was also overtaken** — Aadi chose to run the decisive experiment rather than pause for it, and the answer is 47.2% / +15.3 pp / perception. Recorded as overtaken in `learning-log.md`, same as #1. **#2 is the only prediction still genuinely sealed**, and `--probe-batch` has been deliberately left unrun to keep it that way.
- **#4 is new and belongs to Week 4** — the order of QLoRA / LoRA / DoRA under a matched wall-clock budget, and whether fp16 LoRA's larger footprint helps or hurts it when the budget is time rather than epochs. Nothing has been run against it.

⚠️ **#1 (night zero-shot) was overtaken** — it was the go/no-go and had to run. Recorded as such in `learning-log.md` rather than quietly dropped.

#### 🔧 State of the working tree

**Built and committed this session:**

| File | State |
|---|---|
| `src/train.py` | **New.** Wall-clock budget, padded collation, grad accum, `--probe-batch`, Milestone F tripwire imported unchanged. **Collation verified correct on CPU** — supervised spans contiguous, aligned to each example's own prompt boundary, no pad or prompt token ever supervised. ✅ **Run end to end 2026-08-13**: 2,428 steps in a 60 min budget, tripwire silent on the healthy run and firing on `--lr 5.0`. |
| `scripts/build_train_pool.py` | **New.** Day-domain pool + the leak check that found D11. |
| `requirements-kaggle.txt` | **New. Untested** — verified only by running it on Kaggle. Deliberately omits torch. |
| `notebooks/kaggle_qlora_train.ipynb` | **New.** Thin launcher. Never run. |
| `src/eval.py` | `--split night\|day`; `--shard0-only` forces `day`; **new prior-baseline metrics.** |
| `scripts/build_eval_split.py` | `--split`; vocabulary rebuild now opt-in (`--rebuild-vocab`) so the frozen artifact cannot move silently. |
| `src/stability.py` | One **additive, default-preserving** change: `build_trainable(..., lora_alpha=None, lora_dropout=0.0)`. Milestone F's results are unaffected — the defaults reproduce it exactly. Shared rather than copied because it holds the §10 hard-fail on zero trainable vision params. |

**Two defaults deliberately set against the reflex:** `--clip-grad` defaults **off** (it is ladder step 4, a mitigation for a problem F measured as absent at lr 1e-4; enabling it would change the numerics away from the validated configuration and mask the gradients the tripwire watches), and `build_eval_split.py` no longer re-derives the answer vocabulary by default.

**✅ The training pool is built and the leak check passes.** `results/train_pool_manifest.jsonl` (575 KB, tracked) — **1,817 rows over 276 images, 6.58 questions/image, SHARED IMAGES = 0**, 25.9% majority (`yes`), 47.4% binary. Pixels: `outputs/train_pool/`, 138 MB, gitignored and regenerable at a measured 74 images/min.

**All local gates green, verified 2026-08-13:** tripwire halts `train.py` at lr 5.0 naming the vision tower · 60 min QLoRA run completed clean (2,428 steps, no false alarm) · `eval --adapter` on night = **47.2%** · shard-0 regression gate reproduces **35.7 / 35.7 / 22.9 / 75.7 / 67.2 / 11.4** exactly · both manifests verify with **0** sha256 mismatches. **Only `--probe-batch` is unrun**, deliberately — it belongs on the T4.

---

### 🗄️ Previous session — Milestone F (2026-08-11 evening)

**Milestone F is complete. WEEK 2 IS CLOSED.**

- **`src/stability.py` is the fp16 tripwire**, and both criteria passed: a healthy run completes with 6 scaler skips and no false alarm; a sabotage run (lr 5.0) is halted at step 7 naming the offending module. Full record in **§10**.
- **Training fits locally — peak 3.53 GiB** at batch 1, no gradient checkpointing, on the 6 GB 3060. §2 assumed local training was impossible; local *dry-runs* of the Week 3 loop are viable.
- **`eval-protocol.md` corrected** (§4's falsified ICC prediction, §5's missing `n`). No rule changed, no run invalidated — the regression gate still reproduces 35.7 / 22.9 exactly.

**The three things this session found, in order of how expensive they would have been:**

1. **A finite loss is not a healthy run.** The loss never went non-finite in 58 steps across both runs, so `plan.md`'s specified *"halt on non-finite loss"* would have caught **nothing**. Sabotage landed exactly **one** optimizer step, wrecked the adapter, then skipped every step after — weights frozen, loss finite and oscillating, progress bar advancing. On Kaggle: 12 hours and a saved garbage adapter. The **consecutive-skip rule** is what caught it.
2. **The ViT-overflow claim is now measured, and narrower than stated.** **13 of 13 overflows began in the vision tower**; the language tower never overflowed alone. But forward hooks logged **zero** non-finite *activations* — it is the ViT's **gradients**, not its activations. Same shape as the §6 uint8 correction: right number, wrong mechanism.
3. **The vision tower uses `qkv`/`proj`, not `q_proj`/`k_proj`/`v_proj`.** The tutorial-standard LoRA target list would have attached nothing to the ViT → zero vision gradients → a permanently reassuring empty column on the exact site being watched. The harness now hard-fails on that condition.

---

### 🗄️ Previous milestone — Milestone E (2026-08-11, earlier)

**Eval protocol frozen; benchmark row 1 measured.**

- **Protocol frozen in writing:** [`docs/eval-protocol.md`](./eval-protocol.md) — prompt, decoding, extraction rule, split, vocabulary, metrics. Changing it is a documented decision, not an edit.
- **Frozen eval split:** `day-validation` shards 0–7 = **1,117 rows** (not 1,120 — shards 5–7 hold 139). Identity committed in `results/eval_split_manifest.jsonl` with a sha256 per image; pixels in gitignored `outputs/eval_split/` (85 MB PNG).
- **Answer vocabulary frozen from `day-train`**, not from the eval split. **Saturated at 29 classes** — independently matching the nuScenes-QA paper.
- **Benchmark row 1 (zero-shot, 4-bit NF4): 35.1% strict** vs a **26.3%** baseline = **+8.8 pp**. Two runs, all 1,117 raw outputs byte-identical.
- `src/eval.py` is the harness every future row uses, `--adapter` included.

**The three things it found that Milestone D got wrong, all from measuring more rows:**

1. **The delta over baseline is a third smaller than reported.** +12.9 pp → **+8.8 pp**. Accuracy barely moved (35.7 → 35.1) but the majority baseline rose 22.9 → 26.3. *Shard 0 was not a representative sample.*
2. **Binary accuracy was overstated.** 67.2% (n=61) → **59.2%** (n=524). Milestone D's "+17 pp of real signal above chance" is really **+9.2 pp**. Nothing about the model changed — only n did.
3. **Scene clustering turned out to be a non-issue, measured.** Feared design effect, built the machinery, measured **ICC = 0.000** on 270 scenes → deff 1.00, n_eff = 1,117. The naive CI stands. (Shard 0's ICC of 0.259 was an artifact of near-singleton clusters.) Machinery kept for W4, where adapters may induce correlation.

---

### 🗄️ Earlier session — 2026-08-10

**Milestone D is complete. The dataset question is settled.**

Last session ran the Week 2 dataset survey end to end. Ten candidates checked against their real pages (not recall), then the winner verified by **running the model**, not by reading a dataset card.

- **Dataset chosen:** [`KevinNotSmile/nuscenes-qa-mini`](https://huggingface.co/datasets/KevinNotSmile/nuscenes-qa-mini) · fallback SUTD-TrafficQA · full record in **§6**
- **Verified by measurement:** zero-shot Cosmos-Reason2-2B scored **35.7%** strict exact-match vs a **22.9%** majority-class baseline (+12.9 pp) on 140 validation rows. The dataset can rank methods. It is not a floor.
- **New decisions locked:** **D7** (US-collected, image before video), **D8** (we do not build our own dataset), **D9** (gated datasets disqualified as primary) — all in §7
- **Repo renamed** `featherweigh-ai` → `featherweight-ai`; remote and Kaggle clone URL both updated
- **`results/` convention established** — tracked run records with `git_sha`; see `results/README.md`

**The two things last session found that are easy to forget and expensive to rediscover:**

1. **The images are 224×224**, not nuScenes' native 1600×900. Binary yes/no questions score **67.2%** (chance 50%) while open-ended ones collapse to **11.4%**. The blended 35.7% hides this completely — *always disaggregate*. → ⚠️ *Magnitudes superseded by §9: binary is **59.2%**, open-ended **13.8%**. The disaggregation lesson holds; these n=140 numbers do not.*
2. **Much of the coming finetuning gain will be output-vocabulary alignment, not better perception.** 39 of 90 zero-shot errors are the model answering sensibly in the wrong words (`bike`→`bicycle`, `zero`→`0`). Carry format-compliance as its own benchmark column so the two effects stay separable, and do not report the jump as a perception win. → ⚠️ *Superseded by §9: the format share is **28.8%**, not 43%. Real, but smaller — most headroom is perception.*

### ▶️ Next action — Week 3, the first end-to-end QLoRA run on Kaggle

Week 2 built every prerequisite: a frozen protocol, a scored baseline row, and a tripwire. W3 is the first run that produces **benchmark row 2**.

- **Port the loop to Kaggle.** `requirements-kaggle.txt` does not exist yet — Kaggle is torch 2.10.0+cu128 / Python 3.12.13 against local 2.13.0+cu130 (§2). This is the one genuinely untested piece.
- **Import the tripwire from `src/stability.py`.** It is not decoration: §10 shows the failure it catches is invisible to a loss check, and a 12 hr session is exactly where that costs the most.
- **Checkpoint/resume across the 12 hr session cap.** `/kaggle/working` is lost on teardown beyond the save.
- **Budget from measured numbers, not guesses:** peak 3.53 GiB at batch 1 local, ~1.0 s/step on a 3060; the T4 has 14.6 GiB. Batch size is a prediction Aadi owes in `learning-log.md` before the run.
- **Score it with `src/eval.py --adapter`, unmodified.** A row scored by different code is not a row.
- ⚠️ **Expect the vision tower to be where trouble starts** — 13 of 13 overflows began there (§10). `--fp32-vision` is ladder step 2 and already implemented.

**Open for W3/W4, not now:** open-ended accuracy is **13.8%**. If the five methods cannot be separated there, report binary and open-ended as separate columns (open question #9).

### ⚠️ Owed — `learning-log.md`, all of it waiting on Aadi

**Still the one thread no session has closed.** Read the file; every prompt is already written in place. Ordered by what decays fastest — the Milestone F entries are the freshest and the most perishable, because Aadi watched those runs happen.

- **Milestone F — prediction reconciled, three questions + one W3 prediction open.** ⏳ *Newest.* He predicted both outcomes **correctly** (the scale falls; a gradient goes non-finite before the loss) — but both answers were a bare direction with no mechanism, so neither could be partly right or partly wrong. The reconcile is written; the three questions are (1) why can no single-step check separate the baseline's 6 harmless skips from sabotage's 5 fatal ones, (2) what would have happened on Kaggle had the tripwire only checked the loss, (3) why did steps stop landing after step 2, and why is a model that has stopped changing more dangerous than one visibly exploding. **Plus a Week 3 prediction owed before the first QLoRA run:** batch size on a 14.6 GiB T4, given 3.53 GiB at batch 1 locally, and what he expects to be the binding constraint.
- ⚠️ **The standing pattern worth naming:** predictions so far give a direction and no mechanism. That was cheap in Weeks 1–2, where a wrong guess cost a rerun. From Week 3 the runs cost GPU-hours against a 30 hr/week quota, and a prediction with no mechanism cannot be checked against anything except the outcome — so it teaches nothing when it happens to be right, which both of Milestone F's were.

- **Milestone B — four `⚠️ Reconcile owed` blocks.** Answered 2026-08-10, all four wrong on *mechanism*; his original answers are kept verbatim, not deleted, because the gap is the content. **The one that must be fixed before Week 3:** *"training costs more because the loop repeats"* — it does not; each iteration frees the last one's activations, and Milestone B's own 57-token generation loop stayed flat. The cost is that **backward needs every forward activation kept alive** for the chain rule. Also corrected there: the load peak is two copies of each layer (fp16 source + NF4 output), not training or a dataset; and a missing `reset_peak_memory_stats()` breaks the *measurement* silently — no error, no hallucination.
- **Milestone E — prediction reconciled, two questions open.** He predicted **25%**; measured **35.1%**. Direction right, magnitude off by ~10 pp, and the stated reasoning ("since it is zero-shot") explains the absolute level rather than the split-to-split change. The two questions left for him: (1) accuracy barely moved but delta-over-baseline fell a third because the *baseline* rose — did the model get worse, and what exactly did? (2) binary fell 67.2% → 59.2% with nothing about the model changing — what does that say about a percentage quoted without its n?
- **Milestones C and D — no entries at all.** Scaffold prompts in place.

### ✍️ Commit convention *(set 2026-08-11)*

**Commit messages end at the last line of the body. No `Co-Authored-By: Claude` trailer.** Author stays `AadiPathak23 <aadipathak2323@gmail.com>`. This is a portfolio project and the work should read as Aadi's.

⚠️ **Do not retroactively rewrite the 9 older commits that carry the trailer.** Aadi's explicit call: *"what has been done has been done."* Rewriting would change every SHA in the repo and break the `git_sha` provenance stored inside `results/*.json` — a discipline §9 and `results/README.md` depend on. The detailed why-not-what message style stays; only the trailer goes.

### 📁 Repo map

| Path | What it is |
|---|---|
| `scripts/check_env.py` | Milestone A. Runs unmodified on both local and Kaggle; reports `bf16_supported` vs `bf16_native` separately |
| `scripts/infer_local.py` | Milestone B. 4-bit load + single-image inference. **Source of the `vram()` reset discipline — reuse it** |
| `scripts/inspect_dataset.py` | Milestone D. Pulls ONE 457 MB shard, reports schema/QA/storage; dumps images to `outputs/dataset_peek/` |
| `scripts/zeroshot_probe.py` | Milestone D go/no-go. **Superseded by `src/eval.py`** — kept as the record of the dataset decision |
| **`docs/eval-protocol.md`** | **FROZEN + AMENDMENT 1 (2026-08-12).** How accuracy is computed, for every benchmark row. **Read Amendment 1 first** — the eval split moved day → night |
| **`src/eval.py`** | **The eval harness.** Every benchmark row comes from here. `--split night\|day`, `--adapter`, `--compare` (McNemar), `--shard0-only` (regression gate, forces `day`) |
| `scripts/build_eval_split.py` | Builds + freezes an eval split and the answer vocabulary. `--split night\|day`, `--verify` re-checks every sha256, `--rebuild-vocab` is off by default |
| **`scripts/build_train_pool.py`** | **W3.** Builds the day-domain training pool + the **hard image-leak check that found D11**. `--verify` |
| **`src/train.py`** | **W3. The QLoRA loop.** Wall-clock budget, padded batching + grad accum, `--probe-batch`, imports the Milestone F tripwire unchanged |
| `requirements-kaggle.txt` | Kaggle pins. **Deliberately excludes torch** — the image's is CUDA-matched. Untested until run on Kaggle |
| `notebooks/kaggle_qlora_train.ipynb` | W3 launcher: clone → install → build data → probe → train → eval → McNemar → copy artifacts out |
| `results/` | **Tracked** run records, small JSON only. Schema + rules in `results/README.md` |
| `outputs/` | **Gitignored** scratch — images, dumps, anything regenerable |
| **`src/stability.py`** | **Milestone F. The fp16 tripwire.** `--sabotage` proves it fires; ladder flags `--init-scale` / `--fp32-vision` / `--clip-grad`. **Week 3 imports this** |
| `notebooks/kaggle_smoke_test.ipynb` | Milestone C. Thin launcher: clones the repo, runs `check_env.py` |

**Repo:** <https://github.com/AadiPathak23/featherweight-ai> — public. Renamed 2026-08-10 from `featherweigh-ai` (old name was missing the `t`); GitHub redirects the old URL but nothing in the repo relies on that. `gh` CLI is **not** installed; pushes use stored HTTPS credentials. Git identity `AadiPathak23 / aadipathak2323@gmail.com`.

### 🗓️ Milestone board

| | Milestone | Status |
|---|---|---|
| **A** | Local env | ✅ 2026-07-28. Venv, pinned stack, `check_env.py` passes on the 3060 |
| **B** | 4-bit local inference | ✅ 2026-08-06. Peak **2.10 GiB** vs <5.0 GB target; **7–10 tok/s** (the Week 5 edge-target figure) |
| **C** | Kaggle bridge | ✅ 2026-08-07. T4 x2; `check_env.py` runs unmodified from a clone; **bf16 confirmed absent in hardware** |
| **D** | Dataset | ✅ 2026-08-10. `nuscenes-qa-mini`, verified by probe (§6) |
| **E** | Eval protocol | ✅ 2026-08-11. Frozen in `eval-protocol.md`; row 1 = **35.1%** vs 26.3% baseline, reproducible (§9). ⚠️ Split retired 2026-08-12 by D11 — row 1 stands, the split does not |
| **F** | fp16 stability harness | ✅ 2026-08-11. Tripwire catches deliberate divergence; **a finite loss would not have** (§10) |
| **W3** | First end-to-end QLoRA run | 🟡 **LOCAL RUN COMPLETE 2026-08-13** — QLoRA **47.2%** vs a 31.9% prior (+15.3 pp, p=6.5e-09) on the 3060. **D11 found and fixed mid-build.** Kaggle/T4 run + `--probe-batch` still owed; `requirements-kaggle.txt` still untested |
| W4+ | DoRA/LoRA under matched budgets, edge latency, write-up | ⬜ |

⚠️ **Security note (2026-08-07):** an HF token was briefly pasted into a notebook markdown cell and a chat log. It was **revoked immediately** and replaced. Standing rule: credentials are injected at runtime from Kaggle Secrets / env vars, never typed into a file that gets saved, committed or shared. Applies doubly to the **write** token needed in Week 6.

---

## 2. Environment facts *(verified 2026-07-28)*

### Local machine — DEV/TEST ONLY, cannot train a VLM to completion

| Item | Value | Source |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU | `nvidia-smi` |
| VRAM | **6144 MiB** total; **426 MiB already used** by `explorer.exe` + `EpicGamesLauncher.exe` | `nvidia-smi` |
| Compute capability | sm_86 (Ampere) — bf16 supported locally | arch |
| Driver | **592.27** | `nvidia-smi` |
| CUDA (driver-reported max) | **13.1** | `nvidia-smi` |
| OS | Windows 11 Home 10.0.26200 | env |
| Python | **3.13.0** (default) + 3.10 also installed | `py -0p` |
| torch | **NOT installed** | import failed |
| git | 2.51.0.windows.2 | `git --version` |
| Editor | VS Code + Claude Code | — |

> ⚠️ Close Epic Games Launcher before any VRAM-sensitive run.

### Cloud — Kaggle, where training actually happens

| Item | Value |
|---|---|
| GPUs offered | **T4 x2** (sm_75) or **P100** (sm_60) |
| ✅ **Measured** (2026-08-07) | 2 × Tesla T4, `capability=(7, 5)`, **14.6 GiB each** — not the 16 GB on the spec sheet. Budget against 14.6. |
| ✅ **Kaggle stack** | `torch 2.10.0+cu128`, CUDA **12.8** — differs from local `2.13.0+cu130`, hence a separate `requirements-kaggle.txt` (Week 3). |
| Quota | ~**30 GPU-hr/week**, **12 hr** max session |
| Choice | **T4 x2.** P100 is a trap: sm_60 has no fp16 tensor cores, and `LLM.int8()` requires sm_75+ |
| **bf16** | ❌ **CONFIRMED ABSENT IN HARDWARE, measured on a live T4 2026-08-07.** See the trap below and §5. |
| Account status | ✅ Created + phone-verified (2026-08-07), email `aadipathak2323@gmail.com`. Environment set to **pin to original** so the base image can't drift mid-benchmark. |
| ✅ Kaggle runtime | **Linux, Python 3.12.13** (local is Windows/3.13.0) — the same `check_env.py` runs unmodified on both. |
| ✅ CUDA-context overhead | Only **0.10–0.20 GiB** per T4, vs **~1.0 GiB** on the Windows 3060. Usable headroom is really ~5.0 GiB local vs ~14.4 GiB Kaggle. |

### Constraints

- **$0 budget.** No paid cloud, no bought hardware. Student.
- Local 6 GB → inference + dry-runs only.
- GitHub is the local↔Kaggle bridge.
- User is **new to cloud notebooks** → Kaggle steps need to be beginner-explicit.

---

## 3. Model facts — `nvidia/Cosmos-Reason2-2B` *(verified 2026-07-28)*

> **Base model is locked — see §7 D10 (2026-08-11).** "Cosmos 3" is NVIDIA's generation/world-model line, not a newer Cosmos-Reason. No `Cosmos-Reason3` exists. Reason2 remains current (32B variant added 2026-04-29).

| Item | Value | Source |
|---|---|---|
| Exists | ✅ 448k downloads, 135 likes | HF API |
| Parameters | **2,438,696,960** (~2.44B) | model card |
| Base architecture | **`Qwen/Qwen3-VL-2B-Instruct`**; arch tag `qwen3_vl`; class `Qwen3VLForConditionalGeneration` | HF `base_model:` tag |
| **Gated** | ✅ **YES — `gated: auto`.** Unauthenticated raw fetch of `config.json` returns *"Access to model nvidia/Cosmos-Reason2-2B is restricted. You must have access to it and be authenticated."* One click to accept; access auto-granted. | HF API + direct fetch |
| License | NVIDIA Open Model License (commercially usable) | model card |
| Weights | single `model.safetensors`, ~4.9 GB (bf16) | HF file listing |
| Inputs | image (jpg), video (mp4), text | model card |
| Card's stated HW req | **24 GB GPU, BF16, Hopper/Blackwell, Linux** — this is for bf16 + long video context, **not** applicable to 4-bit single-image | model card |
| `library_name` | `cosmos` — validate plain-`transformers` path first | HF metadata |
| Min transformers | **`>=4.57.0`** for Qwen3-VL | cosmos-reason2 repo + Qwen3-VL docs |
| Prior art | `embedl/Cosmos-Reason2-2B-W4A16` exists on HF → evidence 4-bit Cosmos-Reason2 is viable; **cite as related work** | HF search |

**✅ MEASURED 4-bit footprint (2026-08-06, Milestone B):** resident weights **1.47 GiB** — the 1.4–1.6 GB estimate was accurate. Worst-case peak **2.10 GiB**, well inside the <5.0 GB target.

| Fact | Value |
|---|---|
| Quantized | **1.81B** params → uint8-packed NF4, 300 tensors |
| **Left fp16** | **0.32B** params, 325 tensors |
| Largest unquantized tensor | `model.language_model.embed_tokens.weight`, **311.2M params ≈ 622 MB = 42% of resident weights** |
| `tie_word_embeddings` | **True** — `lm_head` *is* `embed_tokens`, one tensor. (Card's 2.44B counts it twice; only 2.13B distinct params exist in memory.) |
| Vision geometry | patch 16, merge 2 → **1024 px per vision token**. ⚠️ `plan.md`'s `256*28*28` was **Qwen2-VL** geometry; Qwen3-VL differs. Derive from `processor.image_processor` at runtime, never hardcode. |

**Peak occurs at LOAD, not inference** (2.10 GiB transient while unpacking+quantizing, vs 1.55 GiB during generation). If this model ever OOMs, expect it at load time.

**Activations were a non-issue at inference:** 247 vision tokens cost +0.01 GiB, 57-token KV cache +0.01 GiB. This does **not** clear the risk for training, where activations are retained for the backward pass — that remains a Week 3 concern.

---

## 4. Library versions

### Latest on PyPI as of 2026-07-28 *(reference only — not what we pin)*

`torch 2.13.0` · `transformers 5.14.1` · `peft 0.19.1` · `bitsandbytes 0.50.0` · `accelerate 1.14.0` · `trl 1.9.1` · `datasets 5.0.0` · `qwen-vl-utils 0.0.14`

### Intended pins (pre-Milestone-A)

| Package | Pin | Why |
|---|---|---|
| torch | cu130 index (`--index-url https://download.pytorch.org/whl/cu130`) | driver 13.1 forward-compatible; cu130 is PyTorch's stable channel as of 2.11+ |
| **transformers** | **`>=4.57,<5`** | **Deliberate.** All Cosmos/Qwen3-VL examples target 4.5x. 5.x is a breaking major release. Not staleness. |
| peft, bitsandbytes, accelerate, datasets, qwen-vl-utils, pillow, huggingface_hub[cli] | latest compatible | — |
| trl | deferred | not needed until training |

### ✅ Actually resolved & verified versions

*(`pip freeze`, local venv, Python 3.13.0, 2026-07-28 — Milestone A.)*

```
torch==2.13.0+cu130          torchvision==0.28.0+cu130
transformers==4.57.6         peft==0.20.0
bitsandbytes==0.50.0         accelerate==1.14.0
datasets==5.0.1              qwen-vl-utils==0.0.14
huggingface_hub==0.36.2      tokenizers==0.22.2
safetensors==0.8.0           numpy==2.4.4              pillow==12.2.0
```

**Resolution notes**

- `torch.version.cuda = 13.0`. The cu130 index had a cp313 win_amd64 wheel — **no Python 3.13 wheel gap, no cu130 packaging gap.** Both fallbacks (cu128, Python 3.10) went unused.
- **`huggingface_hub` must be pinned `<1.0`.** transformers 4.57.x requires `huggingface-hub<1.0,>=0.34.0`, but a bare `huggingface_hub[cli]` resolves to 1.25 and sends pip backtracking down through every 1.x release. `requirements-local.txt` now pins `huggingface_hub[cli]>=0.34,<1.0`, which resolves in a single pass. **This pin is coupled to the `transformers<5` pin — the two move together.**
- torch emits a benign `triton not found; flop counting will not work` warning on Windows. Triton has no Windows wheel; nothing we use needs it.

---

## 5. Known risks

### 🚨 The bf16 trap — measured on a live T4, 2026-08-07

```
torch 2.10.0+cu128, CUDA 12.8, Tesla T4, capability=(7, 5)

is_bf16_supported()                          = True    <-- LIES (fp32 emulation)
is_bf16_supported(including_emulation=False) = False   <-- truth
capability >= (8, 0)                         = False   <-- the hardware reason
```

**Never call `torch.cuda.is_bf16_supported()` without `including_emulation=False`.** Bare, it reports `True` on a T4 because PyTorch will emulate bf16 through fp32 — functional, slow, and none of the benefit. Trusting it would make this project's central constraint look imaginary.

**It is silicon, not software.** Kaggle runs CUDA **12.8** — nearly current — and still has no bf16, while the local 3060 (`sm_86`, Ampere) has it on a *different* CUDA. bf16 tensor cores arrived with **Ampere (sm_80, 2020)**; the T4 is **Turing (sm_75, 2018)**. No driver, toolkit or torch upgrade can add it. This is why every training run here is fp16 — permanently, not pending a fix.

| Risk | Severity | Notes / mitigation |
|---|---|---|
| **No bf16 on free Kaggle** | 🔴 **Highest** | T4 = sm_75, P100 = sm_60 → no hardware bf16, but the model card says BF16 required. All training must be **fp16** (`bnb_4bit_compute_dtype=torch.float16`, AMP + GradScaler, fp32 master weights). **The ViT tower is the usual overflow site.** Stability harness = Week 2 deliverable. ✅ **Confirmed empirically on live hardware 2026-08-07** — see the trap above. |
| transformers 5.x breakage | 🟠 | Pinned to `<5` on purpose. Upgrade is a separate verified task. |
| Model is gated | 🟡 | One-click accept, `gated: auto`. Token needed locally *and* in Kaggle Secrets. Blocks Milestone B. |
| 6 GB inference headroom | 🟡 | Cap `max_pixels`. Close Epic Games Launcher. Target peak < 5.0 GB. |
| bitsandbytes on Windows | ✅ **CLOSED 2026-07-28** | Plain `pip install bitsandbytes` (0.50.0) worked; `Linear4bit` nf4 forward pass on sm_86 succeeded, **no DLL error**. The "use a community fork" advice is confirmed obsolete. |
| cu130 index packaging gaps | ✅ **CLOSED 2026-07-28** | cp313 win_amd64 wheels present for torch 2.13.0 + torchvision 0.28.0. cu128 fallback unused. |
| Python 3.13 long-tail wheel gaps | ✅ **CLOSED 2026-07-28** | Entire local stack installed on 3.13.0. The 3.10 fallback is unused. |
| huggingface_hub 1.x vs transformers 4.5x | 🟢 | Must pin `<1.0` or pip backtracks endlessly. Pinned. Coupled to the `transformers<5` pin. |
| Kaggle quota burn | 🟢 | 30 hr/week; stop sessions explicitly. `/kaggle/working` lost on teardown beyond save. |
| ViT quantization hurts perception | 🟡 | More than LLM quantization does. Planned W4 ablation: exclude ViT from quantization. |

---

## 6. Dataset survey — Milestone D *(verified 2026-08-10)*

✅ **DECIDED:** [`KevinNotSmile/nuscenes-qa-mini`](https://huggingface.co/datasets/KevinNotSmile/nuscenes-qa-mini) · **Fallback:** SUTD-TrafficQA (accepting its gating cost).
Confirmed by measurement, not metadata — zero-shot **35.7%** vs a **22.9%** majority baseline (see the probe below).

### Revised §8 criteria

The original criteria were re-derived mid-survey after two constraints were added (D7, D8). Hard requirements now:

1. **Ungated + open license** — a gated dataset breaks deliverable #4 ("anyone can re-run for $0"). This is a *reproducibility* argument, not a convenience one, and it is the criterion that eliminated the most candidates.
2. **Ready-made QA pairs** — no dataset construction (D8).
3. **Image-based** — video is Week 4+ (D7).
4. Small enough for free Kaggle.
5. Traffic/urban domain, US-collected preferred — but **not** at the cost of 1–3.
6. *Nice-to-have:* published baselines to sanity-check the harness.

### Candidates surveyed

| Candidate | Ungated | Image | QA ready | Verdict |
|---|---|---|---|---|
| **nuscenes-qa-mini** | ✅ | ✅ | ✅ | ✅ **recommended** |
| SUTD-TrafficQA | ❌ Zenodo request form | ❌ video | ✅ | 🟡 fallback |
| RoadSceneVQA | ❌ **no public release found** | ✅ | ✅ | ❌ best domain fit in the survey, unobtainable |
| NVIDIA AI City Challenge | ❌ email approval | ❌ video | ✅ | ❌ unbeatable narrative, breaks reproducibility |
| DriveLM | ❌ "agree to share contact information" | ✅ | ✅ | ❌ also ships no images (4.86 GB = annotations only) |
| VRU-Accident | ✅ Apache-2.0 | ❌ video | ✅ MCQ | ❌ sources are MM-AU / DADA-2000 / DoTA → **not US footage** |
| SurveillanceVQA-589K | ✅ MIT | ❌ video | ✅ | ❌ 2.31 GB is annotations only; videos from UCF-Crime etc. Viewer broken. |
| TU-DAT | ✅ CC-BY-4.0 | ❌ video | ❌ **boxes only** | ❌ roadside CCTV + permissive, but zero QA pairs; also mixes BeamNG.drive *simulation* |
| InterAct-Video | ❌ Google Form, weekly batch | ❌ video | ✅ | ❌ |
| Indian Traffic VQA | ✅ CC-BY-4.0, 50.5 MB | ✅ | ✅ | ❌ India-collected (D7); signboard OCR, not reasoning |

### Verified facts — `nuscenes-qa-mini`

| Item | Value | Source |
|---|---|---|
| Gated | **False** | HF API `gated: False`, `private: False` |
| License | **CC-BY-NC-SA-4.0** | HF API tags |
| Size | **19.8 GB** total; `day-train` alone **7.63 GB** | HF tree API |
| Format | **Arrow** shards — `day-train`/`day-validation` 16 each, `night-train`/`night-validation` 5 each | HF file listing |
| **Shard size** | **479 MB** — so inspection needs *one shard*, not the full 19.8 GB | HF tree API |
| Schema | `token`, `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_BACK_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_FRONT_LEFT`, `LIDAR_TOP`, `question`, `answer` | `dataset_info.json` |
| Splits | day/night × train/validation — a **free robustness axis** for the benchmark table | file listing |
| Scenes | Boston + Singapore (nuScenes) | nuScenes-QA paper |
| Answer space | **29 classes, closed** | nuScenes-QA (AAAI 2024) |
| Baselines | published (arXiv 2305.14836) | HF tags |

**Why the 19.8 GB is not the real cost:** the schema carries **6 camera views + LiDAR**. We need `CAM_FRONT`, `question`, `answer` and nothing else. Preprocess once, drop LiDAR and 5 of 6 views, attach the result as a Kaggle Dataset so it is never re-downloaded per session. ⚠️ *Reduction factor is an estimate until measured on a real shard.*

**Why the closed 29-class answer space matters more than it looks:** it makes Milestone E's hardest problem — scoring free-form reasoning traces — mechanical. Exact-match on a fixed vocabulary + constrained output format, with extraction-failure rate as its own metric. Directly retires the `plan.md` risk *"reasoning traces defeat exact-match"*, and avoids paying for a judge.

### Known weaknesses — accepted deliberately, to be stated in the paper

- **Questions are programmatically generated** from 3D scene annotations → existence / counting / status queries. This is a largely **perception** benchmark being used to evaluate a **reasoning**-specialized model.
- **Ego-vehicle AV footage, not roadside smart-city.** Weaker Metropolis fit; stretches D5.
- **Boston + Singapore**, so only partly US-collected.
- **NC license** — fine for research, blocks commercial use.

*Accepted because the thesis is about finetuning efficiency under matched budgets; the dataset is instrumentation, not the finding (D8). The survey found **no** ungated, reasoning-heavy, US-collected, image-based traffic VQA set — the intersection is empty, and that is itself a reportable survey result.*

### ✅ MEASURED on a real shard *(2026-08-10, `scripts/inspect_dataset.py`, `day-validation/data-00000-of-00016.arrow`)*

| Fact | Value |
|---|---|
| Rows per shard | **140** → day-val ≈ 2,240, day-train ≈ 2,240, night ≈ 700 each. **Total ≈ 5,776** |
| Split counts | ✅ **Reconciled.** The "3,068 total" figure was wrong; ~2,229 day + ~659 night per split is right. |
| **`CAM_FRONT` resolution** | 🚨 **224×224** — pre-resized for CNN-era models. **nuScenes native is 1600×900.** |
| Image storage | nested lists (H,W,3), *not* encoded JPEG/PNG. ⚠️ **CORRECTED 2026-08-11 — see below.** |
| Waste factor | **22.3×** — 343.9 KB/row as stored vs **15.5 KB/row** as JPEG q92 |
| **Real size** | **~87 MB** front-camera-only JPEG for the whole dataset, vs 19.8 GB published |
| Distinct answers (shard) | 24 (paper claims 29 overall) |
| **Majority-class baseline** | **22.9%** (`yes`). `yes`+`no` = **43.6%** of all answers — the benchmark is largely binary. |
| Scenes | Boston confirmed visually in the sample (brick rowhouses, US crosswalk markings) |

**Size is a non-issue and always was.** 19.8 GB → ~87 MB after dropping LiDAR + 5 of 6 views and encoding as JPEG. Preprocess once, attach as a Kaggle Dataset. ✅ *Confirmed 2026-08-11: the frozen 1,117-row split is **85 MB** as lossless PNG.*

#### ⚠️ Correction (2026-08-11) — the storage mechanism above was wrong

The table said "nested **int64** lists — 8 bytes per value that needs 1." Read off the Arrow schema directly: the type is **`list<item: list<item: list<item: uint8>>>`**. The values are **uint8**, exactly as the dataset card declares.

The 344 KB/row is real, but it comes from **int32 offset buffers on the nested lists** — roughly 4 bytes of offset per pixel triple — not from wide values. 224×224×3 = 147 KB of pixels carrying ~197 KB of list offsets.

**The 22.3× waste-vs-JPEG conclusion stands; the stated cause did not.** Worth keeping as a caution: a number can be measured correctly and still be attributed to the wrong mechanism, and only the second kind of error survives into a paper's explanation section.

### 🚨 The 224×224 problem — the real risk, found only by downloading

The §6 recommendation was made on metadata. Inspecting the actual images changes the picture:

- Questions ask about **distant pedestrians, vehicle status, and spatial relations** ("what status is the truck to the front of the with rider thing"). At 224×224, with 1600×900 crushed to square (aspect ratio destroyed), **a human often cannot answer them from the image.**
- If the *input* cannot support the *question*, every run row collapses toward the 22.9% majority baseline — a **floor effect**. The benchmark then cannot rank QLoRA vs DoRA vs LoRA, which is the entire thesis.
- This dataset was packaged for **ResNet-era 224×224 CNN pipelines**, not for a VLM. Cosmos-Reason2's perception capacity is wasted on it.

**Not yet fatal, but unproven.** Coarse questions ("are there any barriers?") may survive the resolution loss. The cheap decider is empirical, not analytical: **run zero-shot Cosmos-Reason2 on ~100 examples and compare against the 22.9% majority baseline.** Meaningfully above → the dataset discriminates, commit. At or near chance → floor effect confirmed, switch.

### Confirmed weaknesses

- ✅ **Questions are templated and perception-only** — confirmed by reading them. Existence, counting, status, spatial relations, generated from scene graphs. Phrasing is stilted ("the with rider thing"). No causal or predictive reasoning anywhere in the sample.

### ✅ RESOLVED — zero-shot probe *(2026-08-10, `scripts/zeroshot_probe.py`)*

**Verdict: the floor effect does NOT bite. The dataset discriminates. Milestone D closed, `nuscenes-qa-mini` committed.**

Cosmos-Reason2-2B, 4-bit NF4, greedy, `max_new_tokens=48`, all **140 rows** of `day-validation` shard 0:

| Metric | Value |
|---|---|
| **Zero-shot strict exact-match** | **35.7%** |
| Majority-class baseline | 22.9% (`yes`) |
| **Delta over baseline** | **+12.9 pp** |
| Lenient (gold anywhere in output) | 35.7% — **identical to strict** |
| Format compliance (in-vocabulary) | 72.1% |
| Throughput | 1.1 s/example · peak VRAM **1.50 GiB** |

**Strict == lenient exactly.** No hidden format-refusal masking blindness — when the model is wrong, the gold answer is genuinely absent from its output. That was the caveat the probe was built to detect, and it is clear.

#### The result is two different results

> ⚠️ **These n=140 figures were superseded on 2026-08-11 by the 1,117-row frozen split (§9). The qualitative split is real; the magnitudes were not. Binary is 59.2%, not 67.2%. Cite §9.**

| Question type | n | Accuracy | Reference |
|---|---|---|---|
| **Binary (yes/no)** | 61 | **67.2%** | chance = 50% → +17 pp — *later measured at 59.2% (n=524), i.e. +9.2 pp* |
| **Open-ended** | 79 | **11.4%** | near-collapse — *later 13.8% (n=593)* |

224×224 supports coarse existence/presence judgements but **not** fine-grained identity, status or counting. That is exactly the resolution story, now measured rather than feared.

#### Error decomposition (90 errors)

| Cause | n | Meaning |
|---|---|---|
| Prediction **outside** the answer vocabulary | 39 | format/synonym loss — `bike`→`bicycle`, `zero`→`0`, `disabled`→`parked` |
| Prediction **in** vocabulary but wrong | 51 | genuine misperception |

The model also says `taxi` 11 times for objects that are not taxis — a default guess when it cannot resolve the object.

#### ⚠️ Interpretive caveat that MUST go in the paper

**A large share of the finetuning gain will be output-vocabulary alignment, not improved perception.** 39 of 90 errors are the model answering sensibly in the wrong words; a LoRA adapter learns a 29-word answer vocabulary within a few hundred steps. So a 35.7% → ~70% jump would mostly mean *"learned to say `bicycle` instead of `bike`"*.

This does **not** invalidate the benchmark — every run row pays the same easy win, so QLoRA vs DoRA vs LoRA remain rankable, and the large headroom is precisely what makes them rankable. But reporting the jump as a perception improvement would be dishonest. **Mitigation:** report format compliance (in-vocab %) as a separate column so the vocabulary gain is visible and separable from the perception gain.

### Unresolved / deferred

- Open-ended accuracy at 11.4% is thin. If W3 shows the five methods cannot be separated on open-ended questions, consider **reporting binary and open-ended as separate benchmark columns** rather than one blended accuracy.
- Escape route if this ever fails: **NuScenes-QA annotations (ungated, GitHub) + full-resolution nuScenes-mini images**. ⚠️ nuScenes requires account registration — decide whether that trips D9 before relying on it.
- The probe used shard 0 of `day-validation` only (140 of ~2,229 rows). Milestone E must freeze a proper eval split.

---

## 7. Key decisions

### 2026-07-28

**D1 — Full-FT dropped as a run row.** Replaced by same-model frozen baselines: zero-shot Cosmos-Reason2-2B + prompt-tuning / linear probe.
*Why:* full FT of 2.44B with AdamW needs ~30–40 GB. Does not fit 16 GB T4, and FSDP across T4 x2 on free Kaggle is a multi-day rabbit hole with high failure odds.

**D2 — Full-FT included as a labeled analytical estimate.** VRAM + cost computed from parameter counts, citing published full-FT figures. Marked *"not run — exceeds the free-tier/edge budget by design."*
*Why:* keeps the comparison honest and present without faking a run.

**D3 — No proxy model. Rejected substituting a smaller VLM for the full-FT row.**
*Why:* every run row must stay on Cosmos-Reason2-2B. Same-model rows are exactly what makes the budget-matched claim apples-to-apples; a smaller substitute would silently break it.

**D4 — Framing: full-FT infeasibility is supporting evidence for the edge-efficiency thesis,** not a limitation to apologize for.

**D5 — Week 1 stays dataset-agnostic.** Dataset survey in Week 2, restricted to **Metropolis-aligned domains** (smart-city / traffic / surveillance / industrial-safety video VQA). Explicitly **not** robotics/embodied, despite that being Cosmos Reason's native domain.
*Why:* narrative fit with the NVIDIA Metropolis target. Shortlist 2–3, biased strongly toward the **smallest viable set** — narrative fit and zero-cost feasibility beat size and leaderboard prestige. Criteria in `plan.md` §8.

**D6 — Default Kaggle accelerator is T4 x2, not P100.**
*Why:* P100 (sm_60) has no fp16 tensor cores and `LLM.int8()` requires sm_75+. The single-device 16 GB is a trap.

### 2026-08-10

**D7 — Datasets must be US/North-America collected, and image-based before video.**
*Why (US):* the paper targets a US audience and NVIDIA Metropolis is a US deployment story; a model finetuned on non-US road furniture, signage and vehicle mix demos badly against it. *Why (image-first):* stepping stone — proving the cheap case before the expensive one depends on it, the same ordering Week 1 used. If video turns out infeasible on a T4, an image benchmark is already banked instead of nothing.
*Note:* the US constraint is a **narrative/career** requirement, not a requirement of the research claim — the QLoRA-vs-DoRA-vs-LoRA comparison is indifferent to footage geography. It may rule out a poor-fit dataset; it must **not** push us into a gated or unusable one.

**D8 — We do not build our own dataset. Rejected generating VQA pairs from an annotated US roadside set (TU-DAT et al.).**
*Why:* the problem statement is *"quantization-aware PEFT is competitive under matched VRAM and wall-clock budgets."* A self-built dataset is a second, unrelated claim that dilutes the first and hands a reviewer a new attack surface. The dataset is instrumentation, not the contribution. Aadi's call, and correct — this reverses an earlier recommendation of mine.
*Consequence:* ready-made QA pairs become a hard requirement.

**D9 — Gated datasets are disqualified from being the primary.**

*Why:* deliverable #4 is "reproducible Kaggle notebooks — anyone can re-run the whole thing for $0." If a reader must submit a form and wait for presigned URLs, that claim is dead. This is a reproducibility argument and eliminated SUTD-TrafficQA, InterAct-Video, AI City Challenge and DriveLM as primaries.

### 2026-08-11

**D10 — Stay on `Cosmos-Reason2-2B`. "Cosmos 3" is a different product line, not a newer version of our model.** *(All facts below from the HF API, queried 2026-08-11 — not recall.)*

*The premise checks out but points elsewhere:* NVIDIA did ship a Cosmos**3** family (`Cosmos3-Nano` 2026-03-10, `Cosmos3-Super` 2026-03-10, `Cosmos3-Edge` 2026-07-01, plus Text2Image / Image2Video / Policy-DROID variants). **There is no `Cosmos-Reason3` — zero matches across all of HuggingFace.**

| | Cosmos-Reason2-2B (ours) | Cosmos3-Edge | Cosmos3-Nano | Cosmos3-Super |
|---|---|---|---|---|
| `model_type` | `qwen3_vl` | `cosmos3_edge` | `cosmos3_omni` | `cosmos3_omni` |
| Architecture | `Qwen3VLForConditionalGeneration` | `Cosmos3EdgeForConditionalGeneration` | `Cosmos3ForConditionalGeneration` | `Cosmos3ForConditionalGeneration` |
| Pipeline tag | **`image-text-to-text`** | none (diffusers) | none (diffusers) | none (diffusers) |
| Task | VLM reasoning / VQA | text, image, video, audio **and action generation** | same | same |
| Params | **2.44B** | 3.86B | **15.75B** | **64.6B** |
| Weights on disk | **4.88 GB** | 9.13 GB | 34.89 GB | 132.62 GB |

*Why we do not switch:*

1. **Wrong task.** Cosmos3 is the *generation / world-model* line (successor to Cosmos-Predict/Transfer), tagged "text, image, video, audio, and action generation" and built on diffusers. Our benchmark is exact-match VQA on a closed 29-class vocabulary. Cosmos-Reason2 is the only line tagged `image-text-to-text`.
2. **It does not fit the budget, and the budget *is* the thesis.** The smallest Cosmos3 is 3.86B / 9.13 GB; Nano is 15.75B; Super is 64.6B. ⚠️ **"Nano" is a trap — it is 15.75B, 6× our model.** The names are family-relative, not absolute. §3 measures Reason2-2B at 1.47 GiB resident in NF4 on a 6 GB card. A model that cannot train on a 14.6 GiB T4 does not complicate this project, it deletes it.
3. **Tooling risk is the real killer.** Reason2 is Qwen3-VL underneath, so plain `transformers` + `peft` + `bitsandbytes` work — **proven by measurement in Milestone B**. `Cosmos3ForConditionalGeneration` under `library_name: cosmos` + diffusers has no demonstrated QLoRA/DoRA path. Week 3 would become a porting project.
4. **Reason2 is not stale.** `Cosmos-Reason2-32B` was published 2026-04-29 and the whole Reason2 family was updated 2026-04-30. It is the current reasoning line (615k downloads on the 2B).
5. **D3 already covers this.** Every run row must stay on one model. Switching now discards the Milestone B footprint measurement and the entire Milestone D probe.

*What we do instead:* cite Cosmos3 as **concurrent work** in the paper — it is strong evidence that NVIDIA is investing in exactly this space. `Cosmos3-Edge` (3.86B, explicitly edge-targeted) is a legitimate **Week 5+ / Project #2 extension**, not a Week 2 swap. ⚠️ Recheck for a `Cosmos-Reason3` before the write-up; if one lands mid-project it goes in related work, **not** into the benchmark table — changing the base model mid-benchmark invalidates every row already measured.

---

### 2026-08-12

**D11 — The eval split moves `day-validation` → `night-validation`. The dataset's own train/validation split leaks images.** *(All figures measured 2026-08-12 by `scripts/build_train_pool.py`'s leak check and the follow-up analysis — not inferred.)*

*What was found.* The Week 3 training-pool builder ran an image-overlap check written in the expectation that it would print `0`. It printed **235**.

| | |
|---|---|
| Images in `day-train` shards 0–3 | 241 |
| …also present in the frozen day eval split | **235** |
| …with **byte-identical PNGs** (sha256) | **235 of 235** |
| `day-train` images *outside* the eval split | **6 images / 10 rows** |
| Shared `(image, question, answer)` triples | **0 of 560** |
| Distinct images across 13 day shards (1,817 rows) | **276** |

**`day-train` and `day-validation` are two sets of *questions* about one shared pool of ~276 keyframes, not two sets of images.** The answers do not leak. The pixels leak almost totally. There is no usable clean training data inside `day-train`: six images.

*Why night.* Measured: `day ∩ night-validation = 0` images, `day ∩ night-train = 0` images. Day and night are different drives, so the day/night axis is disjoint **by construction** — unlike the dataset's own train/validation labels, which this exercise proved cannot be trusted. Cost: one download. Price paid: the benchmark is now a **day → night domain-shift** evaluation and must be reported as that claim, not as in-domain accuracy.

*What survives.* Benchmark row 1 (35.1% on day) **stands** — it is zero-shot, so nothing leaked; it is simply not comparable to a night row. The shard-0 regression gate stands on the day split (`--shard0-only` forces `--split day`). The 29-class answer vocabulary needed **no change**: night contains **0 out-of-vocabulary answers**. That §5 of `eval-protocol.md` insisted the vocabulary be external to the eval split is precisely what made the eval split replaceable at zero cost.

*Alternative rejected:* re-partitioning the day domain by image. It preserves the in-domain claim but still requires re-freezing the split and re-measuring row 1, and it cannot remove the near-duplicate-frame problem either.

*Three things not to lose:*

1. **`token` is a keyframe id, not a scene id.** §9 and `eval-protocol.md` §4 both call it a scene. A nuScenes drive contributes many near-identical frames ~0.5 s apart. **The mislabel is what hid the leak** — "270 distinct scenes" reads as 270 independent situations, and "train vs validation" reads as a split over them.
2. **§9's ICC = 0.000 was measured under that mislabel, on the retired split.** It says nothing about night, which is **5.73 questions/image over 115 images** — far more clustered than day's 4.14 over 270.
3. **Night is disjoint but not independent.** 115 keyframes are not 115 situations. Consecutive frames of one night drive are nearly the same picture. Quote the clustered interval.

*The check itself is the lesson.* It was a hard failure, not a warning, purely on the principle that this project has twice measured a right number with a wrong mechanism. Had it printed a warning above a successful build, every W3 and W4 row would have trained on its own eval images — and the accuracy number would have looked completely normal.

---

## 8. State log

*(Append-only, newest last.)*

- **2026-07-28** — Planning pass complete. Verified: Cosmos-Reason2-2B exists, is gated (`gated: auto`), built on Qwen3-VL-2B-Instruct, 2.44B params. Verified local env (3060 6 GB, driver 592.27, CUDA 13.1, Python 3.13, no torch, git present). Verified bitsandbytes now officially supports Windows x86-64 + sm_86. Identified the no-bf16-on-Kaggle problem as the project's largest technical risk. Locked decisions D1–D6. Created `docs/plan.md` + `docs/memory.md`. **Nothing installed.**
- **2026-07-28 — Milestone A ✅ PASSED.** Built `.venv` on Python 3.13.0; torch 2.13.0+cu130 from the cu130 index; rest of the stack from `requirements-local.txt`. Wrote `scripts/check_env.py`, `.gitignore`, `requirements-local.txt`; `git init` + first commit. `check_env.py` output: `cuda_available=True`, `device=NVIDIA GeForce RTX 3060 Laptop GPU`, `capability=(8, 6)`, `total_vram=6.00 GiB`, `bf16_native=True`, bitsandbytes 0.50.0 `Linear4bit` nf4 forward pass OK, exit 0. Closed three 🟢 risks (bnb-on-Windows, cu130 gaps, py3.13 gaps); found and pinned one new one (huggingface_hub `<1.0`).
  - `check_env.py` reports **`bf16_supported`** (torch, `including_emulation=False`) *and* **`bf16_native`** (derived from compute capability) as separate lines. Reason: `torch.cuda.is_bf16_supported()` counts *emulated* bf16 and can return `True` on a T4 — which would look like it falsifies the no-bf16 constraint (§5) when the hardware reality is unchanged. **On Kaggle, `bf16_native=False` is the line that matters.**
  - Measured on the 3060 **after CUDA context init**: 5.00 GiB free of 6.00 GiB. The ~1.0 GiB unavailable is other processes *plus torch's own CUDA context*. Relevant to the Milestone B budget: the `< 5.0 GB` target is `max_memory_allocated()`, which excludes the context, so context overhead sits on top of it.
  - Pushed to <https://github.com/AadiPathak23/featherweigh-ai> (public) as commit `d829ff3`; `main` tracks `origin/main`. Git identity `AadiPathak23 / aadipathak2323@gmail.com`. **Milestone A is 5/5 — all steps done.**
- **2026-08-06 — Milestone B ✅ PASSED.** HF account `Aadi58`; license gate accepted. Note: a valid token alone still returned **403** — **gate acceptance and authentication are separate things**. `scripts/infer_local.py` loads Cosmos-Reason2-2B in 4-bit NF4 + double-quant with fp16 compute, and describes the traffic-cop scene correctly (orange vest, red vehicle, motion-blurred white bus). Peak **2.10 GiB**, resident **1.47 GiB**, 57 tokens in 8.2 s (**7–10 tok/s**). Load: 439 s first run (incl. 4.9 GB download), **14.3 s cached**.
  - **Instrumentation bug found and fixed — carries forward.** `max_memory_allocated()` is a high-water mark that never decreases, so reading it at successive checkpoints *without* `reset_peak_memory_stats()` reports every later phase as the earliest spike and makes all phase deltas zero. `vram()` now resets after each read. **Every future VRAM measurement must do this**, or the Week 4/5 benchmark columns will be silently wrong.
  - `transformers` 4.57 deprecates `torch_dtype=` in favour of `dtype=`.
  - Benign Windows warnings, non-blocking: HF cache symlinks unavailable (more disk used), `hf_xet` missing (slower downloads), `triton` missing.
- **2026-08-07 — Milestone C ✅ PASSED. Week 1 complete.** Kaggle account created + phone-verified; notebook on **T4 x2** with internet on, environment **pinned**; `HF_TOKEN` supplied via Add-ons → Secrets. `notebooks/kaggle_smoke_test.ipynb` cloned the repo and ran `scripts/check_env.py` **unmodified** → PASS: `device_count=2`, 2 × Tesla T4, `capability=(7, 5)`, 14.56 GiB each, `bf16_supported=False`, `bf16_native=False`, bitsandbytes 0.50.0 `Linear4bit` nf4 forward pass OK.
  - **The bf16 trap was caught live:** bare `is_bf16_supported()` returned **True** on the T4 while `including_emulation=False` returned **False**. Had the notebook printed only the bare call, the project's central constraint would have looked imaginary. See §5.
  - **Silicon, not software — settled empirically.** Kaggle runs CUDA 12.8 (near-current) and still has no bf16; the local 3060 has it on a different CUDA. The variable is compute capability (sm_75 Turing vs sm_86 Ampere), nothing installable.
  - Kaggle stack: Linux, Python 3.12.13, torch 2.10.0+cu128. bitsandbytes installs cleanly via `pip install -q bitsandbytes` (not in the base image).
  - A token was leaked into a notebook cell and revoked immediately; see the security note in §1.
- **2026-08-10 — Milestone D dataset survey (steps 1–3 of 5).** Ten candidates surveyed and verified against source pages, not recall. Locked D7 (US-collected + image-first), D8 (no self-built dataset), D9 (no gated primaries). Recommendation: `nuscenes-qa-mini`; fallback SUTD-TrafficQA. Full table + verified facts in §6. **Steps 4–5 (download, inspect ~20 examples, record) still outstanding — the pick is not final.**
  - **The survey's main finding is a negative one: the intersection of {US-collected, image-based, ungated, ready-made QA, roadside/smart-city} is empty.** Every candidate fails at least one. This is worth one paragraph in the paper — it is a real gap in the public dataset landscape, not a shortcoming of the search.
  - **Two datasets advertise a size that is annotations only.** DriveLM's 4.86 GB and SurveillanceVQA-589K's 2.31 GB both exclude the images/videos, which come from a separate (and far larger) download. **Always check whether media is in the repo before trusting a size figure.**
  - **`gated` is not one thing.** HF's `gated` flag, a Zenodo request form, and a Google Form with weekly batch review are three different costs. DriveLM reports ungated in search results but shows "You need to agree to share your contact information" on the repo page — check the page, not the summary.
  - RoadSceneVQA (AAAI 2026) was the best domain fit found — roadside, regulation-aware reasoning, image-based, 34,736 QA — with **no locatable public release**. Recheck later; if it lands, it is a strong Week 5+ addition.
- **2026-08-10 — Milestone D step 4: shard downloaded and inspected.** `scripts/inspect_dataset.py` pulls **one** 457 MB Arrow shard (not the full 19.8 GB), reports schema, QA pairs, answer distribution, true storage breakdown, and dumps CAM_FRONT images to `outputs/dataset_peek/`. Measurements in §6.
  - **Step 4 earned its place.** Metadata said "19.8 GB, 6-view images + LiDAR, 29 answer classes" — all true and all misleading. Only downloading revealed **224×224** images and an **int64** storage format. plan.md's *"a dataset that looks right on paper and wrong in practice is the normal failure"* was exactly right.
  - **Published size figures can be ~200× off what you need.** 19.8 GB → ~87 MB after dropping LiDAR + 5 of 6 views and encoding JPEG. **Never reject a dataset on its published size before checking what that size is made of.**
  - **Measure the majority-class baseline before any model runs.** 22.9% here. Without it, a 45% accuracy result reads as success rather than as barely beating "always answer yes".
  - Two instrumentation bugs in the first version of the script, both silent: `json.dumps` byte-counting gave a wrong 10.8× reduction (true: 22.3× waste vs JPEG), and the PIL check failed because images are nested int64 lists, not an HF `Image` feature. **Sizing a column by serializing it is not measuring it** — use `table.column(name).nbytes` and `np.array(...).shape`.
- **2026-08-10 — Milestone D ✅ COMPLETE. Zero-shot probe run; dataset committed.** `scripts/zeroshot_probe.py`: **35.7%** strict vs **22.9%** majority baseline over all 140 rows of `day-validation` shard 0, 1.1 s/example, peak 1.50 GiB. Floor-effect hypothesis **rejected**. Full breakdown in §6.
  - **The headline number hid two opposite results.** Binary yes/no **67.2%** (vs 50% chance) against open-ended **11.4%**. A single blended accuracy would have concealed that 224×224 supports presence judgements but not identity or counting. **Always disaggregate before trusting an aggregate.**
  - **Writing the pass/fail thresholds into the script before running it** (≥10 pp commit, 3–10 pp marginal, <3 pp abort) meant the verdict could not be rationalised after seeing 35.7%. Same discipline Milestone E needs, and cheap to apply.
  - **`strict == lenient` was the check that mattered.** Had lenient run far above strict, the low score would have been a prompt-format problem misdiagnosed as a bad dataset. Scoring only one way would not have distinguished them.
  - Prompt-format lesson: a reasoning-tuned model must be told explicitly to emit *only* the answer, or exact-match scores near zero for reasons unrelated to comprehension. Format compliance still only **72.1%** zero-shot.
- **2026-08-10 — Repo renamed `featherweigh-ai` → `featherweight-ai`.** Done on GitHub by Aadi; local remote re-pointed with `git remote set-url` and verified against `git ls-remote`. Updated the two places that matter: the current-state URL in §1 and the **clone command in `notebooks/kaggle_smoke_test.ipynb`** — a stale clone URL in the Kaggle bridge would have worked silently via GitHub's redirect and then broken whenever the redirect lapsed.
  - The Milestone A state-log entry above still shows the old URL. **Left deliberately** — this log is append-only and that URL was correct on 2026-07-28. GitHub redirects it, so it still resolves.
  - `results/zeroshot_probe.json` records `git_sha` but **not** the remote URL, so no results file needed rewriting. Provenance that captures the commit rather than the hosting location survives a rename.
  - Fixed while cheap: the name is not yet cited in the paper, an HF model card, or the Kaggle notebook's public URL. After those exist, a rename means chasing citations.
- **2026-08-11 — "Cosmos 3" checked against the HF API; base model unchanged (D10).** A Cosmos**3** family does exist (Nano/Super/Edge + Text2Image/Image2Video/Policy), but it is the **generation/world-model line**, not a successor to Cosmos-Reason. **No `Cosmos-Reason3` exists anywhere on HF.** Full comparison table and the five reasons to stay are in §7 D10.
  - **Model family names do not imply size.** `Cosmos3-Nano` is **15.75B params / 34.89 GB** — 6× our 2.44B model. "Nano" is its rank *within the Cosmos3 family*, nothing more. Reading it as "small" would have wasted a session discovering it cannot load.
  - **Newer is a different axis from applicable.** Cosmos3 is newer *and* useless here: wrong task (generation, not `image-text-to-text`), wrong size, and no proven `peft`/`bitsandbytes` path off the Qwen3-VL architecture that Milestone B already validated.
  - Standing rule reaffirmed: **check the API, not recall.** Claude's training cutoff predates the Cosmos3 releases, so every fact in D10 came from `huggingface.co/api/models` queried live.
  - `learning-log.md` restructured — Aadi's Milestone B answers moved out of HTML comments so they render, with `⚠️ Reconcile owed` blocks marking the four wrong mechanisms. **The wrong answers are kept, not deleted**; the gap is the pedagogical content.

- **2026-08-11 — Milestone E ✅ COMPLETE. Eval protocol frozen; benchmark row 1 measured.** `docs/eval-protocol.md` written before any adapter exists. Frozen split = `day-validation` shards 0-7, **1,117 rows**, committed as a manifest with a sha256 per image. Answer vocabulary frozen from `day-train`, **saturated at 29 classes**. `src/eval.py` scored **35.1%** strict vs a **26.3%** baseline (+8.8 pp), reproduced byte-identically across two runs. Full record in §9.
  - **The regression gate earned its place.** Before touching the new split, `src/eval.py --shard0-only` had to reproduce the probe's 35.7 / 35.7 / 22.9 / 67.2 / 11.4 on the same 140 rows — and did, exactly. Refactoring the harness and changing the data in one step would have made any discrepancy unattributable. **Change one thing at a time, and prove it.**
  - **A bigger split did not change the score; it changed the claim.** Strict moved 35.7 → 35.1 (0.6 pp), but delta-over-baseline fell **+12.9 → +8.8 pp** because the majority baseline rose 22.9 → 26.3. **A metric can look stable while the thing you actually assert about it moves by a third.** Always re-check the baseline when the split changes.
  - **Binary accuracy was overstated by 8 pp at n=61.** 67.2% → **59.2%** at n=524. Milestone D's "+17 pp of real signal" is **+9.2 pp**. Nothing about the model changed. **A percentage quoted without its n is not yet a result.**
  - **Scene clustering: feared, instrumented, measured absent.** 4.14 questions/scene made correlated errors plausible, which would have inflated every confidence interval. Measured **ICC = 0.000** over 270 scenes → deff 1.00. Shard 0's ICC of 0.259 was an artifact of near-singleton clusters. Machinery kept for W4. **Build the instrument, then let it decide — do not assume the worry or dismiss it.**
  - **Deriving the answer vocabulary from the eval split was a real bug in the probe**, not a stylistic preference: it made format compliance a moving target and leaked split information into a reported metric. Fixing it moved the number 72.1% → 81.3%, which is the size of an effect that could easily have been mistaken for a finding.
  - **A `⚠️` in a progress message killed a 20-minute build** via Windows cp1252 `UnicodeEncodeError`, *after* every expensive step had succeeded. Fixed structurally with `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` in both entry points. **A logging call must never be able to destroy the run it reports on.**
  - Loading PNGs instead of decoding nested int lists made eval **3.8× faster** (1.1 → 0.33 s/example), so a full 1,117-row run is ~6 min and the two-run determinism check stays cheap for the rest of the project.
  - `hf_xet` installed (optional, download-path only) after repeated `Read timed out` against the HF CDN; recorded in `requirements-local.txt` with the reason.
  - Corrected §6: `CAM_FRONT` values are **uint8**, not int64. The 344 KB/row is nested-list **int32 offset buffers**. The 22.3× conclusion stands, the mechanism did not — **a correctly measured number can still carry a wrong explanation, and only the explanation reaches the paper.**

- **2026-08-11 (end of session) — housekeeping.** Commit convention set: no `Co-Authored-By: Claude` trailer, author stays `AadiPathak23`. Existing 9 commits carrying it are **deliberately left alone** — rewriting them would change every SHA and invalidate the `git_sha` provenance inside `results/*.json`. `learning-log.md` is the only open thread: Milestone B reconciles, the Milestone E follow-up questions, and empty C/D scaffolds, all waiting on Aadi.

- **2026-08-11 — Milestone F ✅ COMPLETE. Week 2 closed.** `src/stability.py` built and both criteria met: a healthy fp16 LoRA run completes with 6 scaler skips and no false alarm; a sabotage run (lr 5.0) is halted at step 7 naming `model.visual.patch_embed.proj`. Full record in §10.
  - **A finite loss is not a healthy run — the single most important thing this milestone found.** The loss never went non-finite in 58 steps across both runs, so `plan.md`'s specified *"halt on non-finite loss"* rule would have caught **nothing**. The sabotage run landed exactly **one** optimizer step, wrecked the adapter with it, then skipped every subsequent step — weights frozen, loss finite and oscillating, progress bar advancing. On Kaggle that is 12 hours and a saved adapter of pure garbage. **What caught it was the consecutive-skip rule**, and that rule only exists because the design started from "how does this differ from the scaler working correctly?" rather than from "what does a broken run look like?"
  - **Instrument the thing you fear, then let it answer.** §5's "the ViT is the overflow site" was inherited from other people's write-ups. Measured here: **13 of 13 overflows across both runs began in the vision tower; the language tower never overflowed alone.** The claim is now this project's own measurement.
  - **...but the mechanism was narrower than the claim.** Forward hooks on all 24 vision blocks logged **zero** non-finite activations. The overflow is in the **backward** pass — ViT *gradients*, not ViT activations. Same shape of error as the §6 uint8/int32-offset correction: the number was right and the stated mechanism was not, and only the mechanism reaches the paper.
  - **Module names were discovered, not assumed, and that decided the milestone.** The vision tower uses `qkv`/`proj`; the language model uses `q_proj`/`k_proj`/`v_proj`/`o_proj`. The tutorial-standard target list would have put LoRA nowhere near the ViT → no vision gradients → a permanently clean vision column on the exact site being watched. `build_trainable()` now hard-fails if trainable vision params = 0. **A blind instrument reporting "fine" is worse than no instrument.**
  - **Training fits locally after all: peak 3.53 GiB** at batch 1 with no gradient checkpointing, on a 6 GB card. `prepare_model_for_kbit_training` costs **+0.59 GiB** — the fp32 upcast of `embed_tokens`, exactly the tensor §3 flagged as 42% of resident weights. Local dry-runs of the Week 3 loop are viable, which §2 had assumed they would not be.
  - Aadi predicted both outcomes correctly (scale falls; a gradient goes non-finite before the loss) — direction right, mechanism not yet written. Reconcile and the three follow-up questions are in `learning-log.md`.

- **2026-08-12 (session close) — Milestone F pushed; Week 2 complete and backed up.** Commit `26173d9` on `origin/main`, author `AadiPathak23`, **no `Co-Authored-By` trailer** — the convention set 2026-08-11 held on its first real use. 11 files, +1,808/−38: `src/stability.py`, both stability results, the `eval-protocol.md` corrections, and the 1,120 → 1,117 sweep.
  - **Verification at close, all five green:** baseline PASS · sabotage PASS (halts, names the module) · no tripwire false alarm on 6 legitimate scaler skips · `build_eval_split.py --verify` 0 sha256 mismatches · `src/eval.py --shard0-only` reproduces 35.7 / 35.7 / 22.9 / 75.7 / 67.2 / 11.4 with **byte-identical per-example records** to the previously committed run. The frozen split and the harness are provably undisturbed by Milestone F.
  - **Working tree clean, `main` tracks `origin/main`.** `outputs/stability_set/` (64 day-train PNGs) is correctly gitignored and regenerable from `build_stability_set()`.
  - **Next session starts at Week 3**, and its first task is **not** training: `requirements-kaggle.txt` does not exist, and Kaggle's torch 2.10.0+cu128 / Python 3.12.13 differs from local 2.13.0+cu130. Cheap to get wrong locally, expensive to discover mid-session against a 12 hr cap.
  - **New capability unlocked this session, easy to forget:** training now fits on the 3060 (3.53 GiB peak of ~5.0 usable). §2 said the local box was inference-only. **W3 should be dry-run locally before any Kaggle hour is spent** — every bug found on the laptop is a GPU-hour kept.

- **2026-08-11 — `eval-protocol.md` corrected (not changed).** Two statements in the frozen document were wrong or unusable. **No protocol rule moved and no run was invalidated** — §9's re-run of the shard-0 regression gate still reproduces 35.7 / 35.7 / 22.9 / 75.7 / 67.2 / 11.4 exactly.
  - **§4's scene-correlation conclusion was falsified by its own measurement.** The section predicted a "substantially larger design effect" on the full split; the measurement returned **ICC = 0.000, deff 1.00**. The document's history on that one quantity runs *mild* (draft, unmeasured) → *not mild* (§4, from shard 0) → *absent* (measured). **The first claim was right, and the correction was more wrong than the thing it corrected — and more confident, because it arrived with a number attached.** Shard 0's ICC of 0.259 was near-singleton clusters manufacturing structure: at 1.33 questions/scene the ANOVA estimator has no within-cluster variance to work with. **A small sample can invent a structure as easily as it can hide one.**
  - **§5 was never wrong — it was missing its `n`, which was enough to make it unusable.** "Format compliance moved 72.1% → 75.7%" compares the *same 140 rows* under two vocabularies, which is the correct comparison. But it reads as a claim about the benchmark column, and the benchmark column is **81.3%** (n=1,117). It was misread that way within a day of being frozen — by the next session, reading it as a contradiction of the committed run. §9's own rule (*a percentage quoted without its n is not yet a result*) applies to the document that states it.
  - Also fixed: `~50 MB` → **85 MB** for `outputs/eval_split/`, and five stale **1,120**s (the planned count) → **1,117** (the measured one) across `results/README.md`, `src/eval.py` and `scripts/build_eval_split.py`. Aadi's learning-log entries keep 1,120 deliberately — they are the historical record of what he predicted at the time.

- **2026-08-12 — Week 3 half built, and stopped by a finding nobody predicted. `src/train.py`, `scripts/build_train_pool.py`, `requirements-kaggle.txt` and `notebooks/kaggle_qlora_train.ipynb` written; `src/eval.py` and `scripts/build_eval_split.py` gained `--split night|day`. No training step has run.**
  - **The leak check fired on its first run and killed the plan it was part of.** It was written expecting to print `0` and printed **235**: of 241 images in `day-train` shards 0–3, 235 also sit in the frozen day eval split, **byte-identical by sha256**, with only 6 images (10 rows) outside it — while **0 of 560 `(image, question, answer)` triples** are shared. `day-train` and `day-validation` are two sets of *questions about the same ~276 keyframes*. **The answers never leaked; the pixels almost entirely did.** Eval moved to `night-validation` (D11, `eval-protocol.md` Amendment 1); day∩night = 0 images, measured both ways.
  - **The check earned its keep by being a hard failure rather than a warning.** Had it printed a warning above a successful build, it would have been scrolled past, and every W3/W4 row would have trained on its own eval images — **with an accuracy number that looked completely normal.** The failures worth instrumenting are the ones that do not announce themselves.
  - **One wrong word hid it for a week: `token` is a KEYFRAME id, not a scene id.** "270 distinct scenes" reads as 270 independent situations; it is 270 frames, and a nuScenes drive contributes many near-identical frames ~0.5 s apart. Under that mislabel, "train vs validation" sounded like a split over situations. §9's ICC = 0.000 was measured over the wrong grouping and does not transfer to night (5.73 questions/image over just 115 images; measured ICC there is **0.024**, deff 1.11, n_eff 593 — small, but no longer zero).
  - 🚨 **The bigger finding: the benchmark's baseline was a straw man, and it flattered every number.** Night zero-shot is **31.9%** against a 24.9% global majority (+7.0 pp). But the **per-question-type prior** — answer `yes` to yes/no questions and `car` to everything else, which needs no image, no training and no understanding, since question type is readable off the question text — scores **31.9%** on night and **33.4%** on day. So the real deltas are **+0.0 pp (night)** and **+1.7 pp (day)**, not +7.0 and +8.8. On night *binary* questions the model scores **51.2%** against a 54.1% trivial constant — **worse than the straw man.** `src/eval.py` now reports `prior_baseline` and `delta_over_prior_pp` on every run. **A baseline that no real system would adopt is not a baseline.**
  - **This does not yet condemn the dataset, and saying so precisely matters.** Zero-shot is a lower bound: format compliance is 80.6% and the model has never seen the task. The decisive experiment is a local ~60 min QLoRA run — free, no Kaggle quota, since Milestone F showed training fits at 3.53 GiB — scored against **31.9%**, not 24.9%. Aadi's call; it is the first thing the next session does.
  - **Measured while building, worth keeping:** sequences are only **~96 tokens** (≈49 vision + ~40 text), not the ~247 vision tokens §3 quotes — `max_pixels` caps 224×224 inputs well below the model's ceiling. Directly relevant to the still-sealed T4 batch-size prediction. Pool decode rate is **74 images/min**, so the 1,817-row pool is ~25 min of CPU.
  - `src/stability.py` took its only change since Milestone F: `build_trainable` gained `lora_alpha` / `lora_dropout` kwargs whose **defaults reproduce F exactly**. Shared rather than copied, because that function holds the §10 hard-fail on zero trainable vision params, and the copy that drifts is the one that goes blind.
  - **Training pool built and clean:** 1,817 rows over 276 images (6.58 questions/image), **0 shared images with the night eval split**, 25.9% majority (`yes`) vs night's 24.9% — close enough that the adapter is not learning a prior mismatched to what it is scored against. `results/train_pool_manifest.jsonl` tracked at 575 KB; 138 MB of PNG in gitignored `outputs/train_pool/`.

- **2026-08-13 — Week 3 loop RUN, and the viability question answered: the dataset can rank methods.** Local QLoRA, 60 min wall-clock budget on the 3060, trained on the day pool and scored on night: **47.2% strict vs a 31.9% per-type prior = +15.3 pp**, McNemar **p = 6.5e-09** (b=98, c=199). Full record in §11.
  - **The gain is perception, not wording — and only the prior baseline could show that.** Format compliance went **80.6% -> 100.0%**, so the adapter captured *all* the vocabulary headroom open question #8 feared. But the per-type prior is **already 100% format-compliant by construction**, so a purely vocabulary-driven model would have converged *on* the prior. It finished +15.3 pp above it. **Open question #8 now has a number: the vocabulary gain is real, fully captured, and worth 0 pp of the reported delta.** Against the old majority baseline the headline would read +22.3 pp and would have been unfalsifiable as to which effect produced it.
  - **Open-ended moved most, which contradicts the standing expectation.** 15.4% -> **31.2%** (+2.5 pp -> +18.3 pp above its trivial constant), while binary went 51.2% -> 66.0% (-3.0 pp -> +11.9 pp). Milestone D called open-ended a near-collapse at 224x224 and treated binary as the part that worked; finetuning bought the most where the least was expected.
  - **The tripwire works inside the training loop, not just in the harness that invented it.** `--lr 5.0` halted `src/train.py` at step 6 on 5 consecutive skips, naming `visual.patch_embed.proj.lora_A` — the same vision-tower signature Milestone F measured 13/13 times. The healthy 60 min run skipped 7 steps while the scaler settled to **1024** (F's exact value) and correctly did not fire.
  - **An early throughput estimate was wrong by 2.6x, and the wall-clock budget absorbed it silently.** The first 18 steps ran at 3.91 s/step, implying ~920 steps for the hour; the run settled to ~0.65 s/step and delivered **2,428**. Warmup steps are not representative of throughput, and a wall-clock budget is exactly the design that does not care — the step count is what gives, and the results file records both.
  - ⚠️ **This row's accuracy is a benchmark number; its wall-clock and VRAM columns are 3060 numbers.** W4's matched-budget comparison must be all-T4. 2,428 steps is not what a T4 hour buys, and `--probe-batch` was deliberately left unrun so that prediction stays sealed for the T4.
  - **All gates green:** shard-0 regression gate reproduces 35.7 / 35.7 / 22.9 / 75.7 / 67.2 / 11.4 exactly; night and train-pool manifests both verify with **0** sha256 mismatches. Nothing in Week 3 disturbed the frozen artifacts.

---

## 9. Milestone E — frozen eval protocol & benchmark row 1 *(2026-08-11)*

Protocol: [`docs/eval-protocol.md`](./eval-protocol.md). Harness: `src/eval.py`. Raw record: `results/eval_zeroshot.json`.

> ⚠️ **RETIRED AS THE BENCHMARK SPLIT, 2026-08-12 (D11).** Everything in this section is a correct measurement **of the day split**, and row 1 is still valid — it is zero-shot, so the leak D11 describes cannot touch it. But the day split is no longer where rows are scored, and a night row must never be compared to these numbers.
>
> **Two words in this section are wrong throughout: "scene" means *keyframe*.** The 270 "distinct scenes" are 270 *frames*, and a nuScenes drive contributes many near-identical frames roughly half a second apart. That mislabel is what allowed `day-train` vs `day-validation` to read as a split over independent situations when it is a split over questions about the same pictures. **The ICC = 0.000 below was therefore measured over the wrong grouping**, and it says nothing about the night split, which is 5.73 questions/image over only 115 images.

### The frozen split

| Item | Value |
|---|---|
| Split | `day-validation` shards 0–7, **1,117 rows** (shards 0–4 hold 140, shards 5–7 hold 139) |
| Frozen by | `results/eval_split_manifest.jsonl` — sha256 per image; `--verify` re-checks |
| Pixels | `outputs/eval_split/`, **85 MB lossless PNG**, gitignored and regenerable |
| Scenes | **270** distinct `token`s → **4.14 questions/scene** |
| Reserve | shards 8–15 deliberately never downloaded or scored |
| Answer vocabulary | **29 classes**, from `day-train`, saturated (+24, +4, +1, +0, +0 over 5 shards) |

### Benchmark row 1 — zero-shot Cosmos-Reason2-2B, 4-bit NF4

| Metric | Value |
|---|---|
| **Strict exact-match** | **35.1%**  95% CI [32.4, 37.9] |
| Majority baseline | **26.3%** (always `yes`) |
| **Delta over baseline** | **+8.8 pp** |
| Lenient | 35.2% — 1 example above strict |
| Format compliance (in-vocab) | 81.3% |
| Binary yes/no | **59.2%** (n=524, chance 50%) |
| Open-ended | **13.8%** (n=593) |
| Throughput | 0.33 s/example, 370 s total, peak VRAM **1.50 GiB** |
| Determinism | ✅ two runs, **all 1,117 raw outputs byte-identical** |

### What the bigger split changed — the point of Milestone E

| | shard 0 (n=140) | frozen split (n=1,117) |
|---|---|---|
| Strict | 35.7% | 35.1% |
| Majority baseline | 22.9% | **26.3%** |
| **Delta over baseline** | **+12.9 pp** | **+8.8 pp** |
| Binary | 67.2% (n=61) | **59.2%** (n=524) |
| Open-ended | 11.4% (n=79) | **13.8%** (n=593) |
| Questions/scene | 1.33 | **4.14** |

**Accuracy barely moved; the claim did.** Headline accuracy fell 0.6 pp, but the delta over baseline fell by a third because the baseline rose 3.4 pp. Milestone D's "+17 pp of real binary signal" is really **+9.2 pp**. Nothing about the model changed between those two numbers — only n. **Shard 0 was not a representative sample of `day-validation`.**

### Scene clustering — feared, measured, absent

Questions cluster 4.14-per-scene, so correctness *could* have been correlated, inflating confidence. Measured: **ICC = 0.000** across 270 scenes (one-way ANOVA estimator, non-positive → clamped), design effect **1.00**, n_eff = 1,117. **The naive CI stands.**

Shard 0 reported ICC = 0.259, an artifact of near-singleton clusters (1.33/scene); the full-split estimate is the trustworthy one. The machinery stays in `src/eval.py` — adapters in W4 may induce correlation that the zero-shot model does not.

⚠️ **McNemar assumes independent pairs.** If clustering ever becomes non-zero, its p-values are optimistic by roughly the design effect, and a scene-level cluster bootstrap is required for any W4 comparison near p ≈ 0.05.

### Error decomposition (725 errors)

| Cause | n | Share |
|---|---|---|
| Prediction **outside** the vocabulary | 209 | 28.8% — format/synonym loss |
| Prediction **in** vocabulary but wrong | 516 | **71.2% — genuine misperception** |

Milestone D put the format share at 43% (39/90). At 8× the sample it is **28.8%**. The vocabulary-alignment concern (open question #8) is real but **smaller than Milestone D estimated** — most of the headroom is perception, not wording. Most common wrong predictions: `no` (128), `stopped` (88), `yes` (84), `taxi` (55).

---

## 10. Milestone F — fp16 stability harness *(2026-08-11)*

Harness: `src/stability.py`. Records: `results/stability_baseline.json`, `results/stability_sabotage.json`.
Both runs on the 3060, LoRA r=8, batch 1, 64 `day-train` rows, **no gradient checkpointing**.

### Result — both success criteria met

| Run | Outcome | Exit |
|---|---|---|
| baseline (lr 1e-4, 50 steps) | completed, all finite, **6 skipped steps correctly not treated as divergence** | PASS |
| sabotage (lr 5.0) | **tripwire halted it at step 7**, naming `model.visual.patch_embed.proj` | PASS |

The harness is self-verifying: exit 0 only when the outcome matches what the mode expects. **A `--sabotage` run that completes cleanly is a failed milestone**, because a tripwire that misses deliberate divergence would miss the real thing.

### 🚨 The most important finding — a finite loss is not a healthy run

**The loss never went non-finite. Not once, in 58 steps across both runs.**

`plan.md` §5-F specified the tripwire as *"halt on non-finite loss or grad-norm"*. **The loss half of that rule would have caught nothing.** The sabotage run:

| step | loss | what happened |
|---|---|---|
| 0–1 | 3.84, 0.85 | scaled-gradient overflow, steps skipped — the scaler working normally |
| 2 | 0.02 | scale reaches 16384, **the only step that ever lands**; lr=5.0 wrecks the adapter |
| 3–7 | 36.0, 72.9, 35.9, 72.7, 36.0 | every step skipped. Weights frozen at the wreckage; loss just oscillates by example |

Exactly **one** optimizer step landed in the entire run. Afterwards nothing changed — the loss is not diverging, it is **stuck**. No exception, no warning, a finite loss, and a run that would have advanced happily for 12 hours on Kaggle, saved an adapter, and burned quota to produce the step-2 wreckage.

**What caught it was the consecutive-skip rule** (5 in a row), which exists only because the design had to separate *"the scaler is working"* from *"the scaler has given up"*. Both look identical at any single step.

### ✅ The ViT-overflow claim is now measured on this model, not inherited

§5 has asserted since day one that "the ViT tower is the usual overflow site", on the authority of other people's write-ups. Measured here:

| | overflows starting in vision | language-only |
|---|---|---|
| baseline (6 skips) | **6 / 6** | **0** |
| sabotage (7 skips) | **7 / 7** | **0** |

The language tower **never** overflowed without the vision tower overflowing first. Sabotage named `model.visual.patch_embed.proj` — the patch embedding, the first module to touch a pixel.

⚠️ **But the mechanism is narrower than "the ViT overflows".** Forward hooks on all 24 vision blocks recorded **zero** non-finite activations in either run. The overflow is in the **backward pass** — it is the ViT's *gradients*, not its activations. Anything mitigating this (open question #4, the fp32-vision ladder step) must target gradient magnitude, not forward numerics.

### Measured numbers worth carrying to Week 3

| Fact | Value |
|---|---|
| **Peak VRAM, training** | **3.53 GiB** — batch 1, no gradient checkpointing, on a 6 GB card |
| `prepare_model_for_kbit_training` cost | **+0.59 GiB** resident (1.47 → 2.06 GiB) |
| Trainable params (LoRA r=8) | 4,411,392 — 3,211,264 language + **1,200,128 vision** |
| Throughput | ~1.0 s/step local; the 50-step baseline ran in 58 s |
| Scaler settling point | **1024** (from 65536, six halvings, all within the first 17 steps) |

**The +0.59 GiB is `embed_tokens`,** as §3 predicted it would be: `prepare_model_for_kbit_training` upcasts every non-4bit param to fp32, and that tensor is 311.2M params = 42% of resident weights. Measured, not discovered by an OOM.

### ⚠️ The trap that would have made this harness blind

LoRA target modules were **discovered from the model, not assumed**. The vision tower names its attention projections **`qkv` / `proj`**; the language model uses `q_proj` / `k_proj` / `v_proj` / `o_proj`. Every QLoRA tutorial lists the second set.

Targeting only those would have attached **no adapter anywhere near the ViT** → zero vision-tower gradients → a permanently empty, permanently reassuring vision column, on the exact site the milestone exists to watch. `build_trainable()` now **refuses to start** if the trainable vision-param count is zero, because a blind instrument that reports "fine" is worse than no instrument.

### Mitigation ladder — decided in advance, not under pressure

1. lower `--init-scale` — overflow in the first 0–2 steps
2. `--fp32-vision` — vision norms go non-finite before language norms *(the case actually observed)*
3. lower `--lr` — loss climbs steadily before going non-finite
4. `--clip-grad` — norms spike intermittently but recover

All four are implemented flags, so the response to a Week 3 divergence is a command line, not a redesign.

---

## 11. Week 3 — the QLoRA run, and the answer to the viability question *(2026-08-13)*

Training: `src/train.py`, record `results/train_qlora.json`. Scoring: `src/eval.py --split night --adapter`, record `results/eval_qlora_night.json`.

### 🎯 The finding — the dataset can rank methods after all, and the gain is perception, not wording

| Night split, n=659 | zero-shot | **QLoRA (1 h, day-trained)** | change |
|---|---|---|---|
| **strict exact-match** | 31.9% | **47.2%**  95% CI [43.4, 51.0] | **+15.3 pp** |
| global majority (`always yes`) | 24.9% | 24.9% | — |
| **per-question-type prior** | 31.9% | 31.9% | — |
| **delta over the prior** | **+0.0 pp** | **+15.3 pp** | **the whole result** |
| binary (n=303, trivial 54.1%) | 51.2% → **−3.0 pp** | 66.0% → **+11.9 pp** | +14.8 pp |
| open-ended (n=356, trivial 12.9%) | 15.4% → +2.5 pp | 31.2% → **+18.3 pp** | +15.8 pp |
| format compliance | 80.6% | **100.0%** | +19.4 pp |
| scene ICC / deff / n_eff | 0.024 / 1.11 / 593 | 0.011 / 1.05 / 626 | clustering stays negligible |

**Paired McNemar: p = 6.5×10⁻⁹** (b=98, c=199, χ²=33.67) — significant, and measured on the same 659 examples.

### Why this is a perception result and not a vocabulary result

This is exactly what the prior baseline was built to decide, one day earlier.

Format compliance went **80.6% → 100.0%**: the adapter captured *all* of the vocabulary headroom that open question #8 worried about. If the gain were only vocabulary alignment, the score would have converged on the per-type prior — because **the prior is already 100% format-compliant by construction**; it only ever emits in-vocabulary answers. Instead the model finished **+15.3 pp above it**.

**The delta over the prior is therefore net of the vocabulary effect.** Had we kept quoting the majority-class baseline, the headline would read +22.3 pp and would have been unfalsifiable as to which effect produced it. Open question #8 is now answered with a number: **the vocabulary gain is real, fully captured, and worth 0 pp of the reported delta.**

The largest movement is where the least was expected: **open-ended, 15.4% → 31.2%**, from +2.5 pp above its trivial constant to **+18.3 pp**. Milestone D called open-ended a near-collapse at 224×224 and it is where finetuning bought the most.

### Measured run facts worth carrying to Kaggle

| Fact | Value |
|---|---|
| Optimizer steps in the 60 min budget | **2,428** — 1 full epoch over 1,817 rows |
| Throughput | **1.48 s/step** average; ~0.65 s/step steady-state (the first ~20 steps are ~4 s and are not representative) |
| Loss | 2.0963 (first 10) → **0.4854** (last 10) |
| Scaler | settled at **1024**, 7 skips total, all early — Milestone F's exact value |
| Peak VRAM | **3.60 GiB** at batch 1, no gradient checkpointing |
| Tripwire | did not fire; correctly ignored the 7 settling skips |

⚠️ **This row's accuracy is a benchmark number; its wall-clock and VRAM columns are 3060 numbers.** `eval-protocol.md` §7 and the wall-clock rule both say device-bound quantities are only comparable within a device. **Every W4 matched-budget row must come off a T4**, and this run's 2,428 steps are not the step count a T4 hour will buy.

⚠️ **This is a domain-shift row: trained on day, scored on night.** It must be reported as that claim.

### The tripwire, proven inside the training loop

`python -m src.train --max-steps 12 --lr 5.0 --warmup 1 --grad-accum 1` halted at step 6 on 5 consecutive skips, naming `base_model.model.model.visual.patch_embed.proj.lora_A.default.weight`. Same signature Milestone F measured: **the overflow starts in the vision tower**. Record: `results/train_tripwire_check.json`. Milestone F proved the tripwire works; this proves `src/train.py` actually consults it.
