# featherweight-ai — Project Plan

> Living document. Update as decisions land. Companions: [`memory.md`](./memory.md) (state + measured facts — **authoritative**) and [`learning-log.md`](./learning-log.md) (Aadi's own-words notes).
> Last updated: 2026-08-11 · **Weeks 1 and 2 COMPLETE.** D ✅ dataset · E ✅ eval protocol frozen, benchmark row 1 = 35.1% · F ✅ fp16 tripwire, both criteria met. **Next: Week 3, first end-to-end QLoRA run on Kaggle.**

---

## 1. Goal & thesis

Parameter-efficient, **quantization-aware** finetuning (QLoRA + DoRA) of **`nvidia/Cosmos-Reason2-2B`** for edge deployment, benchmarked against standard LoRA and frozen baselines under **matched VRAM and wall-clock budgets**.

**Thesis:** for edge-class vision-language reasoning, quantization-aware PEFT reaches competitive accuracy at a fraction of the memory and training cost — and full finetuning is not merely expensive but *out of reach* on free-tier hardware, which is itself the point.

Project #1 of a planned arc. Later phases: multi-LoRA edge serving (#2), a standardized efficiency benchmark (#3). Structure everything to extend.

### Locked decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Full-FT is not a run row.** Replaced by same-model frozen baselines: zero-shot Cosmos-Reason2-2B + prompt-tuning (and/or linear probe). | Honest floor, actually runnable on free tier. |
| D2 | **Full-FT appears as a labeled analytical estimate** — VRAM + cost computed from parameter counts, citing published figures. Marked *"not run — exceeds the free-tier/edge budget by design."* | Keeps the comparison in the paper without faking a run. |
| D3 | **No proxy model.** Every run row stays on Cosmos-Reason2-2B. | Same-model rows are what make the budget-matched claim apples-to-apples. A smaller substitute would break it. |
| D4 | Full-FT infeasibility is framed as **supporting evidence for the edge-efficiency thesis**, not a limitation. | It is the finding, not a gap. |
| D5 | Week 1 is **dataset-agnostic**. Dataset survey happens Week 2, restricted to **Metropolis-aligned domains** (smart-city / traffic / surveillance / industrial-safety video VQA). Explicitly **not** robotics/embodied. | Narrative fit with the Metropolis target. |
| D10 | **Base model locked to Cosmos-Reason2-2B.** "Cosmos 3" is the generation/world-model line, not a newer Cosmos-Reason; no `Cosmos-Reason3` exists. Cosmos3-Edge/Nano/Super are 3.86B / 15.75B / 64.6B and diffusers-based. | Wrong task, wrong budget, unproven PEFT path. Full record + table in [`memory.md`](./memory.md) §7. Cosmos3 goes in *related work*; `Cosmos3-Edge` is a Project #2 candidate. |

---

## 2. Deliverables

- **Benchmark table** — accuracy × peak VRAM × latency × wall-clock × training cost across finetuning methods
- **Released adapter weights** on Hugging Face
- **Short paper** (mini-paper / tech report)
- **Reproducible Kaggle notebooks** — anyone can re-run the whole thing for $0

---

## 3. Hardware & budget reality

| | Local (dev/test) | Cloud (training) |
|---|---|---|
| Hardware | RTX 3060 Laptop, **6 GB** (~5.5 usable) | Kaggle **T4 x2** (16 GB each, sm_75) |
| Role | 4-bit inference, tiny dry-runs, latency measurement as the *edge target* | All real training runs |
| Limits | Cannot train a VLM adapter to completion | ~30 GPU-hr/week, 12 hr max session |
| Cost | — | **$0. No paid cloud, no bought hardware.** |

**Constraint that shapes everything:** T4 (sm_75) and P100 (sm_60) have **no hardware bfloat16**. All training is fp16. See §7.

---

## 4. Workflow — the local→GitHub→Kaggle loop

```
  laptop (VS Code)          GitHub              Kaggle notebook
  ───────────────           ──────              ───────────────
  write / edit code    →    git push      →     !git clone <repo>
  4-bit smoke test                              run training (T4 x2)
  read results         ←    git pull      ←     push artifacts / download
```

Rules:
- Code lives in git. Notebooks are **thin launchers** that clone the repo and call into `src/` — never the place logic lives.
- HF token goes in Kaggle **Add-ons → Secrets**. Never hardcoded in a cell, never committed.
- Every run writes its config + measured metrics to a results file, committed back.
- Checkpoint to survive the 12 hr session cap (Week 3 concern).

---

## 5. Roadmap

### Week 1 — foundations ✅ **COMPLETE (2026-08-07)**

Three ordered milestones, all passed. Detail kept below as a record — the documented risks and how they actually resolved are worth preserving.

| Milestone | Result |
|---|---|
| **A** — local env | ✅ Passed 2026-07-28. All three "🟢" risks closed by evidence: bitsandbytes installs plainly on Windows, cu130 had cp313 wheels, whole stack built on Python 3.13. Both fallbacks unused. |
| **B** — 4-bit local inference | ✅ Passed 2026-08-06. Peak **2.10 GiB** vs <5.0 GB target; correct traffic-scene answer; **7–10 tok/s**. |
| **C** — Kaggle bridge | ✅ Passed 2026-08-07. `check_env.py` runs unmodified from a clone on T4 x2. **bf16 confirmed absent in hardware.** |

Measured numbers live in [`memory.md`](./memory.md) §2–3; that file is authoritative over anything restated here.

---

#### Milestone A — Local environment sanity check ✅

**Goal:** a reproducible venv where torch sees the 3060 and bitsandbytes actually quantizes on it.

**Steps**
1. `py -3.13 -m venv .venv`, activate. *(3.13 is fine — all pinned libs support `>=3.10`. Kaggle's Python is separate → two requirements files, not one shared pin.)*
2. torch from the **cu130** index: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`
3. Pin the rest in `requirements-local.txt`:
   - `transformers>=4.57,<5` — **deliberately not 5.x**; every Cosmos/Qwen3-VL example targets 4.5x
   - `peft`, `bitsandbytes`, `accelerate`, `datasets`, `qwen-vl-utils`, `pillow`, `huggingface_hub[cli]`
   - defer `trl` to the training milestone
4. `scripts/check_env.py` — prints torch version, `torch.version.cuda`, `cuda_available`, device name, compute capability, total VRAM, `is_bf16_supported()`, then allocates a `bnb.nn.Linear4bit` on CUDA and runs a forward pass.
5. `git init` + `.gitignore` (`.venv/`, `__pycache__/`, `*.safetensors`, `outputs/`, `.env`) + first commit + push to GitHub.

**✅ Success criterion**
`python scripts/check_env.py` prints `cuda_available=True`, `device=NVIDIA GeForce RTX 3060 Laptop GPU`, `capability=(8, 6)`, `total_vram≈6.0 GB`, `bf16_supported=True`, and completes a `Linear4bit` forward pass with **no `CUDA Setup failed` / DLL error**. Repo pushed to GitHub.

**Risks**
- bitsandbytes DLL load failure on Windows → run `python -m bitsandbytes` diagnostics; check MSVC runtime; fall back to cu128 torch (bnb's newest Windows wheels top out at CUDA Toolkit 13.0)
- cu130 index packaging gaps → fall back to cu128 (fine for sm_86)
- Python 3.13 wheel gap on a long-tail dep → fall back to the installed 3.10
- Version thrash → pin now, record resolved versions in `memory.md` immediately

---

#### Milestone B — Load Cosmos-Reason2-2B in 4-bit locally, single-image inference ✅

**Goal:** prove the exact model we're finetuning runs on 6 GB, so local iteration is real, not aspirational.

**Steps**
1. **Accept the license gate** (blocking, ~30 s): log in to huggingface.co → open `nvidia/Cosmos-Reason2-2B` → *Agree and access repository*. Gate is `auto`, so access is granted immediately.
2. Create an HF token → `hf auth login` (`huggingface-cli` is deprecated in `huggingface_hub` ≥0.34). Same token later goes into Kaggle Secrets. **Note: a valid token is not enough — gate acceptance is separate, and a token alone returns 403.**
3. `scripts/infer_local.py` — `Qwen3VLForConditionalGeneration.from_pretrained(...)` with:
   ```python
   BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=torch.float16,   # fp16, not bf16 — match Kaggle
   )
   ```
   - **Critical for 6 GB:** cap vision tokens via the processor's `min_pixels`/`max_pixels`. ⚠️ **`256*28*28` was wrong** — that is Qwen2-VL geometry. Qwen3-VL uses patch 16 + merge 2 = **1024 px per vision token**. Derive it at runtime from `processor.image_processor.patch_size`/`.merge_size`; never hardcode.
   - *Measured outcome:* capping barely mattered at inference — 247 vision tokens cost **+0.01 GiB**. The activation risk is real but lands in **training**, where activations are retained for the backward pass.
   - `max_new_tokens` ~128–256 for a smoke test (the card suggests 4096 for full reasoning traces).
4. Run on one image with a simple scene-reasoning prompt. Log `torch.cuda.max_memory_allocated()`.
5. **Close Epic Games Launcher and other GPU consumers first** — 426 MiB is gone before we even start.

**✅ Success criterion**
Model loads in 4-bit, emits a coherent on-topic answer about the image, **peak allocated VRAM < 5.0 GB**, no OOM. Peak + latency recorded in `memory.md`.

**Risks**
- *Does 4-bit fit in 6 GB?* 2.44B params at NF4 ≈ **1.4–1.6 GB** + ViT tower + unquantized layers. Comfortable **if** vision tokens are capped. The card's "24 GB" is for bf16 + long video context — not applicable here. Evidence it works: `embedl/Cosmos-Reason2-2B-W4A16` already exists on HF (also: cite as related work).
- Card says Linux-only/untested elsewhere → expect friction; image inference should be fine, video decoding may not be
- `library_name: cosmos` → validate the plain-`transformers` path first; pull NVIDIA's package only if the chat template demands it
- Quantizing the ViT hurts perception more than quantizing the LLM → Week 4 ablation (exclude ViT from quantization)
- Gate rejection (unlikely) → fall back to ungated `Qwen/Qwen3-VL-2B-Instruct`; methodology survives, NVIDIA narrative doesn't

---

#### Milestone C — Kaggle account + one trivial GPU notebook ✅

**Goal:** demystify the cloud environment *before* anything depends on it. Deliberately trivial — this is a confidence milestone.

**Steps**
1. Create a free Kaggle account → **complete phone verification**. GPU + internet stay locked until you do. This is the #1 first-timer stumble.
2. New Notebook → right sidebar → **Accelerator: GPU T4 x2** → **Internet: On**.
3. Add HF token via **Add-ons → Secrets**. Verify retrieval with `UserSecretsClient` (not a hardcoded cell).
4. Run ~10 lines: `!nvidia-smi`, then torch version, device name, `get_device_capability()`, and **`torch.cuda.is_bf16_supported(including_emulation=False)`** — empirically confirms the bf16 constraint rather than trusting analysis.
   - 🚨 **The bare call is a trap and was caught live:** `is_bf16_supported()` returned **True** on the T4 (PyTorch emulates bf16 via fp32), while `including_emulation=False` returned **False**. Always pass the argument, or the project's central constraint looks imaginary.
5. **Test the bridge:** `!git clone <repo>` and run `scripts/check_env.py` unmodified. This is the actual thing being validated.
6. Locate the GPU-hour meter so the 30 hr/week budget is visible from day one.

**✅ Success criterion**
A saved notebook that prints two T4s from `nvidia-smi`, reports `capability=(7, 5)` and `bf16_supported=False`, reads the HF token from Secrets, and runs `check_env.py` cloned from GitHub.

**Risks**
- Phone verification gates GPU + internet → do it first
- Quota: 30 hr/week, weekly reset; a forgotten idle session burns it → stop sessions explicitly
- T4 x2 is *two* devices; naive code sees `cuda:0` only → fine now, matters Week 3
- **P100 trap:** 16 GB in one device looks appealing, but sm_60 has no fp16 tensor cores and `LLM.int8()` needs sm_75+ → default to T4 x2
- `/kaggle/working` is lost on teardown beyond the save → checkpoint strategy is a Week 3 concern

---

### Week 2 — dataset, protocol, stability *(next — detailed)*

Character shift: Week 1 was plumbing with binary pass/fail. Week 2 is **judgment**, and the decisions made here determine whether the benchmark is defensible. Three milestones, ordered by what blocks what.

---

#### Milestone D — Metropolis dataset shortlist ✅ **COMPLETE 2026-08-07→10**

**Result:** 10 candidates surveyed; **`KevinNotSmile/nuscenes-qa-mini`** chosen, SUTD-TrafficQA as fallback. Confirmed by a zero-shot probe (**35.7%** vs a **22.9%** majority baseline) rather than by metadata. Full record and the accepted weaknesses live in [`memory.md`](./memory.md) §6.

**Goal:** 2–3 candidate datasets meeting the §8 criteria, with a recommendation and a stated fallback.

**Steps**
1. Survey Metropolis-aligned VQA datasets — smart-city / traffic / surveillance / industrial-safety. **Not** robotics/embodied (D5).
2. For each candidate report: exact size · license (must allow research use **and** redistribution of results) · access method (flag gated/application-required) · annotation format · iteration speed.
3. Verify each claim against the actual dataset page or paper. **No sizes or licenses from recall** — this is the single most common place to get burned.
4. Download the smallest candidate and eyeball ~20 examples. A dataset that looks right on paper and wrong in practice is the normal failure.
5. Recommend one, name a fallback, record both in `memory.md` with the reasoning.

**✅ Success criterion**
A written shortlist with verified size/license/access per candidate, one recommendation, one fallback, and ~20 examples actually inspected.

**Risks**
- Nothing fits → widen to generic urban-scene VQA and lean harder on the *method* narrative than the domain one
- Everything good is gated or application-walled → check access **before** falling in love with a dataset
- Too large for free Kaggle → bias to the smallest viable set, per §8; subsample rather than abandon
- Annotations are captions, not QA pairs → either reformulate into VQA or drop it; decide deliberately, not silently

---

#### Milestone E — Eval protocol ✅ **COMPLETE 2026-08-11**

**Result:** [`docs/eval-protocol.md`](./eval-protocol.md) frozen before any adapter exists. `src/eval.py` is the harness for every benchmark row. Frozen split = `day-validation` shards 0–7, **1,117 rows**, committed as a manifest with a sha256 per image. Answer vocabulary frozen from `day-train`, **saturated at 29 classes**.

**Benchmark row 1 — zero-shot, 4-bit NF4: 35.1% strict** vs a **26.3%** majority baseline (**+8.8 pp**), 95% CI [32.4, 37.9], format compliance 81.3%, binary 59.2% (n=524), open-ended 13.8% (n=593). **Two runs, all 1,117 raw outputs byte-identical** — success criterion met. Full record in [`memory.md`](./memory.md) §9.

**What the larger split changed:** accuracy moved only 0.6 pp (35.7 → 35.1), but delta-over-baseline fell **+12.9 → +8.8 pp** because the baseline rose 22.9 → 26.3, and binary accuracy fell **67.2% → 59.2%** as n went 61 → 524. Shard 0 was not a representative sample. **This is what Milestone E was for.**

**Goal:** a frozen, written definition of how accuracy is computed — locked before any training run exists to be tempted by.

**Why it matters:** the model emits reasoning traces, not labels. "Accuracy" is therefore a *choice*, and choosing it after seeing results is how benchmarks become dishonest. This is open question #6.

**What Milestone D already settled — start from here, do not re-derive:**

- **The judge-vs-exact-match question is closed.** The answer space is a **closed 29-class vocabulary**, so exact-match on a normalized answer works. No judge, no cost, no nondeterminism.
- **`scripts/zeroshot_probe.py` is the working skeleton.** Normalize rule, strict/lenient scoring, greedy decoding and the results schema are built and validated.
- **Determinism is already demonstrated** — two runs reproduced 35.7 / 22.9 / 72.1 exactly. The §E success criterion is half-met before E starts.
- **Keep scoring strict AND lenient.** They came out identical (35.7% both), which is what proved the low open-ended score was genuine blindness rather than format refusal. Collapsing to one number destroys that diagnostic.
- **Report format compliance (in-vocabulary %) as its own column.** 72.1% zero-shot. Without it, vocabulary learning masquerades as perception gain — see `memory.md` §6.
- **Always store the majority-class baseline beside the metric.** 22.9% here.

**Two things E must decide that D deliberately left open:**

1. **Freeze a real eval split.** The probe used only shard 0 of `day-validation` (140 of ~2,229 rows). Pick the split, write it down, never touch it in training.
2. **Blended or disaggregated accuracy?** Binary yes/no scores **67.2%** while open-ended collapses to **11.4%**. If the five methods cannot be separated on open-ended questions, report the two as **separate benchmark columns** rather than one average that hides the split.

**Steps**
1. Decide the answer-extraction rule: exact-match on a final answer (constrained output format) vs. a judged protocol. Trade-off: exact-match is cheap and reproducible but punishes correct answers that are formatted oddly; judging is fairer but needs a judge, which costs money or reproducibility.
2. Fix the **prompt template**, decoding parameters (greedy, `do_sample=False`) and `max_new_tokens`. These are part of the protocol — changing them mid-benchmark invalidates comparisons.
3. Fix the eval split and **freeze it**. Never evaluate on anything touched during training.
4. Write `src/eval.py` implementing exactly this, plus a results schema (JSON per run: config + metrics + timings + git SHA).
5. Baseline it: run zero-shot Cosmos-Reason2-2B through the harness end to end. That is benchmark row 1 and it validates the harness before any adapter exists.

**✅ Success criterion**
`src/eval.py` produces a scored zero-shot number on the frozen eval split, written to a results file, reproducible across two runs with identical output.

**Risks**
- Protocol drift → freeze it in writing, in the repo, and treat changes as a documented decision
- Reasoning traces defeat exact-match → constrain the output format in the prompt and report the extraction-failure rate as its own metric
- Judged protocol adds cost/nondeterminism → if used, pin the judge and publish the judge prompt
- Eval too slow to iterate → subsample a fixed dev subset for iteration, keep the full split for final numbers only

---

#### Milestone F — fp16 stability harness ✅ **COMPLETE 2026-08-11**

**Result:** `src/stability.py`. Both criteria met — a healthy fp16 LoRA run completes with 6 scaler skips and **no false alarm**; a sabotage run (lr 5.0) is **halted at step 7** naming `model.visual.patch_embed.proj`. The harness is self-verifying: exit 0 only when the outcome matches the mode's expectation. Full record in [`memory.md`](./memory.md) §10.

> 🚨 **Step 2 of this plan was wrong, and the milestone found it.** It specified *"halt immediately on non-finite loss or grad-norm"*. **The loss never went non-finite** — not once in 58 steps across both runs. The sabotage run landed exactly **one** optimizer step, wrecked the adapter with it, then skipped every subsequent step: weights frozen, loss finite and oscillating between ~36 and ~73, progress bar advancing. A loss-only check would have let that train for 12 hours on Kaggle and save a garbage adapter. What caught it was the **consecutive-skip rule**, which only exists because the design started from *"how does this differ from the scaler working correctly?"*
>
> **Also measured, not inherited:** **13 of 13 overflows began in the vision tower**, the language tower never overflowing alone — §7's ViT claim is now this project's own measurement. But forward hooks logged **zero** non-finite *activations*, so it is the ViT's **gradients**, not its activations.
>
> **And training fits locally after all:** peak **3.53 GiB** at batch 1 with no gradient checkpointing, on the 6 GB 3060. §3 assumed local training was impossible; local dry-runs of the W3 loop are viable.

**Goal:** detect divergence early and automatically, **before** Week 3 depends on it. This is insurance against the project's highest risk.

**Steps**
1. Log per step: loss, grad-norm, and the live `GradScaler` scale factor.
2. **NaN/Inf tripwire** — halt immediately on non-finite loss or grad-norm, and dump the step index plus the offending module. A run that silently NaNs and trains to completion on garbage is the expensive failure.
3. Watch the scaler: repeated halving means recurring overflow. Log every scale change rather than only the final value.
4. Instrument the **vision tower separately** — it is the documented overflow site; per-module grad-norms make that visible instead of inferred.
5. Prepare the mitigation ladder in advance so it isn't invented under pressure: lower initial scale → fp32 vision tower → lower LR → gradient clipping.
6. **Reuse `vram()` from `scripts/infer_local.py`**, including its `reset_peak_memory_stats()` — without the reset, phase deltas silently read zero and the Week 4/5 VRAM columns become wrong.

**✅ Success criterion — MET.** A short deliberately-unstable run (inflated LR) trips the tripwire and halts with a useful diagnostic, rather than producing quiet garbage.

**Risks — how they actually resolved**
- Harness itself perturbs timing → ✅ non-issue. 24 vision forward hooks + per-group norms cost ~1.0 s/step total; logging is not measurable against the forward pass.
- Tripwire too sensitive → 🎯 **this was the real design problem, and the plan called it correctly.** The baseline run skipped **6 of 50** steps and had to not halt; the sabotage run skipped **5 consecutively** and had to halt. Both are "the scaler skipped a step" and **no single-step check can separate them** — the rule has to be about persistence.
- fp16 turns out to be unfixable → ✅ not triggered. A sane LR is stable; the scaler settles at **1024** from 65536 within 17 steps. The escalation path is implemented as flags (`--init-scale`, `--fp32-vision`, `--lr`, `--clip-grad`) so a W3 divergence is answered with a command line, not a redesign.
- ⚠️ **Unlisted risk that nearly made the harness blind:** LoRA target modules. The vision tower names its projections `qkv`/`proj`; the language model uses `q_proj`/`k_proj`/`v_proj`/`o_proj`. The tutorial-standard list attaches **nothing** to the ViT, giving zero vision gradients and a permanently clean vision column on the exact site being watched. `build_trainable()` now hard-fails if trainable vision params = 0.

---

### Weeks 3+ *(outline)*

- **W3** — first end-to-end QLoRA run on Kaggle; checkpoint/resume across the 12 hr cap; eval harness green. **Import the tripwire from `src/stability.py`** — §5-F shows its failure mode is invisible to a loss check. `requirements-kaggle.txt` still does not exist and is the one untested piece (torch 2.10.0+cu128 / py3.12.13 vs local 2.13.0+cu130).
- **W4** — DoRA + LoRA runs under matched VRAM/wall-clock budgets; seed variance; ViT-quantization ablation
- **W5** — latency/VRAM measurement on the 3060 as the *edge target*; adapter merge + 4-bit inference profiling
- **W6+** — write-up, release adapters, repo polish; hooks for Project #2 (multi-LoRA serving)

---

## 6. Benchmark design

| Run row | Base | Precision | Status |
|---|---|---|---|
| Zero-shot | Cosmos-Reason2-2B | 4-bit NF4 | run |
| Prompt-tuning / linear probe | Cosmos-Reason2-2B | frozen | run |
| LoRA | Cosmos-Reason2-2B | fp16 | run |
| **QLoRA** | Cosmos-Reason2-2B | 4-bit NF4 | run |
| **DoRA (4-bit)** | Cosmos-Reason2-2B | 4-bit NF4 | run |
| Full finetuning | Cosmos-Reason2-2B | — | **estimated only — not run (D2)** |

Measured per row: **accuracy** · **peak VRAM** · **wall-clock** · **inference latency / tokens-per-sec** · **adapter size on disk** · **training cost ($0, but GPU-hours)**.

Budget matching: runs are compared at equal wall-clock *and* equal peak VRAM, not equal epochs. That's the whole point.

---

## 7. Key technical constraints

### 🚨 No bfloat16 on free Kaggle — ✅ **CONFIRMED ON LIVE HARDWARE 2026-08-07**
T4 = Turing (sm_75), P100 = Pascal (sm_60). **Neither has hardware bf16.** The Cosmos model card says BF16 is required.

Measured on a Kaggle T4: bare `is_bf16_supported()` → **True** (fp32 emulation), `including_emulation=False` → **False**, `capability=(7, 5)`. Kaggle runs CUDA **12.8** and still has no bf16, while the local sm_86 3060 has it on a different CUDA — so this is **silicon, not software**, and no upgrade fixes it. Therefore every training run uses **fp16**: `bnb_4bit_compute_dtype=torch.float16`, fp16 AMP + GradScaler, fp32 master weights. Qwen-VL-family QLoRA on T4 is widely done, but **the ViT tower is the usual overflow site**. Mitigation harness is a Week 2 deliverable.

### transformers 4.5x vs 5.x
PyPI latest is **5.14.1**, a major release with breaking changes. Every Cosmos/Qwen3-VL example targets `>=4.57`. We pin `>=4.57,<5` deliberately. Upgrading to 5.x is a separate, individually-verified task — **the pin is not staleness**.

### The model is gated
`nvidia/Cosmos-Reason2-2B` has `gated: auto`. Unauthenticated file access is refused. One click to accept; token needed locally *and* in Kaggle Secrets.

---

## 8. Week 2 dataset survey criteria — **revised 2026-08-10**

> Survey executed 2026-08-10. Results, verified facts and the recommendation live in [`memory.md`](./memory.md) §6, which is authoritative. Criteria below were revised *during* the survey after decisions D7–D9 landed.

**Hard requirements** (in elimination order — 1 killed the most candidates):

1. **Ungated + openly licensed.** A gated dataset breaks deliverable #4 ("anyone can re-run for $0"). Reproducibility argument, not convenience (D9).
2. **Ready-made QA pairs.** No dataset construction — that is a second claim and dilutes the problem statement (D8).
3. **Image-based** before video. Video is Week 4+ (D7).
4. **Small enough for free Kaggle**, counting *media*, not just annotations.
5. **US/North-America collected**, traffic/urban domain — preferred, but never at the cost of 1–3 (D7).

*Nice-to-have:* published baselines, so a wrong harness is visible rather than silent.

**Bias strongly toward the smallest viable set.** Zero-cost feasibility and reproducibility beat dataset size and leaderboard prestige.

⚠️ **Two traps found the hard way, both now standing rules:**
- **A stated size may be annotations only.** DriveLM (4.86 GB) and SurveillanceVQA-589K (2.31 GB) both exclude their media. Confirm the images/videos are actually in the repo before trusting a size.
- **Check the dataset page, not the search summary, for gating.** DriveLM reads as ungated in search results and gates on the page.

---

## 9. Open questions

**✅ Resolved**
1. ~~Accept the license gate + create a token~~ — done 2026-08-06. Learned: **gate acceptance ≠ authentication**; a valid token alone still 403s.
3. ~~Is fp16-only training really forced?~~ — **Yes, settled on live hardware 2026-08-07.** T4 is sm_75 Turing; bf16 tensor cores arrived with Ampere (sm_80). Kaggle runs CUDA 12.8 and still lacks it, so it is silicon, not software. *Escalation if W3 proves unstable:* fp32 vision tower, tighter loss scaling, or a free Ada-class tier (L4 has bf16) as a **supplementary** runner — Kaggle stays primary.

**Still open**
2. transformers 4.5x vs 5.x — on 4.57.6; upgrade is a separate, individually-verified task. Note `huggingface_hub` is pinned `<1.0` **because** of this, so the two move together.
8. **From Milestone D, resized by E:** how much of the finetuning gain is **output-vocabulary alignment** rather than improved perception? At n=140 the format share of errors looked like 43%; at n=1,117 it is **28.8%** (209 of 725). Real, but **smaller than feared — 71.2% of errors are genuine misperception**, so most of the headroom is perception after all. Format compliance stays its own column; quantify the split once adapters exist.
9. **From Milestone D, updated by E:** open-ended accuracy is **13.8%** (n=593), binary **59.2%** (n=524). If the five methods cannot be separated on open-ended questions, report binary and open-ended as separate columns. Decide with W3 data, not now.
10. **New (from Milestone E):** scene clustering measured **ICC = 0.000** for the zero-shot model, so the naive CI is valid today. Do **adapters** induce scene-level correlation that the base model lacks? `src/eval.py` records `scene_icc` per run, so this answers itself in W4 — and if it turns non-zero, McNemar p-values need a cluster bootstrap.
4. Vision-tower quantization policy — quantize the ViT or keep it fp16? Affects accuracy *and* the VRAM story. Decide with data in W4.
7. **New (from Milestone B):** `embed_tokens` is 311.2M params left in fp16 ≈ 622 MB = **42% of resident weights**, tied to `lm_head`. Is quantizing or shrinking the embedding table a legitimate lever for the edge story, or does it wreck quality? Nobody benchmarks this. Potentially a genuine contribution — worth a W4/W5 ablation.

**✅ Resolved (cont.)**
6. ~~Eval metric definition — exact-match vs judged protocol~~ — **done 2026-08-11, Milestone E.** Exact-match on a normalized answer against a **frozen 29-class vocabulary derived from `day-train`**; no judge, no cost, no nondeterminism. Frozen in [`eval-protocol.md`](./eval-protocol.md) before any adapter existed. Learned: the harder half of this question was not *which* metric but *which rows and which vocabulary* — deriving the answer set from the eval split was a genuine leak, and fixing it moved format compliance 72.1% → 81.3%, an effect easily large enough to be mistaken for a finding.
5. ~~Metropolis dataset shortlist (§8)~~ — **done 2026-08-10, Milestone D.** `nuscenes-qa-mini`, fallback SUTD-TrafficQA. Learned: the intersection of {US-collected, image-based, ungated, ready-made QA, roadside} is **empty** in the public landscape — that is a reportable finding, not a failed search.
