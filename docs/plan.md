# featherweight-ai — Project Plan

> Living document. Update as decisions land. Companions: [`memory.md`](./memory.md) (state + measured facts — **authoritative**) and [`learning-log.md`](./learning-log.md) (Aadi's own-words notes).
> Last updated: 2026-08-07 · **Week 1 complete; Week 2 is next (§5, Milestones D–F).**

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

#### Milestone D — Metropolis dataset shortlist

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

#### Milestone E — Eval protocol *(the one that decides defensibility)*

**Goal:** a frozen, written definition of how accuracy is computed — locked before any training run exists to be tempted by.

**Why it matters:** the model emits reasoning traces, not labels. "Accuracy" is therefore a *choice*, and choosing it after seeing results is how benchmarks become dishonest. This is open question #6.

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

#### Milestone F — fp16 stability harness

**Goal:** detect divergence early and automatically, **before** Week 3 depends on it. This is insurance against the project's highest risk.

**Steps**
1. Log per step: loss, grad-norm, and the live `GradScaler` scale factor.
2. **NaN/Inf tripwire** — halt immediately on non-finite loss or grad-norm, and dump the step index plus the offending module. A run that silently NaNs and trains to completion on garbage is the expensive failure.
3. Watch the scaler: repeated halving means recurring overflow. Log every scale change rather than only the final value.
4. Instrument the **vision tower separately** — it is the documented overflow site; per-module grad-norms make that visible instead of inferred.
5. Prepare the mitigation ladder in advance so it isn't invented under pressure: lower initial scale → fp32 vision tower → lower LR → gradient clipping.
6. **Reuse `vram()` from `scripts/infer_local.py`**, including its `reset_peak_memory_stats()` — without the reset, phase deltas silently read zero and the Week 4/5 VRAM columns become wrong.

**✅ Success criterion**
A short deliberately-unstable run (inflated LR) trips the tripwire and halts with a useful diagnostic, rather than producing quiet garbage.

**Risks**
- Harness itself perturbs timing → keep logging cheap; wall-clock is a benchmark column and must stay honest
- Tripwire too sensitive → occasional scaler halving is *normal*; alert on sustained patterns, not single events
- fp16 turns out to be unfixable → escalation path in open question #3: fp32 vision tower, tighter scaling, or a supplementary bf16-capable free tier with Kaggle still primary

---

### Weeks 3+ *(outline)*

- **W3** — first end-to-end QLoRA run on Kaggle; checkpoint/resume across the 12 hr cap; eval harness green
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

## 8. Week 2 dataset survey criteria

Shortlist **2–3** candidates, Metropolis-aligned (smart-city / traffic / surveillance / industrial-safety video VQA). For each, report:

- **Exact size** — flag anything over ~a few GB (must be feasible to download and use on free Kaggle)
- **License** — must permit research use *and* redistribution of results
- **Access method** — flag anything gated or application-required
- **Format / annotation type** — must suit VQA-style eval of a VLM
- **Iteration speed** — small enough to iterate quickly

**Bias strongly toward the smallest viable set.** Narrative fit and zero-cost feasibility beat dataset size and leaderboard prestige.

---

## 9. Open questions

**✅ Resolved**
1. ~~Accept the license gate + create a token~~ — done 2026-08-06. Learned: **gate acceptance ≠ authentication**; a valid token alone still 403s.
3. ~~Is fp16-only training really forced?~~ — **Yes, settled on live hardware 2026-08-07.** T4 is sm_75 Turing; bf16 tensor cores arrived with Ampere (sm_80). Kaggle runs CUDA 12.8 and still lacks it, so it is silicon, not software. *Escalation if W3 proves unstable:* fp32 vision tower, tighter loss scaling, or a free Ada-class tier (L4 has bf16) as a **supplementary** runner — Kaggle stays primary.

**Still open**
2. transformers 4.5x vs 5.x — on 4.57.6; upgrade is a separate, individually-verified task. Note `huggingface_hub` is pinned `<1.0` **because** of this, so the two move together.
4. Vision-tower quantization policy — quantize the ViT or keep it fp16? Affects accuracy *and* the VRAM story. Decide with data in W4.
6. **Eval metric definition** — exact-match on final answer vs. judged protocol. Now Milestone E, and **must be frozen before any training run exists to tempt post-hoc choices**.
7. **New (from Milestone B):** `embed_tokens` is 311.2M params left in fp16 ≈ 622 MB = **42% of resident weights**, tied to `lm_head`. Is quantizing or shrinking the embedding table a legitimate lever for the edge story, or does it wreck quality? Nobody benchmarks this. Potentially a genuine contribution — worth a W4/W5 ablation.

**Week 2**
5. Metropolis dataset shortlist (§8) → Milestone D.
