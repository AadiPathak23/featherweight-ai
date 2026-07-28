# featherweight-ai — Project Plan

> Living document. Update as decisions land. Companion file: [`memory.md`](./memory.md) (state + environment facts).
> Last updated: 2026-07-28

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

### Week 1 — foundations *(detailed)*

Three ordered milestones. Each is small and independently verifiable.

---

#### Milestone A — Local environment sanity check

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

#### Milestone B — Load Cosmos-Reason2-2B in 4-bit locally, single-image inference

**Goal:** prove the exact model we're finetuning runs on 6 GB, so local iteration is real, not aspirational.

**Steps**
1. **Accept the license gate** (blocking, ~30 s): log in to huggingface.co → open `nvidia/Cosmos-Reason2-2B` → *Agree and access repository*. Gate is `auto`, so access is granted immediately.
2. Create an HF token → `huggingface-cli login`. Same token later goes into Kaggle Secrets.
3. `scripts/infer_local.py` — `Qwen3VLForConditionalGeneration.from_pretrained(...)` with:
   ```python
   BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=torch.float16,   # fp16, not bf16 — match Kaggle
   )
   ```
   - **Critical for 6 GB:** cap vision tokens via the processor's `min_pixels`/`max_pixels` (start low, e.g. `max_pixels ≈ 256*28*28`). Unbounded image resolution is what OOMs you — not the weights.
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

#### Milestone C — Kaggle account + one trivial GPU notebook

**Goal:** demystify the cloud environment *before* anything depends on it. Deliberately trivial — this is a confidence milestone.

**Steps**
1. Create a free Kaggle account → **complete phone verification**. GPU + internet stay locked until you do. This is the #1 first-timer stumble.
2. New Notebook → right sidebar → **Accelerator: GPU T4 x2** → **Internet: On**.
3. Add HF token via **Add-ons → Secrets**. Verify retrieval with `UserSecretsClient` (not a hardcoded cell).
4. Run ~10 lines: `!nvidia-smi`, then torch version, device name, `get_device_capability()`, and **`torch.cuda.is_bf16_supported()`** — empirically confirms the bf16 constraint rather than trusting analysis.
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

### Weeks 2+ *(outline)*

- **W2** — Metropolis dataset survey → shortlist of 2–3 (criteria in §8); lock eval protocol & metrics; build the fp16-stability harness (loss-scale monitoring, NaN tripwire)
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

### 🚨 No bfloat16 on free Kaggle
T4 = Turing (sm_75), P100 = Pascal (sm_60). **Neither has hardware bf16.** The Cosmos model card says BF16 is required. Therefore every training run uses **fp16**: `bnb_4bit_compute_dtype=torch.float16`, fp16 AMP + GradScaler, fp32 master weights. Qwen-VL-family QLoRA on T4 is widely done, but **the ViT tower is the usual overflow site**. Mitigation harness is a Week 2 deliverable.

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

**Blocking Milestone B (needs user, ~2 min)**
1. Accept the `nvidia/Cosmos-Reason2-2B` license gate on HF + create an access token

**Resolve during Week 1**
2. transformers 4.5x vs 5.x — starting on 4.5x; revisit as a deliberate upgrade
3. fp16-only training — confirmed empirically by Milestone C step 4. If unstable in W3: fp32 vision tower, tighter loss scaling, or a free Ada-class tier (L4 has bf16) as a *supplementary* runner. Kaggle stays primary.
4. Vision-tower quantization policy — quantize the ViT or keep it fp16? Affects accuracy *and* the VRAM story. Decide with data in W4.

**Week 2**
5. Metropolis dataset shortlist (§8)
6. Eval metric definition — reasoning-trace outputs need either exact-match-on-final-answer or a judged protocol. Choice affects reproducibility claims.
