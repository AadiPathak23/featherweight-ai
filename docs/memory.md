# featherweight-ai — Memory / Context Log

> **Re-read this file first** to regain context in a new session. Companion: [`plan.md`](./plan.md) (roadmap).
> Everything below traces to a command output or URL captured on the stated date — nothing asserted from recall.
> Last updated: 2026-07-28

---

## 1. Current state

**Phase:** Week 1, **Milestone B ✅ COMPLETE** (2026-08-06). Cosmos-Reason2-2B loads in 4-bit on the 6 GB 3060 and produces a correct, on-topic description of a traffic scene. Peak **2.10 GiB** vs the <5.0 GB target. Latency **7–10 tok/s** (the edge-target figure for Week 5). Details in §3.

**Next action:** Milestone C — Kaggle account + phone verification, T4 x2 notebook, HF token in Secrets, `git clone` + run `check_env.py` unmodified. See `plan.md` §5.

**Milestone A ✅ COMPLETE** (2026-07-28). Venv built, full local stack installed, `scripts/check_env.py` passes on the 3060, committed and pushed.

**Repo:** <https://github.com/AadiPathak23/featherweigh-ai> — public. ⚠️ **The repo name is missing the `t`** ("featherweigh"), while the project, the local folder, and the docs all say *featherweight*. Not yet renamed; GitHub redirects after a rename, so the fix stays cheap until the URL is cited in the paper or an HF model card. `gh` CLI is **not** installed; pushes use stored HTTPS credentials.

**Next action:** Milestone B — accept the `nvidia/Cosmos-Reason2-2B` license gate + create an HF token (**blocking, needs user**), then `huggingface-cli login` and write `scripts/infer_local.py`. See `plan.md` §5.

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
| GPUs offered | **T4 x2** (16 GB each, sm_75) or **P100** (16 GB, sm_60) |
| Quota | ~**30 GPU-hr/week**, **12 hr** max session |
| Choice | **T4 x2.** P100 is a trap: sm_60 has no fp16 tensor cores, and `LLM.int8()` requires sm_75+ |
| **bf16** | ❌ **Not supported in hardware on either.** See §5. |
| Account status | Not yet created (Milestone C) — **phone verification required** to unlock GPU + internet |

### Constraints

- **$0 budget.** No paid cloud, no bought hardware. Student.
- Local 6 GB → inference + dry-runs only.
- GitHub is the local↔Kaggle bridge.
- User is **new to cloud notebooks** → Kaggle steps need to be beginner-explicit.

---

## 3. Model facts — `nvidia/Cosmos-Reason2-2B` *(verified 2026-07-28)*

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

| Risk | Severity | Notes / mitigation |
|---|---|---|
| **No bf16 on free Kaggle** | 🔴 **Highest** | T4 = sm_75, P100 = sm_60 → no hardware bf16, but the model card says BF16 required. All training must be **fp16** (`bnb_4bit_compute_dtype=torch.float16`, AMP + GradScaler, fp32 master weights). **The ViT tower is the usual overflow site.** Stability harness = Week 2 deliverable. Confirmed empirically in Milestone C step 4. |
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

## 6. Key decisions

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

---

## 7. State log

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
