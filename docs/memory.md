# featherweight-ai — Memory / Context Log

> **Re-read this file first** to regain context in a new session. Companions: [`plan.md`](./plan.md) (roadmap; Week 2 = Milestones D–F) and [`learning-log.md`](./learning-log.md) (Aadi's own-words notes).
>
> 🧭 **How we work — read before doing anything.** Aadi is deliberately working above his current level and wants to *learn*, not receive finished code. Per milestone: **frame** briefly → **Aadi writes a prediction before anything runs** → **build in small pieces**, explaining each non-obvious decision where it appears → **reconcile** prediction vs reality, digging hardest where he was wrong → **Aadi logs it in his own words**. Label each thing **Tier 1** (learn deeply: LoRA/DoRA math, quantization, fp16 stability, experiment design), **Tier 2** (know the shape: HF/PEFT APIs), or **Tier 3** (plumbing: venv, git, Kaggle UI). No black boxes — every flag and magic number gets a reason. Say plainly what is measured fact vs. estimate vs. untested bet. Where he can reason it out, ask and wait. One milestone per session; depth over throughput.
> Everything below traces to a command output or URL captured on the stated date — nothing asserted from recall.
> Last updated: 2026-08-10

---

## 1. START HERE — session resume

### 🔖 Where we left off — end of session 2026-08-10

**Milestone D is complete. The dataset question is settled.**

Last session ran the Week 2 dataset survey end to end. Ten candidates checked against their real pages (not recall), then the winner verified by **running the model**, not by reading a dataset card.

- **Dataset chosen:** [`KevinNotSmile/nuscenes-qa-mini`](https://huggingface.co/datasets/KevinNotSmile/nuscenes-qa-mini) · fallback SUTD-TrafficQA · full record in **§6**
- **Verified by measurement:** zero-shot Cosmos-Reason2-2B scored **35.7%** strict exact-match vs a **22.9%** majority-class baseline (+12.9 pp) on 140 validation rows. The dataset can rank methods. It is not a floor.
- **New decisions locked:** **D7** (US-collected, image before video), **D8** (we do not build our own dataset), **D9** (gated datasets disqualified as primary) — all in §7
- **Repo renamed** `featherweigh-ai` → `featherweight-ai`; remote and Kaggle clone URL both updated
- **`results/` convention established** — tracked run records with `git_sha`; see `results/README.md`

**The two things last session found that are easy to forget and expensive to rediscover:**

1. **The images are 224×224**, not nuScenes' native 1600×900. Binary yes/no questions score **67.2%** (chance 50%) while open-ended ones collapse to **11.4%**. The blended 35.7% hides this completely — *always disaggregate*.
2. **Much of the coming finetuning gain will be output-vocabulary alignment, not better perception.** 39 of 90 zero-shot errors are the model answering sensibly in the wrong words (`bike`→`bicycle`, `zero`→`0`). Carry format-compliance as its own benchmark column so the two effects stay separable, and do not report the jump as a perception win.

### ▶️ Next action — Milestone E, freeze the eval protocol

**Do this first, before any adapter exists to tempt a post-hoc metric choice.** Write down and freeze: the answer-extraction rule, the prompt template, decoding params (greedy, `do_sample=False`), `max_new_tokens`, and the eval split. Then build `src/eval.py`.

- **Start from `scripts/zeroshot_probe.py`** — it is already most of the skeleton. Strict/lenient scoring, the normalize rule, the results schema and greedy decoding are all in place and working.
- **Already banked:** re-running the probe reproduced **35.7 / 22.9 / 72.1 exactly**, so Milestone E's "identical across two runs" criterion is demonstrated for this harness.
- **Still to decide:** how to freeze a proper eval split (the probe used only shard 0 of `day-validation`, 140 of ~2,229 rows), and whether to report binary and open-ended as separate columns rather than one blended accuracy.

Then **Milestone F** — fp16 stability harness (NaN/Inf tripwire, loss-scale monitoring, per-module vision-tower grad-norms).

### ⚠️ Owed

- **`learning-log.md` has no entries for Milestones B, C or D.** The reconcile step of the working protocol below has not been closed since 2026-07-28. Milestone B's prediction bucket (2–3.5 GB) was right but the *reasoning* was about backprop, while the real peak came from load-time unpacking — that gap is the entry worth writing.

### 📁 Repo map

| Path | What it is |
|---|---|
| `scripts/check_env.py` | Milestone A. Runs unmodified on both local and Kaggle; reports `bf16_supported` vs `bf16_native` separately |
| `scripts/infer_local.py` | Milestone B. 4-bit load + single-image inference. **Source of the `vram()` reset discipline — reuse it** |
| `scripts/inspect_dataset.py` | Milestone D. Pulls ONE 457 MB shard, reports schema/QA/storage; dumps images to `outputs/dataset_peek/` |
| `scripts/zeroshot_probe.py` | Milestone D go/no-go, and the **Milestone E skeleton** |
| `results/` | **Tracked** run records, small JSON only. Schema + rules in `results/README.md` |
| `outputs/` | **Gitignored** scratch — images, dumps, anything regenerable |
| `notebooks/kaggle_smoke_test.ipynb` | Milestone C. Thin launcher: clones the repo, runs `check_env.py` |

**Repo:** <https://github.com/AadiPathak23/featherweight-ai> — public. Renamed 2026-08-10 from `featherweigh-ai` (old name was missing the `t`); GitHub redirects the old URL but nothing in the repo relies on that. `gh` CLI is **not** installed; pushes use stored HTTPS credentials. Git identity `AadiPathak23 / aadipathak2323@gmail.com`.

### 🗓️ Milestone board

| | Milestone | Status |
|---|---|---|
| **A** | Local env | ✅ 2026-07-28. Venv, pinned stack, `check_env.py` passes on the 3060 |
| **B** | 4-bit local inference | ✅ 2026-08-06. Peak **2.10 GiB** vs <5.0 GB target; **7–10 tok/s** (the Week 5 edge-target figure) |
| **C** | Kaggle bridge | ✅ 2026-08-07. T4 x2; `check_env.py` runs unmodified from a clone; **bf16 confirmed absent in hardware** |
| **D** | Dataset | ✅ 2026-08-10. `nuscenes-qa-mini`, verified by probe (§6) |
| **E** | Eval protocol | ⬜ **NEXT** |
| **F** | fp16 stability harness | ⬜ |
| W3+ | QLoRA run, DoRA/LoRA, edge latency, write-up | ⬜ |

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
| Image storage | nested **`int64`** lists (H,W,3) — *not* encoded JPEG/PNG. 8 bytes per value that needs 1. |
| Waste factor | **22.3×** — 343.9 KB/row as stored vs **15.5 KB/row** as JPEG q92 |
| **Real size** | **~87 MB** front-camera-only JPEG for the whole dataset, vs 19.8 GB published |
| Distinct answers (shard) | 24 (paper claims 29 overall) |
| **Majority-class baseline** | **22.9%** (`yes`). `yes`+`no` = **43.6%** of all answers — the benchmark is largely binary. |
| Scenes | Boston confirmed visually in the sample (brick rowhouses, US crosswalk markings) |

**Size is a non-issue and always was.** 19.8 GB → ~87 MB after dropping LiDAR + 5 of 6 views and encoding as JPEG. Preprocess once, attach as a Kaggle Dataset.

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

| Question type | n | Accuracy | Reference |
|---|---|---|---|
| **Binary (yes/no)** | 61 | **67.2%** | chance = 50% → **+17 pp of real signal** |
| **Open-ended** | 79 | **11.4%** | near-collapse |

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
