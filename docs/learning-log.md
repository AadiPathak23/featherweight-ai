# Learning log

> Aadi's own-words notes. Append-only, newest at the bottom.
>
> **Rule: write it in your own words.** Not copied from chat, not polished.
> If you can't write the entry, you don't have the concept yet — that's the
> signal to go back and ask, not to move on.
>
> Different from [`memory.md`](./memory.md): that file records *what is true*
> (versions, measurements, decisions). This one records *what I understand* —
> including the things I got wrong, which are the most useful entries here.

**Format per entry:** what I predicted → what actually happened → what I now
understand that I didn't before. 3–5 lines is plenty.

**Convention:** Aadi's writing is plain text. Prompts still to be answered stay
in `<!-- HTML comments -->`. A `> ⚠️ Reconcile owed` block is Claude flagging a
first answer that turned out to be wrong — the corrected version is Aadi's to
write, and **the wrong answer is never deleted**, because seeing the gap between
the two is the point of this file.

---

## Milestone A — local environment (2026-07-28)

*Mostly Tier 3 plumbing (venv, pip, git), done before this log existed. Nothing
owed here. The one idea worth keeping: `bf16_supported` and `bf16_native` are
different questions, because torch counts emulated bf16 and the hardware
doesn't do it.*

---

## Milestone B — 4-bit inference on the 3060

**Prediction (committed 2026-07-28, before any code was run):**

`torch.cuda.max_memory_allocated()` peak → **2 – 3.5 GB**

*My reasoning at the time:*

> so the thing was if the back propogation comes into play and training and all
> that comes that the model pushes the output back into the training log so the
> memory peak would increase by 1-1.5 gb ad that would disrupt everything, that
> is one issue that would come.

**What actually happened:**

Measured: peak **2.10 GiB**, resident weights **1.47 GiB**. The bucket was right.

*My answer (2026-08-10) — where did the 2.10 GiB come from:*

> The peak came from normal training and loading the dataset and this is
> supposed to happen.

> ⚠️ **Reconcile owed.** Milestone B had no training (no optimizer, no backward
> pass, `inference_mode()`) and no dataset — it ran on one JPEG,
> `assets/sample_traffic.jpg`. Dataset loading costs host RAM, not VRAM.
> The real source: the weights ship as a ~4.9 GB **fp16** file and must become
> 1.47 GiB of NF4, so bitsandbytes converts layer by layer and briefly holds
> **two representations of the same layer at once**. That transient is the
> ~0.6 GiB above resident.
>
> The prediction reasoning above was directionally right about backprop — it was
> just applied to a run that had no backprop in it. That gap is the entry.

**What I understand now:**

*1. Why does peak VRAM happen at LOAD rather than during generation?*

> Because VRAM is the driver when the LOAD occurs and not during generation when
> we are loading stuff in the transformar that is where peak VRAM is used.

> ⚠️ **Reconcile owed** — this restates the question rather than giving a
> mechanism. The mechanism: **at load, two copies of each layer exist at once
> (fp16 source + NF4 output); at generation, one copy plus scraps** (247 vision
> tokens = +0.01 GiB, 57-token KV cache = +0.01 GiB). Consequence worth keeping:
> if this model ever OOMs, it OOMs at load, before generating a single token.

*2. Activations cost +0.01 GiB here. Why will that be different during training?*

> because in training it will enter the same loop again and again not just once
> that is why the operational cost will be changed.

> ⚠️ **Reconcile owed — fix this one first; Week 3 depends on it.** Repeating a
> loop does *not* accumulate memory; each iteration frees the last one's
> activations. Milestone B's own generation looped 57 times and stayed flat.
> The real reason is **lifetime, not repetition**: backward needs each layer's
> input activation to compute that layer's weight gradient (chain rule), so the
> whole forward pass must stay alive instead of being freed layer by layer.
> One layer alive → all layers alive, times batch × sequence length. Then AdamW
> adds 2 fp32 moments per *trainable* param on top — which is exactly why QLoRA
> fits and full-FT (D1) does not.

*3. `max_memory_allocated()` without `reset_peak_memory_stats()` in between?*

> A read write error will occur and the chache memory will get filled and the
> model will start to hallucinate so the reset peak memory stat function is
> critical.

> ⚠️ **Reconcile owed.** Nothing breaks at runtime — it is a passive counter,
> reading it allocates nothing and cannot affect model output; hallucination is
> unrelated. What breaks is the **measurement**: it is a high-water mark, so
> after a 2.10 load peak, generation still reports 2.10 and its delta reads 0.00.
> No error, no warning, and the numbers look plausible — a silent instrumentation
> bug. It matters because peak VRAM is a benchmark column: without the reset,
> all five Week 4 rows report the same load spike. Conclusion was right, reason
> was not.

---

## Milestone C — Kaggle bridge (2026-08-07)

*Mostly Tier 3 plumbing, but one Tier 1 idea hides in it.*

**What actually happened:**

<!-- The same check_env.py ran unmodified on both machines and reported
     different, correct facts. And the bf16 trap fired live. -->


**What I understand now:**

<!-- The one that matters: torch.cuda.is_bf16_supported() returned True on the
     T4 and False with including_emulation=False. In your own words —
     why does torch say True when the hardware cannot do it, and what would
     have happened to this project if the notebook had only printed the
     bare call? -->


---

## Milestone D — dataset (2026-08-10)

*No prediction was committed before this one — worth noticing why that
happened, and whether it should have.*

**What I decided, and why:**

<!-- You made two calls that changed the project's direction:
     1. You killed the build-your-own-dataset plan on scope grounds.
        Write the argument in your own words — why does a second contribution
        WEAKEN a paper rather than strengthen it?
     2. You insisted on image before video. What did that ordering actually
        buy us, concretely, in this session? -->


**What surprised me:**

<!-- Candidates: the dataset's real images are 224x224 despite nuScenes being
     1600x900; the published 19.8 GB is ~87 MB of stuff we need; the blended
     35.7% hid binary 67.2% vs open-ended 11.4%. Which of these would you have
     caught by reading the dataset card? -->


**What I understand now:**

<!-- Reach for the general rule, not the specific fact:
     1. Why is a majority-class baseline (22.9%) worth computing BEFORE any
        model runs?
     2. Why were the pass/fail thresholds written into the script before it
        ran, rather than judged after seeing 35.7%?
     3. The model answers 'bike' when the gold answer is 'bicycle'. Finetuning
        will fix that fast. Why is that a REPORTING problem and not a win? -->
