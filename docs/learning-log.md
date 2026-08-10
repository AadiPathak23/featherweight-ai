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

<!-- your reasoning, in your own words: why that bucket and not the others
so the thing was if the back propogation comes into play and training and all that comes that the model pushes the output back into the training log so the memory peak would increase by 1-1.5 gb ad that would disrupt everything, that is one issue that would come.-->


**What actually happened:**

<!-- Measured: peak 2.10 GiB, resident weights 1.47 GiB. Your BUCKET was right.
     But read your reasoning above again — it was about backprop and training.
     This run had no backward pass at all. The peak came from somewhere else.
     In your own words: where did the 2.10 GiB actually come from? -->


**What I understand now:**

<!-- Three things worth reaching for, in your own words:
     1. Why does peak VRAM happen at LOAD rather than during generation?
     2. Activations cost +0.01 GiB here. Why will that number be completely
        different during training, and what changes?
     3. max_memory_allocated() is a high-water mark. What breaks if you read it
        at several checkpoints without reset_peak_memory_stats() in between? -->


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
