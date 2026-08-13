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


---

## Milestone E — eval protocol (2026-08-11)

**Prediction (write this BEFORE the harness runs):**

Zero-shot strict exact-match on the frozen **1,120-row** split will be **25%**,
versus the **35.7%** measured on shard 0 alone (n=140).

*My reasoning (committed 2026-08-11, before the harness ran):*

> 25% it will be lower since it is zero shot

> 📌 Noted at prediction time, not after: the question was why *1,120 rows* would
> differ from *140 rows of the same split, same model, same protocol*. "It is
> zero-shot" is true of both, so it explains the absolute level (~36% rather than
> ~90%) but not the change. The prediction stands as written; what it is missing
> is a claim about whether shard 0 is representative of day-validation.

**Second prediction — the one that actually decides the benchmark:**

*Asked: if QLoRA and LoRA land 2 pp apart, can this benchmark separate them, and
under what test?*

> I cant tell them apart and dont know how to answer this one

> 📌 Honest, and the right answer to give. Worked through below in
> "What I understand now" — this is the Week 4 machinery, so it is worth owning
> rather than nodding at.


**What actually happened:**

Measured on the frozen 1,117-row split, 2026-08-11. Two runs, all 1,117 raw
outputs byte-identical.

| | shard 0 (n=140) | frozen split (n=1,117) |
|---|---|---|
| Strict exact-match | 35.7% | **35.1%** |
| Majority baseline | 22.9% | **26.3%** |
| **Delta over baseline** | **+12.9 pp** | **+8.8 pp** |
| Binary yes/no | 67.2% (n=61) | **59.2%** (n=524) |
| Open-ended | 11.4% (n=79) | **13.8%** (n=593) |

**Your direction was right, your magnitude was not.** You predicted 25%, i.e. a
drop of ~10.7 pp. The real drop was **0.6 pp**. Accuracy barely moved.

<!-- But look at the third row before concluding the prediction was "nearly right
     in the wrong way". Accuracy held almost constant while the delta over
     baseline fell by a third, because the BASELINE rose 3.4 pp.

     In your own words: if the model's score is unchanged but the baseline moved
     up, has the model got worse? What exactly got worse? -->


<!-- The other thing worth sitting with: binary accuracy fell 67.2% -> 59.2%.
     Milestone D reported "+17 pp of real signal above chance" from n=61.
     The true figure is +9.2 pp. Nothing about the model changed between those
     two numbers. What changed, and what does that tell you about how much to
     trust a percentage quoted without its n? -->


**What I understand now:**

<!-- 1. Why did the protocol have to be frozen BEFORE any adapter existed?
        What specifically could have gone wrong if we had written eval.py after
        the first QLoRA run?
     2. The probe built its answer vocabulary from the eval split's own answers.
        Name the two separate things that go wrong when you do that.
     3. Shards 8-15 were deliberately left undownloaded. Why does it matter that
        the reserve rows have never been scored? -->


---

## Milestone F — fp16 stability harness (2026-08-11)

*This is the first milestone with a **backward pass** in it. Everything before
this was forward-only inference, so fp16 could not really hurt us. Now it can.*

**Prediction (committed 2026-08-11, before either run):**

*1. Over ~50 steps of a healthy run, does the GradScaler scale factor go up, down,
or stay flat?*

> scale goes down

*2. Under a deliberately inflated LR, which goes non-finite first — the loss or a
gradient?*

> gradient goes first

> 📌 Noted at prediction time, not after: both answers give a direction with no
> mechanism. That is fine as a first pass, but it means neither can be *partly*
> right — there is nothing to check except the outcome. The reconcile below is
> where the mechanism gets built, and question 2 in particular has a specific
> chain of events behind it that is worth being able to trace.


**What actually happened:**

**Both predictions were right.** Measured 2026-08-11 on the 3060, 50 steps, LoRA r=8
on 64 `day-train` rows.

*1. Scale: 65536 → 1024.* Down, as you said. Six halvings, **all of them inside the
first 17 steps**, then dead flat for the remaining 33.

The mechanism behind the direction: 65536 is not a tuned starting point, it is
deliberately *too high*. The scaler is **searching downward** for the largest
multiplier that does not overflow fp16, and halving is how it searches. It is not
decay. It also tries to climb back — `growth_interval=2000` steps without a skip
doubles it again — which is why 50 steps only ever shows the downward half of the
behaviour. On a real Week 3 run of thousands of steps you would see it settle and
then oscillate up.

*2. A gradient went non-finite first — and the loss never did at all.* Not in either
run, not once, in 58 steps total.

That is a stronger version of your answer than you claimed. Trace the sabotage run:

| step | loss | what happened |
|---|---|---|
| 0–1 | 3.84, 0.85 | scaled gradients overflow, steps skipped — the scaler working |
| 2 | 0.02 | scale reached 16384, **the step lands** — and lr=5.0 moves LoRA enormously |
| 3 | **35.99** | the wrecked weights produce huge gradients → overflow |
| 4–7 | 72.9, 35.9, 72.7, 36.0 | every step skipped. Weights frozen. Loss just oscillates by example |

Exactly **one** optimizer step ever landed, and it destroyed the adapter. After that
nothing changed at all — the loss is not diverging, it is *stuck*, bouncing between
~36 and ~73 depending on which question came up.

**This is the failure the whole milestone exists to catch, and it is worse than a
NaN.** A NaN loss is loud. This run has a finite loss, no exception, no warning, and a
progress bar that would have advanced happily for twelve hours on Kaggle, saved an
adapter, and burned a chunk of a 30 hr/week quota — to produce the step-2 wreckage.
`plan.md` specified the tripwire as *"halt on non-finite loss or grad-norm"*. **The
loss half of that rule would never have fired here.** What caught it was the
consecutive-skip rule, which exists only because the design had to distinguish "the
scaler is working" from "the scaler has given up".

*3. The unplanned finding — the ViT claim is no longer inherited, it is measured.*

`memory.md` §5 has said since day one that "the ViT tower is the usual overflow site",
on the authority of other people's write-ups. Now:

| | overflows starting in vision | in language only |
|---|---|---|
| baseline (6 skips) | **6 / 6** | 0 |
| sabotage (7 skips) | **7 / 7** | 0 |

The language tower **never** overflowed without the vision tower overflowing too. The
first non-finite parameter the sabotage run named was
`model.visual.patch_embed.proj` — the patch embedding, the very first thing that
touches a pixel.

And a negative result from the forward hooks, which is why they were worth adding:
**vision activations stayed finite in every step of both runs.** The overflow is in
the *backward* pass, not the forward. "The ViT overflows" is true but imprecise —
it is the ViT's *gradients*, not its activations.


**What I understand now:**

<!-- 1. The healthy run skipped 6 of its 50 steps and that was fine. The sabotage
        run skipped 5 in a row and got halted. Both are "the scaler skipped a
        step". In your own words: what is the difference between them, and why
        can no single-step check tell them apart?

     2. The sabotage run's loss stayed finite the whole time. Say plainly what
        would have happened on Kaggle if the tripwire had only checked the loss.
        Include what you'd have had at the end of it.

     3. Only ONE optimizer step landed in the sabotage run, at step 2, and it did
        all the damage. Why did the steps AFTER it stop landing -- and why is a
        model that has stopped changing more dangerous than one that is visibly
        exploding? -->


**Prediction for Week 3 (write before the first real QLoRA run):**

<!-- You now know: overflow starts in the vision tower every time, the scaler
     settles around 1024, and peak VRAM here was 3.53 GiB on a 6 GB card at
     batch size 1 with NO gradient checkpointing.

     Kaggle's T4 has 14.6 GiB. Predict the batch size you think Week 3 can run
     at, and say what you expect to be the thing that actually stops you going
     higher. -->


---

## Week 3 — the leak, the straw-man baseline, and benchmark row 2 (2026-08-12/13)

*Two findings arrived before any prediction could be made about them, because
nobody thought to predict them. That is itself the lesson of this entry: the
things that nearly sank the project were not the risks on the risk register.*

---

### Finding 1 — the dataset's own train/validation split leaks its images

The pool builder ran a train/eval image-overlap check written in the expectation
that it would print `0`. It printed **235**. Of the 241 images in `day-train`
shards 0–3, **235 also sit in the frozen eval split, byte-identical by sha256**;
only 6 images are outside it. But **0 of 560 `(image, question, answer)` triples**
are shared. `day-train` and `day-validation` are not two sets of images — they are
two sets of *questions about the same ~276 keyframes*. The answers never leaked.
The pixels almost entirely did.

Fix: the eval split moved to `night-validation` (day ∩ night = **0** images,
measured both ways) and the training pool became the whole day domain. Record:
`memory.md` D11, `eval-protocol.md` Amendment 1.

**Questions — answer in your own words:**

<!-- 1. The check was a HARD FAILURE: exit non-zero, refuse to write the manifest.
        Say what would have happened if it had been a warning printed above a
        successful build. Be specific about what the ACCURACY NUMBER would have
        looked like, and why that is the dangerous part.

     2. `token` was called a "scene" everywhere in the docs. It is a KEYFRAME.
        Explain how that one wrong word is what let this sit unnoticed through the
        whole of Milestone E. What did "270 distinct scenes" make you picture, and
        what is actually there?

     3. Row 1 (35.1% zero-shot) is NOT invalidated by the leak, but any finetuned
        row would have been. Why does the leak damage one and not the other?
        Answer in terms of what training does that zero-shot does not.

     4. The answer vocabulary needed no change at all — night has 0 answers
        outside it. Milestone E insisted it be derived from day-train rather than
        from the eval split, which at the time looked like pedantry about a 3.6 pp
        number. What did that decision buy here? -->

---

### Finding 2 — the baseline we had been reporting against was a straw man

This is the more important of the two, and the more uncomfortable, because it had
been inside every number since Milestone E.

The majority-class baseline answers `yes` to **everything** — including *"what
colour is the truck"*. No real system does that. Answering the most common answer
**of each question type** needs no image, no training and no understanding, since
question type is readable straight off the question text:

|  | strict | majority baseline | **per-type prior** | delta vs majority | **delta vs prior** |
|---|---|---|---|---|---|
| day, zero-shot | 35.1% | 26.3% | **33.4%** | +8.8 pp | **+1.7 pp** |
| night, zero-shot | 31.9% | 24.9% | **31.9%** | +7.0 pp | **+0.0 pp** |

On night the model scored **210/659 — exactly what the prior scores.** On night
*binary* questions it scored 51.2% against a 54.1% constant: **worse than the straw
man.** `src/eval.py` now reports `prior_baseline` and `delta_over_prior_pp` on
every run.

**Questions — this is the part most worth getting right:**

<!-- 1. In your own words: why does a model that cannot see ANYTHING still beat
        the majority-class baseline by ~7 pp on this dataset? What is it doing?

     2. Milestone D's own lesson was "measure the majority-class baseline before
        any model runs", and we did exactly that. It still was not enough. What is
        the sharper version of that rule that this session forces?

     3. A baseline is supposed to be the thing you must beat in order to have
        shown anything. Say what makes a baseline HONEST rather than flattering —
        and how you would spot a flattering one before it reaches a paper. -->

---

### The result — benchmark row 2

60 min wall-clock budget on the 3060, trained on day, scored on night:

| night, n=659 | zero-shot | **QLoRA** |
|---|---|---|
| strict | 31.9% | **47.2%** |
| delta over the per-type prior | +0.0 pp | **+15.3 pp** |
| binary (trivial 54.1%) | 51.2% | 66.0% |
| open-ended (trivial 12.9%) | 15.4% | **31.2%** |
| format compliance | 80.6% | **100.0%** |

Paired McNemar **p = 6.5e-09**. The dataset can rank methods.

---

**Prediction 1 — zero-shot on night.** ⚠️ **Overtaken.**

> It was the go/no-go for the whole split change, so it had to run before the
> prediction was captured. Its predictive value is gone; the reason is recorded
> here rather than quietly dropped. Answer it as "explain this", not "call this".

<!-- Measured: night zero-shot 31.9% strict — +7.0 pp over the 24.9% majority
     baseline, but +0.0 pp over the per-type prior.

     The interesting part is WHICH HALF broke. Day -> night:
         binary      59.2% -> 51.2%   (from +3.1 pp above its trivial constant
                                       to -3.0 pp, i.e. BELOW it)
         open-ended  13.8% -> 15.4%   (slightly BETTER)

     memory.md §6 established that 224x224 supports presence judgements but not
     identity or counting. Darkness attacks something different from what
     resolution attacks. Explain why darkness destroyed the BINARY half — the half
     that was working — and left open-ended alone. -->

**Prediction 2 — batch size on a T4.** 🔒 **Still sealed. Write it before the Kaggle
session; `--probe-batch` has been deliberately left unrun so this stays a real
prediction.**

<!-- Measured inputs, all from this project. NOTE that two were corrected 08-13:

       peak VRAM, batch 1, no gradient checkpointing, 3060 : 3.60 GiB
       4-bit NF4 weights, resident                        : 1.47 GiB
       fp32 upcast of embed_tokens (311M params)          : +0.59 GiB
       trainable LoRA params (r=8)                        : 4,411,392
       sequence length per example                        : ~96 tokens
                                                            (~49 vision + ~40 text)
       T4                                                 : 14.6 GiB, ~14.4 usable

     THE CORRECTED FIGURE MATTERS: sequences are ~96 tokens, NOT the ~247 vision
     tokens memory.md §3 quotes. max_pixels caps a 224x224 image well below the
     model's ceiling, so each example is far cheaper than the headline geometry
     suggests.

     What physical batch size fits on the T4, and what is the binding constraint?

     The MECHANISM is the answer, not the number: which parts of that 3.60 GiB
     scale with batch size and which do not — and WHY those particular parts fall
     on either side of the line. --probe-batch measures the curve, so your
     reasoning gets checked against two points rather than one. -->

**Prediction 3 — QLoRA day → night.** ⚠️ **Overtaken, deliberately.**

> You chose to run the decisive experiment rather than pause for the prediction.
> That was the right call — the viability of the entire dataset was riding on it.

<!-- Measured: 47.2% strict, +15.3 pp over the 31.9% prior, McNemar p = 6.5e-09.
     Format compliance 80.6% -> 100.0%.
     Open-ended moved most: 15.4% -> 31.2%. Binary: 51.2% -> 66.0%.

     1. The prior baseline is ALREADY 100% format-compliant by construction — it
        only ever emits in-vocabulary answers. Say in your own words why that
        single fact is what proves the +15.3 pp is perception and not vocabulary.
        What could we have concluded if we were still quoting the majority
        baseline, which would have read +22.3 pp?

     2. Open question #8 feared that most of the finetuning gain would be
        output-vocabulary alignment. Format compliance DID go to 100%, so that
        gain was real and fully captured. Explain how it can be both REAL and
        worth 0 pp of the reported delta.

     3. Milestone D called open-ended a "near-collapse" at 224x224 and treated
        binary as the half that worked. Finetuning bought +15.8 pp on open-ended
        and +14.8 pp on binary — the most where the least was expected. What does
        that say about reading a ZERO-SHOT score as a measure of what the images
        can support? -->

**Prediction 4 — Week 4. Write before the LoRA and DoRA rows run:**

<!-- QLoRA (4-bit base + LoRA adapter) scored 47.2% on night under a 60 min budget.
     Week 4 runs LoRA (fp16 base) and DoRA (4-bit base) under the SAME wall-clock
     budget on the same T4.

     Predict the ORDER of the three and the size of the gaps.

     Then the part that decides whether the thesis holds: LoRA in fp16 has a much
     bigger memory footprint, so under a MATCHED WALL-CLOCK budget it may fit
     fewer steps. Say whether you expect that to help or hurt it, and why — and
     what it would mean for the thesis if the cheapest method won.

     Also: eval.py measures scene ICC per run. Zero-shot on night was 0.024 and
     QLoRA 0.011 — both negligible. Do you expect an ADAPTER to induce more
     scene-level correlation than the base model, or less? (Open question #10.) -->
