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

<!-- your reasoning, in your own words: why that bucket and not the others? -->


**What actually happened:**

<!-- fill in after we run it -->

**What I understand now:**

<!-- fill in after we run it -->
