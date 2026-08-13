"""Week 3 — the QLoRA training loop. This is what produces benchmark row 2.

Everything Week 2 built exists to make this file safe to run on somebody else's
GPU for an hour at a time:

  src/stability.py   the fp16 tripwire, imported UNCHANGED. §10 measured the
                     failure it catches -- one optimizer step lands, every step
                     after it is silently skipped, the loss stays finite and
                     oscillates, and the progress bar keeps moving. On Kaggle
                     that is 12 hours and a saved adapter of pure garbage.
  src/eval.py        build_model(), so the base model and its quantization are
                     byte-for-byte what docs/eval-protocol.md froze.
  scripts/build_train_pool.py
                     the rows, with a hard train/eval scene-leak check.

--- The budget is wall-clock, and that decides more than it looks like ---------

plan.md §6: rows are compared at equal WALL-CLOCK and equal PEAK VRAM, not equal
epochs. Two consequences are baked into this file:

1. A wall-clock budget FORBIDS ANY LR SCHEDULE THAT NEEDS THE HORIZON.
   Cosine decay needs the total step count up front. A wall-clock run does not
   have one -- it has however many steps the hour buys. So the default schedule
   is warmup -> constant, and --cosine is rejected unless --max-steps is given.
   This is not a simplification; a schedule that silently mis-estimates its
   horizon decays to the wrong LR and the run underperforms for a reason that
   never appears in the logs.

2. WALL-CLOCK IS ONLY COMPARABLE ACROSS RUNS ON THE SAME DEVICE.
   A 60-minute run on the 3060 and a 60-minute run on a T4 are not the same
   budget. Every results file records torch.cuda.get_device_name(0) for exactly
   this reason, and W4's matched-budget rows must all come off a T4.

--- Why grad accumulation AND real batching ------------------------------------

Physical batch 1 + accumulation would have reproduced Milestone F's measured
3.53 GiB exactly and dodged padding entirely. It would also have made the
question "what batch size fits on a 14.6 GiB T4?" unanswerable, which is the
Week 3 prediction owed in learning-log.md. So this file does real padded
collation (--batch-size) with accumulation on top (--grad-accum), and
--probe-batch measures the VRAM/batch-size curve directly instead of arguing
about it.

Usage:
    python -m src.train --probe-batch                  # VRAM vs batch size, then exit
    python -m src.train --max-steps 60 --save-every 30 # local dry-run
    python -m src.train --max-steps 12 --lr 5.0        # the tripwire MUST halt this
    python -m src.train --budget-minutes 60            # the real run (Kaggle)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Windows stdout is cp1252 when piped; a non-ASCII progress message would raise
# UnicodeEncodeError and kill the run. It has happened once in this repo already.
# A training run must never die inside the code that reports on it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.eval import GIB, MODEL_ID, PROMPT_TEMPLATE, REPO_ROOT, git_sha, vram
from src.stability import (
    LANG_TARGETS,
    VISION_TARGETS,
    ActivationWatch,
    Tripwire,
    build_trainable,
    grad_norms,
    jsonable,
)

TRAIN_MANIFEST = REPO_ROOT / "results" / "train_pool_manifest.jsonl"
TRAIN_IMG_DIR = REPO_ROOT / "outputs" / "train_pool"
ADAPTER_ROOT = REPO_ROOT / "outputs" / "adapters"

DEFAULT_LR = 1e-4          # the LR Milestone F measured as stable in fp16 on this model
DEFAULT_WARMUP = 20        # optimizer steps
DEFAULT_BUDGET_MIN = 60.0
DEFAULT_SAVE_EVERY = 200

# The results file must stay small -- results/ is TRACKED (results/README.md) and a
# 2,000-step run would otherwise commit ~600 KB of per-step telemetry. Keep every
# step up to this many, then stride-sample, but ALWAYS keep the anomalous steps:
# a skipped step is the whole diagnostic and sampling it away would defeat §10.
FULL_LOG_STEPS = 200
SAMPLED_LOG_TARGET = 300


def line(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_pool(limit: int | None) -> list[dict]:
    if not TRAIN_MANIFEST.exists():
        raise SystemExit(f"missing {TRAIN_MANIFEST}\nRun: python scripts/build_train_pool.py")
    rows = [json.loads(x) for x in TRAIN_MANIFEST.read_text(encoding="utf-8").splitlines() if x.strip()]
    missing = [r["image"] for r in rows if not (TRAIN_IMG_DIR / r["image"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} pool images missing from {TRAIN_IMG_DIR} "
                         f"(e.g. {missing[0]})\nRun: python scripts/build_train_pool.py")
    return rows[:limit] if limit else rows


def collate(rows: list[dict], processor, device: str = "cuda:0"):
    """A padded supervised batch. Generalizes stability.build_batch to batch > 1.

    LABELS MASK THE PROMPT TO -100 so the loss is taken on the ANSWER tokens only.
    Without the mask the model is also scored on reproducing the question and the
    instruction sentence -- which is most of the sequence -- so the loss would be
    dominated by tokens nobody cares about and would barely move when the answer
    is wrong.

    n_prompt is measured PER EXAMPLE by running the processor on that example's
    prompt with THAT example's image, so the vision-token expansion is identical
    in both passes and the mask boundary lands on the right token. Counting text
    tokens and hoping would be off by the ~250 vision tokens.

    PADDING SIDE IS FORCED RIGHT. The processor's default suits generation, where
    left padding keeps the continuation adjacent to the prompt. In training that
    would put pad tokens BEFORE the prompt, so the per-example `labels[:n_prompt]`
    mask would land on padding instead of on the prompt -- the model would be
    trained to predict the question. Same tensor shapes, silently wrong loss.
    """
    from PIL import Image

    images, prompts, fulls = [], [], []
    for row in rows:
        # Image.open with no convert() -- byte-identical to src/eval.py's read path.
        images.append(Image.open(TRAIN_IMG_DIR / row["image"]))
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_TEMPLATE.format(question=row["question"])},
            ],
        }]
        p = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(p)
        fulls.append(p + row["answer"] + (processor.tokenizer.eos_token or ""))

    n_prompt = [processor(text=[p], images=[im], return_tensors="pt")["input_ids"].shape[-1]
                for p, im in zip(prompts, images)]

    previous_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "right"
    try:
        inputs = processor(text=fulls, images=images, return_tensors="pt", padding=True)
    finally:
        processor.tokenizer.padding_side = previous_side
    inputs = inputs.to(device)

    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100      # never learn to emit padding
    for j, n in enumerate(n_prompt):
        labels[j, :n] = -100                          # never learn to emit the prompt
    inputs["labels"] = labels

    n_answer = [int((labels[j] != -100).sum()) for j in range(len(rows))]
    return inputs, n_prompt, n_answer


class Batcher:
    """Seeded, epoch-shuffled sampler over the pool.

    Shuffled rather than sequential because the pool is built shard by shard and a
    shard is contiguous in collection order -- consecutive rows are the same drive,
    often the same scene. Feeding those in order makes each optimizer step see one
    narrow slice of the world, which is exactly the correlation the eval harness
    measures as ICC and the training loop should not manufacture.
    """

    def __init__(self, rows: list[dict], batch_size: int, seed: int):
        self.rows, self.bs = rows, batch_size
        self.rng = random.Random(seed)
        self.order: list[int] = []
        self.pos = 0
        self.epoch = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = list(range(len(self.rows)))
        self.rng.shuffle(self.order)
        self.pos = 0

    def next(self) -> list[dict]:
        if self.pos + self.bs > len(self.order):
            # Drop the short tail and reshuffle: a ragged final batch would change
            # the effective batch size for one step per epoch, which is a small but
            # entirely avoidable inconsistency in a budget-matched comparison.
            self.epoch += 1
            self._reshuffle()
        idx = self.order[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return [self.rows[i] for i in idx]


# --------------------------------------------------------------------------- #
# Learning-rate schedule
# --------------------------------------------------------------------------- #

def lr_at(step: int, base_lr: float, warmup: int, cosine_over: int | None) -> float:
    """Warmup -> constant (default), or warmup -> cosine when the horizon is known.

    Warmup exists because of §10, not because tutorials do it: the GradScaler starts
    at 65536 and spends its first ~17 steps halving down to a workable scale. Taking
    full-size optimizer steps during that settling period is where a run has its best
    chance of wrecking the adapter before the loss has told anyone anything.
    """
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if cosine_over is None:
        return base_lr
    progress = min(1.0, (step - warmup) / max(1, cosine_over - warmup))
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------- #
# Batch-size probe
# --------------------------------------------------------------------------- #

def probe_batch(model, processor, rows, optimizer, scaler, max_bs: int) -> list[dict]:
    """Measure peak VRAM against physical batch size until it OOMs.

    This is a measurement, not a safety feature. Milestone F measured 3.53 GiB at
    batch 1 on a 6 GB card; the open question is what that implies for a 14.6 GiB
    T4, and the honest answer is that it CANNOT be read off one point. Fixed costs
    (4-bit weights 1.47 GiB, the fp32 upcast +0.59 GiB, optimizer state) do not
    scale with batch size; retained activations do. Two points separate those; one
    point cannot.

    Each iteration runs a real forward + backward + optimizer step, because it is
    the BACKWARD pass that holds every layer's activations alive, and the optimizer
    state only exists after a step has landed.
    """
    import torch

    results = []
    bs = 1
    while bs <= max_bs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            inputs, n_prompt, n_answer = collate(rows[:bs], processor)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**inputs).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            peak = torch.cuda.max_memory_allocated() / GIB
            seq = int(inputs["input_ids"].shape[-1])
            results.append({"batch_size": bs, "peak_gib": peak, "seq_len": seq, "ok": True})
            print(f"  batch {bs:>3}  seq {seq:>5}  peak {peak:>6.2f} GiB")
            del inputs, loss
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.OutOfMemoryError) and "out of memory" not in str(exc).lower():
                raise  # a real bug must not be swallowed as if it were an OOM
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            results.append({"batch_size": bs, "peak_gib": None, "ok": False})
            print(f"  batch {bs:>3}  OOM")
            break
        bs *= 2

    fitted = [r for r in results if r["ok"]]
    if len(fitted) >= 2:
        # Two points give the split the single measurement cannot: a per-example
        # slope and a fixed intercept. The intercept is what does NOT scale.
        (b0, p0), (b1, p1) = ((fitted[0]["batch_size"], fitted[0]["peak_gib"]),
                              (fitted[-1]["batch_size"], fitted[-1]["peak_gib"]))
        slope = (p1 - p0) / (b1 - b0)
        intercept = p0 - slope * b0
        print(f"\n  linear fit over batch {b0}..{b1}:")
        print(f"    fixed cost      = {intercept:.2f} GiB   (weights + fp32 upcast + optimizer state)")
        print(f"    per-example cost= {slope:.3f} GiB  (retained activations)")
        if slope > 0:
            t4_usable = 14.4  # 14.6 GiB total minus ~0.2 GiB CUDA context (memory.md §2)
            print(f"    -> extrapolated max batch on a 14.6 GiB T4 "
                  f"(~{t4_usable} usable): {int((t4_usable - intercept) / slope)}")
        print("    ⚠️  an extrapolation, not a measurement. It assumes activation cost is")
        print("        linear in batch size and that nothing else changes on the T4.")
    return results


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    # budget
    ap.add_argument("--budget-minutes", type=float, default=DEFAULT_BUDGET_MIN,
                    help="wall-clock training budget — the primary stopping rule")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="optimizer-step cap; a guard for dry-runs, and required by --cosine")
    # optimization
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--cosine", action="store_true",
                    help="cosine decay — needs a known horizon, so requires --max-steps")
    ap.add_argument("--batch-size", type=int, default=1, help="PHYSICAL batch size")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="micro-batches per optimizer step; effective batch = batch-size * this")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=None, help="default 2*r")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    # mitigation ladder — memory.md §10, all four already implemented in stability.py
    ap.add_argument("--init-scale", type=float, default=65536.0, help="ladder step 1")
    ap.add_argument("--fp32-vision", action="store_true", help="ladder step 2")
    # OFF by default, deliberately. Clipping at 1.0 is the reflex for a training loop,
    # but it is ladder step 4 -- a MITIGATION -- and Milestone F measured that lr=1e-4
    # is stable in fp16 on this model without it (6 skips in 50 steps, no divergence).
    # Turning it on by default would apply a fix to a problem not yet observed, change
    # the numerics away from the configuration F actually validated, and mask the very
    # gradient behaviour the tripwire is watching for. Reach for it when the ladder
    # says to, not before.
    ap.add_argument("--clip-grad", type=float, default=None, help="ladder step 4")
    ap.add_argument("--grad-checkpointing", action="store_true")
    # plumbing
    ap.add_argument("--rows", type=int, default=None, help="limit the pool (iteration only)")
    ap.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY,
                    help="save the adapter every N optimizer steps")
    ap.add_argument("--run-name", type=str, default="qlora")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--probe-batch", action="store_true",
                    help="measure peak VRAM vs batch size, write the record, and exit")
    ap.add_argument("--probe-max", type=int, default=32)
    args = ap.parse_args()

    if args.cosine and args.max_steps is None:
        # See the module docstring. Failing loudly beats silently decaying toward a
        # horizon that was guessed.
        raise SystemExit("--cosine requires --max-steps: cosine decay needs the total "
                         "step count up front, and a wall-clock budget does not have one.")

    import torch

    torch.manual_seed(args.seed)

    line("QLoRA training — Week 3")
    print(f"Device  : {torch.cuda.get_device_name(0)}")
    print(f"Budget  : {args.budget_minutes:.0f} min wall-clock"
          + (f", capped at {args.max_steps} optimizer steps" if args.max_steps else ""))
    print(f"Optim   : lr={args.lr} warmup={args.warmup} "
          f"schedule={'cosine' if args.cosine else 'constant'} clip={args.clip_grad}")
    print(f"Batch   : {args.batch_size} physical x {args.grad_accum} accum "
          f"= {args.batch_size * args.grad_accum} effective")
    print(f"LoRA    : r={args.lora_r} alpha={args.lora_alpha or 2 * args.lora_r} "
          f"dropout={args.lora_dropout}")

    line("training pool (day-train — never the eval split)")
    rows = load_pool(args.rows)
    print(f"  {len(rows)} rows, {len({r['token'] for r in rows})} distinct scenes")

    line("model")
    torch.cuda.reset_peak_memory_stats()
    model, processor, n_train, n_vis = build_trainable(
        args.lora_r, args.fp32_vision, args.grad_checkpointing,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )
    vram("after LoRA attach")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", init_scale=args.init_scale)

    # ------------------------------------------------------------- probe mode
    if args.probe_batch:
        line("batch-size probe (peak VRAM vs physical batch size)")
        model.train()
        probe = probe_batch(model, processor, rows, optimizer, scaler, args.probe_max)
        out = REPO_ROOT / "results" / (args.out or "train_batch_probe.json")
        out.write_text(json.dumps({
            "run": "src/train.py --probe-batch",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": git_sha(),
            "device": torch.cuda.get_device_name(0),
            "total_vram_gib": torch.cuda.get_device_properties(0).total_memory / GIB,
            "config": {"model": MODEL_ID, "lora_r": args.lora_r,
                       "trainable_params": n_train, "trainable_params_vision": n_vis,
                       "fp32_vision": args.fp32_vision,
                       "grad_checkpointing": args.grad_checkpointing},
            "probe": probe,
        }, indent=1), encoding="utf-8")
        print(f"\nwrote {out.relative_to(REPO_ROOT)}")
        return 0

    # ------------------------------------------------------------- training
    adapter_dir = ADAPTER_ROOT / args.run_name
    adapter_dir.mkdir(parents=True, exist_ok=True)
    live_log = adapter_dir / "train_log.jsonl"
    live_log.write_text("", encoding="utf-8")   # a killed session still leaves this

    batcher = Batcher(rows, args.batch_size, args.seed)
    watch = ActivationWatch(model)
    trip = Tripwire()
    model.train()

    line("training")
    print(f"{'step':>5} {'loss':>9} {'lr':>9} {'|g|':>9} {'|g|vis':>9} {'|g|lang':>9} "
          f"{'scale':>8} {'skip':>5} {'ep':>3} {'s/step':>7}")

    steps: list[dict] = []
    budget_s = args.budget_minutes * 60
    t_start = time.perf_counter()
    halted_at = None
    stop_reason = None
    opt_step = 0

    while True:
        elapsed = time.perf_counter() - t_start
        if args.max_steps is not None and opt_step >= args.max_steps:
            stop_reason = f"max-steps ({args.max_steps}) reached"
            break
        if elapsed >= budget_s:
            stop_reason = f"wall-clock budget ({args.budget_minutes:.0f} min) exhausted"
            break

        t0 = time.perf_counter()
        watch.reset()

        lr_now = lr_at(opt_step, args.lr, args.warmup, args.max_steps if args.cosine else None)
        for group in optimizer.param_groups:
            group["lr"] = lr_now

        # --- accumulate --------------------------------------------------- #
        micro_losses = []
        n_answer_tokens = 0
        for _ in range(args.grad_accum):
            inputs, n_prompt, n_answer = collate(batcher.next(), processor)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**inputs).loss
            loss_val = loss.item()
            micro_losses.append(loss_val)
            n_answer_tokens += sum(n_answer)

            if trip.check_loss(opt_step, loss_val):
                halted_at = opt_step
                break
            # Divide before scaling: accumulating grad_accum full-size gradients and
            # stepping on their sum would multiply the effective LR by grad_accum.
            scaler.scale(loss / args.grad_accum).backward()

        if trip.tripped:
            steps.append({"step": opt_step, "loss": jsonable(micro_losses[-1]), "halted": True})
            break

        # --- step ----------------------------------------------------------- #
        # Unscale BEFORE reading any gradient: until this call the grads still carry
        # the scale factor (~65536x) and every norm would be meaningless.
        scaler.unscale_(optimizer)
        g_tot, g_vis, g_lang, bad_param = grad_norms(model)
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)      # a no-op when the gradients are non-finite
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        optimizer.zero_grad(set_to_none=True)

        dt = time.perf_counter() - t0
        loss_mean = sum(micro_losses) / len(micro_losses)
        record = {
            "step": opt_step, "loss": jsonable(loss_mean), "lr": lr_now,
            "grad_norm": jsonable(g_tot), "grad_norm_vision": jsonable(g_vis),
            "grad_norm_language": jsonable(g_lang),
            "scale": scale_before, "skipped": skipped,
            "first_nonfinite_param": bad_param,
            "vision_blocks_nonfinite": watch.hits[:5],
            "epoch": batcher.epoch, "answer_tokens": n_answer_tokens,
            "seconds": dt,
        }
        steps.append(record)
        with live_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

        if opt_step % 10 == 0 or skipped or watch.hits:
            flag = "SKIP" if skipped else ""
            if watch.hits:
                flag += f" vis×{len(watch.hits)}"
            print(f"{opt_step:>5} {loss_mean:>9.4f} {lr_now:>9.2e} {g_tot:>9.3f} "
                  f"{g_vis:>9.3f} {g_lang:>9.3f} {scale_before:>8.0f} {flag:>5} "
                  f"{batcher.epoch:>3} {dt:>7.2f}")

        if trip.check_step(opt_step, skipped, scaler.get_scale(), bad_param,
                           watch.hits, g_vis, g_lang):
            halted_at = opt_step
            stop_reason = "tripwire"
            break

        opt_step += 1
        if args.save_every and opt_step % args.save_every == 0:
            model.save_pretrained(adapter_dir)
            print(f"      ... adapter saved at step {opt_step} -> "
                  f"{adapter_dir.relative_to(REPO_ROOT)}")

    elapsed = time.perf_counter() - t_start
    watch.remove()
    peak = torch.cuda.max_memory_allocated() / GIB

    # ----------------------------------------------------------------- verdict
    line("verdict")
    fired = trip.tripped is not None
    if fired:
        t = trip.tripped
        print(f"🚨 TRIPWIRE FIRED at step {t['step']}: {t['reason']}")
        print(f"   {t['detail']}")
        if t.get("first_nonfinite_param"):
            print(f"   first non-finite parameter : {t['first_nonfinite_param']}")
        if t.get("vision_blocks_nonfinite"):
            print(f"   vision blocks non-finite   : {t['vision_blocks_nonfinite']}")
        print(f"   mitigation                 : {t['mitigation']}")
        print("\n   The adapter on disk is from the last periodic save and is NOT")
        print("   trustworthy — a divergent run's most recent weights are the wreckage.")
    else:
        model.save_pretrained(adapter_dir)
        finite = [s for s in steps if isinstance(s.get("loss"), float)]
        print(f"✅ stopped: {stop_reason}")
        print(f"   {len(steps)} optimizer steps, {batcher.epoch} full epochs over "
              f"{len(rows)} rows")
        print(f"   scaler skipped {trip.total_skips} step(s) — "
              + ("normal fp16 behaviour, correctly not treated as divergence"
                 if trip.total_skips else "no overflow at this LR"))
        if len(finite) >= 2:
            head = sum(s["loss"] for s in finite[:10]) / min(10, len(finite))
            tail = sum(s["loss"] for s in finite[-10:]) / min(10, len(finite))
            print(f"   loss {head:.4f} (first 10 steps) -> {tail:.4f} (last 10)")
        print(f"   adapter -> {adapter_dir.relative_to(REPO_ROOT)}")

    print(f"\n{len(steps)} steps in {elapsed/60:.1f} min   "
          f"{elapsed/max(len(steps),1):.2f} s/step   peak VRAM {peak:.2f} GiB")

    # ----------------------------------------------------------------- record
    # results/ is tracked, so the per-step log is thinned -- but never at the cost
    # of a skipped step, which is the diagnostic §10 exists for.
    if len(steps) <= FULL_LOG_STEPS:
        sampled = steps
    else:
        stride = max(1, len(steps) // SAMPLED_LOG_TARGET)
        keep = {i for i, s in enumerate(steps)
                if i % stride == 0 or s.get("skipped") or s.get("halted")
                or s.get("vision_blocks_nonfinite")}
        sampled = [steps[i] for i in sorted(keep)]

    finite = [s for s in steps if isinstance(s.get("loss"), float)]
    out = REPO_ROOT / "results" / (args.out or f"train_{args.run_name}.json")
    out.write_text(json.dumps({
        "run": "src/train.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "config": {
            "model": MODEL_ID, "quantization": "4-bit NF4 + double quant, compute fp16",
            "amp": "fp16 autocast + GradScaler", "init_scale": args.init_scale,
            "lr": args.lr, "warmup_steps": args.warmup,
            "schedule": "cosine" if args.cosine else "warmup+constant",
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "effective_batch": args.batch_size * args.grad_accum,
            "lora_r": args.lora_r, "lora_alpha": args.lora_alpha or 2 * args.lora_r,
            "lora_dropout": args.lora_dropout,
            "lora_targets": VISION_TARGETS + LANG_TARGETS,
            "trainable_params": n_train, "trainable_params_vision": n_vis,
            "fp32_vision": args.fp32_vision, "clip_grad": args.clip_grad,
            "grad_checkpointing": args.grad_checkpointing,
            "seed": args.seed, "pool_rows": len(rows),
            "budget_minutes": args.budget_minutes, "max_steps": args.max_steps,
            # Wall-clock budgets are only comparable within one device. Never drop this.
            "device": torch.cuda.get_device_name(0),
        },
        "metrics": {
            "stop_reason": stop_reason, "tripwire_fired": fired,
            "halted_at_step": halted_at,
            "optimizer_steps": len(steps), "epochs": batcher.epoch,
            "scaler_skips_total": trip.total_skips,
            "loss_first": jsonable(finite[0]["loss"]) if finite else None,
            "loss_last": jsonable(finite[-1]["loss"]) if finite else None,
            "scale_last": finite[-1].get("scale") if finite else None,
        },
        "tripwire": trip.tripped,
        "timings": {"seconds_total": elapsed,
                    "seconds_per_step": elapsed / max(len(steps), 1),
                    "minutes_total": elapsed / 60},
        "peak_vram_gib": peak,
        "adapter_dir": str(adapter_dir.relative_to(REPO_ROOT)),
        "steps_logged": len(sampled), "steps_total": len(steps),
        "steps": sampled,
    }, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}   (full per-step log: "
          f"{live_log.relative_to(REPO_ROOT)})")

    # Exit non-zero on a tripwire so a notebook cell chain stops instead of going on
    # to evaluate an adapter that is known to be wreckage.
    return 1 if fired else 0


if __name__ == "__main__":
    raise SystemExit(main())
