"""Milestone F — the fp16 stability harness.

This is insurance against the project's highest risk (memory.md §5). The T4 has no
hardware bf16 -- measured on live silicon, unfixable by any driver or torch upgrade --
so every Week 3+ training run is fp16 + AMP + GradScaler, and the ViT tower is the
documented overflow site.

THE FAILURE THIS EXISTS TO PREVENT IS NOT A CRASH. A crash is cheap: you see it and
fix it. The expensive failure is a run that silently goes non-finite at step 40, keeps
training on garbage for four more hours, saves an adapter, and only reveals itself when
eval reports a number near the majority baseline -- by which point a chunk of a
30 GPU-hr/week quota is gone and the cause is hours behind you.

So this file is a training loop whose PURPOSE is its instrumentation. It is deliberately
not a Week 3 training run: 50 steps, batch of 1, a tiny LoRA. What Week 3 imports from
here is the tripwire, not the hyperparameters.

WHAT IS MEASURED PER STEP
    loss, global grad-norm, VISION-TOWER grad-norm separately, the live GradScaler
    scale factor, whether the optimizer step was skipped, and VRAM.

    The vision tower gets its own column because memory.md §5 names it as the overflow
    site. A per-module norm makes that VISIBLE instead of inferred -- and forward hooks
    on the 24 vision blocks catch a non-finite ACTIVATION at its source, which is a
    strictly better diagnostic than noticing a non-finite gradient afterwards.

THE TRIPWIRE, AND THE FALSE POSITIVE IT MUST NOT HAVE
    GradScaler works BY producing inf gradients on overflow and skipping that step.
    An occasional skip is the scaler doing its job, not divergence. A tripwire that
    halted on the first inf gradient would fire on correct behaviour, get switched off
    by whoever is trying to work, and then be absent for the real failure. So:

      non-finite LOSS                  -> halt now. Nothing legitimate produces this.
      non-finite grad + step skipped   -> log, continue. This is the scaler working.
      N consecutive skips (default 5)  -> halt. Sustained overflow, not a transient.
      scale below the floor            -> halt. The scaler has run out of room.

MITIGATION LADDER -- decided in advance, so it is not invented under pressure:
    1. lower --init-scale        (overflow at step 0-2; the scaler starts too high)
    2. --fp32-vision             (vision-tower norms blow up before the language ones)
    3. lower --lr                (loss climbs steadily before going non-finite)
    4. --clip-grad               (norms spike intermittently but recover)

Usage:
    python -m src.stability                # baseline: sane LR, must complete clean
    python -m src.stability --sabotage     # inflated LR, MUST trip the tripwire
    python -m src.stability --steps 20 --lr 3e-4

EXIT CODE IS SELF-VERIFYING: 0 when the observed outcome matches what the mode
expects (baseline completes / sabotage trips), 1 when it does not. A --sabotage run
that finishes cleanly is a FAILED milestone, not a passed one -- it means the tripwire
would also have missed a real divergence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# See scripts/build_eval_split.py: on Windows stdout defaults to cp1252 and a non-ASCII
# progress message raises UnicodeEncodeError. That killed one 20-minute build AFTER all
# the expensive work had succeeded (memory.md §8). A logging call must never be able to
# destroy the run it is reporting on -- least of all this one, whose entire job is logging.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.eval import GIB, MODEL_ID, PROMPT_TEMPLATE, REPO_ROOT, build_model, git_sha, vram

# Training rows come from day-TRAIN. The eval split (day-validation shards 0-7) is frozen
# for scoring and shards 8-15 are the untouched reserve -- docs/eval-protocol.md §4:
# "Never evaluate on anything touched during training." This is the other side of that rule.
TRAIN_SHARD = "day-train/data-00000-of-00016.arrow"
STAB_DIR = REPO_ROOT / "outputs" / "stability_set"
STAB_MANIFEST = STAB_DIR / "manifest.json"
N_TRAIN_ROWS = 64

# LoRA targets, DISCOVERED from the model rather than assumed (2026-08-11).
# The vision tower names its attention projections `qkv` and `proj`; the language model
# uses `q_proj`/`k_proj`/`v_proj`/`o_proj`. Targeting only the language names -- the
# reflex, since that is what every QLoRA tutorial lists -- would put no adapter anywhere
# near the ViT, so there would be NO vision-tower gradients to measure, and this harness
# would report a clean vision column for the exact site it was built to watch.
# `proj` cannot collide with `q_proj` etc: PEFT matches on a `.`-delimited suffix.
VISION_TARGETS = ["qkv", "proj"]
LANG_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

DEFAULT_LR = 1e-4
SABOTAGE_LR = 5.0        # ~4 orders of magnitude too high. Should diverge within a few steps.
DEFAULT_STEPS = 50
MAX_CONSECUTIVE_SKIPS = 5
SCALE_FLOOR = 1.0


def line(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


def jsonable(x):
    """inf/nan are not valid JSON. Store them as strings rather than emitting bare
    NaN literals, which json.dump will happily write and many parsers then reject --
    a results file that cannot be read back is not a record of anything."""
    if isinstance(x, float) and not math.isfinite(x):
        return "inf" if x > 0 else ("-inf" if x < 0 else "nan")
    return x


# --------------------------------------------------------------------------- #
# Training rows
# --------------------------------------------------------------------------- #

def build_stability_set(n_rows: int) -> list[dict]:
    """Dump n_rows of day-train to PNG. Cached, so this costs one download ever.

    Reuses fetch() and read_shard() from scripts/build_eval_split.py -- the retry/resume
    download and the exact uint8 -> PNG conversion the eval split uses. Rewriting either
    would risk the two paths drifting apart.
    """
    if STAB_MANIFEST.exists():
        rows = json.loads(STAB_MANIFEST.read_text(encoding="utf-8"))
        if len(rows) >= n_rows and all((STAB_DIR / r["image"]).exists() for r in rows[:n_rows]):
            print(f"  cached: {n_rows} day-train rows in {STAB_DIR}")
            return rows[:n_rows]

    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from build_eval_split import fetch, read_shard  # noqa: E402

    STAB_DIR.mkdir(parents=True, exist_ok=True)
    ds = read_shard(fetch(TRAIN_SHARD))
    tokens, questions, answers = ds["token"], ds["question"], ds["answer"]

    rows = []
    for i in range(min(n_rows, len(ds))):
        name = f"train_{i:04d}.png"
        path = STAB_DIR / name
        if not path.exists():
            Image.fromarray(np.array(ds[i]["CAM_FRONT"], dtype=np.uint8)).save(path, format="PNG")
        rows.append({
            "image": name, "shard": TRAIN_SHARD, "row_index": i,
            "token": tokens[i], "question": questions[i], "answer": answers[i],
        })

    STAB_MANIFEST.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"  built: {len(rows)} day-train rows -> {STAB_DIR}")
    return rows


def build_batch(row: dict, processor):
    """One supervised example: the frozen eval prompt, with the gold answer as the target.

    Labels mask the prompt to -100 so loss is taken on the ANSWER tokens only. Without
    the mask the model is also scored on reproducing the question and the instruction
    sentence, which is most of the sequence -- the loss would be dominated by tokens we
    do not care about and would barely move when the answer is wrong.

    n_prompt is measured by running the processor on the prompt alone with the SAME
    image, so the image-token expansion is identical in both passes and the boundary
    lands where it should. Counting text tokens and hoping would be off by the ~250
    vision tokens.
    """
    from PIL import Image

    image = Image.open(STAB_DIR / row["image"])
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": PROMPT_TEMPLATE.format(question=row["question"])},
        ],
    }]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = prompt_text + row["answer"] + (processor.tokenizer.eos_token or "")

    n_prompt = processor(text=[prompt_text], images=[image], return_tensors="pt")["input_ids"].shape[-1]
    inputs = processor(text=[full_text], images=[image], return_tensors="pt").to("cuda:0")

    labels = inputs["input_ids"].clone()
    labels[:, :n_prompt] = -100
    inputs["labels"] = labels
    return inputs, n_prompt, labels.shape[-1] - n_prompt


# --------------------------------------------------------------------------- #
# Instrumentation
# --------------------------------------------------------------------------- #

def grad_norms(model) -> tuple[float, float, float, str | None]:
    """(global, vision-tower, language, first non-finite parameter name).

    MUST be called after scaler.unscale_(). Before that the gradients still carry the
    scale factor (~65536x), so every norm would be meaningless and the finiteness check
    would flag overflow that only exists because of the scaling.

    "First" is in named_parameters() order, which tracks module definition order and so
    is close to forward order -- good enough to name the offending region, which is the
    point. The forward hooks below are the precise instrument.
    """
    import torch

    tot = vis = lang = 0.0
    first_bad = None
    for name, p in model.named_parameters():
        if p.grad is None or not p.requires_grad:
            continue
        g = p.grad.detach()
        if first_bad is None and not torch.isfinite(g).all():
            first_bad = name
        sq = g.float().norm(2).item() ** 2
        tot += sq
        if "visual" in name:
            vis += sq
        else:
            lang += sq
    return math.sqrt(tot), math.sqrt(vis), math.sqrt(lang), first_bad


class ActivationWatch:
    """Forward hooks on the vision blocks, so a non-finite ACTIVATION is caught where it
    is born rather than inferred later from a gradient.

    This is the difference between "some gradient was inf" and "model.visual.blocks.17
    emitted a non-finite activation" -- the second tells you which mitigation to reach
    for. Cost is one isfinite() over a ~0.5 MB tensor per block per step: negligible
    against a forward pass, and `plan.md` warns that a harness which perturbs timing
    corrupts the wall-clock benchmark column.
    """

    def __init__(self, model):
        self.hits: list[str] = []
        self.handles = []
        for name, mod in model.named_modules():
            if "visual.blocks." in name and name.count(".") == 3:  # one hook per block
                self.handles.append(mod.register_forward_hook(self._make(name)))

    def _make(self, name):
        import torch

        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t) and not torch.isfinite(t).all():
                self.hits.append(name)
        return hook

    def reset(self):
        self.hits = []

    def remove(self):
        for h in self.handles:
            h.remove()


class Tripwire:
    """Halts the run on divergence -- but not on the scaler doing its job.

    See the module docstring. The distinction between "inf gradient, step skipped" and
    "divergence" is the whole design: the first is the mechanism working as intended and
    happens routinely at the start of an fp16 run.
    """

    def __init__(self, max_skips: int = MAX_CONSECUTIVE_SKIPS, scale_floor: float = SCALE_FLOOR):
        self.max_skips = max_skips
        self.scale_floor = scale_floor
        self.consecutive_skips = 0
        self.total_skips = 0
        self.tripped: dict | None = None

    def check_loss(self, step: int, loss: float) -> bool:
        if not math.isfinite(loss):
            self.tripped = {
                "step": step, "reason": "non-finite loss", "loss": loss,
                "detail": "the forward pass itself produced inf/nan -- the weights are "
                          "already corrupt, so the previous step is what broke",
                "mitigation": "lower --lr (ladder step 3), then --clip-grad (step 4)",
            }
            return True
        return False

    def check_step(self, step: int, skipped: bool, scale: float, bad_param: str | None,
                   vision_hits: list[str], grad_v: float, grad_l: float) -> bool:
        if skipped:
            self.consecutive_skips += 1
            self.total_skips += 1
        else:
            self.consecutive_skips = 0

        if self.consecutive_skips >= self.max_skips:
            vision_first = bool(vision_hits) or (math.isfinite(grad_l) and not math.isfinite(grad_v))
            self.tripped = {
                "step": step, "reason": f"{self.consecutive_skips} consecutive skipped steps",
                "first_nonfinite_param": bad_param,
                "vision_blocks_nonfinite": vision_hits[:5],
                "detail": "sustained overflow -- the scaler has halved repeatedly and is "
                          "still overflowing, so no optimizer step is landing and the run "
                          "is burning wall-clock without training",
                "mitigation": "--fp32-vision (ladder step 2)" if vision_first
                              else "lower --init-scale (step 1), then lower --lr (step 3)",
            }
            return True

        if scale < self.scale_floor:
            self.tripped = {
                "step": step, "reason": f"scale collapsed below {self.scale_floor}",
                "scale": scale, "first_nonfinite_param": bad_param,
                "vision_blocks_nonfinite": vision_hits[:5],
                "detail": "the scaler has halved its way to the floor and has no range "
                          "left -- gradients now underflow instead of overflowing",
                "mitigation": "--fp32-vision (ladder step 2), then lower --lr (step 3)",
            }
            return True
        return False


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_trainable(lora_r: int, fp32_vision: bool, grad_checkpointing: bool):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model, processor, patch_area = build_model(adapter=None)  # same quantization the protocol froze
    resident_before, _ = vram("after 4-bit load")

    # Upcasts every non-4bit param to fp32 (layernorms, embeddings) and freezes the base.
    # NOT free: memory.md §3 measured embed_tokens at 311.2M params = 42% of resident
    # weights, so fp16 -> fp32 on the unquantized 0.32B costs ~+0.6 GiB before a single
    # step runs. Logged rather than discovered, because on a 6 GB card that is the
    # difference between fitting and not.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_checkpointing)
    resident_after, _ = vram("after fp32 upcast")
    print(f"  fp32 upcast cost: {resident_after - resident_before:+.2f} GiB resident")

    if fp32_vision:
        # Mitigation ladder step 2. The ViT is the documented overflow site; taking it
        # out of fp16 removes the overflow at the cost of memory and speed.
        n = 0
        for name, p in model.named_parameters():
            if "visual" in name and p.dtype == torch.float16:
                p.data = p.data.to(torch.float32)
                n += 1
        print(f"  --fp32-vision: upcast {n} vision-tower tensors to fp32")

    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.0, bias="none",
        target_modules=VISION_TARGETS + LANG_TARGETS,
        task_type="CAUSAL_LM",
    ))
    model.config.use_cache = False  # incompatible with gradient checkpointing, useless in training
    if grad_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    n_vis = sum(p.numel() for n_, p in model.named_parameters() if p.requires_grad and "visual" in n_)
    n_all = sum(p.numel() for _, p in model.named_parameters() if p.requires_grad)
    print(f"  trainable: {n_all:,} params ({n_all - n_vis:,} language + {n_vis:,} vision)")
    if n_vis == 0:
        # The blind spot this harness exists to avoid. Fail loudly rather than report a
        # reassuring empty vision column.
        raise SystemExit("no trainable vision-tower params — the vision grad column would "
                         "be identically zero and this harness would be blind to the "
                         "documented overflow site. Check VISION_TARGETS against the model.")
    return model, processor, n_all, n_vis


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sabotage", action="store_true",
                    help=f"inflate LR to {SABOTAGE_LR} — the tripwire MUST fire")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--rows", type=int, default=N_TRAIN_ROWS)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--init-scale", type=float, default=65536.0, help="ladder step 1")
    ap.add_argument("--fp32-vision", action="store_true", help="ladder step 2")
    ap.add_argument("--clip-grad", type=float, default=None, help="ladder step 4")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="trade compute for activation memory if 6 GB is not enough")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    lr = args.lr if args.lr is not None else (SABOTAGE_LR if args.sabotage else DEFAULT_LR)
    mode = "sabotage" if args.sabotage else "baseline"
    run_name = f"stability_{mode}"

    import torch

    torch.manual_seed(0)

    line("stability harness — Milestone F")
    print(f"Mode    : {mode}   (expected outcome: "
          f"{'TRIPWIRE FIRES' if args.sabotage else 'completes clean'})")
    print(f"Device  : {torch.cuda.get_device_name(0)}")
    print(f"LR      : {lr}   steps={args.steps}   lora_r={args.lora_r}")
    print(f"Scaler  : init_scale={args.init_scale}   "
          f"halt on {MAX_CONSECUTIVE_SKIPS} consecutive skips or scale < {SCALE_FLOOR}")

    line("training rows (day-train — never the eval split)")
    rows = build_stability_set(args.rows)

    line("model")
    torch.cuda.reset_peak_memory_stats()
    model, processor, n_train, n_vis = build_trainable(args.lora_r, args.fp32_vision,
                                                       args.grad_checkpointing)
    vram("after LoRA attach")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scaler = torch.amp.GradScaler("cuda", init_scale=args.init_scale)
    watch = ActivationWatch(model)
    trip = Tripwire()
    model.train()

    line("training")
    print(f"{'step':>4} {'loss':>9} {'|g|':>10} {'|g|vis':>10} {'|g|lang':>10} "
          f"{'scale':>9} {'skip':>5} {'s/step':>7}")

    steps: list[dict] = []
    t_start = time.perf_counter()
    halted_at = None

    for step in range(args.steps):
        row = rows[step % len(rows)]
        t0 = time.perf_counter()
        watch.reset()

        inputs, n_prompt, n_answer = build_batch(row, processor)

        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**inputs).loss
        loss_val = loss.item()

        if trip.check_loss(step, loss_val):
            halted_at = step
            steps.append({"step": step, "loss": jsonable(loss_val), "halted": True})
            break

        scaler.scale(loss).backward()
        # Unscale BEFORE reading any gradient: until this call the grads are still
        # multiplied by the scale factor.
        scaler.unscale_(optimizer)
        g_tot, g_vis, g_lang, bad_param = grad_norms(model)

        if args.clip_grad is not None:  # ladder step 4
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           args.clip_grad)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)   # a no-op when the grads are non-finite
        scaler.update()
        scale_after = scaler.get_scale()
        skipped = scale_after < scale_before
        optimizer.zero_grad(set_to_none=True)

        dt = time.perf_counter() - t0
        steps.append({
            "step": step, "loss": jsonable(loss_val),
            "grad_norm": jsonable(g_tot), "grad_norm_vision": jsonable(g_vis),
            "grad_norm_language": jsonable(g_lang),
            "scale": scale_before, "skipped": skipped,
            "first_nonfinite_param": bad_param,
            "vision_blocks_nonfinite": watch.hits[:5],
            "n_prompt_tokens": n_prompt, "n_answer_tokens": n_answer,
            "seconds": dt,
        })

        flag = "SKIP" if skipped else ""
        if watch.hits:
            flag += f" vis-nonfinite×{len(watch.hits)}"
        print(f"{step:>4} {loss_val:>9.4f} {g_tot:>10.3f} {g_vis:>10.3f} {g_lang:>10.3f} "
              f"{scale_before:>9.0f} {flag:>5} {dt:>7.2f}")

        if trip.check_step(step, skipped, scale_after, bad_param, watch.hits, g_vis, g_lang):
            halted_at = step
            break

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
    else:
        skip_note = ("normal fp16 behaviour, correctly not treated as divergence"
                     if trip.total_skips else "no overflow occurred at this LR")
        print(f"✅ completed {len(steps)} steps, all finite")
        print(f"   scaler skipped {trip.total_skips} step(s) — {skip_note}")
        if steps:
            print(f"   loss {steps[0]['loss']:.4f} -> {steps[-1]['loss']:.4f}   "
                  f"scale {steps[0]['scale']:.0f} -> {steps[-1]['scale']:.0f}")

    expected_fire = args.sabotage
    ok = (fired == expected_fire)
    print()
    if ok and expected_fire:
        print("PASS — sabotage diverged and the harness caught it with a named cause.")
    elif ok:
        print("PASS — a healthy fp16 run completes without false alarms.")
    elif expected_fire:
        print("FAIL — sabotage ran to completion. The tripwire would ALSO miss a real "
              "divergence, which is the failure this milestone exists to prevent.")
    else:
        print("FAIL — the tripwire fired on a run that was supposed to be healthy. "
              "It is too sensitive; a harness that cries wolf gets switched off.")

    print(f"\n{len(steps)} steps in {elapsed:.0f}s   peak VRAM {peak:.2f} GiB")

    # ----------------------------------------------------------------- record
    out = REPO_ROOT / "results" / (args.out or f"{run_name}.json")
    finite = [s for s in steps if isinstance(s.get("loss"), float)]
    out.write_text(json.dumps({
        "run": "src/stability.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "config": {
            "model": MODEL_ID, "mode": mode, "lr": lr, "steps_requested": args.steps,
            "lora_r": args.lora_r, "lora_targets": VISION_TARGETS + LANG_TARGETS,
            "trainable_params": n_train, "trainable_params_vision": n_vis,
            "quantization": "4-bit NF4 + double quant, compute fp16",
            "amp": "fp16 autocast + GradScaler", "init_scale": args.init_scale,
            "fp32_vision": args.fp32_vision, "clip_grad": args.clip_grad,
            "grad_checkpointing": args.grad_checkpointing,
            "train_shard": TRAIN_SHARD, "n_rows": len(rows), "batch_size": 1,
            "device": torch.cuda.get_device_name(0),
        },
        "metrics": {
            "expected_tripwire": expected_fire, "tripwire_fired": fired,
            "outcome_matches_expectation": ok,
            "steps_completed": len(steps), "halted_at_step": halted_at,
            "scaler_skips_total": trip.total_skips,
            "loss_first": jsonable(finite[0]["loss"]) if finite else None,
            "loss_last": jsonable(finite[-1]["loss"]) if finite else None,
            "scale_first": finite[0]["scale"] if finite and "scale" in finite[0] else None,
            "scale_last": finite[-1]["scale"] if finite and "scale" in finite[-1] else None,
        },
        "tripwire": trip.tripped,
        "timings": {"seconds_total": elapsed,
                    "seconds_per_step": elapsed / max(len(steps), 1)},
        "peak_vram_gib": peak,
        "steps": steps,
    }, indent=1), encoding="utf-8")
    print(f"wrote {out.relative_to(REPO_ROOT)}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
