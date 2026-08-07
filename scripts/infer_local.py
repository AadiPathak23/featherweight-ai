"""4-bit single-image inference on the local 6 GB card — Milestone B.

Answers one question: does nvidia/Cosmos-Reason2-2B, quantized to 4-bit NF4,
load and generate coherent output about an image inside ~5 GB of VRAM?

If yes, local iteration is real and the laptop is a usable edge target.
If no, everything has to happen on Kaggle and the workflow changes shape.

Memory is reported at three checkpoints rather than one, so we can see *where*
it goes -- weights vs image tokens vs KV cache -- not just whether it fit.

Usage:
    python scripts/infer_local.py                     # default: 256 vision tokens
    python scripts/infer_local.py --max-pixels-tokens 512
    python scripts/infer_local.py --image path/to/your.jpg
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

MODEL_ID = "nvidia/Cosmos-Reason2-2B"

# Traffic cop directing cars at a busy intersection. Public domain (US National
# Archives, via Wikimedia Commons) -- no attribution burden, and it sits in the
# smart-city / traffic domain this project targets (plan.md D5).
SAMPLE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/"
    "ORANGE-VESTED_TRAFFIC_COP_KEEPS_CARS_MOVING_AT_BUSY_INTERSECTION_-_NARA_-_546656.jpg/"
    "960px-ORANGE-VESTED_TRAFFIC_COP_KEEPS_CARS_MOVING_AT_BUSY_INTERSECTION_-_NARA_-_546656.jpg"
)
SAMPLE_IMAGE_PATH = Path(__file__).resolve().parent.parent / "assets" / "sample_traffic.jpg"

PROMPT = (
    "Describe what is happening in this scene. "
    "Who is present, and what are they doing?"
)

GIB = 1024**3


def vram(label: str) -> tuple[float, float]:
    """Print resident and phase-peak VRAM, then RESET the peak counter.

    Two different numbers, easy to confuse:
      current -- what is resident right now (what you still hold)
      peak    -- high-water mark SINCE THE LAST RESET (what you needed, briefly)

    max_memory_allocated() never decreases on its own, so without the reset below
    every later phase reports the largest earlier spike and phase-over-phase
    deltas come out as zero. Resetting makes each phase's peak independent.

    Both count tensors only -- NOT the ~0.5-1 GB CUDA context -- so both are
    optimistic relative to nvidia-smi.
    """
    current = torch.cuda.memory_allocated() / GIB
    peak = torch.cuda.max_memory_allocated() / GIB
    print(f"  [VRAM] {label:<34} resident={current:5.2f} GiB   phase peak={peak:5.2f} GiB")
    torch.cuda.reset_peak_memory_stats()
    return current, peak


def report_quantization(model) -> None:
    """Show what actually got quantized. bitsandbytes stores NF4 as packed uint8,
    so dtype is the tell: uint8 = quantized, float16 = left alone.

    Not everything is quantized, by design -- embeddings, layernorms and the
    lm_head stay in fp16, because quantizing them costs more accuracy than the
    memory it saves.
    """
    buckets: dict[str, list[int]] = {}
    for name, p in model.named_parameters():
        kind = "4-bit (uint8)" if p.dtype == torch.uint8 else str(p.dtype)
        buckets.setdefault(kind, []).append(p.numel())

    print("\n  parameter storage:")
    for kind, sizes in sorted(buckets.items(), key=lambda kv: -sum(kv[1])):
        # 4-bit params are packed 2-per-byte, so element count is half the params
        n = sum(sizes) * 2 if "uint8" in kind else sum(sizes)
        print(f"    {kind:<16} {n/1e9:5.2f}B params across {len(sizes):4d} tensors")

    unquantized = [
        (n, p.numel()) for n, p in model.named_parameters()
        if p.dtype != torch.uint8 and p.numel() > 5_000_000
    ]
    if unquantized:
        print("  largest tensors left UNQUANTIZED:")
        for n, size in sorted(unquantized, key=lambda x: -x[1])[:4]:
            print(f"    {size/1e6:7.1f}M  {n}")


def fetch_sample_image() -> Path:
    if SAMPLE_IMAGE_PATH.exists():
        return SAMPLE_IMAGE_PATH
    SAMPLE_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading sample image -> {SAMPLE_IMAGE_PATH.name}")
    req = urllib.request.Request(SAMPLE_IMAGE_URL, headers={"User-Agent": "featherweight-ai/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r, open(SAMPLE_IMAGE_PATH, "wb") as f:
        f.write(r.read())
    return SAMPLE_IMAGE_PATH


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=None, help="image path (default: sample)")
    ap.add_argument(
        "--max-pixels-tokens",
        type=int,
        default=256,
        help="cap on vision tokens. THE lever for VRAM -- activations scale with this, "
             "not with the weights. Expressed in tokens; converted to pixels below.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()
    print(f"\nModel : {MODEL_ID}")
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    # --- quantization config -------------------------------------------------
    # nf4  : 16 codebook levels at the quantiles of a normal distribution, which
    #        is where weights actually cluster -- not 16 evenly spaced values.
    # double_quant : also quantizes the per-block scale factors, reclaiming
    #        ~0.4 bits/weight (~120 MB here). Standard since the QLoRA paper, so
    #        keeping it on also keeps our VRAM numbers comparable to published ones.
    # compute_dtype=float16 : the dtype NF4 is unpacked INTO for each matmul.
    #        fp16 -- not bf16 -- even though this 3060 (sm_86) supports bf16,
    #        because Kaggle's T4 (sm_75) does not. Local numbers must predict
    #        Kaggle behaviour or local iteration is pointless. See plan.md §7.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # --- processor -----------------------------------------------------------
    # A vision token covers a patch of pixels, so the pixel budget is
    # tokens * patch_area. Qwen3-VL's effective patch is 32x32 (16px patches,
    # merged 2x2), but we read it off the processor rather than hardcoding it.
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    ip = processor.image_processor
    patch_area = (ip.patch_size * ip.merge_size) ** 2
    ip.min_pixels = 4 * patch_area
    ip.max_pixels = args.max_pixels_tokens * patch_area
    print(f"patch={ip.patch_size} merge={ip.merge_size} -> {patch_area} px/token")
    print(f"vision token cap = {args.max_pixels_tokens}  (max_pixels={ip.max_pixels:,})\n")

    # --- load ----------------------------------------------------------------
    print("Loading model in 4-bit (first run downloads ~4.9 GB)...")
    t0 = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        dtype=torch.float16,   # `torch_dtype` is deprecated in transformers 4.57
        device_map="cuda:0",   # explicit: one GPU. "auto" could silently spill to CPU
                               # and quietly wreck the VRAM measurement.
    )
    model.eval()
    print(f"Loaded in {time.perf_counter() - t0:.1f}s")
    resident_weights, peak_load = vram("after load (weights only)")
    report_quantization(model)

    # --- prepare input -------------------------------------------------------
    image_path = args.image or fetch_sample_image()
    image = Image.open(image_path).convert("RGB")
    print(f"\nImage : {image_path.name}  {image.size[0]}x{image.size[1]}")

    messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")

    n_tokens = inputs["input_ids"].shape[-1]
    print(f"Prompt: {n_tokens} tokens total", end="")
    if "image_grid_thw" in inputs:
        grid = inputs["image_grid_thw"][0].tolist()
        # grid is [t, h, w] in patches; the 2x2 merge divides the token count by 4
        print(f"  (image grid {grid} -> {grid[0]*grid[1]*grid[2] // 4} vision tokens)")
    else:
        print()
    resident_proc, peak_proc = vram("after processing image")

    # --- generate ------------------------------------------------------------
    print(f"\nGenerating (max_new_tokens={args.max_new_tokens})...")
    t0 = time.perf_counter()
    with torch.inference_mode():   # no autograd graph -- we are not training
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,       # greedy: deterministic, so reruns are comparable
        )
    torch.cuda.synchronize()       # CUDA is async; sync before trusting the clock
    elapsed = time.perf_counter() - t0

    generated = out[0][n_tokens:]  # strip the prompt; keep only new tokens
    answer = processor.decode(generated, skip_special_tokens=True).strip()
    n_new = len(generated)
    resident_gen, peak_gen = vram("after generation")

    # --- report --------------------------------------------------------------
    print("\n" + "=" * 72)
    print("ANSWER:")
    print(answer)
    print("=" * 72)
    print(f"\nnew tokens      : {n_new}")
    print(f"latency         : {elapsed:.1f}s  ({n_new / elapsed:.1f} tok/s)")

    # The number that decides whether we OOM is the largest single-phase peak,
    # not the sum -- phases do not hold their transients at the same time.
    overall_peak = max(peak_load, peak_proc, peak_gen)

    print("\nmemory, by phase (each peak measured independently):")
    print(f"  load       resident {resident_weights:5.2f} GiB   transient peak {peak_load:5.2f} GiB")
    print(f"  process    resident {resident_proc:5.2f} GiB   transient peak {peak_proc:5.2f} GiB")
    print(f"  generate   resident {resident_gen:5.2f} GiB   transient peak {peak_gen:5.2f} GiB")
    print(f"\n  weights held throughout : {resident_weights:5.2f} GiB")
    print(f"  image tokens added      : {resident_proc - resident_weights:+5.2f} GiB")
    print(f"  KV cache after {n_new:3d} tok : {resident_gen - resident_proc:+5.2f} GiB")
    print(f"  worst-case peak         : {overall_peak:5.2f} GiB")

    target = 5.0
    ok = overall_peak < target
    print(f"\n{'PASS' if ok else 'FAIL'} - peak {overall_peak:.2f} GiB vs {target:.1f} GiB target")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
