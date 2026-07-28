"""Environment sanity check — Milestone A.

Prints the facts the whole project depends on, then proves bitsandbytes can
actually quantize on this GPU (a 4-bit forward pass, not just an import).

Designed to run UNMODIFIED on both targets:
  * local  — RTX 3060 Laptop, 6 GB, sm_86  -> native bf16 available
  * Kaggle — T4 x2, 16 GB each, sm_75      -> NO native bf16 (see docs/plan.md §7)

Exit code 0 = all checks passed. Non-zero = something the project relies on is broken.
"""

from __future__ import annotations

import platform
import sys

GIB = 1024**3
NATIVE_BF16_MIN_CAPABILITY = (8, 0)  # Ampere+; Turing (7,5) and Pascal (6,0) have none


def line(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 60 - len(title)))


def check_torch():
    import torch

    line("torch / CUDA")
    print(f"python           = {platform.python_version()}  ({platform.system()})")
    print(f"torch            = {torch.__version__}")
    print(f"torch.version.cuda = {torch.version.cuda}")

    available = torch.cuda.is_available()
    print(f"cuda_available   = {available}")
    if not available:
        print("\nFAIL: torch cannot see a CUDA device.")
        print("      A CPU-only wheel is the usual cause — reinstall from the cu130 index.")
        return None

    print(f"device_count     = {torch.cuda.device_count()}")
    return torch


def report_devices(torch) -> list[tuple[int, int]]:
    """Print per-device facts. Returns each device's compute capability."""
    capabilities = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        cap = (props.major, props.minor)
        capabilities.append(cap)

        line(f"cuda:{i}")
        print(f"device           = {props.name}")
        print(f"capability       = {cap}")
        print(f"total_vram       = {props.total_memory / GIB:.2f} GiB")

        # NB: this is read after torch has initialised its CUDA context, which itself
        # costs a few hundred MB. So "unavailable" = other processes + torch's own
        # context, not other processes alone. Don't read it as headroom for weights.
        free, total = torch.cuda.mem_get_info(i)
        print(f"free_vram        = {free / GIB:.2f} GiB of {total / GIB:.2f} GiB "
              f"({(total - free) / GIB:.2f} GiB unavailable: other processes "
              f"+ this process's CUDA context)")

        # The distinction that matters for this project: torch's is_bf16_supported()
        # can answer True on a Turing card because it counts *emulated* bf16, which is
        # not what we can train on. Ask for the hardware truth separately.
        native = cap >= NATIVE_BF16_MIN_CAPABILITY
        try:
            torch_says = torch.cuda.is_bf16_supported(including_emulation=False)
        except TypeError:  # older torch has no such keyword
            torch_says = torch.cuda.is_bf16_supported()
        print(f"bf16_supported   = {torch_says}   (torch, excluding emulation)")
        print(f"bf16_native      = {native}   (derived from compute capability)")
        if not native:
            print("  -> NO hardware bf16. All training on this device must be fp16.")

    return capabilities


def check_bitsandbytes(torch) -> bool:
    """Allocate a Linear4bit on CUDA and run a forward pass through it."""
    line("bitsandbytes 4-bit")
    try:
        import bitsandbytes as bnb
    except Exception as exc:
        print(f"FAIL: could not import bitsandbytes: {exc!r}")
        print("      Run `python -m bitsandbytes` for diagnostics; check the MSVC runtime.")
        return False

    print(f"bitsandbytes     = {bnb.__version__}")

    try:
        layer = bnb.nn.Linear4bit(
            64,
            32,
            bias=False,
            compute_dtype=torch.float16,  # fp16, not bf16 — match the Kaggle constraint
            quant_type="nf4",
        )
        layer = layer.to("cuda")  # quantization happens on the move to GPU
        x = torch.randn(4, 64, device="cuda", dtype=torch.float16)
        y = layer(x)
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"FAIL: 4-bit forward pass raised {type(exc).__name__}: {exc}")
        return False

    print(f"quant_storage    = {layer.weight.dtype} (packed nf4)")
    print(f"forward pass     = OK, output {tuple(y.shape)} {y.dtype}")

    if not torch.isfinite(y).all():
        print("FAIL: output contains NaN/Inf.")
        return False
    print("output finite    = True")
    return True


def main() -> int:
    torch = check_torch()
    if torch is None:
        return 1

    report_devices(torch)
    ok = check_bitsandbytes(torch)

    line("result")
    print("PASS — environment is ready." if ok else "FAIL — see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
