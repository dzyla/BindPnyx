"""Build Protenix's fused LayerNorm CUDA kernel on a toolchain where it currently fails (GCC 15 + PyTorch 2.11 headers).

Cause: torch/include/ATen/core/List_inl.h:202 casts with `typename decltype(impl_->list)::difference_type`, which GCC 15 rejects
("need 'typename' ... [-Wtemplate-body]"; -fpermissive does not help, g++-13 is not installed). For std::vector that type is std::ptrdiff_t, so a one-line
OVERRIDE of that header (placed ahead of torch's include dir; torch's own file is NOT edited) fixes the build.

  build_layernorm.py              build into a scratch dir and report (no environment changes)
  build_layernorm.py --install    also copy fast_layer_norm_cuda_v2*.so next to protenix/model/layer_norm/layer_norm.py, which imports it first
                                  and so skips the ~80 s failed compile at every Protenix start.
After --install run:  build_layernorm.py --verify   (needs an idle GPU: max|fused - torch.nn.functional.layer_norm| on random data)."""
import argparse, os, shutil, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--install", action="store_true"); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
import torch, protenix
LN = Path(protenix.__file__).parent / "model" / "layer_norm"; K = LN / "kernel"
if a.verify:
    import importlib, torch.nn.functional as F
    ln = importlib.import_module("protenix.model.layer_norm.layer_norm")
    print("fused kernel loaded:", ln.fast_layer_norm_cuda_v2 is not None)
    assert ln.fast_layer_norm_cuda_v2 is not None, "kernel not importable: the fused path is NOT in use"
    worst = 0.0
    for shape, dt in [((4, 256, 128), torch.float32), ((1, 300, 300, 128), torch.float32), ((8, 512, 384), torch.bfloat16), ((1, 120, 120, 128), torch.bfloat16)]:
        for affine in (True, False):
            m = ln.FusedLayerNorm(shape[-1], create_scale=affine, create_offset=affine).cuda() if "create_scale" in ln.FusedLayerNorm.__init__.__code__.co_varnames else ln.FusedLayerNorm(shape[-1]).cuda()
            x = torch.randn(*shape, device="cuda", dtype=dt)
            ref = F.layer_norm(x.float(), [shape[-1]], getattr(m, "weight", None), getattr(m, "bias", None), 1e-5)
            err = float((m(x).float() - ref).abs().max()); worst = max(worst, err); print(f"  {str(shape):22s} {str(dt):15s} affine={affine!s:5s} max|fused - torch| = {err:.2e}")
    import time; x = torch.randn(1, 400, 400, 128, device="cuda"); m = ln.FusedLayerNorm(128).cuda()
    for name, f in (("fused", lambda: m(x)), ("torch", lambda: F.layer_norm(x, [128], m.weight, m.bias, 1e-5))):
        for _ in range(5): f()
        torch.cuda.synchronize(); t = time.time()
        for _ in range(50): f()
        torch.cuda.synchronize(); print(f"  {name:6s} {1000*(time.time()-t)/50:.3f} ms / call")
    print("WORST ERROR", worst); sys.exit()
work = REPO / ".pxd" / "ln_build"; ov = work / "override" / "ATen" / "core"; ov.mkdir(parents=True, exist_ok=True); (work / "build").mkdir(exist_ok=True)
hdr = Path(torch.__file__).parent / "include" / "ATen" / "core" / "List_inl.h"
src = hdr.read_text(); fixed = src.replace("static_cast<typename decltype(impl_->list)::difference_type>(pos)", "static_cast<std::ptrdiff_t>(pos)")
if fixed == src: print("header line not found (torch version changed?): nothing to override; try the build as-is"); 
(ov / "List_inl.h").write_text(fixed)
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
from torch.utils.cpp_extension import load
inc = f"-I{work / 'override'}"; t = time.time()
load(name="fast_layer_norm_cuda_v2", sources=[str(K / "layer_norm_cuda.cpp"), str(K / "layer_norm_cuda_kernel.cu")], extra_include_paths=[str(K)],
     extra_cflags=["-O3", "-DVERSION_GE_1_1", "-DVERSION_GE_1_3", "-DVERSION_GE_1_5", inc],
     extra_cuda_cflags=["-O3", "--use_fast_math", "-DVERSION_GE_1_1", "-DVERSION_GE_1_3", "-DVERSION_GE_1_5", "-std=c++17", "-maxrregcount=32", "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__", "--expt-relaxed-constexpr", "--expt-extended-lambda", inc], build_directory=str(work / "build"))
so = next((work / "build").glob("fast_layer_norm_cuda_v2*.so")); print(f"built {so} in {time.time()-t:.0f}s")
if a.install: shutil.copy2(so, LN / so.name); print("installed ->", LN / so.name, "(run --verify on an idle GPU)")
else: print("not installed (use --install)")
