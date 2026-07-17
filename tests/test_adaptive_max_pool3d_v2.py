"""
Test cases for adaptive_max_pool3d — full correctness coverage.

Covers all dispatch paths in the latest 2-kernel architecture:
  Guard 0 — Empty:             any dim == 0
  Guard 1 — Identity:          in_d==out_d, in_h==out_h, in_w==out_w
  Guard 2 — Global pool:       (1,1,1) && spatial >= 4 → _kernel_cooperative
  Guard 3 — out_d=1 + in_d>1:
      A: work_per_thread ≤ 10k → _kernel_direct (full 3D scan)
      B: work_per_thread > 10k → input.max(dim=2) D-reduce, then dispatch
  Coop   — total_output ≤ 64 && win_size ≥ 1024 → _kernel_cooperative
  Direct — _kernel_direct, autotune over (OUT_PER_BLOCK, CHAN_PER_BLOCK)
           M=16/64/128/256, K=1/4/8
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

# ============================================================================
# SHAPE GROUPS — organized by dispatch path
# ============================================================================

# --- Guard 1: Identity ---
IDENTITY_SHAPES = [
    ((2, 256, 2, 14, 14),   (2, 14, 14),   "Identity: exact"),
    ((2, 128, 8, 16, 16),   (8, 16, 16),   "Identity: power-of-2"),
    ((2, 256, 4, 7, 7),     (4, 7, 7),     "Identity: odd spatial"),
    ((4, 64, 8, 32, 32),    (8, 32, 32),   "Identity: large"),
    ((4, 3, 32, 112, 112),  (32, 112, 112),"Identity: C=3, large"),
]

# --- Guard 2: Global pool → _kernel_cooperative ---
# direct global pool (in_d=1):
GLOBAL_POOL_DIRECT = [
    ((1, 256, 1, 64, 64),   (1, 1, 1),     "Global: spatial=4096"),
    ((2, 128, 1, 128, 64),  (1, 1, 1),     "Global: spatial=8192"),
    ((1, 512, 1, 56, 112),  (1, 1, 1),     "Global: spatial=6272"),
    ((2, 512, 1, 8, 8),     (1, 1, 1),     "Global: spatial=64 boundary"),
    ((1, 256, 1, 14, 14),   (1, 1, 1),     "Global: spatial=196"),
    ((2, 128, 1, 7, 14),    (1, 1, 1),     "Global: spatial=98"),
    ((1, 64, 1, 28, 28),    (1, 1, 1),     "Global: spatial=784"),
    ((2, 512, 1, 7, 7),     (1, 1, 1),     "Global: spatial=49"),
    ((2, 2048, 1, 4, 4),    (1, 1, 1),     "Global: spatial=16"),
    ((2, 512, 1, 4, 4),     (1, 1, 1),     "Global: spatial=16"),
    ((1, 1024, 1, 4, 4),    (1, 1, 1),     "Global: spatial=16"),
    ((1, 128, 1, 3, 5),     (1, 1, 1),     "Global: spatial=15"),
    ((1, 256, 1, 2, 2),     (1, 1, 1),     "Global: spatial=4 boundary"),
    ((1, 256, 1, 1, 3),     (1, 1, 1),     "Global: spatial=3 (<4, no fast path)"),
]

# global pool via Guard 3 (in_d>1 → D-reduce then global):
GLOBAL_POOL_VIA_GUARD3 = [
    ((1, 4096, 64, 64, 64), (1, 1, 1),     "B→Global: spatial=4096"),
    ((1, 8, 32, 128, 128),  (1, 1, 1),     "B→Global: spatial=16384"),
    ((1, 4, 8, 24, 32),     (1, 1, 1),     "B→Global: spatial=768"),
    ((1, 128, 16, 28, 28),  (1, 1, 1),     "B→Global: spatial=784"),
    ((1, 1024, 8, 7, 7),    (1, 1, 1),     "B→Global: I3D Mixed_5c (sp=49)"),
    ((1, 256, 32, 7, 7),    (1, 1, 1),     "B→Global: SlowFast fast (sp=49)"),
    ((1, 2048, 4, 7, 7),    (1, 1, 1),     "B→Global: SlowFast slow (sp=49)"),
    ((1, 256, 30, 7, 7),    (1, 1, 1),     "B→Global: R3D-18 (sp=49)"),
    ((1, 256, 14, 7, 7),    (1, 1, 1),     "B→Global: sp=49"),
    ((1, 256, 10, 7, 7),    (1, 1, 1),     "B→Global: sp=49"),
    ((1, 4096, 4, 4, 4),    (1, 1, 1),     "B→Global: huge C, sp=16"),
    ((1, 4096, 2, 4, 4),    (1, 1, 1),     "B→Global: huge C, sp=16"),
    ((1, 1, 5, 9, 11),      (1, 1, 1),     "B→Global: C=1, sp=99"),
    ((128, 768, 4, 4, 4),   (1, 1, 1),     "B→Global: huge batch, sp=16"),
    ((2, 3, 7, 13, 17),     (1, 1, 1),     "B→Global: prime, sp=221"),
]

# --- Guard 3: out_d=1 + in_d>1 ---
# 3A: work ≤ 10k → _kernel_direct (full 3D scan)
GUARD3A_SHAPES = [
    ((1, 64, 8, 56, 56),    (1, 56, 56),   "3A: work=36, spatial preserved"),
    ((1, 256, 64, 112, 112),(1, 112, 112), "3A: work=260, spatial preserved"),
    ((8, 64, 16, 112, 112), (1, 7, 7),     "3A: work=4913, aggressive reduction"),
    ((1, 32, 8, 64, 64),    (1, 32, 32),   "3A: work=81"),
    # near 10k boundary:
    ((1, 16, 32, 128, 128), (1, 8, 8),     "3A: work=9537 (just under 10k)"),
]

# 3B: work > 10k → D-reduce first, then dispatch
GUARD3B_SHAPES = [
    ((1, 8, 64, 256, 256),  (1, 7, 7),     "3B: work=93860, classic extreme"),
    ((1, 16, 32, 140, 140), (1, 8, 8),     "3B: work=10692 (just over 10k)"),
    ((1, 8, 48, 112, 112),  (1, 7, 7),     "3B: work=14161, moderate D"),
    # more variants
    ((2, 512, 8, 56, 56),   (1, 28, 28),   "3B: then spatial"),
    ((1, 128, 32, 48, 64),  (1, 12, 16),   "3B: non-aligned after reduction"),
    ((2, 256, 16, 112, 112),(1, 56, 56),   "3B: large HW after reduction"),
]

# --- Cooperative: total_output ≤ 64 + win_size ≥ 1024 → _kernel_cooperative ---
COOPERATIVE_SHAPES = [
    ((1, 2, 64, 64, 64),    (2, 2, 2),     "Coop: total=16, win=35937"),
    ((1, 4, 64, 64, 64),    (1, 1, 1),     "Coop: total=4, win=274625"),
    ((1, 1, 128, 128, 128), (2, 2, 2),     "Coop: total=8, win=274625"),
    # boundary cases:
    ((1, 4, 32, 32, 32),    (2, 2, 2),     "Coop: total=32<64, win=4913"),
    ((1, 8, 32, 32, 32),    (2, 2, 2),     "Coop: total=64=64, win=4913"),
    ((1, 16, 32, 32, 32),   (2, 2, 2),     "Coop: total=128>64, win=4913→direct"),
    # win boundary:
    ((1, 2, 8, 8, 8),       (2, 2, 2),     "Coop: win=125<1024→direct"),
    ((1, 2, 32, 32, 32),    (2, 2, 2),     "Coop: win=4913≥1024→coop"),
]

# --- Direct dispatch: _kernel_direct (autotune over M + K) ---
DIRECT_GENERAL = [
    # general 3D compression
    ((1, 1280, 48, 36, 50), (8, 8, 8),     "Direct: Qwen2.5-VL, win=245"),
    ((2, 64, 6, 16, 32),    (2, 8, 16),    "Direct: asymmetric, win=27"),
    ((2, 640, 64, 42, 72),  (16, 14, 14),  "Direct: Qwen moderate, win=120"),
    ((2, 768, 96, 32, 32),  (16, 8, 8),    "Direct: VideoLLaMA long-T, win=175"),
    ((2, 512, 64, 64, 64),  (4, 8, 8),     "Direct: medical 3D, win=1377"),
    ((1, 8, 64, 256, 256),  (64, 7, 7),    "Direct: T keep, win=2888"),
    ((2, 64, 64, 256, 256), (2, 32, 32),   "Direct: win=2673"),
    ((1, 32, 32, 128, 128), (4, 8, 8),     "Direct: win=2601"),
    ((2, 64, 16, 64, 64),   (2, 8, 8),     "Direct: win=729"),
    # temporal compression (H/W preserved)
    ((1, 768, 160, 32, 32), (8, 32, 32),   "Direct: PLLaVA T=160→8"),
    ((2, 1024, 32, 48, 48), (16, 48, 48),  "Direct: VideoLLaMA T=32→16"),
    ((4, 128, 32, 64, 64),  (8, 64, 64),   "Direct: InternVideo T=32→8"),
    ((2, 256, 64, 40, 40),  (16, 40, 40),  "Direct: temporal compression"),
    ((1, 256, 64, 112, 112),(4, 112, 112), "Direct: T=64→4"),
    # spatial compression (D preserved)
    ((2, 128, 4, 28, 28),   (4, 14, 14),   "Direct: D preserved, HW 28→14"),
    ((8, 64, 16, 56, 56),   (16, 14, 14),  "Direct: D preserved, HW 56→14"),
]

# Channel chunking (K=4/8) boundary:
DIRECT_K_BOUNDARY = [
    ((1, 3, 16, 32, 32),    (4, 8, 8),     "K: C=3 < 4 → K=1 only"),
    ((1, 4, 16, 32, 32),    (4, 8, 8),     "K: C=4 = K=4 threshold"),
    ((1, 7, 16, 32, 32),    (4, 8, 8),     "K: C=7 → K=4 (2 groups)"),
    ((1, 8, 16, 32, 32),    (4, 8, 8),     "K: C=8 = K=8 threshold"),
    ((1, 15, 16, 32, 32),   (4, 8, 8),     "K: C=15 → K=8 (2 groups)"),
]

# Large C for K=4/8 channel chunking:
DIRECT_LARGE_C = [
    ((1, 4096, 4, 4, 4),    (2, 2, 2),     "Direct: C=4096→K=4/8, total=65536"),
    ((1, 2048, 8, 8, 8),    (4, 4, 4),     "Direct: C=2048, total=131072"),
    ((1, 1024, 16, 16, 16), (8, 8, 8),     "Direct: C=1024, large out"),
    ((1, 512, 32, 32, 32),  (4, 8, 8),     "Direct: C=512"),
]

# Small OUT_PER_BLOCK (M=16):
DIRECT_SMALL_M = [
    ((1, 2, 4, 4, 4),       (2, 2, 2),     "M=16: total=16, win=27"),
    ((1, 4, 8, 16, 16),     (4, 8, 8),     "M=16: total=1024, win=27"),
    ((2, 16, 8, 16, 16),    (4, 8, 8),     "M=16: total=8192, win=27"),
]

# Large total_output (grid-heavy, good SM utilization):
DIRECT_LARGE_TOTAL = [
    ((4, 32, 16, 32, 32),   (8, 16, 16),   "Direct: total=262144"),
    ((32, 512, 8, 8, 8),    (4, 4, 4),     "Direct: total=1048576"),
    ((1, 512, 128, 28, 28), (16, 28, 28),  "Direct: long temporal"),
]

# prefer_2d boundary:
PREFER_2D_SHAPES = [
    ((16, 4, 4, 4, 4),      (2, 2, 2),     "prefer_2d: N*C=64"),
    ((8, 8, 4, 4, 4),       (2, 2, 2),     "prefer_2d: N*C=64"),
    ((8, 4, 4, 4, 4),       (2, 2, 2),     "prefer_2d: N*C=32"),
    ((2, 4, 4, 4, 4),       (2, 2, 2),     "prefer_2d: N*C=8"),
    ((1, 8, 4, 4, 4),       (2, 2, 2),     "prefer_2d: N*C=8"),
    ((2, 1, 4, 4, 4),       (2, 2, 2),     "!prefer_2d: N*C=2"),
]

# --- Sparse grid: N×C small, SM underutilized ---
SPARSE_GRID_SHAPES = [
    ((1, 3, 16, 224, 224),  (8, 7, 7),     "Sparse: N×C=3, total=1176, grid≈5"),
    ((1, 1, 16, 224, 224),  (8, 7, 7),     "Sparse: N×C=1, total=392"),
    ((1, 2, 32, 128, 128),  (4, 8, 8),     "Sparse: N×C=2, total=2048"),
]

# --- Edge cases ---
EDGE_CASE_SHAPES = [
    ((2, 256, 16, 1, 14),   (8, 1, 14),    "Edge: H=1"),
    ((2, 256, 16, 14, 1),   (8, 14, 1),    "Edge: W=1"),
    ((2, 256, 1, 1, 14),    (1, 1, 7),     "Edge: D=1, H=1"),
    ((2, 256, 1, 1, 1),     (1, 1, 1),     "Edge: D=H=W=1, global"),
    ((1, 64, 37, 59, 43),   (7, 13, 19),   "Edge: all prime in/out"),
    ((1, 1, 3, 7, 11),      (1, 3, 5),     "Edge: C=1, non-global"),
    ((1, 16, 64, 107, 73),  (7, 12, 18),   "Edge: non-aligned"),
    ((1, 3, 5, 13, 17),     (2, 5, 7),     "Edge: small odd dims"),
    ((1, 16, 15, 33, 33),   (5, 11, 11),   "Edge: in_d not div by out_d"),
    ((1, 64, 8, 17, 31),    (4, 5, 7),     "Edge: non-divisible small"),
]

# --- Stress tests ---
STRESS_SHAPES = [
    ((128, 768, 4, 4, 4),   (1, 1, 1),     "Stress: huge batch→B"),
    ((32, 512, 8, 8, 8),    (4, 4, 4),     "Stress: large batch"),
    ((1, 8192, 2, 2, 2),    (1, 1, 1),     "Stress: extreme C=8192→B"),
    ((256, 3, 8, 14, 14),   (4, 7, 7),     "Stress: extreme batch=256"),
    ((1, 1, 128, 256, 256), (8, 8, 8),     "Stress: C=1, large spatial"),
    ((1, 4096, 2, 2, 2),    (1, 1, 1),     "Stress: huge C, tiny spatial→B"),
]

# --- Known model configs ---
MODEL_CONFIG_SHAPES = [
    ((1, 768, 96, 14, 14),  (1, 1, 1),     "Model: TimeSformer 96f→B"),
    ((1, 768, 16, 14, 14),  (1, 1, 1),     "Model: VideoMAE 16f→B"),
    ((1, 768, 32, 7, 7),    (1, 1, 1),     "Model: VideoSwin→B"),
    ((2, 768, 16, 14, 14),  (4, 7, 7),     "Model: MViT feature map"),
    # R3D backbone
    ((2, 512, 1, 7, 7),     (1, 1, 1),     "Model: R3D-18 layer4"),
    ((2, 256, 2, 14, 14),   (1, 7, 7),     "Model: R3D-18 layer3"),
    ((2, 128, 4, 28, 28),   (2, 14, 14),   "Model: R3D-18 layer2"),
    ((1, 64, 8, 56, 56),    (4, 28, 28),   "Model: R3D-18 layer1"),
    ((1, 2048, 1, 4, 4),    (1, 1, 1),     "Model: R3D-50 layer4"),
    ((2, 512, 1, 4, 4),     (1, 1, 1),     "Model: C3D fc6"),
    ((2, 512, 2, 7, 7),     (1, 4, 4),     "Model: C3D pool5"),
    # I3D
    ((1, 1024, 8, 7, 7),    (1, 1, 1),     "Model: I3D Mixed_5c→B→E"),
    ((1, 832, 16, 14, 14),  (4, 7, 7),     "Model: I3D Mixed_4f"),
    ((1, 480, 32, 28, 28),  (8, 14, 14),   "Model: I3D Mixed_3c"),
    # Medical
    ((1, 256, 16, 16, 16),  (4, 4, 4),     "Model: 3D U-Net bottleneck"),
    ((1, 512, 4, 4, 4),     (2, 2, 2),     "Model: VoxResNet pre-pool"),
    ((1, 256, 8, 32, 32),   (4, 16, 16),   "Model: CT volume encoder"),
]

# --- PLLaVA / Qwen / VideoLLaMA additional variants ---
MORE_MODEL_SHAPES = [
    ((2, 768, 64, 40, 40),  (32, 8, 8),    "PLLaVA variant"),
    ((2, 1280, 64, 42, 72), (8, 14, 14),   "Qwen2.5-VL scaled"),
    ((2, 1280, 128, 48, 80),(16, 8, 8),    "Qwen2.5-VL large T"),
    ((4, 1536, 96, 32, 32), (16, 16, 16),  "VideoLLaMA large"),
    ((4, 768, 128, 40, 40), (32, 8, 8),    "PLLaVA large T"),
]

# --- Win_size boundary near 2048 ---
WIN_BOUNDARY_SHAPES = [
    ((2, 32, 32, 96, 96),   (4, 8, 8),     "Win: 2028≤2048→1D"),
    ((2, 8, 64, 96, 96),    (2, 4, 4),     "Win: 20625>2048→2D"),
]

# --- Cubic output_size (int → (D,D,D)) ---
CUBIC_SHAPES = [
    ((1, 128, 32, 64, 64),  (8, 8, 8),     "Cubic: moderate"),
    ((2, 256, 16, 48, 48),  (4, 4, 4),     "Cubic: small"),
    ((1, 512, 64, 128, 128),(2, 2, 2),     "Cubic: large win"),
    ((1, 32, 32, 32, 32),   (8, 8, 8),     "Cubic: uniform"),
]

# --- Aligned shapes (multiples of 8) ---
ALIGNED_SHAPES = [
    ((2, 64, 16, 64, 64),   (8, 16, 16),   "Aligned: to 8"),
    ((4, 128, 8, 32, 64),   (4, 16, 32),   "Aligned: moderate"),
    ((1, 256, 32, 128, 128),(8, 16, 16),   "Aligned: large"),
]

# --- C3D / additional global pool variants ---
C3D_GLOBAL_VARIANTS = [
    ((1, 256, 20, 7, 7),    (1, 1, 1),     "C3D: T=20→1"),
    ((2, 512, 8, 4, 4),     (1, 1, 1),     "C3D: D=8"),
    ((2, 512, 7, 4, 4),     (1, 1, 1),     "C3D: D=7"),
    ((2, 512, 6, 4, 4),     (1, 1, 1),     "C3D: D=6"),
    ((2, 512, 5, 4, 4),     (1, 1, 1),     "C3D: D=5"),
    ((2, 512, 3, 4, 4),     (1, 1, 1),     "C3D: D=3"),
    ((2, 512, 2, 4, 4),     (1, 1, 1),     "C3D: D=2"),
]

# --- Miscellaneous ---
MISC_SHAPES = [
    ((2, 512, 1, 32, 8),    (1, 1, 1),     "Global: spatial=256"),
    ((1, 1024, 1, 16, 16),  (1, 1, 1),     "Global: spatial=256"),
    ((2, 256, 1, 128, 128), (1, 1, 1),     "Global: spatial=16384"),
    ((1, 64, 1, 256, 256),  (1, 1, 1),     "Global: spatial=65536"),
    ((2, 512, 1, 3, 4),     (1, 1, 1),     "Global: spatial=12"),
    ((1, 64, 8, 112, 112),  (8, 56, 56),   "Spatial compression"),
    ((2, 256, 4, 56, 56),   (4, 28, 28),   "Spatial compression"),
    ((1, 3, 32, 224, 224),  (16, 56, 56),  "Video pyramid"),
    ((8, 128, 2, 7, 7),     (1, 7, 7),     "Batch stress"),
    ((1, 4, 8, 16, 48),     (2, 4, 12),    "Asymmetric spatial"),
    ((1, 32, 128, 4, 4),    (8, 2, 2),     "Deep volume, small spatial"),
    ((1, 32, 4, 128, 128),  (2, 32, 32),   "Shallow, wide"),
    ((1, 64, 64, 64, 64),   (4, 4, 4),     "1D kernel: win=4913, total=4096"),
    ((1, 8, 64, 128, 128),  (4, 8, 8),     "Large window: win=4913"),
    ((1, 16, 64, 128, 128), (4, 8, 8),     "Large window: win=4913"),
    ((2, 64, 16, 64, 64),   (8, 16, 16),   "Aligned"),
    ((4, 128, 8, 32, 64),   (4, 16, 32),   "Aligned"),
]

# ============================================================================
# Assemble all configs
# ============================================================================
ALL_CONFIGS = (
    IDENTITY_SHAPES
    + GLOBAL_POOL_DIRECT
    + GLOBAL_POOL_VIA_GUARD3
    + GUARD3A_SHAPES
    + GUARD3B_SHAPES
    + COOPERATIVE_SHAPES
    + DIRECT_GENERAL
    + DIRECT_K_BOUNDARY
    + DIRECT_LARGE_C
    + DIRECT_SMALL_M
    + DIRECT_LARGE_TOTAL
    + PREFER_2D_SHAPES
    + SPARSE_GRID_SHAPES
    + EDGE_CASE_SHAPES
    + STRESS_SHAPES
    + MODEL_CONFIG_SHAPES
    + MORE_MODEL_SHAPES
    + WIN_BOUNDARY_SHAPES
    + CUBIC_SHAPES
    + ALIGNED_SHAPES
    + C3D_GLOBAL_VARIANTS
    + MISC_SHAPES
)

# Deduplicate
_seen = set()
UNIQUE_CONFIGS = []
for shape, out_size, desc in ALL_CONFIGS:
    key = (shape, out_size)
    if key not in _seen:
        _seen.add(key)
        UNIQUE_CONFIGS.append((shape, out_size, desc))
ALL_CONFIGS = UNIQUE_CONFIGS

# ============================================================================
# Parametrized tests
# ============================================================================

@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", ALL_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_forward(shape, output_size, desc, dtype):
    """Forward correctness with return_indices=True."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out, ref_indices = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )
    res_out, res_indices = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )

    utils.gems_assert_close(res_out, ref_out, dtype)

    # Index consistency
    gems_vals = inp.flatten(2)[
        torch.arange(inp.size(0), device=inp.device)[:, None, None, None, None],
        torch.arange(inp.size(1), device=inp.device)[None, :, None, None, None],
        res_indices,
    ]
    ref_vals = ref_inp.flatten(2)[
        torch.arange(ref_inp.size(0), device=ref_inp.device)[:, None, None, None, None],
        torch.arange(ref_inp.size(1), device=ref_inp.device)[None, :, None, None, None],
        ref_indices,
    ]
    assert torch.allclose(gems_vals.float(), res_out.float(), atol=0, rtol=0), (
        f"GEMS indices mismatch for {desc}"
    )
    assert torch.allclose(ref_vals.float(), ref_out.float(), atol=0, rtol=0), (
        f"Reference indices mismatch for {desc}"
    )


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", ALL_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_forward_no_indices(shape, output_size, desc, dtype):
    """Forward correctness with return_indices=False."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=False
    )
    res_out = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=False
    )

    assert isinstance(res_out, torch.Tensor), (
        f"Expected Tensor for return_indices=False. {desc}"
    )
    utils.gems_assert_close(res_out, ref_out, dtype)


# ============================================================================
# Integer output_size
# ============================================================================

INT_OUTPUT_SIZE_CONFIGS = [
    ((1, 16, 8, 8, 8),      4,   "int: cubic 4"),
    ((2, 128, 32, 64, 64),  8,   "int: cubic 8"),
    ((1, 256, 16, 28, 28),  2,   "int: cubic 2"),
    ((2, 64, 6, 16, 32),    4,   "int: cubic 4, asymmetric input"),
    ((1, 8, 64, 256, 256),  7,   "int: cubic 7, extreme reduction"),
    ((1, 1, 5, 9, 11),      1,   "int: cubic 1 (global pool)"),
    ((2, 512, 1, 7, 7),     1,   "int: cubic 1, in_d=1"),
]


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", INT_OUTPUT_SIZE_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_int_output_size(shape, output_size, desc, dtype):
    """Forward correctness with integer output_size → broadcast to (D,D,D)."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out, ref_indices = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )
    res_out, res_indices = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )

    expected_out_size = (output_size, output_size, output_size)
    assert res_out.shape[2:] == torch.Size(expected_out_size), (
        f"Expected output shape {expected_out_size}, got {res_out.shape[2:]}. {desc}"
    )
    utils.gems_assert_close(res_out, ref_out, dtype)

    ref_out = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=False
    )
    res_out = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=False
    )
    assert isinstance(res_out, torch.Tensor), (
        f"Expected Tensor for int output_size with return_indices=False. {desc}"
    )
    utils.gems_assert_close(res_out, ref_out, dtype)


# ============================================================================
# Special value tests
# ============================================================================

SPECIAL_VALUE_SHAPES = [
    ((2, 512, 1, 7, 7),     (1, 1, 1),     "Global: in_d=1"),
    ((1, 8, 64, 256, 256),  (1, 7, 7),     "Guard 3B: out_d=1, work>10k"),
    ((8, 64, 16, 112, 112), (1, 7, 7),     "Guard 3A: out_d=1, work≤10k"),
    ((1, 2, 64, 64, 64),    (2, 2, 2),     "Cooperative: total≤64"),
    ((4, 128, 32, 64, 64),  (8, 64, 64),   "Direct: temporal compression"),
    ((16, 4, 4, 4, 4),      (2, 2, 2),     "Direct: prefer_2d"),
    ((2, 64, 64, 256, 256), (2, 32, 32),   "Direct: large window"),
    ((2, 256, 2, 14, 14),   (2, 14, 14),   "Identity"),
    ((1, 3, 16, 224, 224),  (8, 7, 7),     "Sparse: N×C=3"),
    ((1, 4096, 4, 4, 4),    (2, 2, 2),     "Direct: K=4/8, large C"),
]


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", SPECIAL_VALUE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_nan(shape, output_size, desc, dtype):
    """NaN propagation: output must be NaN where input window contains NaN."""
    torch.manual_seed(42)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    nan_pos = (0, 0, min(shape[2] - 1, 1), min(shape[3] - 1, 1), min(shape[4] - 1, 1))
    inp[nan_pos] = float("nan")
    ref_inp = utils.to_reference(inp, True)

    res_out, _ = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )
    ref_out, _ = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )
    res_nan = torch.isnan(res_out)
    ref_nan = torch.isnan(ref_out)
    assert torch.equal(res_nan, ref_nan), f"NaN mask mismatch for {desc}"
    utils.gems_assert_close(res_out[~res_nan], ref_out[~ref_nan], dtype)


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", SPECIAL_VALUE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_ties(shape, output_size, desc, dtype):
    """Tie-breaking: value must be correct even when multiple elements share max."""
    torch.manual_seed(42)
    inp = torch.ones(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    res_out, res_indices = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )
    ref_out, _ = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )
    utils.gems_assert_close(res_out, ref_out, dtype)

    spatial_total = shape[2] * shape[3] * shape[4]
    assert (res_indices >= 0).all(), f"Negative indices in {desc}"
    assert (res_indices < spatial_total).all(), f"Out-of-range indices in {desc}"


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", SPECIAL_VALUE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_all_negative(shape, output_size, desc, dtype):
    """All-negative values: should pick the largest (least negative)."""
    torch.manual_seed(42)
    inp = -torch.rand(shape, dtype=dtype, device=flag_gems.device) * 100 - 1
    ref_inp = utils.to_reference(inp, True)

    res_out, _ = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )
    ref_out, _ = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )
    utils.gems_assert_close(res_out, ref_out, dtype)
    assert (res_out < 0).all(), f"Expected all negative output for {desc}"


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", SPECIAL_VALUE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_mixed_sign(shape, output_size, desc, dtype):
    """Mixed positive/negative: must pick the true max."""
    torch.manual_seed(42)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 10
    ref_inp = utils.to_reference(inp, True)

    res_out, res_indices = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )
    ref_out, ref_indices = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )
    utils.gems_assert_close(res_out, ref_out, dtype)

    gems_vals = inp.flatten(2)[
        torch.arange(inp.size(0), device=inp.device)[:, None, None, None, None],
        torch.arange(inp.size(1), device=inp.device)[None, :, None, None, None],
        res_indices,
    ]
    assert torch.allclose(gems_vals.float(), res_out.float(), atol=0, rtol=0), (
        f"Index-value mismatch for {desc}"
    )


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", SPECIAL_VALUE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_all_zero(shape, output_size, desc, dtype):
    """All-zero input: output must be zero, indices valid."""
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    res_out, res_indices = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )
    ref_out, _ = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )
    utils.gems_assert_close(res_out, ref_out, dtype)

    spatial_total = shape[2] * shape[3] * shape[4]
    assert (res_indices >= 0).all(), f"Negative indices in {desc}"
    assert (res_indices < spatial_total).all(), f"Out-of-range indices in {desc}"


# ============================================================================
# Empty tensor handling
# ============================================================================

EMPTY_CONFIGS = [
    ((2, 64, 4, 8, 8),  (0, 4, 4),    "Empty: D_out=0"),
    ((2, 64, 4, 8, 8),  (4, 0, 8),    "Empty: H_out=0"),
    ((2, 64, 4, 8, 8),  (4, 4, 0),    "Empty: W_out=0"),
]


@pytest.mark.adaptive_max_pool3d
@pytest.mark.parametrize("shape, output_size, desc", EMPTY_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_max_pool3d_empty_output(shape, output_size, desc, dtype):
    """Empty output tensor handling (zero in output_size)."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    res_out, res_indices = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )
    ref_out, ref_indices = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )
    assert res_out.numel() == 0, f"Expected empty output for {desc}"
    assert res_indices.numel() == 0, f"Expected empty indices for {desc}"
    assert res_out.shape == ref_out.shape, f"Shape mismatch for {desc}"

    res_out = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=False
    )
    ref_out = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=False
    )
    assert res_out.numel() == 0, f"Expected empty output for {desc}"
    assert isinstance(res_out, torch.Tensor), "Expected Tensor"
