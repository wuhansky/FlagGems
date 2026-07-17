import pytest
import torch

import flag_gems

from . import base, consts

# ============================================================================
# Comprehensive (input_shape, output_size) pairs for adaptive_max_pool3d.
#
# Dispatch paths in the latest 2-kernel architecture:
#
#   Guard 0 — Empty:         any dim == 0
#   Guard 1 — Identity:      in_d==out_d && in_h==out_h && in_w==out_w
#   Guard 2 — Global pool:   (1,1,1) && spatial >= 4
#              → _kernel_cooperative (COOP_THREADS autotune)
#   Guard 3 — out_d=1 + in_d>1:
#       A: work_per_thread ≤ 10k → _kernel_direct (full 3D scan)
#       B: work_per_thread > 10k → input.max(dim=2) D-reduce, then dispatch
#   Coop   — total_output ≤ 64 && win_size ≥ 1024 → _kernel_cooperative
#   Direct — _kernel_direct, autotune over (OUT_PER_BLOCK, CHAN_PER_BLOCK)
#            M=16/64/128/256, K=1/4/8
#
# Key metrics:
#   max_win_x = ceil(in_x/out_x) + 1
#   win_size = max_win_d * max_win_h * max_win_w
#   work_per_thread = max_win_d * max_win_h * max_win_w   (for out_d=1 guard)
#   total_output = N * C * D_out * H_out * W_out
# ============================================================================

ADAPTIVE_MAX_POOL3D_BENCH_CONFIGS = [
    # ========================================================================
    # 1. Guard 1 — Identity
    # ========================================================================
    ((2, 256, 2, 14, 14),   (2, 14, 14)),  # exact identity
    ((2, 128, 8, 16, 16),   (8, 16, 16)),  # exact identity, power-of-2
    ((2, 256, 4, 7, 7),     (4, 7, 7)),    # exact identity, odd spatial
    ((4, 3, 32, 112, 112),  (32, 112, 112)),# identity, channel=3

    # ========================================================================
    # 2. Guard 2 — Global pool (1,1,1) → _kernel_cooperative
    #    direct: in_d=1
    #    after-Guard3: in_d>1 → D-reduce first
    # ========================================================================

    # direct global pool (in_d=1):
    ((1, 256, 1, 64, 64),   (1, 1, 1)),    # spatial=4096 — large
    ((2, 128, 1, 128, 64),  (1, 1, 1)),    # spatial=8192
    ((1, 512, 1, 56, 112),  (1, 1, 1)),    # spatial=6272
    ((2, 512, 1, 8, 8),     (1, 1, 1)),    # spatial=64 — boundary (>=4)
    ((1, 256, 1, 14, 14),   (1, 1, 1)),    # spatial=196
    ((2, 128, 1, 7, 14),    (1, 1, 1)),    # spatial=98
    ((1, 64, 1, 28, 28),    (1, 1, 1)),    # spatial=784
    ((2, 512, 1, 7, 7),     (1, 1, 1)),    # spatial=49
    ((2, 2048, 1, 4, 4),    (1, 1, 1)),    # spatial=16
    ((2, 512, 1, 4, 4),     (1, 1, 1)),    # spatial=16
    ((1, 1024, 1, 4, 4),    (1, 1, 1)),    # spatial=16
    ((1, 128, 1, 3, 5),     (1, 1, 1)),    # spatial=15 — very small
    ((1, 256, 1, 2, 2),     (1, 1, 1)),    # spatial=4 — boundary (>=4)

    # global pool via Guard 3 (in_d>1 → D-reduce then global):
    ((1, 4096, 64, 64, 64), (1, 1, 1)),   # D-reduce: spatial=4096
    ((1, 8, 32, 128, 128),  (1, 1, 1)),   # D-reduce: spatial=16384
    ((1, 4, 8, 24, 32),     (1, 1, 1)),   # D-reduce: spatial=768
    ((1, 128, 16, 28, 28),  (1, 1, 1)),   # D-reduce: spatial=784
    ((1, 1024, 8, 7, 7),    (1, 1, 1)),   # I3D Mixed_5c: D-reduce→spatial=49
    ((1, 256, 32, 7, 7),    (1, 1, 1)),   # SlowFast fast: spatial=49
    ((1, 2048, 4, 7, 7),    (1, 1, 1)),   # SlowFast slow: spatial=49
    ((1, 256, 30, 7, 7),    (1, 1, 1)),   # R3D-18: spatial=49
    ((1, 256, 24, 7, 7),    (1, 1, 1)),   # R3D-50: spatial=49
    ((1, 256, 14, 7, 7),    (1, 1, 1)),   # spatial=49
    ((1, 256, 10, 7, 7),    (1, 1, 1)),   # spatial=49
    ((1, 4096, 4, 4, 4),    (1, 1, 1)),   # huge C: D-reduce→spatial=16
    ((1, 4096, 2, 4, 4),    (1, 1, 1)),   # huge C: spatial=16
    ((1, 1, 5, 9, 11),      (1, 1, 1)),   # C=1: D-reduce→spatial=99
    ((128, 768, 4, 4, 4),   (1, 1, 1)),   # huge batch: spatial=16

    # ========================================================================
    # 3. Guard 3 — out_d=1 + in_d>1 (non-global)
    # ========================================================================

    # 3A: work_per_thread ≤ 10k → _kernel_direct (full 3D scan)
    ((1, 64, 8, 56, 56),    (1, 56, 56)),  # work=(8//1+1)*(56//56+1)^2=9*4=36
    ((1, 256, 64, 112, 112),(1, 112, 112)),# work=65*2*2=260
    ((8, 64, 16, 112, 112), (1, 7, 7)),    # work=17*17*17=4913
    ((1, 32, 8, 64, 64),    (1, 32, 32)),  # work=9*3*3=81
    # work near 10k boundary:
    ((1, 16, 32, 128, 128), (1, 8, 8)),    # work=33*17*17=9537 (just under)
    ((1, 16, 32, 140, 140), (1, 8, 8)),    # work=33*18*18=10692 (just over→B)

    # 3B: work_per_thread > 10k → D-reduce first, then dispatch
    ((1, 8, 64, 256, 256),  (1, 7, 7)),    # classic: work=65*38*38=93860

    # ========================================================================
    # 4. Cooperative path — total_output ≤ 64 + win_size ≥ 1024
    # ========================================================================
    ((1, 2, 64, 64, 64),    (2, 2, 2)),    # total=16, win=33*33*33=35937
    ((1, 4, 64, 64, 64),    (1, 1, 1)),    # total=4, win=65*65*65=274625
    ((1, 1, 128, 128, 128), (2, 2, 2)),    # total=8, win=65*65*65=274625

    # ========================================================================
    # 5. Direct dispatch — _kernel_direct, autotune over M + K
    # ========================================================================

    # 5a. Autotune K=1 (no channel chunking) — diverse shapes:
    # --- general 3D compression ---
    ((1, 1280, 48, 36, 50), (8, 8, 8)),     # Qwen2.5-VL, win=7*5*7=245
    ((2, 64, 6, 16, 32),    (2, 8, 16)),    # asymmetric, win=3*3*3=27
    ((2, 640, 64, 42, 72),  (16, 14, 14)),  # Qwen2.5-VL moderate, win=5*4*6=120
    ((2, 768, 96, 32, 32),  (16, 8, 8)),    # VideoLLaMA long-T, win=7*5*5=175
    ((2, 512, 64, 64, 64),  (4, 8, 8)),     # medical 3D, win=17*9*9=1377
    ((1, 8, 64, 256, 256),  (64, 7, 7)),    # T keep, extreme H/W, win=2*38*38=2888
    ((2, 64, 64, 256, 256), (2, 32, 32)),   # win=33*9*9=2673
    ((1, 32, 32, 128, 128), (4, 8, 8)),     # win=9*17*17=2601
    ((2, 64, 16, 64, 64),   (2, 8, 8)),     # win=9*9*9=729

    # --- temporal compression (H/W preserved) ---
    ((1, 768, 160, 32, 32), (8, 32, 32)),   # PLLaVA T=160→8, win=21*2*2=84
    ((2, 1024, 32, 48, 48), (16, 48, 48)),  # VideoLLaMA T=32→16, win=3*2*2=12
    ((4, 128, 32, 64, 64),  (8, 64, 64)),   # InternVideo T=32→8, win=5*2*2=20
    ((2, 256, 64, 40, 40),  (16, 40, 40)),  # win=5*2*2=20

    # --- spatial compression (D preserved) ---
    ((2, 128, 4, 28, 28),   (4, 14, 14)),   # win=2*3*3=18
    ((8, 64, 16, 56, 56),   (16, 14, 14)),  # win=2*5*5=50

    # 5b. Autotune K=4/8 (channel chunking) for large C:
    ((1, 4096, 4, 4, 4),    (2, 2, 2)),     # C=4096→K=4/8, total=65536
    ((1, 2048, 8, 8, 8),    (4, 4, 4)),     # C=2048, total=131072
    ((1, 1024, 16, 16, 16), (8, 8, 8)),     # C=1024, total=524288
    ((1, 512, 32, 32, 32),  (4, 8, 8)),     # C=512, total=131072

    # 5c. Autotune M=16 (small OUT_PER_BLOCK → more blocks, fewer threads/block):
    ((1, 2, 4, 4, 4),       (2, 2, 2)),     # total=16, win=3*3*3=27
    ((1, 4, 8, 16, 16),     (4, 8, 8)),     # total=1024, win=3*3*3=27
    ((2, 16, 8, 16, 16),    (4, 8, 8)),     # total=8192, win=3*3*3=27

    # 5d. Large total_output (grid-heavy, good SM utilization):
    ((4, 32, 16, 32, 32),   (8, 16, 16)),   # total=262144
    ((32, 512, 8, 8, 8),    (4, 4, 4)),     # total=1048576

    # 5e. prefer_2d boundary cases (N×C large vs small):
    ((16, 4, 4, 4, 4),      (2, 2, 2)),     # N*C=64, grid1=2→prefer_2d
    ((8, 8, 4, 4, 4),       (2, 2, 2)),     # N*C=64, grid1=2→prefer_2d
    ((2, 1, 4, 4, 4),       (2, 2, 2)),     # N*C=2, small

    # ========================================================================
    # 6. Sparse grid — N×C small, SM underutilized
    # ========================================================================
    ((1, 3, 16, 224, 224),  (8, 7, 7)),     # N×C=3, total=1176, grid=5 blocks
    ((1, 1, 16, 224, 224),  (8, 7, 7)),     # N×C=1, total=392
    ((1, 2, 32, 128, 128),  (4, 8, 8)),     # N×C=2, total=4096

    # ========================================================================
    # 7. Edge cases — unit dims, non-divisible, extreme ratios
    # ========================================================================
    ((2, 256, 16, 1, 14),   (8, 1, 14)),    # H=1
    ((2, 256, 16, 14, 1),   (8, 14, 1)),    # W=1
    ((2, 256, 1, 1, 14),    (1, 1, 7)),     # D=1, H=1
    ((1, 64, 37, 59, 43),   (7, 13, 19)),   # all prime in/out
    ((1, 64, 8, 17, 31),    (4, 5, 7)),     # non-divisible small
    ((1, 16, 15, 33, 33),   (5, 11, 11)),   # in_d not divisible by out_d
]

# ============================================================================
# COMPREHENSIVE-ONLY shapes.
# ============================================================================
ADAPTIVE_MAX_POOL3D_BENCH_CONFIGS_COMPREHENSIVE = [
    # --- More global pool variants ---
    ((1, 256, 20, 7, 7),    (1, 1, 1)),
    ((2, 512, 8, 4, 4),     (1, 1, 1)),
    ((2, 512, 7, 4, 4),     (1, 1, 1)),
    ((2, 512, 6, 4, 4),     (1, 1, 1)),
    ((2, 512, 5, 4, 4),     (1, 1, 1)),
    ((2, 512, 3, 4, 4),     (1, 1, 1)),
    ((2, 512, 2, 4, 4),     (1, 1, 1)),
    ((2, 3, 7, 13, 17),     (1, 1, 1)),
    ((2, 512, 1, 32, 8),    (1, 1, 1)),
    ((1, 1024, 1, 16, 16),  (1, 1, 1)),
    ((2, 256, 1, 128, 128), (1, 1, 1)),
    ((1, 64, 1, 256, 256),  (1, 1, 1)),
    ((2, 512, 1, 3, 4),     (1, 1, 1)),     # spatial=12

    # --- Direct global pool boundary: spatial=3 (<4, falls through) ---
    ((1, 256, 1, 1, 3),     (1, 1, 1)),     # spatial=3 — no fast path

    # --- More out_d=1 non-global ---
    ((2, 512, 8, 56, 56),   (1, 28, 28)),
    ((1, 128, 32, 48, 64),  (1, 12, 16)),
    ((2, 256, 16, 112, 112),(1, 56, 56)),
    # out_d=1, work near 10k boundary more precisely:
    ((1, 8, 16, 112, 112),  (1, 56, 56)),    # work=17*3*3=153
    ((1, 8, 32, 112, 112),  (1, 14, 14)),    # work=33*9*9=2673
    ((1, 8, 48, 112, 112),  (1, 7, 7)),      # work=49*17*17=14161 (>10k→B)

    # --- Temporal compression ---
    ((1, 256, 64, 112, 112),(4, 112, 112)),
    ((2, 768, 64, 40, 40),  (32, 8, 8)),
    ((1, 512, 128, 28, 28), (16, 28, 28)),

    # --- Full 3D compression (more variants) ---
    ((2, 256, 2, 14, 14),   (2, 7, 7)),
    ((1, 256, 32, 128, 128),(4, 8, 8)),
    ((1, 640, 64, 24, 40),  (8, 8, 8)),
    ((1, 8, 64, 128, 128),  (4, 8, 8)),
    ((1, 16, 64, 128, 128), (4, 8, 8)),

    # --- Spatial compression ---
    ((1, 64, 8, 112, 112),  (8, 56, 56)),
    ((2, 256, 4, 56, 56),   (4, 28, 28)),
    ((1, 3, 32, 224, 224),  (16, 56, 56)),

    # --- Small / odd / non-aligned ---
    ((1, 16, 64, 107, 73),  (7, 12, 18)),
    ((1, 3, 5, 13, 17),     (2, 5, 7)),
    ((1, 1, 3, 7, 11),      (1, 3, 5)),

    # --- Path: out_h>16 or out_w>16 (_kernel_direct M selected accordingly) ---
    ((1, 16, 64, 256, 256), (2, 32, 32)),
    ((1, 8, 128, 256, 256), (2, 17, 17)),

    # --- CHAN_PER_BLOCK boundary ---
    ((1, 3, 16, 32, 32),    (4, 8, 8)),      # C=3 < 4 → K=1 only
    ((1, 4, 16, 32, 32),    (4, 8, 8)),      # C=4 = K=4 threshold
    ((1, 7, 16, 32, 32),    (4, 8, 8)),      # C=7 → K=4 (ceil(7/4)=2 groups)
    ((1, 8, 16, 32, 32),    (4, 8, 8)),      # C=8 = K=8 threshold
    ((1, 15, 16, 32, 32),   (4, 8, 8)),      # C=15 → K=8 (ceil(15/8)=2 groups)

    # --- Cooperative boundary ---
    ((1, 2, 8, 8, 8),       (2, 2, 2)),      # total=16, win=5*5*5=125
    ((1, 2, 32, 32, 32),    (2, 2, 2)),      # total=16, win=17*17*17=4913
    # total=64 boundary:
    ((1, 4, 32, 32, 32),    (2, 2, 2)),      # total=32<64, win=4913→coop
    ((1, 8, 32, 32, 32),    (2, 2, 2)),      # total=64=64, win=4913→coop
    ((1, 16, 32, 32, 32),   (2, 2, 2)),      # total=128>64, win=4913→direct

    # --- Decision boundaries: win_size near key thresholds ---
    ((2, 32, 32, 96, 96),   (4, 8, 8)),      # win=9*13*13=1521
    ((2, 8, 64, 96, 96),    (2, 4, 4)),      # win=33*25*25=20625

    # --- Extreme C for channel chunking stress ---
    ((1, 8192, 2, 2, 2),    (1, 1, 1)),      # C=8192→K=8
    ((256, 3, 8, 14, 14),   (4, 7, 7)),      # extreme batch
    ((1, 1, 128, 256, 256), (8, 8, 8)),      # C=1, large spatial

    # --- Known video model configs ---
    ((1, 768, 96, 14, 14),  (1, 1, 1)),
    ((1, 768, 16, 14, 14),  (1, 1, 1)),
    ((1, 768, 32, 7, 7),    (1, 1, 1)),
    ((2, 768, 16, 14, 14),  (4, 7, 7)),

    # --- More identity ---
    ((4, 64, 8, 32, 32),    (8, 32, 32)),
    ((1, 128, 16, 28, 28),  (16, 28, 28)),

    # --- Cubic output_size ---
    ((1, 128, 32, 64, 64),  (8, 8, 8)),
    ((2, 256, 16, 48, 48),  (4, 4, 4)),
    ((1, 512, 64, 128, 128),(2, 2, 2)),
]


def adaptive_max_pool3d_input_fn(shape, dtype, device):
    """Yield configs directly — actual pairs come from get_input_iter."""
    inp = base.generate_tensor_input(shape, dtype, device)
    yield inp


class AdaptiveMaxPool3dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, cur_dtype):
        configs = list(ADAPTIVE_MAX_POOL3D_BENCH_CONFIGS)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            configs += ADAPTIVE_MAX_POOL3D_BENCH_CONFIGS_COMPREHENSIVE
        for shape, out_size in configs:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            yield inp, out_size


@pytest.mark.adaptive_max_pool3d
def test_perf_adaptive_max_pool3d():
    bench = AdaptiveMaxPool3dBenchmark(
        input_fn=adaptive_max_pool3d_input_fn,
        op_name="adaptive_max_pool3d",
        torch_op=torch.nn.functional.adaptive_max_pool3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
