# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest
import torch

import flag_gems

from . import base

# One-dimensional parameter tensors of realistic optimizer sizes. LAMB (like the
# other fused optimizers) operates on a flattened parameter array, so the element
# count is the only shape dimension that matters.
_LAMB_SHAPES = [
    (1024,),
    (4096,),
    (16384,),
    (65536,),
    (262144,),
    (1048576,),
    (11048576,),
]

# Fixed hyper-parameters shared by both implementations so the comparison is
# apples-to-apples. These mirror DeepSpeed's own fused_lamb defaults.
_LR = 1e-3
_BETA1 = 0.9
_BETA2 = 0.999
_MAX_COEFF = 10.0
_MIN_COEFF = 0.01
_EPS = 1e-8
_GRAD_SCALE = 1.0
_STEP = 1
_MODE = 1  # eps outside the sqrt (DeepSpeed default)
_BIAS_CORRECTION = 1
_DECAY = 0.01


def _deepspeed_root():
    """Locate the DeepSpeed checkout so the fused_lamb CUDA source can be built."""
    root = os.environ.get("DEEPSPEED_ROOT")
    if root and os.path.isdir(root):
        return root
    # Heuristic: DeepSpeed sits next to the FlagGems repository.
    flaggems_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sibling = os.path.join(os.path.dirname(flaggems_root), "DeepSpeed")
    if os.path.isdir(sibling):
        return sibling
    return None


def _load_deepspeed_lamb():
    """Return DeepSpeed's fused ``lamb`` operator.

    Prefer the installed ``deepspeed`` package; otherwise compile the operator
    directly from the DeepSpeed source tree via ``torch.utils.cpp_extension``.
    """
    try:
        from deepspeed.ops.op_builder import FusedLambBuilder

        return FusedLambBuilder().load().lamb
    except Exception:
        pass

    from torch.utils.cpp_extension import load

    root = _deepspeed_root()
    if root is None:
        pytest.skip(
            "DeepSpeed source tree not found; set DEEPSPEED_ROOT to run the LAMB "
            "performance benchmark."
        )

    mod = load(
        name="fused_lamb_ds",
        sources=[
            os.path.join(root, "csrc", "lamb", "fused_lamb_cuda.cpp"),
            os.path.join(root, "csrc", "lamb", "fused_lamb_cuda_kernel.cu"),
        ],
        extra_include_paths=[os.path.join(root, "csrc", "includes")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return mod.lamb


# Build (or import) DeepSpeed's fused_lamb once at module import time so the cost
# of JIT compilation is not counted in the measured kernel latencies.
_deepspeed_lamb = _load_deepspeed_lamb()


class LambBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = _LAMB_SHAPES

    def set_shapes(self, shape_file=None):
        # fused_lamb needs 5 tensors per case (param, p_copy, m, v, g); keep the
        # shapes moderate and explicit to avoid OOM on CI GPUs.
        self.shapes = list(_LAMB_SHAPES)

    def set_more_shapes(self):
        return []


def lamb_input_fn(shape, dtype, device):
    p = torch.randn(shape, dtype=dtype, device=device)
    p_copy = torch.empty((0,), dtype=dtype, device=device)  # skip the copy
    m = torch.zeros(shape, dtype=dtype, device=device)
    v = torch.zeros(shape, dtype=dtype, device=device)
    g = torch.randn(shape, dtype=dtype, device=device)
    yield p, p_copy, m, v, g


def torch_op(p, p_copy, m, v, g):
    """Baseline: DeepSpeed's fused_lamb."""
    return _deepspeed_lamb(
        p,
        p_copy,
        m,
        v,
        g,
        _LR,
        _BETA1,
        _BETA2,
        _MAX_COEFF,
        _MIN_COEFF,
        _EPS,
        _GRAD_SCALE,
        _STEP,
        _MODE,
        _BIAS_CORRECTION,
        _DECAY,
    )


@pytest.mark.lamb
def test_lamb_perf():
    def gems_op(p, p_copy, m, v, g):
        return flag_gems.lamb(
            p,
            p_copy,
            m,
            v,
            g,
            _LR,
            _BETA1,
            _BETA2,
            _MAX_COEFF,
            _MIN_COEFF,
            _EPS,
            _GRAD_SCALE,
            _STEP,
            _MODE,
            _BIAS_CORRECTION,
            _DECAY,
        )

    bench = LambBenchmark(
        input_fn=lamb_input_fn,
        op_name="lamb",
        torch_op=torch_op,
        # fused_lamb only supports float32 parameters/state.
        dtypes=[torch.float32],
    )
    bench.set_gems(gems_op)
    bench.run()
