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

"""cudnn_convolution override for the metax (MACA) backend.

The generic implementation is used unchanged; the only change is that FlagTree
AABS (auto_adjust_block_sizes) is held off for the duration of the dispatch.

AABS rewrites the autotuned block sizes at launch time from the *actual* tensor
extents, shrinking any BLOCK that exceeds the extent it was paired with. It
pairs a block to an extent by matching the kernel's ``tl.cdiv(extent, BLOCK)``
against the argument names, and that pairing is wrong for the convolution
kernels here: they tile the flattened spatial volume
``in_n * out_depth * out_height * out_width``, but the analyzer keys the block
to ``in_n`` alone, so the tile collapses to ``next_power_of_2(in_n)``. Measured
on the 3D core case (2, 16, 16, 16, 16): BLOCK_NI_DO_HO_WO 512 -> 2.

That collapse is fatal on metax, not just slow. MACA tensor-core tiles need
M >= 16 rows (see runtime/backend/_metax/ops/gru.py::_MMA_MIN_BLOCK_B), and the
shrunk M = 2 dot cannot be lowered: the kernel aborts while lowering the
``ttg.convert_layout`` of the dot result from the maca_mma layout back to
blocked, surfacing as "RuntimeError: PassManager::run failed" from the
mlir->llir stage. The metax backend has no branch in
triton/runtime/adjust_kernel_param.py that bumps such a tile back up to the dot
floor ("" / hcu / sunrise do), so whatever AABS shrinks stays shrunk.

The knob is global state read at launch time, so holding it off for the whole
dispatch covers every kernel this operator launches -- the generic
conv1d/conv2d/conv3d implicit-GEMM path as well as the direct / depthwise /
pointwise branches -- and the save/restore in the context manager keeps every
other operator's autotuning untouched. This mirrors the iluvatar override
(runtime/backend/_iluvatar/ops/cudnn_convolution.py), where the same pairing
bug is documented and measured.
"""

import contextlib
import logging

from flag_gems.ops.cudnn_convolution import (
    cudnn_convolution as _generic_cudnn_convolution,
)

try:
    from triton.knobs import autotuning as _autotuning_knobs
except (ImportError, ModuleNotFoundError):
    # Triton < 3.6 has no triton.knobs module (and no AABS either).
    _autotuning_knobs = None

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _aabs_disabled():
    """Hold FlagTree AABS off for the duration of the launch."""
    previous = _autotuning_knobs.adjust_block_size
    _autotuning_knobs.adjust_block_size = False
    try:
        yield
    finally:
        _autotuning_knobs.adjust_block_size = previous


def cudnn_convolution(
    input,
    weight,
    padding,
    stride,
    dilation,
    groups,
    benchmark,
    deterministic,
    allow_tf32,
):
    """CUDNN-compatible no-bias convolution for metax.

    Signature, dispatch and results are those of the generic implementation
    (see flag_gems.ops.cudnn_convolution); AABS is simply held off so the
    autotuned blocks reach the kernel unshrunk.
    """
    logger.debug("GEMS_METAX CUDNN_CONVOLUTION")

    # Triton < 3.6 has no knobs module and therefore no AABS to hold off.
    if _autotuning_knobs is None:
        return _generic_cudnn_convolution(
            input,
            weight,
            padding,
            stride,
            dilation,
            groups,
            benchmark,
            deterministic,
            allow_tf32,
        )

    with _aabs_disabled():
        return _generic_cudnn_convolution(
            input,
            weight,
            padding,
            stride,
            dilation,
            groups,
            benchmark,
            deterministic,
            allow_tf32,
        )
