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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.cudnn_convolution import (
    _output_size,
    _pointwise_conv as _shared_pointwise_conv,
)
from flag_gems.ops.cudnn_convolution import (
    _to_list,
)
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ascend direct path.
#
# Each program owns whole output rows rather than a slice of the flattened
# (n, oh, ow) space, so the inner tl.arange is the output column index and the
# input address ``iw = ow * SW + kw * DW`` is affine in it.  With unit stride
# that is a contiguous load.  Tiling the flattened space -- what the shared
# direct kernels do -- leaves the stride-1 index discontinuous at every row
# boundary, which turns each tap into a vector-indexed gather; the Ascend
# backend charges roughly four orders of magnitude for those, and it showed: the
# same shapes that run in 0.4 ms native took 388 ms.
#
# The address is affine *unpadded* because the padding is materialised instead:
# see _pad_input.  Anything that makes the address non-affine again -- a clamp
# to keep a masked lane in range, say -- costs more than the whole rest of the
# rewrite, which is why the halo is built rather than tested for.
#
# Within that tiling there are two kernels, selected per call by _can_use_dot:
# the 2D kernel's tl.dot branch (see the 3D note below for why 3D has none), and
# a plain FMA form that both kernels keep.
#
# * The tl.dot branch reduces a (BLOCK_C, BLOCK_W) input tile against a
#   (BLOCK_OC, BLOCK_C) weight tile, one dot per tap and channel block.  This is
#   what puts the operator on the cube, and msprof confirms it: the dot kernel
#   reports Task Type=MIX_AIC, while the FMA kernel below reports
#   AI_VECTOR_CORE.  On the 2D core case that is 3.29 ms against 28.0 ms.
#
#   The FMA form is not the more accurate one, which is worth stating plainly
#   because this file previously assumed the opposite.  Against an fp64 CPU
#   reference on (4,64,64,64)/3x3, the dot kernel's error is 1.068e-04 absolute
#   / 9.16e-07 relative -- numerically identical to the vendor conv run with
#   HF32 off, so it is the same exact-fp32 arithmetic, not the cube's
#   reduced-precision mode.  The FMA kernel is 3.052e-05 / 2.62e-07, 3.5x
#   tighter, but that is precision no fp32 tolerance asks for and it costs 3.9x
#   in time.  For scale, the vendor conv with HF32 *on* is 1.694e-02 / 1.45e-04,
#   which is where the ~11-bit-mantissa figure this file used to cite comes
#   from; tl.dot here does not do that.
#
# * The FMA form keeps one tap and one channel per iteration, for the shapes
#   tl.dot cannot express: any tile dimension below 16, i.e. every shape with
#   fewer than 16 output channels per group or fewer than 16 input channels.
#   Those are the small launch-bound cases, where padding up to 16 would
#   multiply a cheap kernel by 16/3.  In 3D it is the only form.
#
# The dot path also has to fit the unified buffer, which is a correctness limit
# rather than a speed consideration: past it BiShengHIR either aborts the build
# with "ub overflow" or, for a tile that only just overruns, emits a kernel that
# faults the device.  _pick_block_c shrinks the channel tile to fit; the budget
# and the measurement behind it are at _UB_TILE_MAX.
#
# Both read the weight from the transposed (KH, KW, C, OC) layout built by
# _prep_weight.  That is not incidental: the output-channel axis is what the
# inner loop indexes, and it is a gather in the native layout and unit-stride in
# this one -- 140.8 ms against 36.1 ms with the kernel otherwise unchanged.
#
# Loop nesting differs between the two kernels on purpose.  The 2D kernels are
# tap-major (static kh/kw outside, the channel loop inside); the 3D one is
# channel-major.  A runtime channel loop wrapped around 27 unrolled taps instead
# of 9 sends the Ascend compiler into a multi-minute unroll -- measured stuck
# for > 15 min on (1,16,4,4,4)/3x3x3.  Nine taps is under that limit; twenty-
# seven is not.
#
# For the same reason the 3D kernel is FMA-only: 27 unrolled taps that each hold
# a tl.dot do not merely compile slowly, they hang the device.  On the 3D core
# case (2,16,16,16,16) bf16 with 16 input and 16 output channels -- every tile at
# tl.dot's minimum size -- a 100-launch loop never returned once in four
# attempts, while the identical loop on the FMA kernel finished in 0.5 s and the
# same 27-tap dot kernel at 1x3x3 (9 taps instead of 27) finished normally.  It
# is the tap count, not the tile size.  Disabling dot for 3D costs little: those
# shapes are launch-bound, and FMA already runs the core case in 0.195 ms
# against 11.99 ms before this rewrite.
# ---------------------------------------------------------------------------

# The spatial tile is the inner dimension, so it is chosen first: a whole output
# row when it fits, capped at _BLOCK_W_MAX registers' worth.  The channel tile
# then takes what is left of a fixed accumulator budget.  Both are picked per
# call because the kernel is launch- and mask-bound at the small end of the
# shape range: a fixed BLOCK_W of 256 over a 6-wide output masks 250 lanes away
# and leaves a single program for the whole device.
_BLOCK_W_MAX = 256
# Floor for the width walk in _pick_blocks.  tl.dot needs every dimension at
# least 16, and a tile narrower than this costs more in programs than the
# channel tile it buys back.
_BLOCK_W_MIN = 32
# Ceiling on the output-channel tile.  This is the kernel's throughput knob,
# and the reason is in the loop's shape: one program loads a (BLOCK_C, BLOCK_W)
# input tile and reuses it for BLOCK_OC output channels, so the bytes it must
# pull per MAC are 2 / BLOCK_OC -- the channel tile cancels out of the load
# entirely and the *count of programs* is what BLOCK_OC divides.  The kernel is
# bound on the MTE2 pipe (aic_mte2_ratio 1.000 against a cube at 0.133), so that
# is the load it is bound on.  Measured on (8,256,64,64)/k3, whole call, bf16:
#
#   BLOCK_OC    16     32     64    128    256
#   us        7052   3441   1711    912    917
#
# i.e. exactly 1/BLOCK_OC up to 128 and flat after, so the ceiling is worth
# setting from the budget below rather than at a constant.  It used to be 32,
# which cost this shape 3.8x and capped every case at OC/32 times the loads it
# needed.
_BLOCK_OC_MAX = 256
# Target accumulator size, i.e. BLOCK_OC * BLOCK_W, in elements.  Raised from
# 4096 to 8192 by measurement: (BLOCK_OC=32, BLOCK_W=256) beats (16, 256) by
# 1.4x on the 2D core case even though both fit.
#
# This is what actually caps BLOCK_OC once _BLOCK_OC_MAX stopped doing it, and
# the two axes are not interchangeable in it: BLOCK_W is pinned by the output
# row, so a wide row is what leaves the channel tile small.  Measured on
# (8,256,64,64)/k3 with the row held at one tile, the three (BLOCK_OC, BLOCK_W)
# pairs under this budget -- (128, 64), (64, 128), (32, 256) -- come out at 912,
# 913 and 918 us, so the budget itself is the thing being tested and it is
# already at the flat part of the curve.
_BLOCK_ELEMS = 8192
# Channel tile for the tl.dot path, in elements.  A single dot over the whole
# channel count is what the sweep preferred (C_IN=64 in one dot 5.35 ms, split
# into 32s 8.50 ms, into 16s 12.86 ms), so this only has to cap how much of a
# very wide channel dim is reduced per dot.
_BLOCK_C_MAX = 64
# tl.dot needs every dimension at least this large; below it there is no dot.
_DOT_MIN = 16
# Ceiling on the tl.dot path's input tile (BLOCK_C * BLOCK_W), in elements.
#
# The Ascend unified buffer is 1572864 bits per vector unit.  Measured, the
# pipelined requirement is four times the tile's fp32 size: BiShengHIR asks for
# 2113792 bits at (BLOCK_C=64, BLOCK_W=256), and 4 * 16384 * 32 = 2097152, so
# the tile must not exceed 1572864 / 128 = 12288 elements -- which for power-of-
# two tiles means 8192.  Exceeding it is a correctness bound, not a speed knob:
# the worst shapes fail the build with "ub overflow", but a tile that only just
# overruns it, like the 16384-element one on (32,64,210,210)/k5/s1/p1, compiles
# and then faults the device's MTE at runtime.  Shrinking BLOCK_C rather than
# BLOCK_W is the cheaper half: measured 0.545 ms against 0.593 ms on
# (64,48,1024)/k5/s2.
_UB_TILE_MAX = 8192
# Unrolled taps a kernel may have and still be handed fp16/bf16 tiles for its
# dots.  Not a tuning knob: see _arith_dtype.  9 is the same ceiling the 3D note
# records for unrolled taps that hold a dot at all, and it is the one the 2D
# kernel was measured against -- 3x3 keeps the native dtype and is 1.66x faster
# for it, 5x5 hangs the device.
_DOT_TAPS_MAX = 9
# Cubes per program.  Named rather than a literal at the launch site so it can
# be swept; see the note in _direct_conv2d.
_NUM_WARPS = 4

# Spare elements left at the end of every buffer the kernels index directly, so
# that a masked lane's address stays inside the allocation.  See _slack.
_SLACK_ELEMS = 8192

# Innermost halo run, in bytes, at which a strided copy_ is still worth its one
# pass over the data.  Below it the run is a per-element gather.  One cache line
# is where the measurements happen to separate; see _pad_input.
_PAD_RUN_BYTES = 128


def _arith_dtype(input, use_dot, taps):
    """The dtype the direct kernels should do their arithmetic in.

    fp32, except that a small-tap kernel taking the ``tl.dot`` branch can keep
    fp16/bf16.

    The upcast this file does for bf16 and fp16 buys nothing on the dot path.
    ``tl.dot``'s products are exact either way: the operands are already bf16 or
    fp16, so their product fits an fp32 accumulator exactly whether the cube was
    handed the native dtype or its fp32 image, and only the accumulation order
    differs.  What the upcast costs is real, though -- a pass over the input, a
    pass over the output, and half again the tile traffic feeding the cube --
    and it is what the benchmark's glue kernels are: on
    (32,64,128,128)/3x3/p2 the call is 3252 us, of which 2471 is the conv kernel
    and 203 is the input upcast plus output downcast.  Keeping the native dtype
    takes that case to 1962 us, 1.66x, and is the same product for the same
    accumulator bits.

    **``taps`` is not a heuristic.**  A bf16 dot does not merely compile slowly
    where an fp32 one is merely slow -- it hangs the *device*.  Same shape, same
    kernel, only the tile dtype changed: (32,64,210,210)/k5/s2/p1 is 25 unrolled
    taps over a C_IN=64 reduction, so 50 dots in the unrolled body, and it runs
    in 10 s from a cold cache in fp32 but never returns in bf16 -- 13 minutes of
    compiler CPU and then aicore timeout 507014, the device wedged.  It is the
    same wall the 3D note below describes, reached from a different side: 9
    unrolled taps is under it, 25 is over, and bf16 lowers what a tap costs
    enough to cross it at a tap count fp32 clears.

    The FMA branch has no such excuse and must stay fp32; the Ascend backend
    does not vectorize a bf16 load feeding a multiply, and the 1D shape that
    runs in 1.06 ms in fp32 takes 87.5 ms in bf16.
    """
    if (
        use_dot
        and taps <= _DOT_TAPS_MAX
        and input.dtype in (torch.float16, torch.bfloat16)
    ):
        return input.dtype
    return torch.float32


def _pad_input(input, padding, tail):
    """Materialise the zero halo the kernels index through.

    The kernels use unpadded tap indices (``oh * SH + kh * DH``), so they need a
    tensor in which the halo already exists.  Building one costs a single pass
    over the input, and it buys two things the alternatives cannot:

    * the loads carry no mask and no bounds test, because a tap that falls in
      the halo reads a real zero -- the padding is done by the data rather than
      by a compare per tap;
    * every address stays affine, which is what the backend's axis analysis
      needs to keep the dots on the cube.  The same addressing expressed as a
      clamp -- ``min(max(ow * SW + kw * DW - PW, 0), W - 1)`` -- is the obvious
      way to keep a masked lane inside the tensor, and it runs the identical
      kernel 160x slower: 616 ms against 3.8 ms on (32,64,128,128)/k3/s2 at the
      same 97% cube utilisation.  See the note above _direct_conv2d_kernel.

    The halo is only worth building when a tap can actually fall outside the
    input, so the caller only builds it when the shape needs one.

    ``tail`` is how far the last row's masked lanes can run past the padded
    row's end; that much slack is appended so those addresses stay inside the
    allocation.  See _slack for why an out-of-allocation address matters even
    though the access is masked.

    Callers hand this an fp32 tensor even for fp16/bf16 inputs, so the halo is
    fp32 like the accumulator.  Having the halo's ``copy_`` do that upcast -- by
    asking for the halo in fp32 and passing the operand as it arrived -- is the
    obvious saving, and it was tried and dropped: it removes a pass over the
    tensor but measured no better on any case (and 4.8% worse on one), because
    what this costs is not the bytes moved but the strided copy itself.  On
    2d-ragged-pad the fusion removes 676 MB of traffic across a 10.8 ms call
    and moves the total by 0.007 ms; the copy is issue-bound, not bandwidth-
    bound.  So the upcast stays where it is, as a separate explicit pass.

    There are two ways to fill the interior and the choice is made on the width
    of the innermost run, ``input.shape[-1]`` in bytes, against one cache line.

    ``copy_`` into the strided view moves every byte once and is the faster of
    the two whenever the run is at least a line.  That is every shape at or
    above 128 bytes -- (16,32,32,32)/p2 at 128 exactly, then 56**2, 64**2 and
    224**2, from 0.76x to 0.90x, and 420 GB/s on (32,64,128,128)/p2, where it is
    235 us against 285.  (32,64,210,210) at 210**2 is the one case in this half
    that is not decided by the copy: it only runs its dots in fp32, so the pair
    costs 1643 us against 1609 and the call comes out level at 0.98x.

    Below a line it collapses.  The run degenerates into a per-element gather
    and some shapes leave the AI Core for it entirely -- 408 us to move 0.8 MB
    on (16,32,24,24)/p2, 718 us for 2.3 MB on the 3-D shapes, 3-5 GB/s against
    300-420 where it stays -- while the shapes that keep it on the AI Core are
    merely slow.  F.pad writes the halo and one contiguous flat copy lands it in
    the slack buffer; that pair pays for the bytes twice but is flat at ~80 us
    on every shape that takes it: 3.0x on (16,32,24,24)/p2, 5.5x on
    (4,16,24,24,24)/p1, 1.55x on the 3-D 16**3s and 1.25x on (32,64,210,210).
    All six shapes under a line are on the F.pad side of every one of those
    ratios, so the boundary has measurements on both sides of it.

    A halo is also built under a zero padding, for the masked lanes the last
    output tile leaves (see the caller); there F.pad is a pure copy of the whole
    tensor on top of the one that follows it, and the four pointwise cases at
    (16,64,56,56) take the strided path instead, 127 us against 143.  The F.pad
    side writes the whole element count, so only the strided side still needs
    the zeros.
    """
    shape = list(input.shape)
    for i, p in enumerate(padding):
        shape[2 + i] += 2 * p
    numel = 1
    for s in shape:
        numel *= s
    if not any(padding) or input.shape[-1] * input.element_size() >= _PAD_RUN_BYTES:
        buf = torch.zeros(
            numel + max(_SLACK_ELEMS, tail), device=input.device, dtype=input.dtype
        )
        out = buf[:numel].view(shape)
        dst = [slice(None), slice(None)] + [
            slice(p, p + n) for p, n in zip(padding, input.shape[2:])
        ]
        out[tuple(dst)].copy_(input)
        return out
    buf = torch.empty(
        numel + max(_SLACK_ELEMS, tail), device=input.device, dtype=input.dtype
    )
    out = buf[:numel].view(shape)
    out.view(-1).copy_(
        torch.nn.functional.pad(
            input, tuple(v for p in reversed(padding) for v in (p, p))
        ).reshape(-1)
    )
    return out


def _pick_blocks(ow, oc_per_group, block_w_cap=None):
    """Pick (BLOCK_OC, BLOCK_W) to minimise the bytes the kernel loads per MAC.

    One program loads a (BLOCK_C, BLOCK_W) input tile and a (BLOCK_OC, BLOCK_C)
    weight tile per tap and reuses them for BLOCK_OC * BLOCK_W outputs, so the
    load is BLOCK_C * (BLOCK_W + BLOCK_OC) elements for BLOCK_C * BLOCK_W *
    BLOCK_OC MACs -- that is, ``1/BLOCK_W + 1/BLOCK_OC`` bytes per MAC, times
    the element size.  The channel tile BLOCK_C cancels.

    ``_BLOCK_ELEMS`` bounds BLOCK_OC * BLOCK_W, so the two are alternatives and
    the split between them matters: they are symmetric in the formula but not in
    what pins them, because BLOCK_W is the output row and BLOCK_OC is capped by
    the output channels.  Taking a whole row whenever it fits and giving the
    channel tile only the remainder -- which is what this did -- is the wrong
    end of the trade whenever the row is wider than the row is useful, i.e. on
    every shape with a long width and fewer than 128 output channels.  Measured
    on (32,64,210,210)/k5/s1 and the 1-D shapes, walking BLOCK_W down from the
    row buys the same 1/BLOCK_OC factor the sweeps show everywhere else.

    The search is a walk down the powers of two, stopping at the first step that
    does not help, which is the minimum: below sqrt(_BLOCK_ELEMS) the channel
    tile is already at its cap for the shape and the input term 1/BLOCK_W is the
    only one still moving, in the wrong direction.  Ties keep the wider tile, so
    a shape that does not benefit is left exactly as it was.
    """
    cap_oc = min(_BLOCK_OC_MAX, max(1, triton.next_power_of_2(oc_per_group)))
    block_w = min(block_w_cap or _BLOCK_W_MAX, max(1, triton.next_power_of_2(ow)))
    block_oc = min(cap_oc, max(1, _BLOCK_ELEMS // block_w))
    while block_w > _BLOCK_W_MIN:
        half = block_w // 2
        oc_half = min(cap_oc, max(1, _BLOCK_ELEMS // half))
        if 1.0 / half + 1.0 / oc_half < 1.0 / block_w + 1.0 / block_oc:
            block_w, block_oc = half, oc_half
        else:
            break
    return block_oc, block_w


def _can_use_dot(block_oc, block_w, c_in):
    """Whether the whole tile satisfies tl.dot's minimum dimension.

    A shape that fails this falls back to the FMA kernel rather than padding up
    to 16: the sub-16 cases are the small, launch-bound ones where padding would
    multiply a cheap kernel by 16/3, and the FMA path is already what runs there
    today.
    """
    return min(block_oc, block_w, c_in) >= _DOT_MIN


def _pick_block_c(c_in, use_dot, block_w):
    """Channel tile for the reduction, in elements.

    Only meaningful on the dot path; the FMA path walks one channel at a time.
    Capped at _BLOCK_C_MAX rather than left at c_in so a very wide channel dim
    is reduced in a few dots instead of one impractically large tile, and
    rounded up so a channel count between powers of two still gets one dot with
    a masked tail rather than a mostly-empty second block.

    Then shrunk, if it has to be, until the input tile fits the unified-buffer
    budget -- see _UB_TILE_MAX.  The channel axis is the one to give up rather
    than BLOCK_W: on (64,48,1024)/k5/s2 halving the channel tile costs 0.545 ms
    against 0.593 ms for halving the width, and on (64,64,1024)/k3/s2 the two
    tie at 0.362 ms.
    """
    if not use_dot:
        return 1
    block_c = min(_BLOCK_C_MAX, triton.next_power_of_2(c_in))
    while block_c > _DOT_MIN and block_c * block_w > _UB_TILE_MAX:
        block_c //= 2
    return block_c


def _densify_depthwise(weight, cin):
    """(OC, 1, *k) -> (OC, OC, *k) with the tap weights on the main diagonal.

    Depthwise is a dense convolution whose weight happens to be block diagonal
    with 1x1 blocks, so writing those blocks out and running the ordinary dense
    kernel computes exactly the same sum -- ``sum_c W[c, oc] * x[c]`` over a W
    that is zero off the diagonal is ``w[oc] * x[oc]``.  Every product is exact
    and no term is dropped; only the order the accumulator adds them in changes.

    It costs ``groups`` times the arithmetic, which sounds like the wrong trade
    until you price the alternative.  ``groups == C_in`` collapses BLOCK_OC to 1,
    so the FMA kernel runs one output channel per program with a scalar weight
    per tap: on (16,32,56,56)/k3/g32 that is 4904 us for 462 MMAC, against
    2.36 TMAC/s for the same arithmetic on the cube -- and the same shape's
    dense neighbour (16,32,32,32)/k3 does its 151 MMAC in 64 us.  Scaling that
    by output count puts the dense form at ~196 us, 25x under the FMA one.  The
    block-diagonal weight is 9216 elements here, which is why the build is two
    small tensor ops and not something worth fusing.

    Only the 1x1-block case is worth this: a group with more than one input or
    output channel already clears tl.dot's minimum dimension on its own and
    takes the dot path without any padding of the arithmetic.
    """
    OC, _, *k = weight.shape
    out = torch.zeros((OC, cin, *k), device=weight.device, dtype=weight.dtype)
    idx = torch.arange(min(OC, cin), device=weight.device)
    out[idx, idx] = weight[:, 0]
    return out


def _prep_weight(weight):
    """Move the output-channel axis last and make it contiguous.

    This is the single largest win in the file, and it is a layout fix rather
    than an arithmetic one.  The kernels below read one weight vector per
    (tap, channel) pair, indexed by output channel.  In the native (OC, C, KH,
    KW) layout that vector has stride C*KH*KW -- a gather, in the innermost loop,
    once per tap and channel.  Permuting to (KH, KW, C, OC) makes it unit-stride.

    Measured on (32,64,128,128)/3x3/pad2, identical kernel otherwise:
    140.8 ms native layout vs 36.1 ms permuted, 3.9x.  The cost is one small
    transpose per call on a tensor that is at most a few hundred thousand
    elements, against one gather per program per tap per channel.

    The permutation lands in a buffer with _SLACK_ELEMS to spare because the
    kernel's masked output-channel lanes still compute an address past the end
    of it.  The tail beyond the tensor is never read or written -- the mask
    suppresses both -- but the MTE faults on it before the mask is consulted if
    it lands outside the allocation; see _slack.
    """
    perm = weight.permute(*range(2, weight.dim()), 1, 0)
    numel = perm.numel()
    buf = torch.empty(numel + _SLACK_ELEMS, device=weight.device, dtype=weight.dtype)
    out = buf[:numel].view(perm.shape)
    out.copy_(perm)
    return out


def _slack(out_c_stride, block_oc, block_w):
    """Spare elements to append to an allocation the kernels index directly.

    A lane whose index is past the end of its tile keeps its address: the masks
    suppress the *access*, not the address computation, and the device's MTE
    faults on an address outside the allocation with 507015 "The DDR address of
    the MTE instruction is out of range" before the mask is consulted.  The
    overshoot is bounded by the tile -- BLOCK_W along the innermost axis, and a
    whole output plane per masked output-channel block on the channel axis --
    and rounding up to _SLACK_ELEMS covers the allocator's own alignment, which
    is what makes the fault look intermittent: whether a call survives depends
    on where the allocator happened to put the block.
    """
    return max(_SLACK_ELEMS, (block_oc - 1) * out_c_stride + block_w)


@libentry()
@triton.jit
def _direct_conv2d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N,
    H,
    W,
    OC,
    OH,
    OW,
    in_n_stride,
    in_c_stride,
    in_h_stride,
    in_w_stride,
    w_c_stride,
    out_n_stride,
    out_c_stride,
    out_h_stride,
    out_w_stride,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    C_IN: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NEED_CMASK: tl.constexpr,
    USE_DOT: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_group = tl.program_id(2)

    oc_per_group = OC // GROUPS
    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_glob = pid_group * oc_per_group + oc_off
    oc_mask = oc_off < oc_per_group

    # One (batch, output row) pair per row id; long rows are cut into segments.
    num_seg = tl.cdiv(OW, BLOCK_W)
    n = pid_row // (OH * num_seg)
    rem = pid_row % (OH * num_seg)
    oh = rem // num_seg
    seg = rem % num_seg

    ow = seg * BLOCK_W + tl.arange(0, BLOCK_W)
    w_ok = ow < OW

    in_row = input_ptr + n * in_n_stride
    in_group = pid_group * C_IN * in_c_stride

    acc = tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32)
    # Tap-major loop order: the tap offset and the tap pointer depend only on
    # (kh, kw), so hoisting them out of the channel loop keeps one address
    # computation per tap instead of one per tap and channel.
    #
    # The tap indices here are the *unpadded* ones -- ``oh * SH + kh * DH``, not
    # ``- PH`` -- so the launcher's zero halo is what makes them valid: the
    # padded tensor's element [ih + PH, iw + PW] is the input's [ih, iw], and
    # every tap that falls in the halo reads a real zero.  That is why the loads
    # below carry no mask and no bounds test.  It is also why they are safe: a
    # lane whose ow is past OW reads whatever the halo or the following row
    # holds, and its column of the accumulator is thrown away by the store's
    # w_ok.
    #
    # This is not a stylistic choice.  The same address expressed as
    # ``clamp(ow * SW + kw * DW - PW, 0, W - 1)`` -- the obvious way to keep a
    # masked lane inside the tensor -- takes the address out of the affine form
    # the backend's axis analysis needs, and the kernel then does the same work
    # an order of magnitude slower: measured 616 ms against 3.8 ms on
    # (32,64,128,128)/k3/s2, at 97% cube utilisation either way.  Materialising
    # the halo costs one pass over the input instead.
    for kh in tl.static_range(KH):
        ih = oh * SH + kh * DH
        for kw in tl.static_range(KW):
            iw = ow * SW + kw * DW
            tap_in = in_group + in_row + ih * in_h_stride + iw * in_w_stride
            # Weight is (KH, KW, C, OC), so the tap selects a plane and the
            # output channel is the unit-stride axis.
            tap_w = weight_ptr + (kh * KW + kw) * C_IN * w_c_stride + oc_glob
            if USE_DOT:
                # One (BLOCK_OC, BLOCK_C) x (BLOCK_C, BLOCK_W) dot per tap and
                # channel block, which is what puts this on the cube. Reducing
                # the same work as FMA instead runs it on the vector unit:
                # measured 3.29 ms vs 28.0 ms on the 2D core case.
                cc = tl.arange(0, BLOCK_C)
                for cb in range(0, C_IN, BLOCK_C):
                    if NEED_CMASK:
                        c_mask = (cb + cc) < C_IN
                        x = tl.load(
                            tap_in + (cb + cc)[:, None] * in_c_stride,
                            mask=c_mask[:, None],
                            other=0.0,
                        )
                        w = tl.load(
                            tap_w[:, None] + (cb + cc)[None, :] * w_c_stride,
                            mask=oc_mask[:, None] & c_mask[None, :],
                            other=0.0,
                        )
                    else:
                        x = tl.load(tap_in + (cb + cc)[:, None] * in_c_stride)
                        w = tl.load(
                            tap_w[:, None] + (cb + cc)[None, :] * w_c_stride,
                            mask=oc_mask[:, None],
                            other=0.0,
                        )
                    acc = tl.dot(w, x, acc)
            else:
                # Deliberately `range`, not `tl.static_range`: see the note above.
                for c in range(C_IN):
                    x = tl.load(tap_in + c * in_c_stride)
                    w = tl.load(tap_w + c * w_c_stride, mask=oc_mask, other=0.0)
                    acc += x[None, :] * w[:, None]

    tl.store(
        output_ptr
        + oc_glob[:, None] * out_c_stride
        + n * out_n_stride
        + oh * out_h_stride
        + ow[None, :] * out_w_stride,
        acc,
        mask=oc_mask[:, None] & w_ok[None, :],
    )


@libentry()
@triton.jit
def _direct_conv3d_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    N,
    D,
    H,
    W,
    OC,
    OD,
    OH,
    OW,
    in_n_stride,
    in_c_stride,
    in_d_stride,
    in_h_stride,
    in_w_stride,
    w_c_stride,
    out_n_stride,
    out_c_stride,
    out_d_stride,
    out_h_stride,
    out_w_stride,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SD: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    DD: tl.constexpr,
    DH: tl.constexpr,
    DW: tl.constexpr,
    C_IN: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_group = tl.program_id(2)

    oc_per_group = OC // GROUPS
    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_glob = pid_group * oc_per_group + oc_off
    oc_mask = oc_off < oc_per_group

    num_seg = tl.cdiv(OW, BLOCK_W)
    n = pid_row // (OD * OH * num_seg)
    rem = pid_row % (OD * OH * num_seg)
    od = rem // (OH * num_seg)
    rem = rem % (OH * num_seg)
    oh = rem // num_seg
    seg = rem % num_seg

    ow = seg * BLOCK_W + tl.arange(0, BLOCK_W)
    w_ok = ow < OW

    in_row = input_ptr + n * in_n_stride
    in_group = pid_group * C_IN * in_c_stride

    acc = tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32)
    # Channel-major, unlike the 2D kernel: putting the runtime channel loop
    # around 27 unrolled taps instead of 9 sends the Ascend compiler into the
    # same kind of multi-minute unroll the note above describes -- measured
    # stuck for > 15 min on (1,16,4,4,4) / 3x3x3.  Nine taps is under the
    # limit; twenty-seven is not.  There is no dot branch here for the related
    # reason measured above: 27 taps that each hold a tl.dot hang the device.
    #
    # The tap indices are unpadded and the loads unmasked, for the reason given
    # in the 2D kernel: the launcher materialises the zero halo, so the halo
    # supplies the padding and every tap address is in bounds.
    for c in range(C_IN):
        in_c_off = in_group + in_row + c * in_c_stride
        for kd in tl.static_range(KD):
            idd = od * SD + kd * DD
            for kh in tl.static_range(KH):
                ih = oh * SH + kh * DH
                for kw in tl.static_range(KW):
                    iw = ow * SW + kw * DW
                    x = tl.load(
                        in_c_off
                        + idd * in_d_stride
                        + ih * in_h_stride
                        + iw * in_w_stride
                    )
                    w = tl.load(
                        weight_ptr
                        + ((kd * KH + kh) * KW + kw) * C_IN * w_c_stride
                        + oc_glob
                        + c * w_c_stride,
                        mask=oc_mask,
                        other=0.0,
                    )
                    acc += x[None, :] * w[:, None]

    tl.store(
        output_ptr
        + oc_glob[:, None] * out_c_stride
        + n * out_n_stride
        + od * out_d_stride
        + oh * out_h_stride
        + ow[None, :] * out_w_stride,
        acc,
        mask=oc_mask[:, None] & w_ok[None, :],
    )


def _direct_conv2d(input, weight, padding, stride, dilation, groups):
    N, _, H, W = input.shape
    OC, weight_c, KH, KW = weight.shape
    PH, PW = padding
    SH, SW = stride
    DH, DW = dilation
    OH = _output_size(H, KH, SH, PH, DH)
    OW = _output_size(W, KW, SW, PW, DW)

    block_oc, block_w = _pick_blocks(OW, OC // groups)
    use_dot = _can_use_dot(block_oc, block_w, weight_c)

    # Depthwise is the one grouped shape the per-group tile cannot carry: with
    # one input and one output channel per group, BLOCK_OC is 1 and the tile
    # never reaches tl.dot.  Densifying the weight puts the same arithmetic on
    # the cube; see _densify_depthwise for what that measures.
    if not use_dot and groups > 1 and OC // groups == 1 and weight_c == 1:
        dense = _densify_depthwise(weight, input.shape[1])
        dense_oc, _ = _pick_blocks(OW, OC)
        if _can_use_dot(dense_oc, block_w, dense.shape[1]):
            return _direct_conv2d(input, dense, padding, stride, dilation, 1)

    block_c = _pick_block_c(weight_c, use_dot, block_w)
    arith = _arith_dtype(input, use_dot, KH * KW)

    # The halo is only built when a tap can leave the input: either the padding
    # is non-zero, or the output row is not a whole number of tiles, which
    # leaves masked lanes whose last tap runs past the end of the row.
    #
    # The upcast happens here rather than inside the pad, which is a pass this
    # does not need in principle and pays for anyway; see _pad_input for why the
    # cheaper ordering was dropped.  ``arith`` is fp32 except on the dot path,
    # where the native dtype is kept; see _arith_dtype.
    if any(padding) or OW % block_w:
        num_seg = triton.cdiv(OW, block_w)
        src = _pad_input(
            input.to(arith),
            padding,
            (num_seg * block_w - 1) * SW + (KW - 1) * DW + 1 - (W + 2 * PW),
        )
    else:
        src = input.to(arith)
    in_s = src.stride()

    out_numel = N * OC * OH * OW
    out_buf = torch.empty(
        out_numel + _slack(OH * OW, block_oc, block_w),
        device=input.device,
        dtype=input.dtype,
    )
    output = out_buf[:out_numel].view(N, OC, OH, OW)
    wt = _prep_weight(weight.to(arith))
    out_s = output.stride()

    grid = (
        N * OH * triton.cdiv(OW, block_w),
        triton.cdiv(OC // groups, block_oc),
        groups,
    )
    _direct_conv2d_kernel[grid](
        src,
        wt,
        output,
        N,
        src.shape[2],
        src.shape[3],
        OC,
        OH,
        OW,
        in_s[0],
        in_s[1],
        in_s[2],
        in_s[3],
        # Stride of the C axis, which sits at -2 in the transposed (*K, C, OC)
        # layout.  Not stride(1): that is the kernel-height axis.
        wt.stride(weight.dim() - 2),
        out_s[0],
        out_s[1],
        out_s[2],
        out_s[3],
        KH,
        KW,
        SH,
        SW,
        DH,
        DW,
        C_IN=weight_c,
        GROUPS=groups,
        BLOCK_OC=block_oc,
        BLOCK_W=block_w,
        BLOCK_C=block_c,
        NEED_CMASK=use_dot and weight_c % block_c != 0,
        USE_DOT=use_dot,
        num_warps=_NUM_WARPS,
    )
    return output


def _direct_conv3d(input, weight, padding, stride, dilation, groups):
    N, _, D, H, W = input.shape
    OC, weight_c, KD, KH, KW = weight.shape
    PD, PH, PW = padding
    SD, SH, SW = stride
    DD, DH, DW = dilation
    OD = _output_size(D, KD, SD, PD, DD)
    OH = _output_size(H, KH, SH, PH, DH)
    OW = _output_size(W, KW, SW, PW, DW)

    # No tl.dot in 3D; see the note above the kernel.
    block_oc, block_w = _pick_blocks(OW, OC // groups)

    # Same halo rule as the 2D launcher; see the note there.
    if any(padding) or OW % block_w:
        num_seg = triton.cdiv(OW, block_w)
        src = _pad_input(
            input.float(),
            padding,
            (num_seg * block_w - 1) * SW + (KW - 1) * DW + 1 - (W + 2 * PW),
        )
    else:
        src = input.float()
    in_s = src.stride()

    out_numel = N * OC * OD * OH * OW
    out_buf = torch.empty(
        out_numel + _slack(OD * OH * OW, block_oc, block_w),
        device=input.device,
        dtype=input.dtype,
    )
    output = out_buf[:out_numel].view(N, OC, OD, OH, OW)
    wt = _prep_weight(weight.float())
    out_s = output.stride()

    grid = (
        N * OD * OH * triton.cdiv(OW, block_w),
        triton.cdiv(OC // groups, block_oc),
        groups,
    )
    _direct_conv3d_kernel[grid](
        src,
        wt,
        output,
        N,
        src.shape[2],
        src.shape[3],
        src.shape[4],
        OC,
        OD,
        OH,
        OW,
        in_s[0],
        in_s[1],
        in_s[2],
        in_s[3],
        in_s[4],
        # Stride of the C axis in the transposed (*K, C, OC) layout; see the
        # 2D launcher.
        wt.stride(weight.dim() - 2),
        out_s[0],
        out_s[1],
        out_s[2],
        out_s[3],
        out_s[4],
        KD,
        KH,
        KW,
        SD,
        SH,
        SW,
        DD,
        DH,
        DW,
        C_IN=weight_c,
        GROUPS=groups,
        BLOCK_OC=block_oc,
        BLOCK_W=block_w,
        num_warps=_NUM_WARPS,
    )
    return output


def _direct_conv(input, weight, padding, stride, dilation, groups, ndim):
    if ndim == 1:
        # Lift to 2D with a leading unit height rather than a trailing unit
        # width: the width axis is the one the kernel vectorizes over, so a
        # unit *width* would leave a single live lane per program.
        return _direct_conv2d(
            input.unsqueeze(2),
            weight.unsqueeze(2),
            [0, padding[0]],
            [1, stride[0]],
            [1, dilation[0]],
            groups,
        ).squeeze(2)
    if ndim == 2:
        return _direct_conv2d(input, weight, padding, stride, dilation, groups)
    return _direct_conv3d(input, weight, padding, stride, dilation, groups)


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
    """
    Ascend implementation of the bias-free cuDNN convolution.

    Dimensions, parameter normalization and the returned layout follow the
    shared implementation in ``flag_gems/ops/cudnn_convolution.py``; only the
    arithmetic and the tiling differ (see the note above the direct kernels).

    ``benchmark``, ``deterministic`` and ``allow_tf32`` are accepted for
    interface compatibility and do not select an algorithm.
    """
    logger.debug("GEMS_ASCEND CUDNN_CONVOLUTION")

    ndim = input.ndim - 2
    if ndim not in (1, 2, 3):
        raise ValueError(
            f"cudnn_convolution only supports 1D, 2D, and 3D convolutions, "
            f"got input with {ndim} spatial dimensions"
        )
    padding = _to_list(padding, ndim)
    stride = _to_list(stride, ndim)
    dilation = _to_list(dilation, ndim)

    # Depthwise used to be handed to the shared kernel on the grounds that it
    # has no cross-channel reduction, so a tile with one output channel per
    # program and a scalar weight per tap would be cheaper than a (BLOCK_OC,
    # BLOCK_W) outer product that is 7/8 masked out.  That is true of the
    # *arithmetic* and false of the memory access: the shared tile addresses its
    # input through a gather per tap, which is the one thing this backend
    # charges for (see the note at the top of this file).  The direct path's row
    # tiling makes those loads contiguous instead, and depthwise is the shape it
    # suits best -- groups == C_in collapses BLOCK_OC to 1, so there is no
    # masked-out outer product left to pay for either way.
    #
    # Measured, wall clock, bf16, against the shared kernel on the same tensors
    # and the same torch reference:
    #
    #   (16,32,56,56)  k3 g=32   28.63 ms -> 4.43 ms   6.5x
    #   (16,32,1024)   k3 g=32    3.55 ms -> 0.45 ms   7.8x
    #   (2,16,12,12,12) k3 g=16   3.10 ms -> 1.82 ms   1.7x
    #
    # all three bit-exact against the shared path (max abs diff 0.0).  Same
    # lesson as the channel bound below: the comparison that picked the shared
    # path was made against the pre-row-tiling kernel and did not survive it.
    if input.dtype == torch.float32:
        return _direct_conv(input, weight, padding, stride, dilation, groups, ndim)

    # --- fp16 / bf16 -------------------------------------------------------
    # A 1x1 kernel with a single group and no padding is a plain GEMM; the
    # shared pointwise kernel already covers it and keeps the tensor core.
    if (
        groups == 1
        and all(k == 1 for k in weight.shape[2:])
        and all(s == 1 for s in stride)
        and all(d == 1 for d in dilation)
        and all(p == 0 for p in padding)
    ):
        return _shared_pointwise_conv(input, weight, ndim)

    # Everything else: the direct kernels above, in the dtype each of them
    # asks for.  The kernels are handed the tensor as it arrived rather than an
    # upcast image of it; _direct_conv2d/_direct_conv3d decide per kernel, and
    # only the tl.dot branch keeps the native dtype (see _arith_dtype).  Handing
    # bf16 to a kernel that is *not* on the dot path is a large loss: the kernel
    # is unchanged but bf16 makes the 1D shape that runs in 1.06 ms in fp32 take
    # 87.5 ms, because the Ascend backend does not vectorize the in-loop bf16
    # load / .to(tl.float32).  Converting once up front costs a linear pass over
    # the operands and buys that 80x back, so those paths still do it.
    #
    # Folding that conversion into the halo's copy is the obvious saving and was
    # measured: no better on any case, 4.8% worse on one.  See _pad_input.
    #
    # It also fixes a correctness hole: the shared conv1d/2d/3d kernels
    # mis-compute 1x1 kernels in every dtype (measured 18.0 absolute error in
    # fp16 and 17.99 in fp32 against an fp64 reference), and a 1x1 kernel
    # escapes the pointwise branch above as soon as it carries padding.  The
    # direct kernels handle 1x1 fine, so those are routed here regardless of
    # channel count.
    #
    # There is deliberately no channel bound on this.  There used to be one --
    # the direct path was capped at C_in <= 64, on the grounds that it walks
    # C_in * KH * KW taps per output where the implicit GEMM amortizes over a
    # larger K.  That was measured against the kernel that clamped every tap to
    # stay in bounds, and materialising the halo instead removed exactly the
    # per-tap cost the bound was reasoning about.  Re-measured on the current
    # kernel (bf16, 3x3, s1, oc == C_in, (16, C_in, 56, 56)):
    #
    #   C_in      8      16      32      64     256
    #   shared  10.5    16.0    32.8    89.1   902.6  ms
    #   direct   0.78    0.45    0.46    0.70    8.66  ms
    #   ratio    13x     35x     71x    127x    104x
    #
    # and at s2 / C_in=256, 81.7 ms against 6.14 ms (13x).  The margin grows
    # with C_in rather than shrinking -- the shared path's cost grows with it
    # while the direct path's stays flat -- so the old bound was not merely
    # stale, it was inverted: it handed the shapes with the most to gain to the
    # slower path.  Every shape measured is 12x-127x the direct path's cost,
    # which is well outside any noise that would make this a close call.
    return _direct_conv(input, weight, padding, stride, dilation, groups, ndim)
