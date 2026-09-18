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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def lamb_part1_kernel(
    p_ptr,
    m_ptr,
    v_ptr,
    g_ptr,
    n,
    grad_scale,
    b1,
    b2,
    eps,
    decay,
    w_l2_i_ptr,
    u_l2_i_ptr,
    mode: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Adam moment update plus per-program partial reduction of ||w||^2 and ||u||^2.

    Mirrors ``lamb_cuda_kernel_part1`` in DeepSpeed: update the first/second
    moments, build the Adam update vector ``update = m/denom + decay*w``, then
    reduce the squared weight/update norms inside this program and emit one
    pair of partial sums per program.
    """
    pid = tle.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    g = tl.load(g_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

    scaled_grad = g / grad_scale
    m_new = b1 * m + (1 - b1) * scaled_grad
    v_new = b2 * v + (1 - b2) * scaled_grad * scaled_grad

    if mode == 0:
        denom = tl.sqrt(v_new + eps)
    else:
        denom = tl.sqrt(v_new) + eps

    update = m_new / denom + decay * p

    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)

    # Masked lanes loaded 0.0, so their squares contribute nothing to the sum.
    reg_w = tl.sum(p * p)
    reg_u = tl.sum(update * update)

    tl.store(w_l2_i_ptr + pid, reg_w)
    tl.store(u_l2_i_ptr + pid, reg_u)


@libentry()
@triton.jit
def lamb_part2_kernel(
    num_blocks,
    w_l2_i_ptr,
    u_l2_i_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce the per-program partial norms down to a single scalar each.

    Mirrors ``lamb_cuda_kernel_part2`` in DeepSpeed: a single program sums the
    ``num_blocks`` partial sums and stores the result back at index 0.
    """
    offsets = tl.arange(0, BLOCK_SIZE)
    w_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    u_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in range(0, num_blocks, BLOCK_SIZE):
        idx = start + offsets
        mask = idx < num_blocks
        w_acc += tl.load(w_l2_i_ptr + idx, mask=mask, other=0.0)
        u_acc += tl.load(u_l2_i_ptr + idx, mask=mask, other=0.0)

    tl.store(w_l2_i_ptr, tl.sum(w_acc))
    tl.store(u_l2_i_ptr, tl.sum(u_acc))


@libentry()
@triton.jit
def lamb_part3_kernel(
    p_ptr,
    p_copy_ptr,
    m_ptr,
    v_ptr,
    n,
    max_coeff,
    min_coeff,
    eps,
    step_size,
    decay,
    w_l2_i_ptr,
    u_l2_i_ptr,
    lamb_coeff_ptr,
    mode: tl.constexpr,
    HAS_P_COPY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute the layer-wise trust ratio and apply the parameter update.

    Mirrors ``lamb_cuda_kernel_part3`` in DeepSpeed: ``trust = ||w|| / ||u||``
    (1.0 when either norm is zero), clamped to ``[min_coeff, max_coeff]``, then
    ``p = p - step_size * trust * update`` with ``update`` recomputed from the
    already-updated moments.
    """
    reg_w = tl.sqrt(tl.load(w_l2_i_ptr))
    reg_u = tl.sqrt(tl.load(u_l2_i_ptr))

    # Guard the division so a zero norm yields coeff == 1.0 (DeepSpeed leaves it
    # unclamped in that case); tl.where evaluates both sides, so keep the
    # denominator finite even when reg_u == 0.
    denom = tl.where(reg_u == 0.0, 1.0, reg_u)
    ratio = reg_w / denom
    ratio = tl.minimum(tl.maximum(ratio, min_coeff), max_coeff)
    lamb_coeff = tl.where((reg_w == 0.0) | (reg_u == 0.0), 1.0, ratio)

    pid = tle.program_id(0)
    if pid == 0:
        tl.store(lamb_coeff_ptr, lamb_coeff)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
    m = tl.load(m_ptr + offsets, mask=mask, other=0.0)
    v = tl.load(v_ptr + offsets, mask=mask, other=0.0)

    if mode == 0:
        denom = tl.sqrt(v + eps)
    else:
        denom = tl.sqrt(v) + eps

    update = m / denom + decay * p
    p_new = p - step_size * lamb_coeff * update

    tl.store(p_ptr + offsets, p_new, mask=mask)
    if HAS_P_COPY:
        tl.store(p_copy_ptr + offsets, p_new.to(p_copy_ptr.dtype.element_ty), mask=mask)


def lamb(
    p,
    p_copy,
    m,
    v,
    g,
    lr,
    beta1,
    beta2,
    max_coeff,
    min_coeff,
    eps,
    grad_scale,
    step,
    mode,
    bias_correction,
    decay,
):
    """Fused LAMB (Layer-wise Adaptive Moments) optimizer step.

    Faithful port of DeepSpeed's ``fused_lamb`` CUDA operator. Performs one
    Adam-style moment update, then scales the step per-layer by the trust ratio
    ``||w|| / ||update||`` (clamped to ``[min_coeff, max_coeff]``), which makes
    training robust to very large batch sizes.

    Args:
        p (Tensor): model parameter (fp32).
        p_copy (Tensor): optional reduced-precision copy of the updated weights;
            pass an empty tensor to skip.
        m (Tensor): first moment estimate (fp32).
        v (Tensor): second moment estimate (fp32).
        g (Tensor): gradient (fp32).
        lr (float): learning rate.
        beta1 (float): coefficient for the running average of the gradient.
        beta2 (float): coefficient for the running average of the squared gradient.
        max_coeff (float): upper bound of the trust ratio.
        min_coeff (float): lower bound of the trust ratio.
        eps (float): term added to the denominator for numerical stability.
        grad_scale (float): factor dividing the gradient before the update.
        step (int): optimizer step, used only for bias correction.
        mode (int): 0 keeps eps under the square root, 1 keeps it outside.
        bias_correction (int): 1 enables Adam bias correction, 0 disables it.
        decay (float): weight decay added to the update vector.

    Returns:
        Tensor: the per-layer trust ratio as a one-element fp32 tensor.
    """
    logger.debug("GEMS LAMB")

    assert p.dtype == torch.float32, "lamb only supports float32 parameters"
    assert p.is_cuda, "lamb only supports CUDA tensors"

    n = p.numel()
    assert m.numel() == n and v.numel() == n and g.numel() == n
    assert p_copy.numel() == 0 or p_copy.numel() == n

    # Adam bias correction, matching DeepSpeed's step_size computation.
    if bias_correction == 1:
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
    else:
        step_size = lr

    # A wide block leaves only a handful of programs for a mid-sized tensor, so
    # the kernels spend their time waiting on memory instead of covering it.
    # Capping the block at 1024 elements keeps enough programs in flight to hide
    # that latency without making the cross-block reduction expensive.
    BLOCK_SIZE = triton.next_power_of_2(n)
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)
    num_blocks = triton.cdiv(n, BLOCK_SIZE)

    w_l2_i = torch.empty((num_blocks,), dtype=torch.float32, device=p.device)
    u_l2_i = torch.empty((num_blocks,), dtype=torch.float32, device=p.device)
    lamb_coeff_val = torch.empty((1,), dtype=torch.float32, device=p.device)

    has_p_copy = p_copy.numel() > 0
    p_copy_in = p_copy if has_p_copy else torch.empty((0,), device=p.device)

    with torch_device_fn.device(p.device):
        lamb_part1_kernel[(num_blocks,)](
            p,
            m,
            v,
            g,
            n,
            grad_scale,
            beta1,
            beta2,
            eps,
            decay,
            w_l2_i,
            u_l2_i,
            mode=mode,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        reduce_block = min(max(triton.next_power_of_2(num_blocks), 32), 2048)
        lamb_part2_kernel[(1,)](
            num_blocks,
            w_l2_i,
            u_l2_i,
            BLOCK_SIZE=reduce_block,
        )

        lamb_part3_kernel[(num_blocks,)](
            p,
            p_copy_in,
            m,
            v,
            n,
            max_coeff,
            min_coeff,
            eps,
            step_size,
            decay,
            w_l2_i,
            u_l2_i,
            lamb_coeff_val,
            mode=mode,
            HAS_P_COPY=has_p_copy,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    return lamb_coeff_val
