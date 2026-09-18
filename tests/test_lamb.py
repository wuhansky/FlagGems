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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.lamb
@pytest.mark.parametrize("shape", [(1024,), (4096,), (16384,)])
@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize("bias_correction", [0, 1])
def test_lamb(shape, mode, bias_correction):
    """Test the fused LAMB optimizer step against a manual PyTorch reference."""
    dtype = torch.float32

    p = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    g = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    m = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    v = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    p_copy = torch.empty((0,), dtype=dtype, device=flag_gems.device)

    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    max_coeff = 10.0
    min_coeff = 0.01
    eps = 1e-8
    grad_scale = 1.0
    decay = 0.01
    step = 1

    # Manual reference (mirrors DeepSpeed's fused_lamb_cuda).
    ref_p = p.clone()
    ref_m = m.clone()
    ref_v = v.clone()

    scaled_grad = g / grad_scale
    ref_m = beta1 * ref_m + (1 - beta1) * scaled_grad
    ref_v = beta2 * ref_v + (1 - beta2) * scaled_grad * scaled_grad

    if mode == 0:
        denom = torch.sqrt(ref_v + eps)
    else:
        denom = torch.sqrt(ref_v) + eps
    update = ref_m / denom + decay * ref_p

    w_norm = torch.sqrt((ref_p * ref_p).sum())
    u_norm = torch.sqrt((update * update).sum())
    if w_norm.item() == 0.0 or u_norm.item() == 0.0:
        coeff = torch.ones((), dtype=dtype, device=flag_gems.device)
    else:
        coeff = (w_norm / u_norm).clamp(min_coeff, max_coeff)

    if bias_correction == 1:
        step_size = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)
    else:
        step_size = lr
    ref_p = ref_p - step_size * coeff * update

    lamb_coeff = flag_gems.lamb(
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
    )

    # The trust ratio reported by the operator must match the reference.
    utils.gems_assert_close(
        utils.to_reference(lamb_coeff),
        utils.to_reference(coeff.reshape(1)),
        dtype,
    )
    # The updated parameters and moments must match the reference.
    utils.gems_assert_close(utils.to_reference(p), utils.to_reference(ref_p), dtype)
    utils.gems_assert_close(utils.to_reference(m), utils.to_reference(ref_m), dtype)
    utils.gems_assert_close(utils.to_reference(v), utils.to_reference(ref_v), dtype)


@pytest.mark.lamb
@pytest.mark.parametrize("shape", [(1024,), (4096,)])
def test_lamb_p_copy(shape):
    """Test the optional reduced-precision output copy."""
    dtype = torch.float32

    p = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    g = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    m = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    v = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    p_copy = torch.empty(shape, dtype=dtype, device=flag_gems.device)

    flag_gems.lamb(
        p,
        p_copy,
        m,
        v,
        g,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        max_coeff=10.0,
        min_coeff=0.01,
        eps=1e-8,
        grad_scale=1.0,
        step=1,
        mode=1,
        bias_correction=1,
        decay=0.0,
    )

    utils.gems_assert_close(utils.to_reference(p_copy), utils.to_reference(p), dtype)


def reference_lamb_step(
    param,
    grad,
    exp_avg,
    exp_avg_sq,
    step,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    max_coeff,
    min_coeff,
    mode,
    bias_correction,
):
    """One LAMB step in fp32, mirroring DeepSpeed's fused_lamb_cuda.

    Keeps the same order of operations as the Triton kernel so the reference and
    the fused implementation agree up to floating-point rounding.
    """
    m = beta1 * exp_avg + (1 - beta1) * grad
    v = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

    if mode == 0:
        denom = torch.sqrt(v + eps)
    else:
        denom = torch.sqrt(v) + eps
    update = m / denom + weight_decay * param

    w_norm = torch.sqrt((param * param).sum())
    u_norm = torch.sqrt((update * update).sum())
    if w_norm.item() == 0.0 or u_norm.item() == 0.0:
        coeff = torch.tensor(1.0, dtype=param.dtype, device=param.device)
    else:
        coeff = (w_norm / u_norm).clamp(min_coeff, max_coeff)

    if bias_correction:
        step_size = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)
    else:
        step_size = lr

    param.copy_(param - step_size * coeff * update)
    exp_avg.copy_(m)
    exp_avg_sq.copy_(v)


@pytest.mark.lamb
@pytest.mark.parametrize("mode", [0, 1], ids=["eps_inside_sqrt", "eps_outside_sqrt"])
@pytest.mark.parametrize(
    "bias_correction", [True, False], ids=["bias_corr", "no_bias_corr"]
)
def test_lamb_matches_reference(mode, bias_correction):
    """Functional test: run LAMB for several steps and compare against a manual reference.

    Mirrors DeepSpeed's ``test_fused_adam_matches_reference``: multiple parameter
    tensors, fresh gradients each step, accumulating first/second moments, with the
    fused operator compared to an independent per-step reference.
    """
    dtype = torch.float32
    torch.manual_seed(0)
    lr, betas, eps, weight_decay = 1e-2, (0.9, 0.999), 1e-8, 0.1
    max_coeff, min_coeff = 10.0, 0.01

    gems_params = [
        torch.randn(1024, dtype=dtype, device=flag_gems.device) for _ in range(3)
    ]
    ref_params = [p.clone() for p in gems_params]
    gems_m = [torch.zeros_like(p) for p in gems_params]
    ref_m = [torch.zeros_like(p) for p in gems_params]
    gems_v = [torch.zeros_like(p) for p in gems_params]
    ref_v = [torch.zeros_like(p) for p in gems_params]

    for step in range(1, 6):
        for i in range(len(gems_params)):
            grad = torch.randn_like(gems_params[i])

            reference_lamb_step(
                ref_params[i],
                grad,
                ref_m[i],
                ref_v[i],
                step,
                lr,
                betas[0],
                betas[1],
                eps,
                weight_decay,
                max_coeff,
                min_coeff,
                mode,
                bias_correction,
            )

            flag_gems.lamb(
                gems_params[i],
                torch.empty((0,), dtype=dtype, device=flag_gems.device),
                gems_m[i],
                gems_v[i],
                grad,
                lr,
                betas[0],
                betas[1],
                max_coeff,
                min_coeff,
                eps,
                1.0,
                step,
                mode,
                int(bias_correction),
                weight_decay,
            )

    for gems_param, ref_param in zip(gems_params, ref_params):
        utils.gems_assert_close(
            utils.to_reference(gems_param), utils.to_reference(ref_param), dtype
        )
    for gems_exp_avg, ref_exp_avg in zip(gems_m, ref_m):
        utils.gems_assert_close(
            utils.to_reference(gems_exp_avg), utils.to_reference(ref_exp_avg), dtype
        )
    for gems_exp_avg_sq, ref_exp_avg_sq in zip(gems_v, ref_v):
        utils.gems_assert_close(
            utils.to_reference(gems_exp_avg_sq),
            utils.to_reference(ref_exp_avg_sq),
            dtype,
        )
