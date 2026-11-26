"""Mamba (selective state space) block in plain PyTorch, for running SEMamba without mamba_ssm.

Written for this repository after the reference (non-CUDA) code paths of
https://github.com/state-spaces/mamba (Apache-2.0, see LICENSE_mamba in se/arch/xlstm_senet/):
`mamba_ssm/modules/mamba_simple.py` (Mamba, Block), `mamba_ssm/ops/selective_scan_interface.py`
(`selective_scan_ref`, `mamba_inner_ref`) and `mamba_ssm/ops/triton/layernorm.py` (`rms_norm_ref`).
Parameter names and shapes are the package's, so a state_dict trained with mamba_ssm loads as is.

Why: mamba_ssm needs CUDA and a compiler to install and has no CPU path, and the submission
must run from `pip install -r requirements.txt` on a CPU. Only what SEMamba uses is here
(Mamba v1, no bias on in_proj/out_proj, RMSNorm, the non-fused Block).

Speed on a CPU. The scan is the recurrence h[t] = exp(delta[t] * A) * h[t-1] + delta[t] * B[t] * u[t]
over the sequence, with h of size [batch, d_inner, d_state]. A sequential pass over t, vectorised
over batch, channels and states, does the least arithmetic there is to do. Measured on 8 CPU
threads for the shapes of a 3-second file (2026-11-26): a log-depth parallel prefix scan
(Hillis-Steele) is about 12 times slower (log2(L) times the arithmetic on [batch, L, d_inner,
d_state] tensors that do not fit in the cache), and computing exp() for 4 to 32 steps at once
is 5 to 20 % slower than one step at a time. So the loop stays sequential, with in-place
operations, and is compiled with torch.jit.script when that is available (about 10-25 % less
time than the Python loop; the numbers are identical). tests/test_semamba.py checks it
against a direct transcription of `selective_scan_ref`.
"""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn

def selective_scan_reference(u, delta, A, B, C):
    """Direct transcription of selective_scan_ref (without D, z). [B, L, D], [B, L, D], [D, N], [B, L, N] x 2."""
    h = u.new_zeros(u.shape[0], u.shape[2], A.shape[1])
    ys = []
    for t in range(u.shape[1]):
        h = torch.exp(delta[:, t, :, None] * A) * h + (delta[:, t] * u[:, t])[:, :, None] * B[:, t, None, :]
        ys.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
    return torch.stack(ys, dim=1)


def _selective_scan(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                    C: torch.Tensor) -> torch.Tensor:
    """y[b, t, d] = sum_n C[b, t, n] * h[b, t, d, n], the same numbers as selective_scan_reference."""
    batch, length, d_inner = u.shape
    h = torch.zeros(batch, d_inner, A.shape[1], dtype=u.dtype, device=u.device)
    y = torch.empty_like(u)
    du = delta * u
    for t in range(length):
        h.mul_(torch.exp(delta[:, t, :, None] * A))
        h.addcmul_(du[:, t, :, None], B[:, t, None, :])
        y[:, t] = torch.bmm(h, C[:, t, :, None]).squeeze(-1)
    return y


try:        # compiled loop; the plain function gives the same numbers if scripting is unavailable
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selective_scan = torch.jit.script(_selective_scan)
except Exception:       # noqa: BLE001
    selective_scan = _selective_scan


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class Mamba(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 layer_idx=None):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state, self.d_conv = d_state, d_conv
        self.dt_rank = math.ceil(d_model / 16)
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
                                  .repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.reference_scan = False     # tests: run the transcription of selective_scan_ref

    def forward(self, hidden_states):
        """[B, L, d_model] -> [B, L, d_model]."""
        length = hidden_states.shape[1]
        x, z = self.in_proj(hidden_states).chunk(2, dim=-1)                  # [B, L, D] each
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :length]).transpose(1, 2).contiguous()
        dt, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))                                 # bias inside, as delta_bias
        A = -torch.exp(self.A_log.float())
        scan = selective_scan_reference if self.reference_scan else selective_scan
        y = scan(x, delta, A, B.contiguous(), C.contiguous())
        y = (y + x * self.D) * F.silu(z)
        return self.out_proj(y)


class Block(nn.Module):
    """Pre-norm residual wrapper (mamba_ssm Block, fused_add_norm=False): returns (mixed, residual)."""

    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False,
                 residual_in_fp32=False):
        super().__init__()
        if fused_add_norm:
            raise NotImplementedError("fused_add_norm needs mamba_ssm's triton kernels")
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

    def forward(self, hidden_states, residual=None):
        residual = hidden_states + residual if residual is not None else hidden_states
        return self.mixer(self.norm(residual)), residual
