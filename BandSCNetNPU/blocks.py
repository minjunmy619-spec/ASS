"""Core blocks for Band-SCNet-NPU.

All blocks operate on 4D tensors ``[B, C, T, F']`` and obey the NPU
constraints documented in ``AGENT.md``:

* only 2D convolutions (never 1D) with kernels/strides in the allowed range
* tensor rank <= 4 in the deployed graph
* causal temporal behaviour with an explicit streaming-state contract
* PReLU / Sigmoid / Softmax / bmm / elementwise arithmetic only
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from spectral_feature_compression.core.model.online_sfc_2d import (
    CausalConv2d,
    RMSNorm2d,
    _runtime_assert,
)


class GatedAct(nn.Module):
    """GLU-style gate ``a * sigmoid(b)`` where the channel dim is split 2-way.

    The input channel count must be even. This is used instead of SiLU in the
    deployed graph so the exported ONNX contains only ``Sigmoid`` + ``Mul``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)


class CrossBandBlock(nn.Module):
    """Cross-band frequency mixer.

    Structure:
        x -> RMSNorm -> Conv2d(C, 2C, (1, Kf)) -> GatedAct -> Conv2d(C, C, 1) -> + residual

    The block is stateless w.r.t. time (no temporal convolution), so it needs
    no streaming state.
    """

    def __init__(self, channels: int, freq_kernel: int = 3):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"channels must be even, got {channels}")
        if freq_kernel % 2 != 1:
            raise ValueError(f"freq_kernel must be odd, got {freq_kernel}")
        if (freq_kernel - 1) > 14:
            raise ValueError(
                f"freq_kernel violates (k-1)*d <= 14 rule: (k-1) = {freq_kernel - 1}"
            )
        self.channels = channels
        self.freq_kernel = freq_kernel
        self.norm = RMSNorm2d(channels)
        self.freq_conv = nn.Conv2d(
            channels,
            2 * channels,
            kernel_size=(1, freq_kernel),
            padding=(0, freq_kernel // 2),
            bias=True,
        )
        self.act = GatedAct()
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        y = self.norm(x)
        y = self.freq_conv(y)
        y = self.act(y)
        y = self.pointwise(y)
        return x + y


class PooledChannelMixer(nn.Module):
    """Parameter-efficient capacity branch with frequency-pooled compute.

    This is a stateless current-frame channel mixer:

        x -> RMSNorm -> mean over F -> Conv2d(C, 2H, 1) -> gate -> Conv2d(H, C, 1)

    The hidden dimension can be large, adding millions of trainable parameters
    without increasing streaming state. Compute stays modest because the
    expensive channel projections run at frequency width 1, then broadcast back
    across the compressed separator frequency axis via Add.
    """

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.norm = RMSNorm2d(channels)
        self.expand = nn.Conv2d(channels, 2 * hidden_channels, kernel_size=1, bias=True)
        self.act = GatedAct()
        self.project = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        y = self.norm(x).mean(dim=3, keepdim=True)
        y = self.expand(y)
        y = self.act(y)
        y = self.project(y)
        return x + y


class BoundedCausalAttn(nn.Module):
    """Temporal causal attention on frequency-pooled features.

    To keep the streaming state under the 192 KiB DSP quota we attend on
    **frequency-pooled** activations: the KV cache is shared across all F'
    bins, while Q is per-frequency so the attention can still modulate the
    feature map along the F' axis via a broadcasted gate.

    State per block: ``[B, num_heads, window, 2 * head_dim]`` (K and V stored
    together along the last axis, packed into a single 4D tensor so NPU rule
    4 stays satisfied). For heads=4, W=16, D=8, fp16 this is 2 KiB per block.

    Outputs have shape ``[B, C, T, 1]`` and are expected to be broadcast-added
    to a ``[B, C, T, F']`` feature map by the caller (ONNX ``Add`` op
    broadcasts without emitting ``Expand``).
    """

    def __init__(
        self,
        channels: int,
        *,
        window: int,
        num_heads: int = 4,
        head_dim: int = 8,
    ):
        super().__init__()
        if window <= 0:
            raise ValueError(f"attention window must be positive, got {window}")
        self.channels = channels
        self.window = window
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim

        self.norm = RMSNorm2d(channels)
        self.qkv_proj = nn.Conv2d(channels, 3 * self.inner_dim, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(self.inner_dim, channels, kernel_size=1, bias=True)
        self._scale = 1.0 / math.sqrt(head_dim)

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _freq_pool(x: torch.Tensor) -> torch.Tensor:
        """[B, C, T, F] -> [B, C, T, 1] via mean over F."""
        return x.mean(dim=3, keepdim=True)

    # -- full-sequence forward ------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full-sequence causal-windowed attention.

        To match streaming semantics exactly, we left-pad the time axis with
        ``W-1`` zero frames (mimicking the zero-initialised KV ring buffer at
        t=0) and then apply a causal+window mask over the padded sequence.
        Only the last T output frames are returned.
        """
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        b, _, t, _ = x.shape
        y = self.norm(x)
        x_pool = self._freq_pool(y)  # [B, C, T, 1]
        qkv = self.qkv_proj(x_pool)  # [B, 3*H*D, T, 1]
        q, k, v = qkv.chunk(3, dim=1)

        pad = self.window - 1
        if pad > 0:
            # Pad K/V on the left with zero frames so streaming==full numerically.
            zero_pad = torch.zeros(b, self.inner_dim, pad, 1, dtype=k.dtype, device=k.device)
            k = torch.cat([zero_pad, k], dim=2)  # [B, H*D, T+W-1, 1]
            v = torch.cat([zero_pad, v], dim=2)
        t_kv = t + pad

        def _bht(t_: torch.Tensor, length: int) -> torch.Tensor:
            return (
                t_.reshape(b, self.num_heads, self.head_dim, length)
                .permute(0, 1, 3, 2)
                .reshape(b * self.num_heads, length, self.head_dim)
            )

        q3 = _bht(q, t)       # [B*H, T, D]
        k3 = _bht(k, t_kv)    # [B*H, T+W-1, D]
        v3 = _bht(v, t_kv)    # [B*H, T+W-1, D]

        scores = torch.bmm(q3, k3.transpose(1, 2)) * self._scale  # [B*H, T, T+W-1]

        # Causal + bounded-window mask. For each query q (index in [0, T)),
        # the allowed keys have padded-indices in [q, q + W).
        q_idx = torch.arange(t, device=x.device).unsqueeze(1)       # [T, 1]
        k_idx = torch.arange(t_kv, device=x.device).unsqueeze(0)    # [1, T+W-1]
        diff = k_idx - q_idx                                         # [T, T+W-1]
        allowed = (diff >= 0) & (diff < self.window)
        neg_inf = torch.full((), float("-inf"), device=x.device, dtype=scores.dtype)
        zero = torch.zeros((), device=x.device, dtype=scores.dtype)
        mask = torch.where(allowed, zero, neg_inf)
        scores = scores + mask.unsqueeze(0)

        attn = torch.softmax(scores, dim=-1)
        out_bh = torch.bmm(attn, v3)  # [B*H, T, D]
        out = out_bh.reshape(b, self.num_heads, t, self.head_dim).permute(0, 1, 3, 2).reshape(
            b, self.inner_dim, t, 1
        )
        return self.out_proj(out)  # [B, C, T, 1]

    # -- streaming API --------------------------------------------------------

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,  # unused but kept for API symmetry with CausalConv2d
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        del freq_bins  # the F axis is pooled out before attention
        return torch.zeros(
            batch_size,
            self.num_heads,
            self.window,
            2 * self.head_dim,
            device=device,
            dtype=dtype,
        )

    def forward_stream(
        self,
        x: torch.Tensor,
        state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        _runtime_assert(x.shape[2] == 1, f"Expected single-frame input, got T={x.shape[2]}")
        b = x.shape[0]
        if state is None:
            state = self.init_stream_state(b, freq_bins=x.shape[-1], device=x.device, dtype=x.dtype)

        y = self.norm(x)
        x_pool = self._freq_pool(y)  # [B, C, 1, 1]
        qkv = self.qkv_proj(x_pool)  # [B, 3*H*D, 1, 1]
        q, k, v = qkv.chunk(3, dim=1)
        # q, k, v: [B, H*D, 1, 1]
        q_bhd = q.reshape(b, self.num_heads, self.head_dim)  # [B, H, D]
        k_bhd = k.reshape(b, self.num_heads, self.head_dim)
        v_bhd = v.reshape(b, self.num_heads, self.head_dim)

        # ring-buffer update: append new [K|V] at the end, drop oldest at the front
        kv_new = torch.cat([k_bhd, v_bhd], dim=-1).unsqueeze(2)  # [B, H, 1, 2*D]
        new_state = torch.cat([state[:, :, 1:, :], kv_new], dim=2)  # [B, H, W, 2*D]

        # split the window cache back into K and V
        k_cache = new_state[:, :, :, : self.head_dim]  # [B, H, W, D]
        v_cache = new_state[:, :, :, self.head_dim :]  # [B, H, W, D]

        # attention: Q: [B*H, 1, D], K: [B*H, W, D], V: [B*H, W, D]
        q3 = q_bhd.reshape(b * self.num_heads, 1, self.head_dim)
        k3 = k_cache.reshape(b * self.num_heads, self.window, self.head_dim)
        v3 = v_cache.reshape(b * self.num_heads, self.window, self.head_dim)

        scores = torch.bmm(q3, k3.transpose(1, 2)) * self._scale  # [B*H, 1, W]
        # Zero-init KV entries behave like pad-with-first-frame attention: the
        # zero K yields a constant score for every past slot, and zero V
        # contributes nothing to the weighted sum. No explicit mask is needed.
        attn = torch.softmax(scores, dim=-1)
        out_bh = torch.bmm(attn, v3)  # [B*H, 1, D]
        out = out_bh.reshape(b, self.num_heads, 1, self.head_dim).permute(0, 1, 3, 2).reshape(
            b, self.inner_dim, 1, 1
        )
        return self.out_proj(out), new_state  # out: [B, C, 1, 1]


class NarrowBandBlock(nn.Module):
    """Narrow-band temporal mixer (causal).

    Structure:
        x -> RMSNorm
          -> CausalConv2d(C, 2C, (Kt, 1), groups=C) [depthwise]
          -> GatedAct
          -> (optional) + BoundedCausalAttn  (broadcast-added across F')
          -> Conv2d(C, C, 1)
          -> + residual

    The depthwise conv carries a streaming state of shape ``[B, C, Kt-1, F']``.
    If ``use_attn=True`` the attention submodule carries its own KV cache
    (shape ``[B, num_heads, window, 2 * head_dim]``, independent of F').
    """

    def __init__(
        self,
        channels: int,
        time_kernel: int = 5,
        *,
        use_attn: bool = False,
        attn_window: int = 16,
        num_heads: int = 4,
        head_dim: int = 8,
    ):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(f"channels must be even, got {channels}")
        self.channels = channels
        self.time_kernel = time_kernel
        self.use_attn = use_attn

        self.norm = RMSNorm2d(channels)
        self.causal_dw = CausalConv2d(
            channels,
            2 * channels,
            kernel_size=(time_kernel, 1),
            groups=channels,
            bias=True,
        )
        self.act = GatedAct()
        if use_attn:
            self.attn: BoundedCausalAttn | None = BoundedCausalAttn(
                channels,
                window=attn_window,
                num_heads=num_heads,
                head_dim=head_dim,
            )
        else:
            self.attn = None
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    # -- full-sequence forward ------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        y = self.norm(x)
        y = self.causal_dw(y)
        y = self.act(y)
        if self.attn is not None:
            # attn output is [B, C, T, 1]; broadcast-add along F' via +=
            y = y + self.attn(x)
        y = self.pointwise(y)
        return x + y

    # -- streaming API --------------------------------------------------------

    def stream_context_frames(self) -> int:
        return self.causal_dw.stream_context_frames()

    def init_stream_state(
        self,
        batch_size: int,
        *,
        freq_bins: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, ...]:
        dw_state = self.causal_dw.init_stream_state(
            batch_size,
            freq_bins=freq_bins,
            device=device,
            dtype=dtype,
        )
        if self.attn is None:
            return (dw_state,)
        attn_state = self.attn.init_stream_state(
            batch_size,
            freq_bins=freq_bins,
            device=device,
            dtype=dtype,
        )
        return (dw_state, attn_state)

    def forward_stream(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, ...] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        _runtime_assert(x.ndim == 4, f"Expected 4D input, got {tuple(x.shape)}")
        if state is None:
            state = self.init_stream_state(
                x.shape[0],
                freq_bins=x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
        y = self.norm(x)
        y, new_dw_state = self.causal_dw.forward_stream(y, state[0])
        y = self.act(y)
        if self.attn is not None:
            attn_out, new_attn_state = self.attn.forward_stream(x, state[1])
            y = y + attn_out
            new_state: tuple[torch.Tensor, ...] = (new_dw_state, new_attn_state)
        else:
            new_state = (new_dw_state,)
        y = self.pointwise(y)
        return x + y, new_state
