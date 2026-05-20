"""
TIGERNPUEdgeV2 -- NPU-optimized variant addressing ONNX graph bloat.

Key changes vs V1:
1. Replace LayerNorm2DOnChannel (5+ ops each) with a lightweight channel-axis
   RMSNorm (3 ONNX ops, no running stats, train/eval identical).
2. Replace BNBlock permute-based norm with a direct Conv2d path (no Transpose).
3. Replace StaticFreqResize2D repeat_interleave (Tile) with ConvTranspose2d.
4. Vectorize attention heads via batched bmm (no Python for-loop unrolling).
5. Fuse the 67-subband encoder/decoder loops into single grouped Conv2d ops.
6. Minimize Slice/Concat by pre-allocating state update shapes.
7. Chunked-T training path: ``forward_sequence`` unrolls in chunks of
   ``chunk_size`` frames via a shared ``forward_chunk`` kernel that is
   T-agnostic, so the GPU sees wide kernel launches during training while the
   T=1 export path (``forward_cell``) stays bit-identical.

The design keeps the TIGER freq/time alternation + sliding-window KV attention
architecture intact, preserving the core ideas from the original paper.

Normalization notes:
- Earlier revisions of this file used ``BatchNorm2d`` everywhere.  That is
  cheap at ONNX export time (BN folds into the preceding Conv2d), but it
  trains badly on this graph: the old ``forward_sequence`` ran one T=1 call
  per audio frame, so BN saw a highly correlated mini-batch of size
  ``B * F * 1`` per step, ran ~1000 updates per clip with momentum=0.1, and
  drifted between train-mode and eval-mode statistics.  Replacing BN with
  ``NPURMSNormChannel`` removes the train/eval split entirely, adds zero
  streaming state, and costs only ~3 extra ONNX ops per norm site.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .npu_edge_utils import sanitize_for_npu_edge


# ---------------------------------------------------------------------------
# NPU-friendly normalization: frame-local RMSNorm over the channel axis.
#
# The old file used ``BatchNorm2d`` here because BN folds into the previous
# Conv2d at export time and collapses to a single ONNX node.  That's fine for
# the exported graph, but on this architecture it is actively harmful during
# training -- see the module-level docstring for the full rationale.
#
# ``NPURMSNormChannel`` normalises along the channel axis of a (B, C, F, T)
# tensor.  It is frame-local (no time reduction), does not track running
# statistics, and exposes the same forward signature as a BN wrapper so call
# sites are mechanical to migrate.  ONNX export of this module emits
# ``Mul -> ReduceMean -> Add -> Rsqrt -> Mul -> Mul`` (6 elementwise ops); that
# is still 100x smaller than the V1 LayerNorm2DOnChannel path.
# ---------------------------------------------------------------------------


class NPURMSNormChannel(nn.Module):
    """Channel-axis RMSNorm for (B, C, F, T) tensors.

    The normalisation is computed over the channel axis only, so there is no
    time-axis reduction -- the operator is trivially causal and streaming-safe.
    A single learnable per-channel scale is applied after normalisation; there
    is no bias term.  No running statistics: eval() and train() produce
    identical outputs by construction.
    """

    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        # Shape (1, C, 1, 1) so it broadcasts cleanly against (B, C, F, T)
        # and folds into ONNX as a static constant Mul.
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ms = (x * x).mean(dim=1, keepdim=True)
        x = x * torch.rsqrt(ms + self.eps)
        return x * self.weight


# NOTE: Earlier revisions of this file used ``NPUBatchNorm2d`` (a thin wrapper
# around ``nn.BatchNorm2d``).  That class has been removed in favour of
# ``NPURMSNormChannel``; the parameter shapes are different (RMSNorm stores a
# single (1, C, 1, 1) weight while BN stores weight/bias/running_mean/
# running_var/num_batches_tracked), so pre-existing V2 checkpoints will not
# load verbatim.  This is a deliberate breaking change: the BN statistics
# accumulated under the old T=1 training loop were poorly conditioned anyway,
# so re-training from scratch is the right migration.


# ---------------------------------------------------------------------------
# NPU-friendly Conv blocks (no permute, no LayerNorm decomposition)
# ---------------------------------------------------------------------------


class NPUConv2dNormAct(nn.Module):
    """Conv2d + NPURMSNormChannel + ReLU. No permute, no LayerNorm decomposition."""

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, dilation=1, bias=False):
        super().__init__()
        padding = ((kSize - 1) * dilation) // 2
        self.conv = nn.Conv2d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=bias,
            groups=groups, dilation=dilation,
        )
        self.bn = NPURMSNormChannel(nOut)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class NPUConv2dNorm(nn.Module):
    """Conv2d + NPURMSNormChannel without activation."""

    def __init__(self, nIn, nOut, kSize, stride=(1, 1), dilation=(1, 1),
                 groups=1, bias=False):
        super().__init__()
        if isinstance(kSize, int):
            kSize = (kSize, kSize)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        pad_h = ((kSize[0] - 1) * dilation[0]) // 2
        pad_w = ((kSize[1] - 1) * dilation[1]) // 2
        self.conv = nn.Conv2d(
            nIn, nOut, kSize, stride=stride, padding=(pad_h, pad_w),
            bias=bias, groups=groups, dilation=dilation,
        )
        self.bn = NPURMSNormChannel(nOut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class NPUCausalConv2dNorm(nn.Module):
    """Time-causal Conv2d on [B,C,F,T]: left-pad only on time axis."""

    def __init__(self, nIn, nOut, kSize_t, dilation_t=1, groups=1, bias=False):
        super().__init__()
        self.lookback = (kSize_t - 1) * dilation_t
        self.conv = nn.Conv2d(
            nIn, nOut, (1, kSize_t), stride=(1, 1), padding=(0, 0),
            dilation=(1, dilation_t), bias=bias, groups=groups,
        )
        self.bn = NPURMSNormChannel(nOut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.lookback > 0:
            x = F.pad(x, (self.lookback, 0, 0, 0))
        return self.bn(self.conv(x))


# ---------------------------------------------------------------------------
# NPU-friendly frequency resize: Resize-style upsample (± depthwise Conv2d),
# strided Conv2d for downsample. Avoids grouped ConvTranspose (ONNX group>1),
# which circle-mlir does not legalize today. Avoids Tile/repeat_interleave.
# ---------------------------------------------------------------------------


class NPUFreqResize2D(nn.Module):
    """
    Frequency-axis resize: upsample via ``interpolate`` (ONNX ``Resize``-style)
    with optional depthwise ``Conv2d`` after each doubling step; downsample via
    strided depthwise ``Conv2d``. Avoids Tile/repeat_interleave.

    Optional kwargs (defaults preserve drop-in behavior): ``upsample_mode``
    (``nearest``, ``linear`` / ``bilinear``), ``use_dw_conv`` (per-step DW conv).
    """

    def __init__(
        self,
        channels: int,
        source_bins: int,
        target_bins: int,
        *,
        upsample_mode: str = "nearest",
        use_dw_conv: bool = True,
    ):
        super().__init__()
        self.source_bins = source_bins
        self.target_bins = target_bins
        self.use_dw_conv = use_dw_conv

        if upsample_mode in ("linear", "bilinear"):
            self._up_interp_mode: str = "bilinear"
        elif upsample_mode == "nearest":
            self._up_interp_mode = "nearest"
        else:
            raise ValueError(
                f"upsample_mode must be 'nearest', 'linear', or 'bilinear', got {upsample_mode!r}"
            )

        if source_bins == target_bins:
            self.mode = "identity"
        elif source_bins > target_bins:
            self.mode = "downsample"
            # Use a series of stride-2 convolutions
            self.down_layers = nn.ModuleList()
            current = source_bins
            while current > target_bins:
                # stride-2 on freq axis
                self.down_layers.append(nn.Conv2d(
                    channels, channels, (3, 1), stride=(2, 1), padding=(1, 0),
                    groups=channels, bias=False,
                ))
                current = (current + 1) // 2
        else:
            self.mode = "upsample"
            # Previous upsample (grouped ConvTranspose2d); ONNX exports group>1 and
            # circle-mlir does not legalize it — kept verbatim for reference:
            # self.up_layers = nn.ModuleList()
            # current = source_bins
            # while current < target_bins:
            #     self.up_layers.append(nn.ConvTranspose2d(
            #         channels, channels, (2, 1), stride=(2, 1), padding=(0, 0),
            #         groups=channels, bias=False,
            #     ))
            #     current = current * 2
            self._up_steps = 0
            current = source_bins
            while current < target_bins:
                self._up_steps += 1
                current = current * 2
            self.up_dw_convs = nn.ModuleList()
            if use_dw_conv:
                for _ in range(self._up_steps):
                    self.up_dw_convs.append(nn.Conv2d(
                        channels, channels, (3, 1), padding=(1, 0),
                        groups=channels, bias=False,
                    ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "identity":
            return x
        if self.mode == "downsample":
            for layer in self.down_layers:
                x = layer(x)
            return x[:, :, :self.target_bins, :]
        # Previous upsample forward (paired with commented ``self.up_layers`` above):
        # for layer in self.up_layers:
        #     x = layer(x)
        align_corners = False if self._up_interp_mode == "bilinear" else None
        for i in range(self._up_steps):
            x = torch.nn.functional.interpolate(
                x,
                scale_factor=(2.0, 1.0),
                mode=self._up_interp_mode,
                align_corners=align_corners,
            )
            if self.use_dw_conv and len(self.up_dw_convs) > 0:
                x = self.up_dw_convs[i](x)
        return x[:, :, :self.target_bins, :]


# ---------------------------------------------------------------------------
# NPU-friendly encoder/decoder: fused across all subbands via grouped Conv2d.
# Eliminates the 67-iteration loop + per-band Slice/permute/Cat.
# ---------------------------------------------------------------------------


class NPUFusedSubbandEncoder(nn.Module):
    """
    Replaces the per-band BNBlock loop.
    Input: [B, 1, 2*enc_dim, T] (full RI spectrum)
    Output: [B, feature_dim, nband, T]

    Uses a factored two-stage projection to keep parameters reasonable:
    Stage 1: 2*enc_dim -> bottleneck (reduces the huge freq dim)
    Stage 2: bottleneck -> feature_dim * nband (expand to band features)
    """

    def __init__(self, band_widths: List[int], feature_dim: int):
        super().__init__()
        self.band_widths = band_widths
        self.nband = len(band_widths)
        self.feature_dim = feature_dim
        self.enc_dim_2 = sum(bw * 2 for bw in band_widths)

        # Factored projection: enc_dim_2 -> mid -> feature_dim*nband
        mid = min(256, self.enc_dim_2 // 4)
        self.proj1 = nn.Conv2d(self.enc_dim_2, mid, 1, bias=True)
        self.bn1 = NPURMSNormChannel(mid)
        self.act1 = nn.ReLU()
        self.proj2 = nn.Conv2d(mid, feature_dim * self.nband, 1, bias=True)
        self.bn2 = NPURMSNormChannel(feature_dim * self.nband)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 2*enc_dim, T]
        B, _, _, T = x.shape
        x = x.squeeze(1).unsqueeze(2)  # [B, 2*enc_dim, 1, T]
        x = self.act1(self.bn1(self.proj1(x)))  # [B, mid, 1, T]
        x = self.bn2(self.proj2(x))  # [B, feature_dim*nband, 1, T]
        x = x.view(B, self.feature_dim, self.nband, T)
        return x


class NPUFusedSubbandDecoder(nn.Module):
    """
    Replaces the per-band MaskBlock loop.
    Input: [B, feature_dim, nband, T] (separator output)
    Output: [B, 4*num_sources, enc_dim, T] (masks for all bands concatenated)

    Uses factored projection to keep parameter count reasonable.
    """

    def __init__(self, band_widths: List[int], feature_dim: int, num_sources: int):
        super().__init__()
        self.band_widths = band_widths
        self.nband = len(band_widths)
        self.feature_dim = feature_dim
        self.num_sources = num_sources
        self.enc_dim = sum(band_widths)
        self.out_channels = 4 * num_sources * self.enc_dim

        # Factored decoder: feature_dim*nband -> mid -> 4*num_sources*enc_dim
        in_ch = feature_dim * self.nband
        mid = min(256, in_ch)
        self.act = nn.ReLU()
        self.proj1 = nn.Conv2d(in_ch, mid, 1, bias=True)
        self.bn1 = NPURMSNormChannel(mid)
        self.act1 = nn.ReLU()
        self.proj2 = nn.Conv2d(mid, self.out_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, feature_dim, nband, T]
        B, C, Fb, T = x.shape
        x = self.act(x)
        x = x.reshape(B, C * Fb, 1, T)  # [B, feature_dim*nband, 1, T]
        x = self.act1(self.bn1(self.proj1(x)))  # [B, mid, 1, T]
        x = self.proj2(x)  # [B, 4*num_sources*enc_dim, 1, T]
        x = x.view(B, 4 * self.num_sources, self.enc_dim, T)
        return x


# ---------------------------------------------------------------------------
# NPU-friendly attention: vectorized across heads (no Python loop)
# ---------------------------------------------------------------------------


class NPUVectorizedFrameAttention(nn.Module):
    """
    Sliding-window causal attention on the frame (time) axis.
    All heads computed in parallel via a single bmm -- no Python for-loop.
    Input/output: [B, C, F, T]
    """

    def __init__(self, in_channels: int, n_heads: int = 4, hid_chan: int = 2,
                 v_hid_chan: int = 2, freq_bins: int = 67, window_size: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.hid_chan = hid_chan
        self.v_hid_chan = v_hid_chan
        self.freq_bins = freq_bins
        self.window_size = window_size
        self.head_dim = hid_chan * freq_bins
        self.v_head_dim = v_hid_chan * freq_bins
        self.attn_scale = float(self.head_dim ** -0.5)

        total_proj = n_heads * (hid_chan * 2 + v_hid_chan)
        # QKV projection: [B, C, F, T] -> [B, total_proj, F, T]
        self.qkv_conv = nn.Sequential(
            nn.Conv2d(in_channels, total_proj, 1, bias=True),
            NPURMSNormChannel(total_proj),
            nn.ReLU(),
        )
        # Output projection
        self.proj_conv = nn.Sequential(
            nn.Conv2d(n_heads * v_hid_chan, in_channels, 1, bias=True),
            NPURMSNormChannel(in_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, past_kv: torch.Tensor,
                past_valid_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, C, F, T] with T=1 for export
        past_kv: [B, n_heads, window_size, head_dim + v_head_dim]
        past_valid_mask: [B, 1, window_size, 1]
        """
        B, C, Fb, T = x.shape

        # QKV projection -- keep freq folded into head_dim
        qkv = self.qkv_conv(x)  # [B, total_proj, F, T]
        # Reshape: [B, n_heads, (2*hid+v_hid), F, T]
        qkv = qkv.view(B, self.n_heads, 2 * self.hid_chan + self.v_hid_chan, Fb, T)
        # Merge F and T: for T=1 export, flatten to [B, n_heads, proj_per_head, F]
        qkv = qkv.squeeze(-1)  # [B, n_heads, proj, F] since T=1

        q = qkv[:, :, :self.hid_chan, :]  # [B, H, hid, F]
        k = qkv[:, :, self.hid_chan:2*self.hid_chan, :]  # [B, H, hid, F]
        v = qkv[:, :, 2*self.hid_chan:, :]  # [B, H, v_hid, F]

        # Flatten spatial dims into head_dim: [B, H, head_dim]
        q_flat = q.reshape(B, self.n_heads, self.head_dim)  # [B, H, D]
        k_flat = k.reshape(B, self.n_heads, self.head_dim)  # [B, H, D]
        v_flat = v.reshape(B, self.n_heads, self.v_head_dim)  # [B, H, Dv]

        # Update KV cache: shift left and append new
        prev_k = past_kv[:, :, :, :self.head_dim]  # [B, H, W, D]
        prev_v = past_kv[:, :, :, self.head_dim:]  # [B, H, W, Dv]

        # Shift: drop oldest, append new. Build the valid marker from the
        # existing mask tensor so ONNX export does not need ConstantOfShape.
        new_k = torch.cat([prev_k[:, :, 1:, :], k_flat.unsqueeze(2)], dim=2)
        new_v = torch.cat([prev_v[:, :, 1:, :], v_flat.unsqueeze(2)], dim=2)
        current_valid = past_valid_mask[:, :, -1:, :] * 0.0 + 1.0
        next_valid_mask = torch.cat([past_valid_mask[:, :, 1:, :], current_valid], dim=2)

        # Attention: q [B*H, 1, D] @ k^T [B*H, D, W] -> [B*H, 1, W]
        q_vec = q_flat.reshape(B * self.n_heads, 1, self.head_dim)
        k_mat = new_k.reshape(B * self.n_heads, self.window_size, self.head_dim).transpose(1, 2)
        attn = torch.bmm(q_vec, k_mat) * self.attn_scale  # [B*H, 1, W]

        # Mask invalid positions with broadcast Add. Avoid repeat(), which
        # exports as Tile/Expand on this graph.
        invalid = (1.0 - next_valid_mask.reshape(B, 1, 1, self.window_size)) * (-1e4)
        attn = attn.reshape(B, self.n_heads, 1, self.window_size) + invalid
        attn = attn.reshape(B * self.n_heads, 1, self.window_size)
        attn = F.softmax(attn, dim=-1)  # [B*H, 1, W]

        # Context: [B*H, 1, W] @ [B*H, W, Dv] -> [B*H, 1, Dv]
        v_mat = new_v.reshape(B * self.n_heads, self.window_size, self.v_head_dim)
        context = torch.bmm(attn, v_mat).squeeze(1)  # [B*H, Dv]

        # Reshape back: [B, H, v_hid, F] -> [B, H*v_hid, F, 1]
        context = context.view(B, self.n_heads, self.v_head_dim)
        out = context.view(B, self.n_heads, self.v_hid_chan, Fb)
        out = out.reshape(B, self.n_heads * self.v_hid_chan, Fb, 1)
        out = self.proj_conv(out)  # [B, C, F, 1]

        next_kv = torch.cat([new_k, new_v], dim=3)
        return out, next_kv, next_valid_mask

    def forward_chunk(self, x: torch.Tensor, past_kv: torch.Tensor,
                      past_valid_mask: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """T-generic sliding-window causal attention for chunked training.

        Given ``T`` consecutive frames, each output frame attends to its own
        trailing window of ``window_size`` frames -- i.e., the same set of
        keys/values that ``forward`` would have seen if the frames had been
        pushed through one at a time.  At ``T == 1`` this method produces
        numerically-identical outputs to ``forward`` (see the parity test in
        ``test_tiger_npu_edge_v2.py``), so callers who need the optimised T=1
        ONNX graph should still invoke ``forward``.  This path is intended
        exclusively for training; it is **not** wired into the ONNX export.

        Shapes match ``forward`` but with arbitrary ``T``:
          x:               (B, C, F, T)
          past_kv:         (B, n_heads, window_size, head_dim + v_head_dim)
          past_valid_mask: (B, 1, window_size, 1)
        returns:
          out:              (B, C, F, T)
          next_kv:          (B, n_heads, window_size, head_dim + v_head_dim)
          next_valid_mask:  (B, 1, window_size, 1)
        """
        B, _, Fb, T = x.shape
        W = self.window_size
        H = self.n_heads

        qkv = self.qkv_conv(x)  # (B, total_proj, F, T)
        qkv = qkv.view(B, H, 2 * self.hid_chan + self.v_hid_chan, Fb, T)

        q = qkv[:, :, :self.hid_chan, :, :]                              # (B, H, hid, F, T)
        k = qkv[:, :, self.hid_chan:2 * self.hid_chan, :, :]             # (B, H, hid, F, T)
        v = qkv[:, :, 2 * self.hid_chan:, :, :]                          # (B, H, v_hid, F, T)

        # Bring T to the front of the per-head payload so we can flatten
        # (hid, F) into head_dim and keep T as a distinct axis.
        q_T = q.permute(0, 1, 4, 2, 3).reshape(B, H, T, self.head_dim)
        k_T = k.permute(0, 1, 4, 2, 3).reshape(B, H, T, self.head_dim)
        v_T = v.permute(0, 1, 4, 2, 3).reshape(B, H, T, self.v_head_dim)

        prev_k = past_kv[:, :, :, :self.head_dim]                        # (B, H, W, head_dim)
        prev_v = past_kv[:, :, :, self.head_dim:]                        # (B, H, W, v_head_dim)

        # Combined KV sequence: prev window followed by the current chunk.
        # Output frame i within the chunk needs to attend to the trailing W
        # frames ending at i; in combined-sequence coordinates that is the
        # slice [i+1, i+W]. See the window-mask construction below.
        combined_k = torch.cat([prev_k, k_T], dim=2)                     # (B, H, W+T, head_dim)
        combined_v = torch.cat([prev_v, v_T], dim=2)                     # (B, H, W+T, v_head_dim)

        new_valid = past_valid_mask.new_ones(B, 1, T, 1)
        combined_valid = torch.cat([past_valid_mask, new_valid], dim=2)  # (B, 1, W+T, 1)

        # Attention scores: (B*H, T, W+T)
        q_bmm = q_T.reshape(B * H, T, self.head_dim)
        k_bmm = combined_k.reshape(B * H, W + T, self.head_dim).transpose(1, 2)
        attn = torch.bmm(q_bmm, k_bmm) * self.attn_scale

        # Causal-window mask: chunk frame i attends to combined positions j
        # with 1 <= (j - i) <= W. Built from arange() each call; this is
        # training-only, so ONNX cost is irrelevant here.
        i_idx = torch.arange(T, device=x.device)
        j_idx = torch.arange(W + T, device=x.device)
        diff = j_idx.unsqueeze(0) - i_idx.unsqueeze(1)                   # (T, W+T)
        window_mask = (diff >= 1) & (diff <= W)                           # (T, W+T)
        window_add = torch.where(
            window_mask,
            torch.zeros((), device=x.device, dtype=attn.dtype),
            torch.full((), -1e4, device=x.device, dtype=attn.dtype),
        )                                                                 # (T, W+T)

        # Combine with per-position validity from the warmup mask.
        valid_flat = combined_valid.reshape(B, 1, W + T)                 # (B, 1, W+T)
        valid_add = (1.0 - valid_flat) * (-1e4)                          # (B, 1, W+T)

        attn = attn.view(B, H, T, W + T)
        attn = attn + window_add.view(1, 1, T, W + T)
        attn = attn + valid_add.view(B, 1, 1, W + T)
        attn = attn.reshape(B * H, T, W + T)
        attn = F.softmax(attn, dim=-1)

        v_bmm = combined_v.reshape(B * H, W + T, self.v_head_dim)
        context = torch.bmm(attn, v_bmm)                                 # (B*H, T, v_head_dim)
        context = context.view(B, H, T, self.v_hid_chan, Fb)
        # (B, H, T, v_hid, F) -> (B, H*v_hid, F, T)
        context = context.permute(0, 1, 3, 4, 2).reshape(
            B, H * self.v_hid_chan, Fb, T,
        )
        out = self.proj_conv(context)                                    # (B, C, F, T)

        # Carry forward the trailing window (last W combined positions) so
        # the next chunk sees the same sliding-window state as frame-by-frame.
        next_kv = torch.cat(
            [combined_k[:, :, -W:, :], combined_v[:, :, -W:, :]], dim=3,
        )
        next_valid_mask = combined_valid[:, :, -W:, :]
        return out, next_kv, next_valid_mask


class NPUVectorizedFreqAttention(nn.Module):
    """
    Self-attention on the frequency axis (no streaming state needed).
    All heads computed in parallel.
    Input/output: [B, C, F, T]
    """

    def __init__(self, in_channels: int, n_heads: int = 4, hid_chan: int = 2,
                 v_hid_chan: int = 2, freq_bins: int = 67):
        super().__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.hid_chan = hid_chan
        self.v_hid_chan = v_hid_chan
        self.freq_bins = freq_bins
        self.attn_scale = float(hid_chan ** -0.5)

        total_proj = n_heads * (hid_chan * 2 + v_hid_chan)
        self.qkv_conv = nn.Sequential(
            nn.Conv2d(in_channels, total_proj, 1, bias=True),
            NPURMSNormChannel(total_proj),
            nn.ReLU(),
        )
        self.proj_conv = nn.Sequential(
            nn.Conv2d(n_heads * v_hid_chan, in_channels, 1, bias=True),
            NPURMSNormChannel(in_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, F, T] with T=1 for export."""
        B, C, Fb, T = x.shape

        qkv = self.qkv_conv(x)  # [B, total, F, T]
        # For T=1: [B, total, F, 1] -> squeeze T
        qkv = qkv.squeeze(-1)  # [B, total, F]
        qkv = qkv.view(B, self.n_heads, 2 * self.hid_chan + self.v_hid_chan, Fb)

        q = qkv[:, :, :self.hid_chan, :]  # [B, H, D, F]
        k = qkv[:, :, self.hid_chan:2*self.hid_chan, :]  # [B, H, D, F]
        v = qkv[:, :, 2*self.hid_chan:, :]  # [B, H, Dv, F]

        # Attention over F: q^T [B*H, F, D] @ k [B*H, D, F] -> [B*H, F, F]
        q_t = q.reshape(B * self.n_heads, self.hid_chan, Fb).transpose(1, 2)  # [BH, F, D]
        k_r = k.reshape(B * self.n_heads, self.hid_chan, Fb)  # [BH, D, F]
        v_t = v.reshape(B * self.n_heads, self.v_hid_chan, Fb).transpose(1, 2)  # [BH, F, Dv]

        attn = torch.bmm(q_t, k_r) * self.attn_scale  # [BH, F, F]
        attn = F.softmax(attn, dim=-1)
        ctx = torch.bmm(attn, v_t)  # [BH, F, Dv]

        # Reshape back: [B, H, Dv, F] -> [B, H*Dv, F, 1]
        ctx = ctx.transpose(1, 2).reshape(B, self.n_heads * self.v_hid_chan, Fb, 1)
        out = self.proj_conv(ctx)  # [B, C, F, 1]
        return out

    def forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        """T-generic frequency self-attention for chunked training.

        The original ``forward`` assumes T=1 and squeezes the time axis.  For
        chunk training we fold T into the batch so each frame still attends
        purely over its own frequency axis (identical semantics).  At T=1
        this is numerically identical to ``forward``; the export path is
        therefore unaffected.
        """
        B, _, Fb, T = x.shape
        H = self.n_heads

        qkv = self.qkv_conv(x)  # (B, total, F, T)
        qkv = qkv.view(B, H, 2 * self.hid_chan + self.v_hid_chan, Fb, T)

        q = qkv[:, :, :self.hid_chan, :, :]                     # (B, H, hid, F, T)
        k = qkv[:, :, self.hid_chan:2 * self.hid_chan, :, :]
        v = qkv[:, :, 2 * self.hid_chan:, :, :]                 # (B, H, v_hid, F, T)

        # Collapse (B, H, T) into the batch dim so each (B,H,T) slot runs an
        # independent F-axis attention with no cross-frame leakage.
        q_t = q.permute(0, 1, 4, 3, 2).reshape(B * H * T, Fb, self.hid_chan)
        k_r = k.permute(0, 1, 4, 2, 3).reshape(B * H * T, self.hid_chan, Fb)
        v_t = v.permute(0, 1, 4, 3, 2).reshape(B * H * T, Fb, self.v_hid_chan)

        attn = torch.bmm(q_t, k_r) * self.attn_scale
        attn = F.softmax(attn, dim=-1)
        ctx = torch.bmm(attn, v_t)                               # (B*H*T, F, v_hid)

        # Reassemble to (B, H*v_hid, F, T).
        ctx = ctx.view(B, H, T, Fb, self.v_hid_chan)
        ctx = ctx.permute(0, 1, 4, 3, 2).reshape(
            B, H * self.v_hid_chan, Fb, T,
        )
        return self.proj_conv(ctx)


# ---------------------------------------------------------------------------
# NPU-friendly Freq/Time U-blocks (using NPURMSNormChannel, no Dropout)
# ---------------------------------------------------------------------------


class NPUFreqUConvBlock(nn.Module):
    """Frequency multi-resolution block using NPU-friendly ops."""

    def __init__(self, out_channels: int, in_channels: int,
                 upsampling_depth: int = 4, nband: int = 67):
        super().__init__()
        self.depth = upsampling_depth
        self.proj = NPUConv2dNormAct(out_channels, in_channels, 1)

        # Downsampling on freq axis
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(NPUConv2dNorm(in_channels, in_channels, (5, 1),
                                          groups=in_channels))
        for _ in range(1, upsampling_depth):
            self.spp_dw.append(NPUConv2dNorm(
                in_channels, in_channels, (5, 1), stride=(2, 1),
                groups=in_channels,
            ))

        # Compute level sizes
        sizes = [nband]
        for _ in range(1, upsampling_depth):
            sizes.append((sizes[-1] + 1) // 2)

        # Global feature aggregation resizers
        self.global_resizers = nn.ModuleList([
            NPUFreqResize2D(in_channels, sizes[i], sizes[-1])
            for i in range(upsampling_depth - 1)
        ])

        # Global MLP
        self.global_mlp = nn.Sequential(
            NPUConv2dNorm(in_channels, in_channels, 1),
            nn.Conv2d(in_channels, in_channels, (5, 1), padding=(2, 0),
                      groups=in_channels, bias=True),
            nn.ReLU(),
            NPUConv2dNorm(in_channels, in_channels, 1),
        )

        # Local-global fusion with resizers
        self.loc_glo_fus_conv = nn.ModuleList()
        self.loc_glo_resizers = nn.ModuleList()
        for i in range(upsampling_depth):
            self.loc_glo_fus_conv.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (1, 1), groups=in_channels, bias=False),
                nn.Sigmoid(),
            ))
            self.loc_glo_resizers.append(NPUFreqResize2D(in_channels, sizes[-1], sizes[i]))

        # Reconstruction path: iterates from depth-2 down to 0.
        # At step i, expanded comes from sizes[i+1] and must resize to sizes[i].
        self.last_resizers = nn.ModuleList()
        self.last_fus = nn.ModuleList()
        for i in range(upsampling_depth - 1):
            # expanded at step i has freq size = sizes[i+1]
            self.last_resizers.append(NPUFreqResize2D(in_channels, sizes[i + 1], sizes[i]))
            self.last_fus.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (1, 1), groups=in_channels, bias=False),
                nn.Sigmoid(),
            ))

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.proj(x)
        outputs = [self.spp_dw[0](h)]
        for k in range(1, self.depth):
            outputs.append(self.spp_dw[k](outputs[-1]))

        # Global features
        global_f = self.global_resizers[0](outputs[0])
        for idx in range(1, self.depth - 1):
            global_f = global_f + self.global_resizers[idx](outputs[idx])
        global_f = global_f + outputs[-1]
        global_f = self.global_mlp(global_f)

        # Fuse local with global
        fused = []
        for i in range(self.depth):
            g_resized = self.loc_glo_resizers[i](global_f)
            gate = self.loc_glo_fus_conv[i](g_resized)
            fused.append(outputs[i] * gate + g_resized)

        # Bottom-up reconstruction
        expanded = fused[-1]
        for i in range(self.depth - 2, -1, -1):
            expanded_resized = self.last_resizers[i](expanded)
            gate = self.last_fus[i](expanded_resized)
            expanded = fused[i] * gate + expanded_resized

        return self.res_conv(expanded) + residual



class NPUCausalTimeBlock(nn.Module):
    """
    Causal time-domain block with context state.
    Uses depthwise causal Conv2d with NPURMSNormChannel. No Dropout.
    
    Architecture: parallel dilated causal convolutions applied to the
    context-prefixed input. Each conv receives the same padded input and
    produces output of length T (the original chunk length).
    """

    def __init__(self, out_channels: int, hidden_channels: int = 8,
                 dilations: Tuple[int, ...] = (1, 1, 2),
                 kernel_size: int = 3, global_kernel_size: int = 5):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.global_kernel_size = global_kernel_size
        self.depth = len(dilations)

        # Context size = total lookback needed by all parallel convs + global
        conv_ctx = max((kernel_size - 1) * d for d in dilations)
        global_ctx = max(0, global_kernel_size - 1)
        self.context_size = conv_ctx + global_ctx

        self.proj = NPUConv2dNormAct(out_channels, hidden_channels, 1)

        # Dilated causal convolutions on time axis (no padding, rely on context)
        self.dw_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, (1, kernel_size),
                          dilation=(1, d), padding=(0, 0),
                          groups=hidden_channels, bias=False),
                NPURMSNormChannel(hidden_channels),
            )
            for d in dilations
        ])
        # Each conv reduces temporal dim by (kernel_size-1)*dilation
        # We pad the input with context so output covers T frames

        # Global causal conv (no padding, uses context)
        self.global_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, (1, global_kernel_size),
                      padding=(0, 0), groups=hidden_channels, bias=False),
            NPURMSNormChannel(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False),
            NPURMSNormChannel(hidden_channels),
        )

        # Fusion gates
        self.fuse_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 1, groups=hidden_channels, bias=False),
                nn.Sigmoid(),
            )
            for _ in range(self.depth)
        ])

        self.res_conv = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, ctx: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, Fb, T = x.shape
        projected = self.proj(x)

        if ctx is None:
            ctx = projected.new_zeros(B, self.hidden_channels, Fb, self.context_size)

        # Concatenate context on time axis
        if self.context_size > 0:
            full = torch.cat([ctx, projected], dim=-1)  # [B, H, F, ctx+T]
        else:
            full = projected

        # Apply dilated convolutions in parallel to full
        # Each conv(full) produces: ctx+T - (k-1)*d output frames
        # We take the last T frames from each
        outputs = []
        for dw_conv in self.dw_convs:
            conv_out = dw_conv(full)
            outputs.append(conv_out[:, :, :, -T:])

        # Global feature (takes last T frames from its output)
        global_out = self.global_conv(full)
        global_f = global_out[:, :, :, -T:]

        # Fuse each branch with global
        fused = []
        for i in range(self.depth):
            gate = self.fuse_gates[i](global_f)
            fused.append(outputs[i] * gate + global_f)

        # Sum and project back
        result = fused[0]
        for f in fused[1:]:
            result = result + f
        result = self.res_conv(result)  # [B, out_channels, F, T]

        # Residual connection
        valid = result + x

        # Next context = last context_size frames of the full projection
        if self.context_size > 0:
            next_ctx = full[:, :, :, -self.context_size:].detach()
        else:
            next_ctx = full[:, :, :, :0].detach()

        return valid, next_ctx


# ---------------------------------------------------------------------------
# NPU-friendly stacked separator
# ---------------------------------------------------------------------------


class NPUFreqTimeStage(nn.Module):
    """One freq + time stage with vectorized attention."""

    def __init__(self, out_channels: int, in_channels: int, nband: int,
                 f_upsampling_depth: int = 4, n_heads: int = 4,
                 att_hid_chan: int = 2, att_val_hid_chan: int = 2,
                 kv_window_size: int = 4,
                 time_hidden_channels: int = 8,
                 time_dilations: Tuple[int, ...] = (1, 1, 2),
                 time_kernel_size: int = 3,
                 time_global_kernel_size: int = 5):
        super().__init__()
        self.freq_block = NPUFreqUConvBlock(out_channels, in_channels,
                                             f_upsampling_depth, nband)
        self.freq_attn = NPUVectorizedFreqAttention(
            out_channels, n_heads, att_hid_chan, att_val_hid_chan, nband,
        )
        self.freq_norm = NPURMSNormChannel(out_channels)

        self.time_block = NPUCausalTimeBlock(
            out_channels, time_hidden_channels, time_dilations,
            time_kernel_size, time_global_kernel_size,
        )
        self.frame_attn = NPUVectorizedFrameAttention(
            out_channels, n_heads, att_hid_chan, att_val_hid_chan, nband,
            kv_window_size,
        )
        self.frame_norm = NPURMSNormChannel(out_channels)

    def forward(self, x: torch.Tensor, past_kv: torch.Tensor,
                past_valid_mask: torch.Tensor, time_ctx: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Freq path
        residual_1 = x
        freq_out = self.freq_block(x)
        freq_attn_out = self.freq_attn(freq_out)
        freq_out = self.freq_norm(freq_attn_out)
        x2 = freq_out + residual_1

        # Time path
        residual_2 = x2
        time_out, next_ctx = self.time_block(x2, ctx=time_ctx)
        frame_out, new_kv, new_valid = self.frame_attn(
            time_out, past_kv, past_valid_mask,
        )
        frame_out = self.frame_norm(frame_out)
        out = frame_out + residual_2

        return out, new_kv, new_valid, next_ctx

    def forward_chunk(self, x: torch.Tensor, past_kv: torch.Tensor,
                      past_valid_mask: torch.Tensor, time_ctx: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """T-generic stage forward used by chunked training.

        The freq block, the time block and the two norms are already
        T-agnostic; only the attention modules were T=1-specialised, so this
        method just routes through their ``forward_chunk`` variants.  At T=1
        the result matches ``forward`` within fp32 tolerance.
        """
        residual_1 = x
        freq_out = self.freq_block(x)
        freq_attn_out = self.freq_attn.forward_chunk(freq_out)
        freq_out = self.freq_norm(freq_attn_out)
        x2 = freq_out + residual_1

        residual_2 = x2
        time_out, next_ctx = self.time_block(x2, ctx=time_ctx)
        frame_out, new_kv, new_valid = self.frame_attn.forward_chunk(
            time_out, past_kv, past_valid_mask,
        )
        frame_out = self.frame_norm(frame_out)
        out = frame_out + residual_2
        return out, new_kv, new_valid, next_ctx


class NPUStackedSeparator(nn.Module):
    """Stacked freq/time separator for NPU deployment."""

    def __init__(self, out_channels: int, in_channels: int, nband: int,
                 num_stages: int = 2, f_upsampling_depth: int = 4,
                 n_heads: int = 4, att_hid_chan: int = 2,
                 att_val_hid_chan: int = 2, kv_window_size: int = 4,
                 time_hidden_channels: int = 8,
                 time_dilations: Tuple[int, ...] = (1, 1, 2),
                 time_kernel_size: int = 3,
                 time_global_kernel_size: int = 5):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.nband = nband
        self.att_hid_chan = att_hid_chan
        self.att_val_hid_chan = att_val_hid_chan
        self.kv_window_size = kv_window_size
        self.num_stages = num_stages

        self.stages = nn.ModuleList([
            NPUFreqTimeStage(
                out_channels, in_channels, nband, f_upsampling_depth,
                n_heads, att_hid_chan, att_val_hid_chan, kv_window_size,
                time_hidden_channels, time_dilations, time_kernel_size,
                time_global_kernel_size,
            )
            for _ in range(num_stages)
        ])
        # Inter-stage mixing (simple 1x1 conv + ReLU, no PReLU)
        self.mix_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1),
                nn.ReLU(),
            )
            for _ in range(max(0, num_stages - 1))
        ])

    def forward(self, x: torch.Tensor, past_kvs: torch.Tensor,
                past_valid_mask: torch.Tensor, time_ctx: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, Fb, T = x.shape
        mixture = x

        kv_dim = (self.att_hid_chan + self.att_val_hid_chan) * self.nband
        ctx_dim = self.stages[0].time_block.hidden_channels
        ctx_width = self.stages[0].time_block.context_size

        new_kvs_list = []
        new_ctxs_list = []
        new_valid_mask = past_valid_mask

        for stage_idx, stage in enumerate(self.stages):
            prev_ctx = time_ctx[:, stage_idx * ctx_dim:(stage_idx + 1) * ctx_dim, :, :]
            past_kv = past_kvs[:, :, :, stage_idx * kv_dim:(stage_idx + 1) * kv_dim]

            if stage_idx == 0:
                stage_input = x
            else:
                stage_input = self.mix_blocks[stage_idx - 1](mixture + x)

            x, new_kv, new_valid_mask, next_ctx = stage(
                stage_input, past_kv, past_valid_mask, prev_ctx,
            )
            new_kvs_list.append(new_kv)
            new_ctxs_list.append(next_ctx)

        new_kvs = torch.cat(new_kvs_list, dim=-1)
        new_ctxs = torch.cat(new_ctxs_list, dim=1)

        return x, new_kvs, new_valid_mask, new_ctxs

    def forward_chunk(self, x: torch.Tensor, past_kvs: torch.Tensor,
                      past_valid_mask: torch.Tensor, time_ctx: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """T-generic stacked forward for chunked training.

        Mirrors ``forward`` line-for-line but dispatches to each stage's
        ``forward_chunk``.  Per-stage slicing of ``past_kvs`` / ``time_ctx``
        and the mix-block routing are identical.
        """
        mixture = x
        kv_dim = (self.att_hid_chan + self.att_val_hid_chan) * self.nband
        ctx_dim = self.stages[0].time_block.hidden_channels

        new_kvs_list: List[torch.Tensor] = []
        new_ctxs_list: List[torch.Tensor] = []
        new_valid_mask = past_valid_mask

        for stage_idx, stage in enumerate(self.stages):
            prev_ctx = time_ctx[:, stage_idx * ctx_dim:(stage_idx + 1) * ctx_dim, :, :]
            past_kv = past_kvs[:, :, :, stage_idx * kv_dim:(stage_idx + 1) * kv_dim]

            if stage_idx == 0:
                stage_input = x
            else:
                stage_input = self.mix_blocks[stage_idx - 1](mixture + x)

            x, new_kv, new_valid_mask, next_ctx = stage.forward_chunk(
                stage_input, past_kv, past_valid_mask, prev_ctx,
            )
            new_kvs_list.append(new_kv)
            new_ctxs_list.append(next_ctx)

        new_kvs = torch.cat(new_kvs_list, dim=-1)
        new_ctxs = torch.cat(new_ctxs_list, dim=1)
        return x, new_kvs, new_valid_mask, new_ctxs



# ---------------------------------------------------------------------------
# Top-level model: TIGERNPUEdgeV2
# ---------------------------------------------------------------------------


def _calculate_band_widths(enc_dim: int, sample_rate: int) -> List[int]:
    """Calculate subband widths matching the original TIGER design."""
    import numpy as np
    if enc_dim < 67:
        raise ValueError(f"TIGER requires at least 67 frequency bins, got enc_dim={enc_dim}.")
    bandwidth_25 = int(np.floor(25 / (sample_rate / 2.0) * enc_dim))
    bandwidth_100 = int(np.floor(100 / (sample_rate / 2.0) * enc_dim))
    bandwidth_250 = int(np.floor(250 / (sample_rate / 2.0) * enc_dim))
    bandwidth_500 = int(np.floor(500 / (sample_rate / 2.0) * enc_dim))
    band_width = [max(1, bandwidth_25)] * 40
    band_width += [max(1, bandwidth_100)] * 10
    band_width += [max(1, bandwidth_250)] * 8
    band_width += [max(1, bandwidth_500)] * 8
    remainder = int(enc_dim - int(np.sum(band_width)))
    if remainder <= 0:
        raise ValueError(
            f"TIGER band split over-allocates enc_dim={enc_dim}; "
            f"minimum nonzero bands need {int(np.sum(band_width)) + 1} bins."
        )
    band_width.append(remainder)
    return band_width


class TIGERNPUEdgeV2(nn.Module):
    """
    NPU Edge V2: Fully NPU-optimized TIGER variant.

    Key architectural choices:
    - Fused subband encoder/decoder (single Conv2d instead of 67-band loop)
    - NPURMSNormChannel everywhere (stream-safe, train-eval identical, ~3 ops)
      instead of LayerNorm (5+ nodes) or BatchNorm (bad under T=1 training)
    - Vectorized attention (no Python for-loop over heads)
    - ConvTranspose2d frequency resizing (no Tile/repeat_interleave)
    - No Dropout, no PReLU (ReLU only)
    - All tensors are 4D [B, C, H, W]
    - Minimal Transpose/Slice/Concat operations
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        num_sources: int = 3,
        win: int = 2048,
        stride: int = 512,
        out_channels: int = 66,
        in_channels: int = 132,
        upsampling_depth: int = 4,
        num_stages: int = 2,
        att_n_head: int = 4,
        att_hid_chan: int = 2,
        att_val_hid_chan: int = 2,
        kv_window_size: int = 4,
        time_hidden_channels: int = 8,
        time_dilations: Tuple[int, ...] = (1, 1, 2),
        time_kernel_size: int = 3,
        time_global_kernel_size: int = 5,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_sources = num_sources
        self.win = win
        self.stride = stride
        self.enc_dim = win // 2 + 1
        self.feature_dim = out_channels

        self.band_width = _calculate_band_widths(self.enc_dim, sample_rate)
        self.nband = len(self.band_width)

        # Fused encoder: [B, 1, 2*enc_dim, T] -> [B, feature_dim, nband, T]
        self.encoder = NPUFusedSubbandEncoder(self.band_width, out_channels)

        # Separator
        self.separator = NPUStackedSeparator(
            out_channels=out_channels,
            in_channels=in_channels,
            nband=self.nband,
            num_stages=num_stages,
            f_upsampling_depth=upsampling_depth,
            n_heads=att_n_head,
            att_hid_chan=att_hid_chan,
            att_val_hid_chan=att_val_hid_chan,
            kv_window_size=kv_window_size,
            time_hidden_channels=time_hidden_channels,
            time_dilations=time_dilations,
            time_kernel_size=time_kernel_size,
            time_global_kernel_size=time_global_kernel_size,
        )

        # Fused decoder: [B, feature_dim, nband, T] -> [B, 4*num_sources, enc_dim, T]
        self.decoder = NPUFusedSubbandDecoder(self.band_width, out_channels, num_sources)

        # Store params for state init
        self._n_heads = att_n_head
        self._kv_window_size = kv_window_size
        self._att_hid_chan = att_hid_chan
        self._att_val_hid_chan = att_val_hid_chan
        self._time_hidden_channels = time_hidden_channels
        self._num_stages = num_stages
        self._time_dilations = time_dilations
        self._time_kernel_size = time_kernel_size
        self._time_global_kernel_size = time_global_kernel_size

        # Chunked training is supported -- ``forward_sequence`` uses a single
        # T-generic kernel (``_forward_chunk``) and slices the total frame
        # range into chunks of ``chunk_size`` frames per GPU call, instead of
        # the old frame-by-frame Python loop.  See ``forward_sequence`` below.
        self.supports_exact_chunk_training = True

    @property
    def _ctx_size(self) -> int:
        return self.separator.stages[0].time_block.context_size

    def init_streaming_state(
        self, batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (past_kvs, past_valid_mask, time_ctx)."""
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        kv_dim = (self._att_hid_chan + self._att_val_hid_chan) * self.nband
        ctx_dim = self._time_hidden_channels
        ctx_width = self._ctx_size

        past_kvs = torch.zeros(
            batch_size, self._n_heads, self._kv_window_size,
            kv_dim * self._num_stages,
            device=device, dtype=dtype,
        )
        past_valid_mask = torch.zeros(
            batch_size, 1, self._kv_window_size, 1,
            device=device, dtype=dtype,
        )
        time_ctx = torch.zeros(
            batch_size, ctx_dim * self._num_stages, self.nband, ctx_width,
            device=device, dtype=dtype,
        )
        return past_kvs, past_valid_mask, time_ctx

    def forward_cell(
        self,
        subband_spec_RIs: torch.Tensor,
        past_kvs: torch.Tensor,
        past_valid_mask: torch.Tensor,
        time_ctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-frame forward for streaming/export. Input T=1.

        This is the ONNX export path and is kept intentionally thin: the
        shapes flowing through ``self.separator`` have ``T == 1`` so the
        frame-attention module takes its T=1 code path with no extra masking
        / windowing machinery. Do not change without re-running the op-count
        assertions in ``test_tiger_npu_edge_v2.py``.
        """
        features = self.encoder(subband_spec_RIs)  # [B, C, nband, 1]
        sep_out, new_kvs, new_valid, new_ctx = self.separator(
            features, past_kvs, past_valid_mask, time_ctx,
        )
        output = self.decoder(sep_out)  # [B, 4*num_sources, enc_dim, 1]
        return output, new_kvs, new_valid, new_ctx

    def _forward_chunk(
        self,
        subband_spec_RIs: torch.Tensor,
        past_kvs: torch.Tensor,
        past_valid_mask: torch.Tensor,
        time_ctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """T-generic chunked forward used by ``forward_sequence``.

        This is the training-only fast path.  For any ``T >= 1`` it routes
        through the same encoder / separator / decoder modules, but with the
        attention modules invoked via their ``forward_chunk`` variants so the
        whole chunk is processed in one fused batched BMM instead of ``T``
        sequential T=1 calls.  Frame-by-frame parity is enforced by
        ``test_chunked_matches_frame_by_frame``.
        """
        features = self.encoder(subband_spec_RIs)  # [B, C, nband, T]
        sep_out, new_kvs, new_valid, new_ctx = self.separator.forward_chunk(
            features, past_kvs, past_valid_mask, time_ctx,
        )
        output = self.decoder(sep_out)  # [B, 4*num_sources, enc_dim, T]
        return output, new_kvs, new_valid, new_ctx

    def forward_sequence(
        self,
        subband_spec_RIs: torch.Tensor,
        past_kvs: Optional[torch.Tensor] = None,
        past_valid_mask: Optional[torch.Tensor] = None,
        time_ctx: Optional[torch.Tensor] = None,
        detach_state: bool = False,
        chunk_size: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Multi-frame forward used for training and offline evaluation.

        Previously this looped frame-by-frame (``T=1`` call per STFT frame),
        which launched ~1 kernel per frame and kept GPU utilisation near
        zero during training.  It now slices the total frame range into
        chunks of ``chunk_size`` frames and pushes each chunk through
        ``_forward_chunk`` in a single call.  Sliding-window KV, causal
        masking and time-block context are preserved exactly across chunk
        boundaries, so training remains strictly causal and the single-frame
        deployment path still matches the training computation.

        ``chunk_size <= 0`` means "whole sequence in one call"; ``1`` falls
        back to the original frame-by-frame loop for exact bit-wise parity
        with ``forward_cell`` (used by the parity test).
        """
        B, _, _, total_frames = subband_spec_RIs.shape

        if past_kvs is None or past_valid_mask is None or time_ctx is None:
            past_kvs, past_valid_mask, time_ctx = self.init_streaming_state(
                B, subband_spec_RIs.device, subband_spec_RIs.dtype,
            )

        if chunk_size is None or chunk_size <= 0:
            chunk_size = total_frames
        chunk_size = min(chunk_size, max(total_frames, 1))

        outputs: List[torch.Tensor] = []
        for start in range(0, total_frames, chunk_size):
            end = min(start + chunk_size, total_frames)
            chunk = subband_spec_RIs[..., start:end]
            if end - start == 1:
                # Preserve exact parity with ``forward_cell`` (used by the
                # frame-by-frame baseline in the parity test).
                out, past_kvs, past_valid_mask, time_ctx = self.forward_cell(
                    chunk, past_kvs, past_valid_mask, time_ctx,
                )
            else:
                out, past_kvs, past_valid_mask, time_ctx = self._forward_chunk(
                    chunk, past_kvs, past_valid_mask, time_ctx,
                )
            outputs.append(out)
            if detach_state:
                past_kvs = past_kvs.detach()
                past_valid_mask = past_valid_mask.detach()
                time_ctx = time_ctx.detach()

        return torch.cat(outputs, dim=-1), past_kvs, past_valid_mask, time_ctx

    def forward(
        self,
        subband_spec_RIs: torch.Tensor,
        past_kvs: Optional[torch.Tensor] = None,
        past_valid_mask: Optional[torch.Tensor] = None,
        time_ctx: Optional[torch.Tensor] = None,
        detach_state: bool = False,
        chunk_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if subband_spec_RIs.shape[-1] == 1:
            if past_kvs is None:
                past_kvs, past_valid_mask, time_ctx = self.init_streaming_state(
                    subband_spec_RIs.shape[0],
                    subband_spec_RIs.device,
                    subband_spec_RIs.dtype,
                )
            return self.forward_cell(
                subband_spec_RIs, past_kvs, past_valid_mask, time_ctx,
            )
        return self.forward_sequence(
            subband_spec_RIs, past_kvs, past_valid_mask, time_ctx,
            detach_state, chunk_size,
        )


# ---------------------------------------------------------------------------
# ONNX export wrapper
# ---------------------------------------------------------------------------


class NPUEdgeV2ExportWrapper(nn.Module):
    """ONNX export wrapper: fixed inputs, no optional streaming tensors."""

    def __init__(self, model: TIGERNPUEdgeV2):
        super().__init__()
        self.model = model

    def forward(
        self,
        subband_spec_RIs: torch.Tensor,
        past_kvs: torch.Tensor,
        past_valid_mask: torch.Tensor,
        time_ctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model.forward_cell(
            subband_spec_RIs, past_kvs, past_valid_mask, time_ctx,
        )


def export_tiger_npu_edge_v2_onnx(
    model: TIGERNPUEdgeV2,
    export_path: Path,
    opset: int = 14,
) -> None:
    """Export V2 model to ONNX."""
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    dummy_ri = torch.zeros(1, 1, model.enc_dim * 2, 1, device=device, dtype=dtype)
    past_kvs, past_valid_mask, time_ctx = model.init_streaming_state(
        batch_size=1, device=device, dtype=dtype,
    )
    wrapper = NPUEdgeV2ExportWrapper(model)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ri, past_kvs, past_valid_mask, time_ctx),
            str(export_path),
            export_params=True,
            opset_version=opset,
            input_names=["subband_spec_RIs", "past_kvs", "past_valid_mask", "time_ctx"],
            output_names=["band_masked_output", "new_kv", "new_valid_mask", "new_time_ctx"],
            do_constant_folding=True,
            dynamo=False,
            external_data=False,
        )
