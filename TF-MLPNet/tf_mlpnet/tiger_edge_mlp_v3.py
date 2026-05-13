"""
TIGEREdgeMLP v3: 2M - 10M parameter NPU-friendly TF-MLP backbone.

Design rationale vs. v2
-----------------------
v2 is deliberately conservative so the separator stays around 0.5M parameters.
For DnR 3-stem separation we want to push toward 3 - 9M parameters while
keeping every v2 deployment property intact.

Quality levers we add (all Conv2d-only, all rank <= 4, all causal in time):

1. **GLU-gated mixers.** Both the frequency mixer and the time mixer use a
   Gated Linear Unit on the expanded hidden channels: ``value * sigmoid(gate)``.
   TF-MLPNet / BS-RoFormer / U-Net-GLU all use this; it is the cheapest
   non-linearity that gives the model real mixing capacity per parameter. Cost
   is one extra 1x1 Conv2d per mixer, still O(C^2 * E).

2. **Pre-LayerNorm.** Channel-only LayerNorm (``mean/var/rsqrt/add/mul`` over
   dim=1, implemented by hand so the exported ONNX has no LayerNorm op) lives
   before every mixer. This is the standard stability fix for deeper stacks.

3. **EMA global state.** v2's global state is just a projection of the latest
   frame; across hundreds of frames that is effectively useless memory. v3
   maintains it with a per-channel learnable low-pass alpha::

       g' = sigmoid(alpha) * g + (1 - sigmoid(alpha)) * update_proj(x_last)

   Same state size as v2 (one frame wide), far richer temporal context.

4. **Wider frequency kernel option (k_f up to 9).** Frequency is acausal, so
   kernel and dilation are free-ish. We cap at `k_f=9` so that
   ``(k-1)*d < 14`` still holds comfortably.

5. **Time dilation pattern (1, 2, 4)** preserved. `(k_t-1)*d = 2*4 = 8 < 14`
   per the NPU constraint.

NPU / export constraints (unchanged from v2, still audited by tests):

- Only Conv2d (point-wise + depthwise), cat, slice, add, mul, sigmoid, relu.
- No batch folding: all tensors stay 4D ``[B, C, F, T]``.
- ``(kernel_size - 1) * dilation < 14`` on every Conv2d.
- No tensor constants baked into the graph; all state flows through the
  explicit 6-tuple state I/O used by ``TIGEREdgeMLPCellExportWrapper``.
- ``forward_sequence`` unrolls frame-by-frame through ``forward_cell`` so the
  training graph is the deployment graph.

State contract (identical shape layout to v2 so existing export / test
harnesses keep working)::

    past_kvs            : [B, n_heads=1, kv_window=1, att_hid_chan=1]   (dummy)
    past_valid_mask     : [B, 1, 1, 1]                                  (dummy)
    prev_states_0       : [B, |group0|*C, nband, (k_t-1)*d[0]]
    prev_states_1       : [B, |group1|*C, nband, (k_t-1)*d[1]]
    prev_states_2       : [B, |group2|*C, nband, (k_t-1)*d[2]]
    prev_global_states  : [B, L*C, nband, 1]

This module is strictly additive: the v2 ``TIGEREdgeMLP`` in
``tiger_edge_mlp.py`` is untouched. The training wrapper presets can opt into
v3 via the ``edge_impl='v3'`` switch (see ``TIGER.training_wrapper``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from TIGER.tiger_online import TIGER


# ---------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------


class _ChannelLayerNorm(nn.Module):
    """Channel-only LayerNorm on ``[B, C, F, T]``.

    Implemented with ``mean/var/rsqrt/add/mul`` so the exported ONNX graph has
    no LayerNorm op (avoids op-allowlist issues on the NPU compiler).
    """

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=1, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)
        return centered * inv_std * self.weight + self.bias


class _GLUFreqMixer(nn.Module):
    """Frequency mixer with GLU gating.

    Pipeline on ``[B, C, F, T]`` with freq padding symmetric (no state):

        value = expand_value(pre_norm(x))      # [B, CE, F, T]
        gate  = expand_gate (pre_norm(x))      # [B, CE, F, T]
        x = value * sigmoid(gate)
        x = dw_freq(x)                         # depthwise Conv2d((k_f, 1))
        x = post(x)                            # 1x1 back to C
        return residual + x
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        freq_kernel_size: int = 7,
        bias: bool = True,
    ):
        super().__init__()
        assert freq_kernel_size >= 1 and freq_kernel_size % 2 == 1, (
            "freq_kernel_size must be an odd positive integer"
        )
        assert (freq_kernel_size - 1) * 1 < 14, (
            "NPU constraint violated: (freq_kernel_size - 1) * dilation_f must be < 14"
        )

        self.channels = channels
        self.hidden = channels * expansion
        pad_f = (freq_kernel_size - 1) // 2

        self.pre_norm = _ChannelLayerNorm(channels)
        self.expand_value = nn.Conv2d(channels, self.hidden, (1, 1), bias=bias)
        self.expand_gate = nn.Conv2d(channels, self.hidden, (1, 1), bias=bias)
        self.dw = nn.Conv2d(
            self.hidden,
            self.hidden,
            kernel_size=(freq_kernel_size, 1),
            padding=(pad_f, 0),
            dilation=(1, 1),
            groups=self.hidden,
            bias=bias,
        )
        self.post = nn.Conv2d(self.hidden, channels, (1, 1), bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.pre_norm(x)
        v = self.expand_value(y)
        g = self.expand_gate(y)
        y = v * torch.sigmoid(g)
        y = self.dw(y)
        y = self.post(y)
        return residual + y


class _GLUTimeMixer(nn.Module):
    """Causal time mixer with explicit state + GLU gating.

    Pipeline on ``[B, C, F, T]``:

        y    = pre_norm(x)                       # channel LN
        combined = cat(state, y, dim=time)       # [B, C, F, state_width + T]
        new_state = combined[..., -state_width:] # update left cache
        y    = dw_time(combined)                 # depthwise (1, k_t), dilation d
        v    = expand_value(y)                   # 1x1 -> CE
        g    = expand_gate(y)                    # 1x1 -> CE
        y    = v * sigmoid(g)
        y    = project(y)                        # 1x1 -> C
        return residual + y, new_state

    State shape: ``[B, C, F, (k_t - 1) * d]`` (same as v2 -- cache is on the
    cheap C-width tensor, not the expanded hidden).
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        time_kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert time_kernel_size >= 1 and time_kernel_size % 2 == 1, (
            "time_kernel_size must be an odd positive integer"
        )
        assert dilation >= 1, "dilation must be >= 1"
        assert (time_kernel_size - 1) * dilation < 14, (
            "NPU constraint violated: (time_kernel_size - 1) * dilation must be < 14"
        )

        self.channels = channels
        self.hidden = channels * expansion
        self.time_kernel_size = time_kernel_size
        self.dilation = dilation
        self.state_width = (time_kernel_size - 1) * dilation

        self.pre_norm = _ChannelLayerNorm(channels)
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, time_kernel_size),
            padding=(0, 0),
            dilation=(1, dilation),
            groups=channels,
            bias=bias,
        )
        self.expand_value = nn.Conv2d(channels, self.hidden, (1, 1), bias=bias)
        self.expand_gate = nn.Conv2d(channels, self.hidden, (1, 1), bias=bias)
        self.project = nn.Conv2d(self.hidden, channels, (1, 1), bias=bias)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.onnx.is_in_onnx_export():
            assert x.dim() == 4, "time mixer expects a 4D input"
            B, C, _, _ = x.shape
            assert C == self.channels, f"expected channels={self.channels}, got {C}"
            assert state is not None and state.dim() == 4, "explicit state required"
            assert state.shape[0] == B
            assert state.shape[1] == C
            assert state.shape[3] == self.state_width

        residual = x
        y = self.pre_norm(x)
        combined = torch.cat([state, y], dim=-1)
        new_state = combined[:, :, :, -self.state_width :]
        y = self.dw(combined)
        v = self.expand_value(y)
        g = self.expand_gate(y)
        y = v * torch.sigmoid(g)
        y = self.project(y)
        return residual + y, new_state


class _EMAGlobalCtx(nn.Module):
    """EMA-updated global context on ``[B, C, F, 1]``.

    The per-channel learnable ``alpha_logit`` parameterises the EMA rate::

        alpha = sigmoid(alpha_logit)              # in (0, 1)
        g' = alpha * g + (1 - alpha) * update_proj(x_last)

    We avoid time folding: ``x_last = x[:, :, :, -1:]``. During training and
    deployment the cell is always run frame-by-frame, so this update is
    applied exactly once per frame.

    Fusion at the current frame:

        gate = sigmoid(gate_proj(g))
        mix  = mix_proj(g)
        x = x + gate * mix

    All three 1x1 Conv2ds keep the graph op-set small.
    """

    def __init__(self, channels: int, bias: bool = True):
        super().__init__()
        self.channels = channels
        self.gate_proj = nn.Conv2d(channels, channels, (1, 1), bias=bias)
        self.mix_proj = nn.Conv2d(channels, channels, (1, 1), bias=bias)
        self.update_proj = nn.Conv2d(channels, channels, (1, 1), bias=bias)
        # Initialise alpha_logit = 0 -> alpha = 0.5; the optimizer picks the
        # optimal decay per-channel during training.
        self.alpha_logit = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(
        self,
        x: torch.Tensor,
        global_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.onnx.is_in_onnx_export():
            B, C, F, T = x.shape
            assert C == self.channels
            assert global_state.shape == (B, C, F, 1)

        gate = torch.sigmoid(self.gate_proj(global_state))
        mix = self.mix_proj(global_state)
        x = x + gate * mix

        alpha = torch.sigmoid(self.alpha_logit)
        x_last = x[:, :, :, -1:]
        new_global = alpha * global_state + (1.0 - alpha) * self.update_proj(x_last)
        return x, new_global


class _TFMLPBlockV3(nn.Module):
    """One TF-MLP v3 block: freq GLU + causal time GLU + EMA global fusion."""

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        freq_kernel_size: int = 7,
        time_kernel_size: int = 3,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.freq_mixer = _GLUFreqMixer(channels, expansion, freq_kernel_size, bias=bias)
        self.time_mixer = _GLUTimeMixer(channels, expansion, time_kernel_size, dilation, bias=bias)
        self.global_ctx = _EMAGlobalCtx(channels, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        time_state: torch.Tensor,
        global_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.freq_mixer(x)
        x, new_time_state = self.time_mixer(x, time_state)
        x, new_global = self.global_ctx(x, global_state)
        return x, new_time_state, new_global


# ---------------------------------------------------------------------------
# separator
# ---------------------------------------------------------------------------


class EdgeTFMLPSeparatorV3(nn.Module):
    """TIGER-compatible separator with GLU TF-MLP blocks.

    Interface-compatible with v2's ``EdgeTFMLPSeparator``:

    - Input:  ``[B, nband, feature_dim, T]``
    - Output: ``[B, nband, feature_dim, T]`` (then decoded per-band by
      ``MaskBlock`` inside ``TIGER._decode_masks``)
    - State 6-tuple shapes are identical to v2 so
      ``TIGEREdgeMLPCellExportWrapper`` and
      ``build_tiger_edge_mlp_dummy_inputs`` keep working as-is.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        nband: int,
        num_output: int,
        num_blocks: int = 8,
        expansion: int = 2,
        freq_kernel_size: int = 7,
        time_kernel_size: int = 3,
        time_dilations: tuple[int, int, int] = (1, 2, 4),
        dummy_kv_channels: int = 1,
        dummy_kv_width: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        assert len(time_dilations) == 3, "time_dilations must be length 3"
        assert num_blocks >= 1
        assert hidden_channels >= 1

        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.nband = nband
        self.num_output = num_output
        self.num_blocks = num_blocks
        self.expansion = expansion
        self.freq_kernel_size = freq_kernel_size
        self.time_kernel_size = time_kernel_size
        self.time_dilations = tuple(int(d) for d in time_dilations)

        # Dummy-attention compatibility attributes expected by outer code.
        self.n_heads = 1
        self.kv_window_size = int(dummy_kv_width)
        self.att_hid_chan = int(dummy_kv_channels)
        self.att_val_hid_chan = 0
        self.iter = self.num_blocks

        # Input / output 1x1 projections.
        self.in_proj = nn.Sequential(
            nn.Conv2d(input_dim, hidden_channels, (1, 1), bias=bias),
            nn.ReLU(),
        )
        # Output returns feature_dim channels so the TIGER mask_decoders keep
        # their stock per-band MaskBlock layout.
        self.out_proj = nn.Conv2d(hidden_channels, input_dim, (1, 1), bias=True)

        self.blocks = nn.ModuleList(
            [
                _TFMLPBlockV3(
                    channels=hidden_channels,
                    expansion=expansion,
                    freq_kernel_size=freq_kernel_size,
                    time_kernel_size=time_kernel_size,
                    dilation=self.time_dilations[i % 3],
                    bias=bias,
                )
                for i in range(num_blocks)
            ]
        )

        # Group bookkeeping (same rotation as v2 so state shapes match).
        self.group_block_indices = [
            [i for i in range(num_blocks) if (i % 3) == g] for g in range(3)
        ]
        self.group_state_widths = [
            (time_kernel_size - 1) * self.time_dilations[g] for g in range(3)
        ]
        for w in self.group_state_widths:
            assert w < 14, "state width violates NPU constraint"
        self.group_state_channels = [
            len(self.group_block_indices[g]) * hidden_channels for g in range(3)
        ]
        self.global_state_channels = num_blocks * hidden_channels
        self.global_state_width = 1

    # --- streaming state helpers --------------------------------------------

    def _make_dummy_kv(self, batch_size: int, device, dtype):
        return torch.zeros(
            batch_size, self.n_heads, self.kv_window_size, self.att_hid_chan,
            device=device, dtype=dtype,
        )

    def _make_dummy_mask(self, batch_size: int, device, dtype):
        return torch.zeros(
            batch_size, 1, self.kv_window_size, 1, device=device, dtype=dtype,
        )

    def init_streaming_state(self, batch_size: int, device=None, dtype=None):
        z = lambda c, w: torch.zeros(
            batch_size, c, self.nband, w, device=device, dtype=dtype
        )
        return (
            self._make_dummy_kv(batch_size, device, dtype),
            self._make_dummy_mask(batch_size, device, dtype),
            z(self.group_state_channels[0], self.group_state_widths[0]),
            z(self.group_state_channels[1], self.group_state_widths[1]),
            z(self.group_state_channels[2], self.group_state_widths[2]),
            z(self.global_state_channels, self.global_state_width),
        )

    def _slice_group(self, packed: torch.Tensor, local_idx: int) -> torch.Tensor:
        c0 = local_idx * self.hidden_channels
        c1 = c0 + self.hidden_channels
        return packed[:, c0:c1, :, :]

    # --- forward -------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        past_kvs: torch.Tensor,
        past_valid_mask: torch.Tensor,
        prev_states_0: torch.Tensor,
        prev_states_1: torch.Tensor,
        prev_states_2: torch.Tensor,
        prev_global_states: torch.Tensor,
    ):
        if not torch.onnx.is_in_onnx_export():
            assert x.dim() == 4, "separator expects [B, nband, feature_dim, T]"
            B, nband_in, c_in, _ = x.shape
            assert nband_in == self.nband, f"Expected nband={self.nband}, got {nband_in}"
            assert c_in == self.input_dim, (
                f"Expected feature_dim={self.input_dim}, got {c_in}"
            )
            assert past_kvs.shape == (B, self.n_heads, self.kv_window_size, self.att_hid_chan)
            assert past_valid_mask.shape == (B, 1, self.kv_window_size, 1)
            assert prev_states_0.shape == (
                B, self.group_state_channels[0], self.nband, self.group_state_widths[0],
            )
            assert prev_states_1.shape == (
                B, self.group_state_channels[1], self.nband, self.group_state_widths[1],
            )
            assert prev_states_2.shape == (
                B, self.group_state_channels[2], self.nband, self.group_state_widths[2],
            )
            assert prev_global_states.shape == (
                B, self.global_state_channels, self.nband, self.global_state_width,
            )

        # [B, nband, feature_dim, T] -> [B, feature_dim, nband, T]
        x = x.permute(0, 2, 1, 3).contiguous()
        h = self.in_proj(x)

        new_group0: list[torch.Tensor] = []
        new_group1: list[torch.Tensor] = []
        new_group2: list[torch.Tensor] = []
        new_global: list[torch.Tensor] = []

        group_counters = [0, 0, 0]
        for block_idx, block in enumerate(self.blocks):
            g = block_idx % 3
            local = group_counters[g]
            group_counters[g] += 1

            if g == 0:
                ts = self._slice_group(prev_states_0, local)
            elif g == 1:
                ts = self._slice_group(prev_states_1, local)
            else:
                ts = self._slice_group(prev_states_2, local)

            g0 = block_idx * self.hidden_channels
            gs = prev_global_states[:, g0 : g0 + self.hidden_channels, :, :]

            h, new_ts, new_gs = block(h, time_state=ts, global_state=gs)

            if g == 0:
                new_group0.append(new_ts)
            elif g == 1:
                new_group1.append(new_ts)
            else:
                new_group2.append(new_ts)
            new_global.append(new_gs)

        def _cat_or_empty(parts: list[torch.Tensor], width: int) -> torch.Tensor:
            if len(parts) > 0:
                return torch.cat(parts, dim=1)
            return h.new_zeros(h.shape[0], 0, self.nband, width)

        new_states_0 = _cat_or_empty(new_group0, self.group_state_widths[0])
        new_states_1 = _cat_or_empty(new_group1, self.group_state_widths[1])
        new_states_2 = _cat_or_empty(new_group2, self.group_state_widths[2])
        new_global_states = torch.cat(new_global, dim=1)

        sep_out = self.out_proj(h)
        sep_out = sep_out.permute(0, 2, 1, 3).contiguous()

        return (
            sep_out,
            past_kvs,
            past_valid_mask,
            new_states_0,
            new_states_1,
            new_states_2,
            new_global_states,
        )


# ---------------------------------------------------------------------------
# top-level model
# ---------------------------------------------------------------------------


class TIGEREdgeMLPV3(TIGER):
    """TIGER with the TF-MLP v3 separator.

    Constructor reuses the stock ``TIGER`` args for encoder / decoder sizing
    (``out_channels``, ``in_channels``, ``upsampling_depth``, ``num_sources``)
    and adds ``edge_*`` knobs for the separator. Defaults target the
    "balance" preset (~6M total parameters); the training wrapper wires the
    small / balance / large presets.
    """

    def __init__(
        self,
        *args,
        edge_hidden_channels: int = 192,
        edge_num_blocks: int = 9,
        edge_expansion: int = 2,
        edge_freq_kernel_size: int = 7,
        edge_time_kernel_size: int = 3,
        edge_time_dilations: tuple[int, int, int] = (1, 2, 4),
        edge_dummy_kv_channels: int = 1,
        edge_dummy_kv_width: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert len(edge_time_dilations) == 3
        for d in edge_time_dilations:
            assert (edge_time_kernel_size - 1) * int(d) < 14, (
                "NPU constraint violated: (time_kernel - 1) * dilation must be < 14"
            )
        assert (edge_freq_kernel_size - 1) < 14, (
            "NPU constraint violated: (freq_kernel - 1) * 1 must be < 14"
        )

        self.separator = EdgeTFMLPSeparatorV3(
            input_dim=self.feature_dim,
            hidden_channels=edge_hidden_channels,
            nband=self.nband,
            num_output=self.num_output,
            num_blocks=edge_num_blocks,
            expansion=edge_expansion,
            freq_kernel_size=edge_freq_kernel_size,
            time_kernel_size=edge_time_kernel_size,
            time_dilations=edge_time_dilations,
            dummy_kv_channels=edge_dummy_kv_channels,
            dummy_kv_width=edge_dummy_kv_width,
            bias=True,
        )
        self.supports_exact_chunk_training = True

    # The streaming / cell / sequence methods are identical to v2's. We
    # duplicate them here so v3 stays self-contained (v2 and v3 are sibling
    # subclasses of TIGER, neither depends on the other).

    def init_streaming_state(self, batch_size, device=None, dtype=None):
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return self.separator.init_streaming_state(
            batch_size=batch_size, device=device, dtype=dtype,
        )

    def forward_cell(
        self,
        subband_spec_RIs,
        past_kvs=None,
        past_valid_mask=None,
        prev_states_0=None,
        prev_states_1=None,
        prev_states_2=None,
        prev_global_states=None,
    ):
        if not torch.onnx.is_in_onnx_export():
            assert subband_spec_RIs.shape[-1] == 1, "forward_cell expects T=1"

        B = subband_spec_RIs.shape[0]
        if (
            past_kvs is None
            or past_valid_mask is None
            or prev_states_0 is None
            or prev_states_1 is None
            or prev_states_2 is None
            or prev_global_states is None
        ):
            (
                past_kvs,
                past_valid_mask,
                prev_states_0,
                prev_states_1,
                prev_states_2,
                prev_global_states,
            ) = self.init_streaming_state(
                B, device=subband_spec_RIs.device, dtype=subband_spec_RIs.dtype,
            )

        subband_features = self._encode_subbands(subband_spec_RIs)
        (
            sep_out,
            new_kv,
            new_valid_mask,
            ns0,
            ns1,
            ns2,
            new_global,
        ) = self.separator(
            subband_features,
            past_kvs,
            past_valid_mask,
            prev_states_0,
            prev_states_1,
            prev_states_2,
            prev_global_states,
        )
        band_masked = self._decode_masks(sep_out)
        return (band_masked, new_kv, new_valid_mask, ns0, ns1, ns2, new_global)

    def forward_sequence(
        self,
        subband_spec_RIs=None,
        past_kvs=None,
        past_valid_mask=None,
        prev_states_0=None,
        prev_states_1=None,
        prev_states_2=None,
        prev_global_states=None,
        detach_state=False,
        chunk_size: int = 1,
    ):
        assert subband_spec_RIs is not None
        B, _, _, total_frames = subband_spec_RIs.shape
        if (
            past_kvs is None
            or past_valid_mask is None
            or prev_states_0 is None
            or prev_states_1 is None
            or prev_states_2 is None
            or prev_global_states is None
        ):
            (
                past_kvs,
                past_valid_mask,
                prev_states_0,
                prev_states_1,
                prev_states_2,
                prev_global_states,
            ) = self.init_streaming_state(
                B, device=subband_spec_RIs.device, dtype=subband_spec_RIs.dtype,
            )

        frame_outs: list[torch.Tensor] = []
        for t in range(total_frames):
            frame = subband_spec_RIs[:, :, :, t : t + 1]
            (
                frame_out,
                past_kvs,
                past_valid_mask,
                prev_states_0,
                prev_states_1,
                prev_states_2,
                prev_global_states,
            ) = self.forward_cell(
                frame,
                past_kvs=past_kvs,
                past_valid_mask=past_valid_mask,
                prev_states_0=prev_states_0,
                prev_states_1=prev_states_1,
                prev_states_2=prev_states_2,
                prev_global_states=prev_global_states,
            )
            frame_outs.append(frame_out)
            if detach_state:
                past_kvs = past_kvs.detach()
                past_valid_mask = past_valid_mask.detach()
                prev_states_0 = prev_states_0.detach()
                prev_states_1 = prev_states_1.detach()
                prev_states_2 = prev_states_2.detach()
                prev_global_states = prev_global_states.detach()

        return (
            torch.cat(frame_outs, dim=-1),
            past_kvs,
            past_valid_mask,
            prev_states_0,
            prev_states_1,
            prev_states_2,
            prev_global_states,
        )

    def forward(
        self,
        subband_spec_RIs=None,
        past_kvs=None,
        past_valid_mask=None,
        prev_states_0=None,
        prev_states_1=None,
        prev_states_2=None,
        prev_global_states=None,
        detach_state=False,
        chunk_size: int = 1,
    ):
        assert subband_spec_RIs is not None
        if subband_spec_RIs.shape[-1] == 1:
            return self.forward_cell(
                subband_spec_RIs,
                past_kvs=past_kvs,
                past_valid_mask=past_valid_mask,
                prev_states_0=prev_states_0,
                prev_states_1=prev_states_1,
                prev_states_2=prev_states_2,
                prev_global_states=prev_global_states,
            )
        return self.forward_sequence(
            subband_spec_RIs,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            prev_states_0=prev_states_0,
            prev_states_1=prev_states_1,
            prev_states_2=prev_states_2,
            prev_global_states=prev_global_states,
            detach_state=detach_state,
            chunk_size=chunk_size,
        )


# ---------------------------------------------------------------------------
# preset factory
# ---------------------------------------------------------------------------


# DnR 3-stem presets targeted at ~3M / ~6M / ~9M total parameters (separator
# sizes dominate). Each is verified to fit the 192 KiB fp16 DSP state budget
# by ``TF-MLPNet/tests/test_tiger_edge_mlp_smoke.py``.
#
# We pass an explicit ``pre_calc_bands`` 8-band split for DnR at 44.1 kHz /
# n_fft=2048 so the streaming state stays small enough for the NPU. The stock
# TIGER ``calculate_band_widths`` produces ~67 single-frequency bands at this
# configuration, which would blow the state quota regardless of channel count.
# ``feature_dim`` values (48 / 72 / 96) are all divisible by ``num_sources=3``
# so the per-band ``MaskBlock`` depthwise groups work out.
_DNR_8_BANDS: tuple[int, ...] = (10, 28, 56, 93, 186, 186, 279, 187)
assert sum(_DNR_8_BANDS) == 1025, "bands must cover enc_dim = n_fft//2 + 1 = 1025"


V3_PRESETS: dict[str, dict] = {
    "v3-small": {
        "out_channels": 48,
        "in_channels": 192,
        "num_blocks": 8,
        "upsampling_depth": 2,
        "pre_calc_bands": _DNR_8_BANDS,
        "edge_hidden_channels": 160,
        "edge_num_blocks": 8,
        "edge_expansion": 2,
        "edge_freq_kernel_size": 5,
        "edge_time_kernel_size": 3,
        "edge_time_dilations": (1, 2, 4),
    },
    "v3-balance": {
        "out_channels": 72,
        "in_channels": 288,
        "num_blocks": 9,
        "upsampling_depth": 2,
        "pre_calc_bands": _DNR_8_BANDS,
        "edge_hidden_channels": 208,
        "edge_num_blocks": 9,
        "edge_expansion": 2,
        "edge_freq_kernel_size": 7,
        "edge_time_kernel_size": 3,
        "edge_time_dilations": (1, 2, 4),
    },
    "v3-large": {
        "out_channels": 96,
        "in_channels": 384,
        "num_blocks": 8,
        "upsampling_depth": 2,
        "pre_calc_bands": _DNR_8_BANDS,
        "edge_hidden_channels": 272,
        "edge_num_blocks": 8,
        "edge_expansion": 2,
        "edge_freq_kernel_size": 7,
        "edge_time_kernel_size": 3,
        "edge_time_dilations": (1, 2, 4),
    },
}


def build_tiger_edge_mlp_v3(preset: str, *, num_sources: int = 3, **overrides) -> TIGEREdgeMLPV3:
    """Build a ``TIGEREdgeMLPV3`` by preset name with optional kwarg overrides."""
    if preset not in V3_PRESETS:
        raise KeyError(
            f"Unknown v3 preset {preset!r}. Available: {sorted(V3_PRESETS)}"
        )
    cfg = dict(V3_PRESETS[preset])
    cfg.update(overrides)
    return TIGEREdgeMLPV3(
        num_sources=num_sources,
        need_streaming=True,
        **cfg,
    )
