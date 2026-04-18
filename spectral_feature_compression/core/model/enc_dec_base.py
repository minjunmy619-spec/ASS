# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange

from spectral_feature_compression.core.model.bandit_split import get_band_specs


class EncoderBase(nn.Module):
    def __init__(
        self,
        d_inner: int,
        d_model: int,
        n_chan: int,
        sample_rate: int,
        n_fft: int,
        n_bands: int,
        band_config: str,
        query_type: str,
    ):
        """
        d_inner: int
            The feature dimension of the encoder/decoder (`D'` in the paper). By setting `d_inner` < `d_model`, we can
            save computational cost.
        d_model: int
            The feature dimension of the separator (`D` in the paper).
        n_chan: int
            The number of channels of the input spectrogram.
        sample_rate: int
            Sampling rate (or frequency) of the input.
        n_fft: int
            FFT size. The number of frequency bins of the input spectrogram must be n_fft//2 + 1
        n_bands: int
            The number of
        query_type: str
            Either `learnable` or `adaptive`.


        """
        super().__init__()

        self.n_chan = n_chan
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_bands = n_bands
        self.query_type = query_type

        self._instantiate_conv_layers()

        assert band_config in ["musical", "tribark", "bark", "erb", "mel"]
        self.band_indices, _, _ = get_band_specs(band_config, n_fft, sample_rate, n_bands=n_bands)

        # check if all the frequency bins are included
        counter = torch.zeros(n_fft // 2 + 1)
        for s, e in self.band_indices:
            counter[s:e] += 1
        assert not torch.any(counter == 0), f"{band_config}{n_bands} does not cover all the frequency bins"

        if self.query_type == "learnable":
            self.query = nn.Parameter(torch.randn(n_bands, d_inner))
        elif self.query_type == "adaptive":
            # widths per band (K,)
            widths = torch.tensor([e - s for s, e in self.band_indices], dtype=torch.long)
            self.register_buffer("widths", widths, persistent=False)

            # Flattened frequency indices across all bands: concat of [s:e] for each band
            flat_idx_list = []
            band_ids_list = []
            for b, (s, e) in enumerate(self.band_indices):
                if e > s:
                    flat_idx_list.append(torch.arange(s, e, dtype=torch.long))
                    band_ids_list.append(torch.full((e - s,), b, dtype=torch.long))

            flat_idx = torch.cat(flat_idx_list) if flat_idx_list else torch.empty(0, dtype=torch.long)
            band_ids = torch.cat(band_ids_list) if band_ids_list else torch.empty(0, dtype=torch.long)

            # Buffers move with the module, no .to(device) needed in forward
            self.register_buffer("flat_idx", flat_idx, persistent=False)  # (sumW,)
            self.register_buffer("band_ids", band_ids, persistent=False)  # (sumW,)

            # Learnable weights for each valid position (initialized to per-band uniform)
            self.freq_weights = nn.Parameter(
                torch.cat([torch.ones(w, dtype=torch.float32) / max(int(w), 1) for w in widths.tolist()])
                if widths.numel() > 0
                else torch.empty(0, dtype=torch.float32)
            )

    def _instantiate_conv_layers(self, d_inner_output_conv=None):
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)

        self.input_conv = nn.Sequential(
            nn.Conv2d(2 * self.n_chan, self.d_inner, ks, padding=padding),
            Rearrange("b d t f -> b t f d"),
            RMSNorm(self.d_inner),
            Rearrange("b t f d-> b d t f"),
        )

        d_inner_output_conv = self.d_inner if d_inner_output_conv is None else d_inner_output_conv
        self.output_conv = nn.Sequential(
            nn.Conv2d(d_inner_output_conv, self.d_model, ks, padding=padding),
            Rearrange("b d t f -> b t f d"),
            RMSNorm(self.d_model),
            Rearrange("b t f d-> b d t f"),
        )

    def _output_proj(self, query):
        return self.output_conv(query.permute(0, 3, 1, 2))

    def _prepare_query(self, emb, query_orig=None):
        if self.query_type == "learnable":
            query = self.query.unsqueeze(0)
        elif self.query_type == "adaptive":
            query = self._vectorized_overlapped_mean_weighted(emb)
        else:
            raise RuntimeError(f"{self.query_type} is not supported.")

        return query

    def _vectorized_overlapped_mean_weighted(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Compute band-wise weighted means without building a padded (K,maxw) matrix.

        Args:
            emb: (B, S, H) full sequence along S
            reducer: OverlapReducer with buffers flat_idx (sumW,), band_ids (sumW,)

        Returns:
            out: (B, K, H) band-wise weighted averages
        """

        B, S, H = emb.shape
        K = int(self.widths.numel())

        # 1) Gather all valid positions once: (B, sumW, H)
        flat = emb.index_select(1, self.flat_idx).reshape(B, -1, H)

        # 2) Apply learnable weights per position
        w = self.freq_weights.to(dtype=emb.dtype)  # (sumW,)
        weighted = flat * w[None, :, None]  # (B, sumW, H)

        # 3) Segment reduce by band using index_add_ over band dimension
        out = emb.new_zeros(B, K, H)  # (B, K, H)
        out.index_add_(1, self.band_ids, weighted)  # sum per band

        # 4) Normalize by per-band weight sums
        denom = emb.new_zeros(K)  # (K,)
        denom.index_add_(0, self.band_ids, w)  # weight sum per band
        out = out / denom[None, :, None].clamp_min(1e-8)
        return out


class DecoderBase(nn.Module):
    def __init__(
        self,
        d_inner: int,
        d_model: int,
        n_src: int,
        n_chan: int,
        sample_rate: int,
        n_fft: int,
        n_bands: int,
        band_config: str,
        query_type: str,
    ):
        super().__init__()

        self.n_src = n_src
        self.n_chan = n_chan
        self.d_model = d_model
        self.d_inner = d_inner
        self.query_type = query_type

        self._instantiate_conv_layers()

        # query position setup
        assert band_config in ["musical", "tribark", "bark", "erb", "mel"]
        self.band_indices, _, _ = get_band_specs(band_config, n_fft, sample_rate, n_bands=n_bands)

        # check if all the frequency bins are included
        counter = torch.zeros(n_fft // 2 + 1)
        for s, e in self.band_indices:
            counter[s:e] += 1
        assert not torch.any(counter == 0), f"{band_config}{n_bands} does not cover all the frequency bins"

        if self.query_type == "learnable":
            self.query = nn.Parameter(torch.randn(n_fft // 2 + 1, d_inner))

    def _instantiate_conv_layers(self, d_inner_output_conv=None):
        t_ksize = 3
        ks, padding = (t_ksize, 3), (t_ksize // 2, 1)
        self.input_conv = nn.ConvTranspose2d(self.d_model, self.d_inner, ks, padding=padding)

        d_inner_output_conv = self.d_inner if d_inner_output_conv is None else d_inner_output_conv
        self.output_conv = nn.ConvTranspose2d(d_inner_output_conv, self.n_chan * self.n_src * 2, ks, padding=padding)

        if self.query_type == "adaptive":
            self.query_mlp = nn.Sequential(
                RMSNorm(d_inner_output_conv), SwiGLUMLP(self.d_inner, self.d_inner * 2, d_inner_output_conv)
            )
        else:
            self.query_mlp = None

    def _output_proj(self, query):
        return self.output_conv(query.permute(0, 3, 1, 2))

    def _prepare_query(self, emb, query_orig=None):
        if self.query_type == "learnable":
            query = self.query[None].expand(emb.shape[0], -1, -1)
        elif self.query_type == "adaptive":
            query = self.query_mlp(query_orig) if self.query_mlp is not None else query_orig
        else:
            raise RuntimeError(f"{self.query_type} is not supported.")

        return query


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        # x: (..., C)
        variance = (x * x).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_inner: int, d_input: int | None = None):
        super().__init__()

        self.d_inner = d_inner
        self.linear1 = nn.Linear(d_model if d_input is None else d_input, d_inner * 2)
        self.linear2 = nn.Linear(d_inner, d_model)

    def forward(self, x):
        a, b = self.linear1(x).chunk(2, dim=-1)
        b = F.silu(b)
        return self.linear2(a * b)
