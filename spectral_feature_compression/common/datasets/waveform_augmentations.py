from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SourceSeparationAugmenter(nn.Module):
    """Training-time waveform augmentations for separated source tensors.

    The expected input shape is ``[n_src, n_chan, n_samples]``.  The augmenter
    returns sources with the same shape so the dataset can recompute the mixture
    as ``ref.sum(dim=0)`` after augmentation.
    """

    def __init__(
        self,
        *,
        p_gain: float = 0.0,
        gain_db_min: float = -6.0,
        gain_db_max: float = 6.0,
        p_polarity: float = 0.0,
        p_channel_swap: float = 0.0,
        p_time_shift: float = 0.0,
        max_time_shift_samples: int = 0,
        p_pitch_time: float = 0.0,
        pitch_time_scale_min: float = 0.98,
        pitch_time_scale_max: float = 1.02,
        p_random_eq: float = 0.0,
        eq_bands: int = 8,
        eq_gain_db_min: float = -3.0,
        eq_gain_db_max: float = 3.0,
        p_band_dropout: float = 0.0,
        band_dropout_width: float = 0.08,
    ) -> None:
        super().__init__()
        self.p_gain = float(p_gain)
        self.gain_db_min = float(gain_db_min)
        self.gain_db_max = float(gain_db_max)
        self.p_polarity = float(p_polarity)
        self.p_channel_swap = float(p_channel_swap)
        self.p_time_shift = float(p_time_shift)
        self.max_time_shift_samples = int(max_time_shift_samples)
        self.p_pitch_time = float(p_pitch_time)
        self.pitch_time_scale_min = float(pitch_time_scale_min)
        self.pitch_time_scale_max = float(pitch_time_scale_max)
        self.p_random_eq = float(p_random_eq)
        self.eq_bands = int(eq_bands)
        self.eq_gain_db_min = float(eq_gain_db_min)
        self.eq_gain_db_max = float(eq_gain_db_max)
        self.p_band_dropout = float(p_band_dropout)
        self.band_dropout_width = float(band_dropout_width)

        if self.eq_bands <= 0:
            raise ValueError("eq_bands must be positive")
        if self.max_time_shift_samples < 0:
            raise ValueError("max_time_shift_samples must be non-negative")
        if not 0.0 < self.pitch_time_scale_min <= self.pitch_time_scale_max:
            raise ValueError("pitch/time scales must satisfy 0 < min <= max")
        for name in (
            "p_gain",
            "p_polarity",
            "p_channel_swap",
            "p_time_shift",
            "p_pitch_time",
            "p_random_eq",
            "p_band_dropout",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    def _bernoulli(self, probability: float, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.rand(shape, device=device) < probability

    def _apply_gain(self, ref: torch.Tensor) -> torch.Tensor:
        if self.p_gain <= 0.0:
            return ref
        mask = self._bernoulli(self.p_gain, (ref.shape[0], 1, 1), ref.device)
        gain_db = torch.empty(ref.shape[0], 1, 1, device=ref.device, dtype=ref.dtype).uniform_(
            self.gain_db_min,
            self.gain_db_max,
        )
        gain = torch.pow(ref.new_tensor(10.0), gain_db / 20.0)
        return torch.where(mask, ref * gain, ref)

    def _apply_polarity(self, ref: torch.Tensor) -> torch.Tensor:
        if self.p_polarity <= 0.0:
            return ref
        flip = self._bernoulli(self.p_polarity, (ref.shape[0], 1, 1), ref.device)
        return torch.where(flip, -ref, ref)

    def _apply_channel_swap(self, ref: torch.Tensor) -> torch.Tensor:
        if self.p_channel_swap <= 0.0 or ref.shape[1] < 2:
            return ref
        out = ref.clone()
        swap = self._bernoulli(self.p_channel_swap, (ref.shape[0],), ref.device)
        if swap.any():
            out[swap] = out[swap].flip(dims=(1,))
        return out

    def _apply_time_shift(self, ref: torch.Tensor) -> torch.Tensor:
        if self.p_time_shift <= 0.0 or self.max_time_shift_samples == 0:
            return ref
        out = ref.clone()
        apply = self._bernoulli(self.p_time_shift, (ref.shape[0],), ref.device)
        for source_idx in torch.nonzero(apply, as_tuple=False).flatten().tolist():
            shift = int(torch.randint(-self.max_time_shift_samples, self.max_time_shift_samples + 1, ()).item())
            if shift != 0:
                out[source_idx] = torch.roll(out[source_idx], shifts=shift, dims=-1)
        return out

    def _time_scale_one(self, source: torch.Tensor, scale: float) -> torch.Tensor:
        n_samples = source.shape[-1]
        scaled_len = max(1, int(round(n_samples * scale)))
        scaled = F.interpolate(
            source.unsqueeze(0),
            size=scaled_len,
            mode="linear",
            align_corners=False,
        ).squeeze(0)
        if scaled_len == n_samples:
            return scaled
        if scaled_len > n_samples:
            start = int(torch.randint(0, scaled_len - n_samples + 1, ()).item())
            return scaled[..., start : start + n_samples]
        pad_total = n_samples - scaled_len
        pad_left = int(torch.randint(0, pad_total + 1, ()).item())
        return F.pad(scaled, (pad_left, pad_total - pad_left))

    def _apply_pitch_time(self, ref: torch.Tensor) -> torch.Tensor:
        if self.p_pitch_time <= 0.0:
            return ref
        out = ref.clone()
        apply = self._bernoulli(self.p_pitch_time, (ref.shape[0],), ref.device)
        for source_idx in torch.nonzero(apply, as_tuple=False).flatten().tolist():
            scale = torch.empty((), device=ref.device).uniform_(
                self.pitch_time_scale_min,
                self.pitch_time_scale_max,
            )
            out[source_idx] = self._time_scale_one(out[source_idx], float(scale.item()))
        return out

    def _frequency_mask(self, ref: torch.Tensor) -> torch.Tensor:
        spec = torch.fft.rfft(ref.float(), dim=-1)
        n_bins = spec.shape[-1]
        gain = ref.new_ones((ref.shape[0], 1, n_bins), dtype=torch.float32)

        if self.p_random_eq > 0.0:
            apply_eq = self._bernoulli(self.p_random_eq, (ref.shape[0],), ref.device)
            edges = torch.linspace(0, n_bins, self.eq_bands + 1, device=ref.device).round().long()
            for band_idx in range(self.eq_bands):
                start = int(edges[band_idx].item())
                end = int(edges[band_idx + 1].item())
                if end <= start:
                    continue
                band_gain_db = torch.empty(ref.shape[0], 1, 1, device=ref.device).uniform_(
                    self.eq_gain_db_min,
                    self.eq_gain_db_max,
                )
                band_gain = torch.pow(ref.new_tensor(10.0), band_gain_db.float() / 20.0)
                gain[..., start:end] = torch.where(apply_eq.view(-1, 1, 1), band_gain, gain[..., start:end])

        if self.p_band_dropout > 0.0:
            apply_dropout = self._bernoulli(self.p_band_dropout, (ref.shape[0],), ref.device)
            width = max(1, int(round(n_bins * self.band_dropout_width)))
            for source_idx in torch.nonzero(apply_dropout, as_tuple=False).flatten().tolist():
                start = int(torch.randint(0, max(1, n_bins - width + 1), ()).item())
                gain[source_idx, :, start : start + width] = 0.0

        return torch.fft.irfft(spec * gain.to(spec.device), n=ref.shape[-1], dim=-1).to(dtype=ref.dtype)

    def forward(self, ref: torch.Tensor) -> torch.Tensor:
        if ref.ndim != 3:
            raise ValueError(f"Expected [n_src, n_chan, n_samples], got {tuple(ref.shape)}")
        out = ref
        out = self._apply_gain(out)
        out = self._apply_polarity(out)
        out = self._apply_channel_swap(out)
        out = self._apply_time_shift(out)
        out = self._apply_pitch_time(out)
        if self.p_random_eq > 0.0 or self.p_band_dropout > 0.0:
            out = self._frequency_mask(out)
        return out
