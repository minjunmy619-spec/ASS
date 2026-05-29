from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F

Resolution = tuple[int, int]


class CompositeSeparationSpectralLoss(nn.Module):
    """Opt-in spectral loss stack for waveform source-separation training.

    Inputs use the repo's separated waveform layout: ``[B, N, C, samples]``.
    The module is training-only and intentionally independent from the exported
    NPU graph.
    """

    def __init__(
        self,
        *,
        n_fft: int,
        hop_length: int,
        complex_ri_weight: float = 0.0,
        log_magnitude_weight: float = 0.0,
        multi_resolution_stft_weight: float = 0.0,
        multi_resolution_stft_resolutions: Sequence[Sequence[int] | Resolution] | None = None,
        transient_weight: float = 0.0,
        eps: float = 1.0e-7,
    ) -> None:
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.complex_ri_weight = float(complex_ri_weight)
        self.log_magnitude_weight = float(log_magnitude_weight)
        self.multi_resolution_stft_weight = float(multi_resolution_stft_weight)
        self.transient_weight = float(transient_weight)
        self.eps = float(eps)

        if multi_resolution_stft_resolutions is None:
            multi_resolution_stft_resolutions = ((self.n_fft, self.hop_length),)
        self.multi_resolution_stft_resolutions = tuple(
            (int(resolution[0]), int(resolution[1])) for resolution in multi_resolution_stft_resolutions
        )

    @property
    def enabled(self) -> bool:
        return any(
            weight > 0.0
            for weight in (
                self.complex_ri_weight,
                self.log_magnitude_weight,
                self.multi_resolution_stft_weight,
                self.transient_weight,
            )
        )

    def _stft(self, wav: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
        wav = wav.float().reshape(-1, wav.shape[-1])
        if wav.shape[-1] < n_fft:
            wav = F.pad(wav, (0, n_fft - wav.shape[-1]))
        window = torch.hann_window(n_fft, device=wav.device, dtype=wav.dtype)
        return torch.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        )

    def _complex_ri_l1(self, est: torch.Tensor, ref: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
        est_spec = self._stft(est, n_fft=n_fft, hop_length=hop_length)
        ref_spec = self._stft(ref, n_fft=n_fft, hop_length=hop_length)
        return F.l1_loss(torch.view_as_real(est_spec), torch.view_as_real(ref_spec))

    def _log_magnitude_l1(self, est: torch.Tensor, ref: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
        est_spec = self._stft(est, n_fft=n_fft, hop_length=hop_length)
        ref_spec = self._stft(ref, n_fft=n_fft, hop_length=hop_length)
        return F.l1_loss(torch.log(est_spec.abs() + self.eps), torch.log(ref_spec.abs() + self.eps))

    def _multi_resolution_stft(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        losses = []
        for n_fft, hop_length in self.multi_resolution_stft_resolutions:
            est_spec = self._stft(est, n_fft=n_fft, hop_length=hop_length)
            ref_spec = self._stft(ref, n_fft=n_fft, hop_length=hop_length)
            complex_loss = F.l1_loss(torch.view_as_real(est_spec), torch.view_as_real(ref_spec))
            log_mag_loss = F.l1_loss(torch.log(est_spec.abs() + self.eps), torch.log(ref_spec.abs() + self.eps))
            losses.append(complex_loss + log_mag_loss)
        return torch.stack(losses).mean()

    def _transient_l1(self, est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if est.shape[-1] < 2:
            return est.new_zeros(())
        est_diff = est[..., 1:] - est[..., :-1]
        ref_diff = ref[..., 1:] - ref[..., :-1]
        return F.l1_loss(est_diff, ref_diff)

    def forward(self, est: torch.Tensor, ref: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if est.shape != ref.shape:
            raise ValueError(f"est/ref shape mismatch: {tuple(est.shape)} vs {tuple(ref.shape)}")

        total = est.new_zeros(())
        components: dict[str, torch.Tensor] = {}

        if self.complex_ri_weight > 0.0:
            value = self._complex_ri_l1(est, ref, self.n_fft, self.hop_length)
            components["complex_ri"] = value
            total = total + self.complex_ri_weight * value

        if self.log_magnitude_weight > 0.0:
            value = self._log_magnitude_l1(est, ref, self.n_fft, self.hop_length)
            components["log_magnitude"] = value
            total = total + self.log_magnitude_weight * value

        if self.multi_resolution_stft_weight > 0.0:
            value = self._multi_resolution_stft(est, ref)
            components["multi_resolution_stft"] = value
            total = total + self.multi_resolution_stft_weight * value

        if self.transient_weight > 0.0:
            value = self._transient_l1(est, ref)
            components["transient"] = value
            total = total + self.transient_weight * value

        return total, components
