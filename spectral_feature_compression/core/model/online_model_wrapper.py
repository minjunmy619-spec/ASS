from __future__ import annotations

import numpy as np

import torch
from torch import autocast
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.transforms import Spectrogram


class CausalISTFTOLA(nn.Module):
    """
    Manual overlap-add iSTFT for strictly causal ``center=False`` pipelines.

    ``torch.istft`` / ``torchaudio.transforms.InverseSpectrogram`` performs a
    strict overlap-add check that fails for common analysis windows such as the
    Hann window when ``center=False``. For the online models we instead do the
    inverse FFT and overlap-add explicitly, then normalize by the accumulated
    squared synthesis window wherever the envelope is non-zero.
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        window_fn=torch.hann_window,
        normalized: bool = False,
        wkwargs: dict | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        if normalized:
            raise ValueError("CausalISTFTOLA currently supports normalized=False only.")

        window = window_fn(n_fft, **(wkwargs or {}))
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalized = normalized
        self.eps = eps
        self.register_buffer("window", window)
        self.register_buffer("window_sq", window.square())

    def forward(self, stft: torch.Tensor, length: int | None = None) -> torch.Tensor:
        # stft: (..., F, T), complex
        if not torch.is_complex(stft):
            raise TypeError(f"CausalISTFTOLA expects a complex STFT tensor, got dtype={stft.dtype}.")

        *lead_shape, n_freq, n_frames = stft.shape
        expected_n_freq = self.n_fft // 2 + 1
        if n_freq != expected_n_freq:
            raise ValueError(f"Expected {expected_n_freq} frequency bins, got {n_freq}.")

        flat = stft.reshape(-1, n_freq, n_frames)
        # frames: (Bflat, n_fft, T)
        frames = torch.fft.irfft(flat, n=self.n_fft, dim=1)
        frames = frames * self.window.view(1, self.n_fft, 1)

        ola_length = self.n_fft + self.hop_length * max(n_frames - 1, 0)
        # fold input: (N, C * kernel_h * kernel_w, L)
        frames_cols = frames
        signal = F.fold(
            frames_cols,
            output_size=(1, ola_length),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).squeeze(-2).squeeze(-2)

        env_cols = self.window_sq.view(1, self.n_fft, 1).expand(flat.shape[0], self.n_fft, n_frames)
        envelope = F.fold(
            env_cols,
            output_size=(1, ola_length),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).squeeze(-2).squeeze(-2)

        signal = torch.where(envelope > self.eps, signal / envelope.clamp_min(self.eps), torch.zeros_like(signal))

        if length is not None:
            if signal.shape[-1] < length:
                signal = F.pad(signal, (0, length - signal.shape[-1]))
            else:
                signal = signal[..., :length]

        return signal.reshape(*lead_shape, signal.shape[-1])


class OnlineModelWrapper(nn.Module):
    """
    Strictly causal waveform wrapper for online models.

    Differences from the generic ModelWrapper:
    - STFT/iSTFT use center=False to avoid future waveform context.
    - global input scaling is forbidden because it leaks future information.
    - the wrapped model is still free to process multiple frames per call, but
      every frame is generated from past/current waveform samples only.
    """

    def __init__(
        self,
        model: nn.Module,
        n_fft: int,
        hop_length: int,
        fs: int,
        scaling: bool = False,
        css_segment_size: int = 6,
        css_shift_size: int = 6,
        css_batch_size: int = 1,
    ):
        super().__init__()
        if scaling:
            raise ValueError("OnlineModelWrapper does not support global scaling in strict realtime mode.")

        self.model = model
        self.stft = nn.Sequential(Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None, center=False))
        self.istft = CausalISTFTOLA(n_fft=n_fft, hop_length=hop_length)

        self.hop_length = hop_length
        self.fs = fs
        self.scaling = False

        self.css_segment_size = css_segment_size
        self.css_shift_size = css_shift_size
        self.css_batch_size = css_batch_size

        # External realtime frontends need at least this many past samples to
        # produce causal STFT frames compatible with the training wrapper.
        self.wave_context_samples = n_fft - hop_length
        # The waveform wrapper pads both sides explicitly so the strictly
        # causal STFT/iSTFT pair has valid overlap-add support at the segment
        # boundaries without relying on center=True.
        self.wave_tail_pad_samples = n_fft - hop_length

    def forward(self, wav, **kwargs):
        left_pad = self.wave_context_samples
        right_pad = self.wave_tail_pad_samples
        wav_pad = F.pad(wav, (left_pad, right_pad))

        with autocast(device_type="cuda", enabled=False):
            # x: (..., F, T_pad), complex STFT with explicit causal boundary padding.
            x = self.stft(wav_pad)

        est_stft = self.model(x, **kwargs)

        with autocast(device_type="cuda", enabled=False):
            est_pad = self.istft(est_stft, wav_pad.shape[-1])

        est = est_pad[..., left_pad : left_pad + wav.shape[-1]]

        return est

    def css(self, speech_mix: torch.Tensor, **kwargs):
        """
        Chunk-wise separation for long recordings.

        This remains an offline evaluation helper; it does not replace the
        stateful frame/chunk streaming APIs implemented on the online cores.
        """

        speech_length = speech_mix.shape[-1]
        if speech_length > self.css_segment_size * self.fs:
            overlap_length = int(np.round(self.fs * (self.css_segment_size - self.css_shift_size)))
            num_segments = int(np.ceil((speech_length - overlap_length) / (self.css_shift_size * self.fs)))
            t = t_total = int(self.css_segment_size * self.fs)
            pad_shape = speech_mix[..., :t_total].shape

            segments = []
            is_silent = []

            for i in range(num_segments):
                st = int(i * self.css_shift_size * self.fs)
                en = st + t_total

                if en >= speech_length:
                    en = speech_length
                    speech_seg = speech_mix.new_zeros(pad_shape)
                    t = en - st
                    speech_seg[..., :t] = speech_mix[..., st:en].clone()
                else:
                    speech_seg = speech_mix[..., st:en].clone()

                segments.append(speech_seg)
                is_silent.append(abs(speech_seg).sum().item() == 0.0)

            enh_waves = [None] * num_segments
            valid_indices = [i for i, silent in enumerate(is_silent) if not silent]

            if len(valid_indices) > 0:
                css_bs = self.css_batch_size
                for mb_st in range(0, len(valid_indices), css_bs):
                    mb_indices = valid_indices[mb_st : mb_st + css_bs]
                    seg_batch = torch.cat([segments[i] for i in mb_indices], dim=0)
                    processed_batch = self(seg_batch, **kwargs)[..., :t_total]
                    for k, seg_i in enumerate(mb_indices):
                        enh_waves[seg_i] = processed_batch[[k]]

            for i in range(num_segments):
                if enh_waves[i] is None:
                    enh_waves[i] = torch.zeros_like(enh_waves[valid_indices[0]])

            waves = enh_waves[0]
            for i in range(1, num_segments):
                if i == num_segments - 1:
                    enh_waves[i][..., t:] = 0
                    enh_waves_res_i = enh_waves[i][..., overlap_length:t]
                else:
                    enh_waves_res_i = enh_waves[i][..., overlap_length:]

                if overlap_length > 0:
                    waves[..., -overlap_length:] = (
                        waves[..., -overlap_length:] + enh_waves[i][..., :overlap_length]
                    ) / 2
                waves = torch.cat([waves, enh_waves_res_i], dim=-1)
        else:
            waves = self(speech_mix, **kwargs)

        return waves
