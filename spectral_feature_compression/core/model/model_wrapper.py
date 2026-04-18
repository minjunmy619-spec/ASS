# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

import numpy as np

import torch
from torch import autocast
import torch.nn as nn

from torchaudio.transforms import InverseSpectrogram, Spectrogram


class ModelWrapper(nn.Module):
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

        self.model = model

        self.stft = nn.Sequential(Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None))
        self.istft = InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)

        self.hop_length = hop_length
        self.fs = fs
        self.scaling = scaling

        self.css_segment_size = css_segment_size
        self.css_shift_size = css_shift_size
        self.css_batch_size = css_batch_size

    def forward(self, wav, **kwargs):
        if self.scaling:
            with autocast(device_type="cuda", enabled=False):
                wav_stft = self.stft(wav)
                scale = wav_stft.abs().square().clip(1e-15)
                scale = (
                    scale.mean(dim=(1, -1), keepdims=True).sqrt()
                    if wav_stft.ndim == 3
                    else scale.mean(dim=(1, 2, 3), keepdims=True).sqrt()
                )
                wav = wav / scale.squeeze(1)

        with autocast(device_type="cuda", enabled=False):
            x = self.stft(wav)[..., : wav.shape[-1] // self.hop_length]

        est_stft = self.model(x, **kwargs)

        with autocast(device_type="cuda", enabled=False):
            est = self.istft(est_stft, wav.shape[-1])  # [B, N, M, T]

        if self.scaling:
            est = est * scale

        return est

    def css(self, speech_mix: torch.Tensor, **kwargs):
        """
        Chunk-wise separation for long recording.
        """
        speech_length = speech_mix.shape[-1]
        if speech_length > self.css_segment_size * self.fs:
            # Segment-wise speech enhancement/separation
            overlap_length = int(np.round(self.fs * (self.css_segment_size - self.css_shift_size)))
            num_segments = int(np.ceil((speech_length - overlap_length) / (self.css_shift_size * self.fs)))
            t = T = int(self.css_segment_size * self.fs)
            pad_shape = speech_mix[..., :T].shape

            # ---------------------------
            # (1) segmenting
            # ---------------------------
            segments = []
            is_silent = []

            for i in range(num_segments):
                st = int(i * self.css_shift_size * self.fs)
                en = st + T

                if en >= speech_length:
                    en = speech_length
                    speech_seg = speech_mix.new_zeros(pad_shape)
                    t = en - st
                    speech_seg[..., :t] = speech_mix[..., st:en].clone()
                else:
                    speech_seg = speech_mix[..., st:en].clone()

                segments.append(speech_seg)
                is_silent.append(abs(speech_seg).sum().item() == 0.0)

            # list to seve chunk-wise outputs
            enh_waves = [None] * num_segments

            # ---------------------------
            # (2) separation with mini-batches
            # ---------------------------
            valid_indices = [i for i, s in enumerate(is_silent) if not s]

            if len(valid_indices) > 0:
                css_bs = self.css_batch_size

                for mb_st in range(0, len(valid_indices), css_bs):
                    mb_indices = valid_indices[mb_st : mb_st + css_bs]

                    # (S_mb, ..., T)
                    seg_batch = torch.cat([segments[i] for i in mb_indices], dim=0)

                    processed_batch = self(seg_batch, **kwargs)  # (n_batch, n_src, n_chan, n_samples)
                    processed_batch = processed_batch[..., :T]

                    # store results
                    for k, seg_i in enumerate(mb_indices):
                        enh_waves[seg_i] = processed_batch[[k]]

            # fill silent segments
            for i in range(num_segments):
                if enh_waves[i] is None:
                    enh_waves[i] = torch.zeros_like(enh_waves[valid_indices[0]])

            assert all(
                (w is not None) and isinstance(w, torch.Tensor) for w in enh_waves
            ), "enh_waves contains None or non-torch.Tensor elements"

            # ---------------------------
            # (3) switching
            # ---------------------------
            waves = enh_waves[0]

            for i in range(1, num_segments):
                if i == num_segments - 1:
                    enh_waves[i][..., t:] = 0
                    enh_waves_res_i = enh_waves[i][..., overlap_length:t]
                else:
                    enh_waves_res_i = enh_waves[i][..., overlap_length:]

                # overlap-and-add (average over the overlapped part)
                if overlap_length > 0:
                    assert waves[..., -overlap_length:].shape == enh_waves[i][..., :overlap_length].shape

                    waves[..., -overlap_length:] = (
                        waves[..., -overlap_length:] + enh_waves[i][..., :overlap_length]
                    ) / 2
                # concatenate the residual parts of the later segment
                waves = torch.cat([waves, enh_waves_res_i], dim=-1)
            # ensure the stitched length is same as input
            assert waves.size(-1) == speech_mix.size(-1), (waves.shape, speech_mix.shape)

        else:
            # normal forward enhance for short audio
            waves = self(speech_mix, **kwargs)

        return waves
