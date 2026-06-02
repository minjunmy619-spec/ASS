from __future__ import annotations

import torch

from spectral_feature_compression.core.model.model_wrapper import ModelWrapper
from spectral_feature_compression.core.model.online_model_wrapper import CausalISTFTOLA, OnlineModelWrapper
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import build_online_soft_band_sfc_system


class _IdentityStftModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x


class _SingleSourceStrictStftModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1)


def test_model_wrapper_css_validation_ignores_reference_kwarg() -> None:
    wrapper = ModelWrapper(
        model=_SingleSourceStrictStftModel(),
        n_fft=64,
        hop_length=16,
        fs=64,
        scaling=False,
        css_segment_size=1,
        css_shift_size=1,
    ).eval()
    wav = torch.randn(1, 1, 96)
    ref = torch.randn(1, 1, 1, 96)

    with torch.no_grad():
        est = wrapper.css(wav, ref=ref)

    assert est.shape == ref.shape


def test_causal_istft_ola_reconstructs_center_false_stft() -> None:
    torch.manual_seed(0)
    n_fft = 2048
    hop_length = 512
    wav = torch.randn(2, 44100)

    wrapper = OnlineModelWrapper(
        model=_IdentityStftModel(),
        n_fft=n_fft,
        hop_length=hop_length,
        fs=44100,
        scaling=False,
    ).eval()

    with torch.no_grad():
        rec = wrapper(wav)

    assert rec.shape == wav.shape
    assert torch.allclose(rec, wav, atol=1e-5, rtol=1e-5)


def test_online_waveform_wrapper_forward_smoke_dnr_shape() -> None:
    torch.manual_seed(0)
    model = build_online_soft_band_sfc_system(
        n_fft=2048,
        hop_length=512,
        fs=44100,
        n_bands=64,
        band_config="musical",
        n_src=3,
        n_chan=1,
        d_model=24,
        n_layers=2,
        causal=True,
        masking=True,
        scaling=False,
    ).eval()
    wav = torch.randn(1, 1, 44100)

    with torch.no_grad():
        est = model(wav)

    assert est.shape == (1, 3, 1, 44100)
    assert torch.isfinite(est).all()
