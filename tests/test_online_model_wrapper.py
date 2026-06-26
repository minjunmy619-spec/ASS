from __future__ import annotations

import torch

from torchaudio.transforms import InverseSpectrogram, Spectrogram

import pytest

from spectral_feature_compression.core.model.model_wrapper import ModelWrapper
from spectral_feature_compression.core.model.online_model_wrapper import OnlineModelWrapper
from spectral_feature_compression.core.model.online_soft_band_sfc_2d import build_online_soft_band_sfc_system
from spectral_feature_compression.core.model.source_separation_postprocess import (
    MISIPhaseConsistency,
    SourceSeparationPostProcessor,
)


class _IdentityStftModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x


class _SingleSourceStrictStftModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1)


class _TwoSourceUnderMixStftModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack((0.25 * x, 0.25 * x), dim=1)


def test_source_separation_postprocessor_enforces_mixture_consistency() -> None:
    torch.manual_seed(0)
    mixture = torch.randn(2, 1, 8, 5, dtype=torch.complex64)
    estimates = torch.stack((0.2 * mixture, -0.1j * mixture), dim=1)

    processor = SourceSeparationPostProcessor(mixture_consistency="uniform")
    refined = processor(estimates, mixture)

    torch.testing.assert_close(refined.sum(dim=1), mixture, rtol=1e-5, atol=1e-5)


def test_source_separation_postprocessor_wiener_and_residual_source() -> None:
    torch.manual_seed(0)
    mixture = torch.randn(1, 1, 8, 6, dtype=torch.complex64)
    estimates = torch.stack((0.35 * mixture, 0.2 * mixture, 0.0 * mixture), dim=1)

    processor = SourceSeparationPostProcessor(
        mixture_consistency="power",
        final_mixture_consistency="power",
        power_smoothing=0.3,
        wiener_blend=0.5,
        residual_source_index=2,
    )
    refined = processor(estimates, mixture)

    torch.testing.assert_close(refined.sum(dim=1), mixture, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(refined[:, 2], mixture - refined[:, :2].sum(dim=1), rtol=1e-5, atol=1e-5)
    assert torch.isfinite(refined.real).all()
    assert torch.isfinite(refined.imag).all()


def test_source_separation_postprocessor_leakage_gate_is_conservative() -> None:
    mixture = torch.ones(1, 1, 2, 2, dtype=torch.complex64)
    dominant = torch.ones_like(mixture)
    leakage = torch.tensor([[[[0.1, 0.6], [0.05, 0.7]]]], dtype=torch.complex64)
    estimates = torch.stack((dominant, leakage), dim=1)

    processor = SourceSeparationPostProcessor(
        leakage_gate_enabled=True,
        leakage_gate_threshold_db=12.0,
        leakage_gate_attenuation_db=6.0,
    )
    refined = processor(estimates, mixture)

    gate_gain = 10.0 ** (-6.0 / 20.0)
    torch.testing.assert_close(refined[:, 0], dominant)
    torch.testing.assert_close(refined[:, 1, :, :, 0], leakage[:, :, :, 0] * gate_gain)
    torch.testing.assert_close(refined[:, 1, :, :, 1], leakage[:, :, :, 1])


def test_source_separation_postprocessor_leakage_gate_keeps_final_consistency() -> None:
    mixture = torch.ones(1, 1, 4, 3, dtype=torch.complex64)
    estimates = torch.stack((0.8 * mixture, 0.08 * mixture), dim=1)
    no_gate = SourceSeparationPostProcessor(final_mixture_consistency="power")(estimates, mixture)
    gated = SourceSeparationPostProcessor(
        leakage_gate_enabled=True,
        leakage_gate_threshold_db=6.0,
        leakage_gate_attenuation_db=12.0,
        final_mixture_consistency="power",
    )(estimates, mixture)

    torch.testing.assert_close(gated.sum(dim=1), mixture, rtol=1e-5, atol=1e-5)
    assert gated[:, 1].abs().mean() < no_gate[:, 1].abs().mean()


def test_source_separation_postprocessor_rejects_ambiguous_config_values() -> None:
    with pytest.raises(ValueError, match="power_beta must be finite"):
        SourceSeparationPostProcessor(power_beta=float("nan"))
    with pytest.raises(TypeError, match="residual_source_index must be an integer"):
        SourceSeparationPostProcessor(residual_source_index=1.5)


def test_model_wrapper_postprocessor_improves_remix_consistency() -> None:
    torch.manual_seed(0)
    wav = torch.randn(1, 1, 256)
    raw_wrapper = ModelWrapper(
        model=_TwoSourceUnderMixStftModel(),
        n_fft=64,
        hop_length=16,
        fs=64,
        scaling=False,
    ).eval()
    refined_wrapper = ModelWrapper(
        model=_TwoSourceUnderMixStftModel(),
        n_fft=64,
        hop_length=16,
        fs=64,
        scaling=False,
        postprocessor=SourceSeparationPostProcessor(final_mixture_consistency="uniform"),
    ).eval()

    with torch.no_grad():
        raw = raw_wrapper(wav).sum(dim=1)
        refined = refined_wrapper(wav).sum(dim=1)

    assert refined.shape == raw.shape == wav.shape
    assert torch.mean(torch.abs(refined - wav)) < torch.mean(torch.abs(raw - wav))


def test_misi_phase_consistency_projects_sources_back_to_mixture() -> None:
    torch.manual_seed(0)
    wav = torch.randn(1, 1, 256)
    stft = torch.nn.Sequential(Spectrogram(n_fft=64, hop_length=16, power=None))
    istft = InverseSpectrogram(n_fft=64, hop_length=16)
    mixture = stft(wav)
    estimates = torch.stack((0.25 * mixture, 0.25 * mixture), dim=1)
    projector = MISIPhaseConsistency(iterations=1)

    refined = projector(
        estimates,
        wav,
        stft=stft,
        istft=istft,
        length=wav.shape[-1],
        target_frames=mixture.shape[-1],
    )
    raw_wave = istft(estimates, wav.shape[-1]).sum(dim=1)
    refined_wave = istft(refined, wav.shape[-1]).sum(dim=1)

    assert refined.shape == estimates.shape
    assert torch.isfinite(refined.real).all()
    assert torch.isfinite(refined.imag).all()
    assert torch.mean(torch.abs(refined_wave - wav)) < torch.mean(torch.abs(raw_wave - wav))


def test_misi_phase_consistency_rejects_ambiguous_config_and_shape_values() -> None:
    wav = torch.randn(1, 1, 256)
    stft = torch.nn.Sequential(Spectrogram(n_fft=64, hop_length=16, power=None))
    istft = InverseSpectrogram(n_fft=64, hop_length=16)
    estimates = torch.stack((stft(wav), stft(wav)), dim=1)

    with pytest.raises(TypeError, match="iterations must be an integer"):
        MISIPhaseConsistency(iterations=1.5)
    with pytest.raises(ValueError, match="Expected target_frames"):
        MISIPhaseConsistency(iterations=1)(
            estimates,
            wav,
            stft=stft,
            istft=istft,
            length=wav.shape[-1],
            target_frames=estimates.shape[-1] - 1,
        )


def test_model_wrapper_phase_consistency_improves_remix_consistency() -> None:
    torch.manual_seed(0)
    wav = torch.randn(1, 1, 256)
    raw_wrapper = ModelWrapper(
        model=_TwoSourceUnderMixStftModel(),
        n_fft=64,
        hop_length=16,
        fs=64,
        scaling=False,
    ).eval()
    refined_wrapper = ModelWrapper(
        model=_TwoSourceUnderMixStftModel(),
        n_fft=64,
        hop_length=16,
        fs=64,
        scaling=False,
        phase_consistency=MISIPhaseConsistency(iterations=1),
    ).eval()

    with torch.no_grad():
        raw = raw_wrapper(wav).sum(dim=1)
        refined = refined_wrapper(wav).sum(dim=1)

    assert refined.shape == raw.shape == wav.shape
    assert torch.mean(torch.abs(refined - wav)) < torch.mean(torch.abs(raw - wav))


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
