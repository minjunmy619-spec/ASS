import torch

import pytest

from spectral_feature_compression.core.model.bandit_split import get_band_specs
from spectral_feature_compression.core.model.crossattn_enc_dec import CrossAttnDecoder, CrossAttnEncoder


def test_vocal64_24k_2048_expected_layout():
    band_specs, freq_weights, overlapping = get_band_specs("vocal64", 2048, 24000, n_bands=64)

    assert overlapping is True
    assert len(band_specs) == 64
    assert len(set(band_specs)) == 64
    assert band_specs[:12] == [
        (0, 5),
        (1, 7),
        (3, 8),
        (6, 12),
        (8, 15),
        (11, 17),
        (13, 20),
        (16, 23),
        (19, 25),
        (21, 27),
        (25, 33),
        (28, 37),
    ]
    assert band_specs[-4:] == [(682, 804), (733, 889), (819, 974), (904, 1025)]

    counter = torch.zeros(2048 // 2 + 1)
    for (start, end), weights in zip(band_specs, freq_weights):
        assert end > start
        assert tuple(weights.shape) == (end - start,)
        counter[start:end] += 1
    assert not torch.any(counter == 0)


def test_speech_vocal_alias_matches_vocal64():
    vocal, _, _ = get_band_specs("vocal64", 2048, 24000, n_bands=64)
    speech_vocal, _, _ = get_band_specs("speech_vocal64", 2048, 24000, n_bands=64)
    assert speech_vocal == vocal


@pytest.mark.parametrize("sample_rate", [16000, 44100, 48000])
def test_vocal64_rejects_unsupported_sample_rates(sample_rate):
    with pytest.raises(ValueError, match="fs=24000 only"):
        get_band_specs("vocal64", 2048, sample_rate, n_bands=64)


def test_crossattn_enc_dec_accept_vocal64():
    encoder = CrossAttnEncoder(
        d_inner=8,
        d_model=16,
        n_chan=1,
        sample_rate=24000,
        n_fft=2048,
        n_bands=64,
        band_config="vocal64",
        query_type="learnable",
        n_heads=2,
    )
    decoder = CrossAttnDecoder(
        d_inner=8,
        d_model=16,
        n_src=3,
        n_chan=1,
        sample_rate=24000,
        n_fft=2048,
        n_bands=64,
        band_config="speech_vocal64",
        query_type="learnable",
        n_heads=2,
    )
    assert len(encoder.band_indices) == 64
    assert len(decoder.band_indices) == 64
