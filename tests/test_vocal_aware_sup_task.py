from pathlib import Path
import sys

import torch

from torchaudio.transforms import Spectrogram

_LOCAL_AIACCEL = Path(__file__).resolve().parents[1] / "aiaccel"
if _LOCAL_AIACCEL.is_dir() and str(_LOCAL_AIACCEL) not in sys.path:
    sys.path.insert(0, str(_LOCAL_AIACCEL))

from spectral_feature_compression.core.tasks.vocal_aware_sup_task import VocalAwareCompositeSupTask  # noqa: E402


def _task(**kwargs):
    torch.nn.Module.__init__(task := VocalAwareCompositeSupTask.__new__(VocalAwareCompositeSupTask))
    task.speech_source_index = 0
    task.speech_robust_logmag_weight = kwargs.get("speech_robust_logmag_weight", 0.05)
    task.speech_robust_logmag_tau = kwargs.get("speech_robust_logmag_tau", 1.0)
    task.vocal_active_frame_weight = kwargs.get("vocal_active_frame_weight", 0.0)
    task.speech_temporal_logmag_gradient_weight = kwargs.get("speech_temporal_logmag_gradient_weight", 0.03)
    task.speech_frequency_logmag_gradient_weight = kwargs.get("speech_frequency_logmag_gradient_weight", 0.02)
    task.speech_gradient_tau = kwargs.get("speech_gradient_tau", 1.0)
    task.speech_inactive_leakage_weight = kwargs.get("speech_inactive_leakage_weight", 0.05)
    task.speech_inactive_threshold_db = kwargs.get("speech_inactive_threshold_db", -45.0)
    task.speech_inactive_softness_db = kwargs.get("speech_inactive_softness_db", 6.0)
    task.speech_robust_logmag_resolutions = kwargs.get("speech_robust_logmag_resolutions", ((256, 64),))
    task.stft = torch.nn.Sequential(Spectrogram(n_fft=256, hop_length=64, power=None))
    return task


def _dummy_band_model(value: float) -> torch.nn.Module:
    wrapper = torch.nn.Module()
    wrapper.model = torch.nn.Module()
    wrapper.model.shared = torch.nn.Parameter(torch.full((1,), value))
    for name in ("encoder", "decoder"):
        module = torch.nn.Module()
        module.block = torch.nn.Module()
        module.block.mixer = torch.nn.Module()
        module.block.mixer.pos_bias = torch.nn.Module()
        module.block.mixer.pos_bias.pos_bias = torch.nn.Parameter(torch.full((1,), value))
        setattr(wrapper.model, name, module)
    wrapper.model.encoder.query = torch.nn.Parameter(torch.full((1,), value))
    return wrapper


def test_vocal_warm_start_preserves_band_dependent_tensors(tmp_path: Path):
    checkpoint_model = _dummy_band_model(1.0)
    checkpoint_path = tmp_path / "musical.ckpt"
    torch.save(
        {"state_dict": {f"model.{key}": value for key, value in checkpoint_model.state_dict().items()}},
        checkpoint_path,
    )
    vocal_model = _dummy_band_model(2.0)

    VocalAwareCompositeSupTask(
        model=vocal_model,
        loss=torch.nn.L1Loss(),
        n_fft=256,
        hop_length=64,
        optimizer_config=object(),
        pretrained_model_path=str(checkpoint_path),
        preserve_initialized_band_layout_on_pretrained_load=True,
    )

    state = vocal_model.state_dict()
    assert state["model.shared"].item() == 1.0
    assert state["model.encoder.query"].item() == 2.0
    assert state["model.encoder.block.mixer.pos_bias.pos_bias"].item() == 2.0
    assert state["model.decoder.block.mixer.pos_bias.pos_bias"].item() == 2.0

    same_layout_model = _dummy_band_model(3.0)
    VocalAwareCompositeSupTask(
        model=same_layout_model,
        loss=torch.nn.L1Loss(),
        n_fft=256,
        hop_length=64,
        optimizer_config=object(),
        pretrained_model_path=str(checkpoint_path),
    )
    assert same_layout_model.state_dict()["model.encoder.query"].item() == 1.0


def test_soft_truncated_l1_saturates_large_errors():
    task = _task()
    err = torch.tensor([0.0, 1.0, 10.0])
    out = task._soft_truncated_l1(err, tau=1.0)
    assert torch.isclose(out[0], torch.tensor(0.0))
    assert out[1] < 1.0
    assert out[2] < 1.0
    assert out[2] > out[1]


def test_speech_robust_losses_are_near_zero_for_identical_signals():
    task = _task()
    wav = torch.randn(2, 1, 2048) * 0.01
    loss, components = task._speech_robust_logmag_losses(wav, wav.clone())
    assert loss.item() < 1.0e-6
    assert components["speech_robust_logmag"].item() < 1.0e-6


def test_vocal_activity_weight_has_expected_broadcast_shape():
    task = _task(vocal_active_frame_weight=2.0)
    ref_mag = torch.rand(2, 1, 129, 33)
    weight = task._vocal_activity_weight(ref_mag)
    assert weight.shape == (2, 1, 1, 33)


def test_speech_inactive_leakage_increases_with_estimated_noise():
    task = _task()
    ref = torch.zeros(1, 1, 2048)
    quiet_est = torch.zeros(1, 1, 2048)
    noisy_est = torch.randn(1, 1, 2048) * 0.1
    quiet_loss = task._speech_inactive_leakage_loss(quiet_est, ref)
    noisy_loss = task._speech_inactive_leakage_loss(noisy_est, ref)
    assert noisy_loss > quiet_loss
