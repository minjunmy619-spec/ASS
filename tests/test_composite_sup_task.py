from functools import partial
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch

from aiaccel.config import load_config, resolve_inherit
from aiaccel.torch.lightning import OptimizerConfig

from spectral_feature_compression.core.loss.composite_separation import CompositeSeparationSpectralLoss
from spectral_feature_compression.core.tasks.composite_sup_task import CompositeSupTask


def _build_task() -> CompositeSupTask:
    return CompositeSupTask(
        model=torch.nn.Identity(),
        loss=torch.nn.L1Loss(),
        n_fft=32,
        hop_length=8,
        optimizer_config=OptimizerConfig(
            optimizer_generator=partial(torch.optim.AdamW, lr=1.0e-4),
        ),
        complex_ri_weight=0.5,
        log_magnitude_weight=0.2,
        multi_resolution_stft_weight=0.3,
        multi_resolution_stft_resolutions=((16, 4), (32, 8)),
        transient_weight=0.1,
        normalize_active_sources_for_aux_loss=True,
        aux_activity_threshold_db=-60.0,
        normalize_mixture_consistency=True,
    )


def test_composite_auxiliary_loss_is_source_scale_invariant_and_skips_inactive() -> None:
    torch.manual_seed(0)
    task = _build_task()
    ref = torch.zeros(1, 3, 1, 128)
    ref[:, 0] = torch.randn(1, 1, 128)
    ref[:, 1] = 0.1 * torch.randn(1, 1, 128)
    wav = ref.sum(dim=1)
    est = (ref + 0.05 * torch.randn_like(ref)).requires_grad_()

    aux_est, aux_ref = task._prepare_auxiliary_sources(wav, est, ref)
    scaled_est, scaled_ref = task._prepare_auxiliary_sources(7.0 * wav, 7.0 * est, 7.0 * ref)
    loss, _ = task.composite_loss(aux_est, aux_ref)
    scaled_loss, _ = task.composite_loss(scaled_est, scaled_ref)

    assert aux_est.shape[0] == 2
    torch.testing.assert_close(loss, scaled_loss, rtol=1.0e-5, atol=1.0e-5)
    loss.backward()
    assert est.grad is not None
    assert torch.isfinite(est.grad).all()


def test_normalized_mixture_consistency_is_scale_invariant() -> None:
    torch.manual_seed(1)
    task = _build_task()
    wav = torch.randn(2, 1, 128)
    est = torch.randn(2, 3, 1, 128, requires_grad=True)

    loss = task._mixture_consistency_l1(wav, est)
    scaled_loss = task._mixture_consistency_l1(5.0 * wav, 5.0 * est)

    torch.testing.assert_close(loss, scaled_loss, rtol=1.0e-6, atol=1.0e-6)
    loss.backward()
    assert est.grad is not None
    assert torch.isfinite(est.grad).all()


def test_speech_leakage_tf_loss_is_reference_anchored_and_has_targeted_gradients() -> None:
    samples = 1024
    timeline = torch.arange(samples, dtype=torch.float32) / samples
    speech = 0.5 * torch.sin(2.0 * torch.pi * 16.0 * timeline)
    reference = torch.zeros(1, 3, 1, samples)
    reference[:, 0, 0] = speech
    clean_estimate = reference.clone().requires_grad_(True)
    loss_module = CompositeSeparationSpectralLoss(
        n_fft=128,
        hop_length=32,
        source_order=("speech", "music", "effects"),
        speech_leakage_weight=1.0,
        speech_leakage_n_fft=128,
        speech_leakage_hop_length=32,
        speech_leakage_speech_active_db=-40.0,
        speech_leakage_target_relative_db=12.0,
    )

    clean_loss, clean_components = loss_module(clean_estimate, reference)

    torch.testing.assert_close(clean_loss, torch.zeros_like(clean_loss))
    torch.testing.assert_close(clean_components["speech_leakage_tf_music"], torch.zeros_like(clean_loss))
    torch.testing.assert_close(clean_components["speech_leakage_tf_effects"], torch.zeros_like(clean_loss))

    leaked_estimate = reference.clone()
    leaked_estimate[:, 1, 0] += 0.1 * speech
    leaked_estimate.requires_grad_(True)
    leaked_loss, leaked_components = loss_module(leaked_estimate, reference)
    leaked_loss.backward()

    assert leaked_components["speech_leakage_tf_music"] > 0.0
    torch.testing.assert_close(leaked_components["speech_leakage_tf_effects"], torch.zeros_like(leaked_loss))
    assert leaked_loss > clean_loss
    assert leaked_estimate.grad is not None
    assert float(leaked_estimate.grad[:, 1].abs().sum()) > 0.0
    torch.testing.assert_close(
        leaked_estimate.grad[:, 0],
        torch.zeros_like(leaked_estimate.grad[:, 0]),
        rtol=0.0,
        atol=1.0e-7,
    )


def test_repaired_sfc_teacher_recipe_resolves_and_instantiates() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "recipes"
        / "dnr"
        / "models"
        / "locoformer-medium.enc-crossattn96dim.dec-crossattn96dim.musical64.learnable-query"
        / "config.yaml"
    )
    unresolved_config = load_config(
        config_path,
        {
            "config_path": str(config_path),
            "working_directory": str(config_path.parent),
            "base_config_path": str(repo_root / "aiaccel/aiaccel/torch/apps/config"),
        },
    )
    assert OmegaConf.is_missing(unresolved_config.task, "pretrained_model_path")
    config = resolve_inherit(
        OmegaConf.merge(unresolved_config, {"task": {"pretrained_model_path": None}})
    )

    for stale_key in ("train_dataset_path", "val_dataset_path", "return_ref", "use_dm_dataset"):
        assert stale_key not in config.datamodule
    assert config.trainer.use_distributed_sampler is True
    assert config.trainer.seed == config.seed == 2026
    assert "build_seeded_trainer" in config.trainer._target_
    datamodule = instantiate(config.datamodule)
    task = instantiate(config.task)

    assert type(datamodule).__name__ == "OnTheFlyStemDataModule"
    assert isinstance(task, CompositeSupTask)
    assert task.normalize_active_sources_for_aux_loss
    assert task.normalize_mixture_consistency
