from __future__ import annotations

import gc
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

import torch

from aiaccel.config import load_config, resolve_inherit
import pytest

from spectral_feature_compression.core.model.crossattn_enc_dec import CrossAttnDecoder
from spectral_feature_compression.core.model.proposed_separation_models import (
    build_source_aware_sfc_locoformer_teacher_system,
)
from spectral_feature_compression.core.model.source_aware_sfc_locoformer_teacher import (
    SourceAwareSFCLocoformerTeacher,
)
from spectral_feature_compression.core.tasks.composite_sup_task import CompositeSupTask


def _build_tiny_system(*, checkpointing: bool = False):
    return build_source_aware_sfc_locoformer_teacher_system(
        n_fft=64,
        hop_length=16,
        fs=8000,
        n_src=3,
        n_chan=1,
        n_bands=8,
        d_inner=16,
        d_model=32,
        encoder_heads=4,
        n_shared_layers=1,
        n_source_layers=1,
        n_heads=4,
        source_attention_heads=4,
        attention_dim=32,
        ffn_hidden_dim=(48, 48),
        num_groups=4,
        dropout=0.0,
        flash_attention=False,
        decoder_feature_channels=4,
        checkpointing=checkpointing,
        scaling=False,
        css_segment_size=0.04,
        css_shift_size=0.02,
    )


def test_source_aware_teacher_stft_invariants_and_backward() -> None:
    torch.manual_seed(0)
    core = _build_tiny_system(checkpointing=True).model.train()
    mixture = torch.randn(2, 1, 33, 9, dtype=torch.complex64)

    estimates, aux = core(mixture, return_aux=True)

    assert estimates.shape == (2, 3, 1, 33, 9)
    assert not torch.allclose(estimates[:, 0], estimates[:, 1])
    assert aux["bounded_mask"].amin() >= -2.0
    assert aux["bounded_mask"].amax() <= 2.0
    torch.testing.assert_close(
        aux["confidence_weights"].sum(dim=1),
        torch.ones_like(aux["confidence_weights"][:, 0]),
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    torch.testing.assert_close(estimates.sum(dim=1), mixture, rtol=1.0e-5, atol=1.0e-5)
    torch.testing.assert_close(core.residual_scale.detach(), torch.tensor(0.05), rtol=1.0e-6, atol=1.0e-6)

    estimates.abs().mean().backward()
    assert core.residual_scale_unconstrained.grad is not None
    assert torch.isfinite(core.residual_scale_unconstrained.grad)
    assert core.source_embeddings.grad is not None
    assert torch.isfinite(core.source_embeddings.grad).all()


def test_source_decoder_is_shared_and_default_parameter_budget_is_met() -> None:
    system = build_source_aware_sfc_locoformer_teacher_system(
        n_fft=2048,
        hop_length=512,
        fs=24000,
        flash_attention=False,
    )
    core = system.model
    parameter_count = sum(parameter.numel() for parameter in system.parameters() if parameter.requires_grad)

    assert isinstance(core, SourceAwareSFCLocoformerTeacher)
    assert sum(isinstance(module, CrossAttnDecoder) for module in core.modules()) == 1
    assert core.decoder.n_src == 1
    assert 20_000_000 <= parameter_count <= 35_000_000


def test_source_aware_teacher_waveform_and_css_lengths_are_finite() -> None:
    torch.manual_seed(1)
    system = _build_tiny_system().eval()
    short_wav = torch.randn(1, 1, 256)
    long_wav = torch.randn(1, 1, 720)

    with torch.no_grad():
        short_estimate = system(short_wav)
        css_estimate = system.css(long_wav)

    assert short_estimate.shape == (1, 3, 1, short_wav.shape[-1])
    assert css_estimate.shape == (1, 3, 1, long_wav.shape[-1])
    assert torch.isfinite(short_estimate).all()
    assert torch.isfinite(css_estimate).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for bf16 mixed-precision coverage")
def test_source_aware_teacher_bf16_forward_backward() -> None:
    torch.manual_seed(2)
    core = _build_tiny_system(checkpointing=True).model.cuda().train()
    mixture = torch.randn(1, 1, 33, 9, dtype=torch.complex64, device="cuda")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        estimates = core(mixture)
        loss = estimates.abs().mean()
    loss.backward()

    assert torch.isfinite(estimates).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in core.parameters())


def _load_recipe(relative_path: str):
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / relative_path
    return resolve_inherit(
        load_config(
            config_path,
            {
                "config_path": str(config_path),
                "working_directory": str(config_path.parent),
                "base_config_path": str(repo_root / "aiaccel/aiaccel/torch/apps/config"),
            },
        )
    )


def test_model_first_recipes_resolve_and_instantiate() -> None:
    source_config = _load_recipe("recipes/dnr/models/source-aware-sfc-locoformer.teacher/config.yaml")
    capacity_config = _load_recipe("recipes/dnr/models/sfc-locoformer-capacity-control.teacher/config.yaml")
    baseline_config = _load_recipe("recipes/dnr/models/sfc-locoformer-current-control.teacher/config.yaml")

    for config in (source_config, capacity_config, baseline_config):
        assert "train_dataset_path" not in config.datamodule
        assert config.datamodule.synthesis.normalize_sources is False
        assert config.task.normalize_active_sources_for_aux_loss is False
        assert config.trainer.max_steps == 20000
        assert config.trainer.use_distributed_sampler is True
        assert config.trainer.seed == config.seed == 2026
        assert "build_seeded_trainer" in config.trainer._target_

    assert source_config.task.mixture_consistency_weight == 0.0
    assert capacity_config.task.mixture_consistency_weight == 0.1
    assert baseline_config.task.mixture_consistency_weight == 0.1
    assert "source_aware_sfc_locoformer_teacher" in source_config.teacher_model._target_
    assert "build_sfc_locoformer_lite_plus_system" in capacity_config.teacher_model._target_
    assert "build_sfc_locoformer_lite_plus_system" in baseline_config.teacher_model._target_
    assert OmegaConf.to_container(source_config.datamodule, resolve=True) == OmegaConf.to_container(
        capacity_config.datamodule, resolve=True
    )
    assert OmegaConf.to_container(source_config.datamodule, resolve=True) == OmegaConf.to_container(
        baseline_config.datamodule, resolve=True
    )
    assert OmegaConf.to_container(source_config.task.optimizer_config, resolve=True) == OmegaConf.to_container(
        capacity_config.task.optimizer_config, resolve=True
    )
    assert OmegaConf.to_container(source_config.task.optimizer_config, resolve=True) == OmegaConf.to_container(
        baseline_config.task.optimizer_config, resolve=True
    )

    task = instantiate(source_config.task)
    assert isinstance(task, CompositeSupTask)
    assert isinstance(task.model.model, SourceAwareSFCLocoformerTeacher)
    del task
    gc.collect()

    capacity_model = instantiate(capacity_config.teacher_model)
    parameter_count = sum(parameter.numel() for parameter in capacity_model.parameters() if parameter.requires_grad)
    assert 20_000_000 <= parameter_count <= 35_000_000

    baseline_model = instantiate(baseline_config.teacher_model)
    baseline_parameter_count = sum(
        parameter.numel() for parameter in baseline_model.parameters() if parameter.requires_grad
    )
    assert 14_000_000 <= baseline_parameter_count <= 20_000_000
