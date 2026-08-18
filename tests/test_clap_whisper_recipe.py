from pathlib import Path
import sys

from omegaconf import OmegaConf

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LOCAL_AIACCEL = _REPO_ROOT / "aiaccel"
if str(_LOCAL_AIACCEL) not in sys.path:
    sys.path.insert(0, str(_LOCAL_AIACCEL))

from aiaccel.config import load_config, resolve_inherit  # noqa: E402


def test_clap_whisper_recipe_requires_both_warm_start_checkpoints() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "recipes/dnr/models"
        / "tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.speech-music-residual-sfx."
        "robust-distill.rt192k.fp512keep475.broadcast-v1.clap-whisper-ft"
        / "config.yaml"
    )
    context = {
        "config_path": str(config_path),
        "working_directory": str(config_path.parent),
        "base_config_path": str(repo_root / "aiaccel/aiaccel/torch/apps/config"),
    }
    unresolved = load_config(config_path, context)

    assert OmegaConf.is_missing(unresolved, "fine_tune_checkpoint_path")
    assert OmegaConf.is_missing(unresolved, "perceptual_teacher_checkpoint_path")

    config = resolve_inherit(
        OmegaConf.merge(
            unresolved,
            {
                "fine_tune_checkpoint_path": "/models/student.ckpt",
                "perceptual_teacher_checkpoint_path": "/models/teacher.ckpt",
            },
        )
    )
    assert config.task.pretrained_model_path == "/models/student.ckpt"
    assert config.task.teacher_checkpoint_path == "/models/teacher.ckpt"
    assert config.task.clap_semantic_loss.prompt_config_path == str(config_path.parent / "clap_prompts.yaml")
    assert config.task.clap_semantic_loss.amodel == "HTSAT-tiny"
    assert config.task.perceptual_loss_compensate_cadence is True


def test_clap_a2a_prompt_bank_recipe_enables_scene_independent_objectives() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = (
        repo_root
        / "recipes/dnr/models"
        / "tvconv-pyramid-sourceaware-sfclite-convgru-smoothup-smoothlogit-npu.speech-music-residual-sfx."
        "robust-distill.rt192k.fp512keep475.broadcast-v1.clap-a2a-bank-whisper-ft"
        / "config.yaml"
    )
    context = {
        "config_path": str(config_path),
        "working_directory": str(config_path.parent),
        "base_config_path": str(repo_root / "aiaccel/aiaccel/torch/apps/config"),
    }
    unresolved = load_config(config_path, context)

    assert OmegaConf.is_missing(unresolved, "fine_tune_checkpoint_path")
    assert OmegaConf.is_missing(unresolved, "perceptual_teacher_checkpoint_path")
    assert OmegaConf.is_missing(unresolved, "clap_a2a_checkpoint_path")
    config = resolve_inherit(
        OmegaConf.merge(
            unresolved,
            {
                "fine_tune_checkpoint_path": "/models/student.ckpt",
                "perceptual_teacher_checkpoint_path": "/models/teacher.ckpt",
                "clap_a2a_checkpoint_path": "/models/clap.pt",
            },
        )
    )

    clap = config.task.clap_semantic_loss
    assert clap.prompt_config_path == str(config_path.parent / "clap_prompts.yaml")
    assert clap.audio_match_weight == 0.05
    assert clap.audio_antibleed_weight == 0.05
    assert clap.audio_antibleed_margin == 0.02
    assert clap.prompt_bank_weight == 0.02
    assert clap.prompt_bank_temperature == 0.07
    assert clap.positive_weight == 0.0
    assert clap.negative_weight == 0.0
    assert clap.amodel == "HTSAT-base"
    assert clap.checkpoint_path == "/models/clap.pt"
    assert clap.allow_download is False
    assert config.task.clap_semantic_loss_weight == 1.0
    assert config.task.whisper_feature_loss_weight == 0.10
    assert config.task.perceptual_loss_compensate_cadence is True
    prompt_config = yaml.safe_load((config_path.parent / "clap_prompts.yaml").read_text(encoding="utf-8"))
    assert set(prompt_config["prompt_banks"]) == {"speech", "music", "effects"}
    assert all(prompt_config["prompt_banks"][source] for source in ("speech", "music", "effects"))
