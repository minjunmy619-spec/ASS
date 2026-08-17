from pathlib import Path
import sys

from omegaconf import OmegaConf

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
