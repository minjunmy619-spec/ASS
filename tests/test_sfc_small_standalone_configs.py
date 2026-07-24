from __future__ import annotations

from pathlib import Path

from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf
import pytest
import yaml

from tools.online.export_onnx_online_model import REPO_ROOT

from aiaccel.config import load_config, resolve_inherit


RECIPE_NAMES = (
    "sfc-small-sameband-dw-bn-npu.musical64.onfly.rt192k",
    "sfc-small-pyramid-dw-bn-npu.musical64.onfly.rt192k",
    "sfc-small-macaron-conv2d-bn-npu.musical36.2l.onfly.rt192k",
    "sfc-small-macaron-conv2d-cln-npu.musical36.2l.onfly.rt192k",
    "sfc-small-macaron-conv2d-cln-lite-npu.musical36.2l.onfly.rt192k",
    "sfc-small-macaron-lrattn-bn-npu.musical36.2l.r2d64g560.onfly.rt192k",
)
RUNTIME_KEYS = {"config_path", "working_directory", "base_config_path"}


def _recipe_path(name: str) -> Path:
    return REPO_ROOT / "recipes" / "dnr" / "models" / name / "config.yaml"


@pytest.mark.parametrize("name", RECIPE_NAMES)
def test_sfc_small_recipe_is_standalone_and_minimal(name: str) -> None:
    raw_text = _recipe_path(name).read_text()
    raw_config = yaml.safe_load(raw_text)

    assert "_base_" not in raw_text
    assert "_inherit_" not in raw_text
    assert set(raw_config) == {"trainer", "datamodule", "task"}
    assert not any(key.startswith("sfc_npu_") for key in raw_config)
    assert raw_config["trainer"]["_target_"] == "lightning.Trainer"
    assert raw_config["datamodule"]["_target_"].endswith(".OnTheFlyStemDataModule")
    assert raw_config["task"]["_target_"].endswith(".SupTask")
    assert len(raw_config["datamodule"]["synthesis"]["synthesis_profiles"]) == 4


@pytest.mark.parametrize("name", RECIPE_NAMES)
def test_sfc_small_recipe_resolves_and_instantiates(name: str) -> None:
    path = _recipe_path(name)
    config = resolve_inherit(
        load_config(
            path,
            {
                "config_path": str(path),
                "working_directory": str(path.parent),
                "base_config_path": str(REPO_ROOT / "aiaccel" / "aiaccel" / "torch" / "apps" / "config"),
            },
        )
    )
    resolved = OmegaConf.to_container(config, resolve=True)

    assert set(resolved) == {"trainer", "datamodule", "task"} | RUNTIME_KEYS
    get_class(config.trainer._target_)
    model = instantiate(config.task.model)
    datamodule = instantiate(config.datamodule)

    assert model.model.core.n_src == 3
    assert model.model.core.n_chan == 1
    assert type(datamodule).__name__ == "OnTheFlyStemDataModule"
