import yaml
import torch

from src.evaluation.evaluate_stage import _load_checkpoint as _load_stage_checkpoint
from src.models.s5.kwo2025 import Kwon2025S5
from src.training.lightningmodule.online_teacher_tse import _load_model_checkpoint as _load_teacher_checkpoint
from src.training.lightningmodule.uss_sc_joint_model_parallel import USSCSJointModelParallelLightning


def _minimal_joint_module(**kwargs):
    args = dict(
        uss_model={"module": "torch.nn", "main": "Identity"},
        sc_model={"module": "torch.nn", "main": "Identity"},
        uss_loss={"module": "torch.nn", "main": "Identity"},
        sc_loss={"module": "torch.nn", "main": "Identity"},
        optimizer_uss={
            "module": "torch.optim",
            "main": "AdamW",
            "args": {"params": None, "lr": 1e-4},
        },
        optimizer_sc={
            "module": "torch.optim",
            "main": "AdamW",
            "args": {"params": None, "lr": 1e-4},
        },
        uss_device="cpu",
        sc_device="cpu",
    )
    args.update(kwargs)
    module = USSCSJointModelParallelLightning(**args)
    module.setup()
    return module


def test_joint_uss_target_uses_foreground_doa_without_feeding_oracle_input():
    module = _minimal_joint_module()
    foreground_doa = torch.randn(2, 3, 3)
    batch = {
        "mixture": torch.randn(2, 4, 256),
        "foreground_waveform": torch.randn(2, 3, 1, 256),
        "interference_waveform": torch.randn(2, 1, 1, 256),
        "noise_waveform": torch.randn(2, 1, 1, 256),
        "class_index": torch.zeros(2, 3, dtype=torch.long),
        "is_silence": torch.zeros(2, 3, dtype=torch.bool),
        "foreground_doa": foreground_doa,
    }

    input_dict = module._uss_input(batch)
    target_dict = module._uss_target(batch)

    assert "spatial_vector" not in input_dict
    assert torch.equal(target_dict["foreground_doa"], foreground_doa)
    assert torch.equal(target_dict["spatial_vector"], foreground_doa)


def test_joint_slot_targets_skip_uncertain_matches_unless_enabled():
    sep = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    batch = {
        "foreground_waveform": sep.clone(),
        "class_index": torch.tensor([[7]]),
        "is_silence": torch.tensor([[False]]),
    }

    module = _minimal_joint_module(
        clean_match_score=100.0,
        uncertain_weight=0.35,
        use_uncertain_matches=False,
    )
    class_idx, is_silence, sample_weight, _, quality_code = module._build_slot_targets(sep, batch)

    assert class_idx.item() == 0
    assert is_silence.item()
    assert sample_weight.item() == 0.0
    assert quality_code.item() == 2

    module = _minimal_joint_module(
        clean_match_score=100.0,
        uncertain_weight=0.35,
        use_uncertain_matches=True,
    )
    class_idx, is_silence, sample_weight, _, quality_code = module._build_slot_targets(sep, batch)

    assert class_idx.item() == 7
    assert not is_silence.item()
    assert sample_weight.item() == torch.tensor(0.35).item()
    assert quality_code.item() == 2


def test_universal_joint_config_uses_universal_uss_and_pretrainedsed_sc():
    with open("config/separation/modified_deft_uss_sc_joint_universal_pretrainedsed_fusion.yaml") as f:
        cfg = yaml.safe_load(f)

    args = cfg["lightning_module"]["args"]

    assert args["uss_model"]["main"] == "UnifiedModifiedDeFTUSS"
    assert args["sc_model"]["main"] == "M2DPretrainedSEDFusionClassifier"
    assert args["freeze_sc"] is False
    assert args["sc_update_every"] == 4
    assert args["use_uncertain_matches"] is False
    assert args["uss_pretrained_ckpt"] == "checkpoint/modified_deft_uss_lite_6s_unified_all_features.ckpt"
    assert args["sc_pretrained_ckpt"] == "checkpoint/m2d_sc_stage1_pretrainedsed_fusion.ckpt"
    assert cfg["train"]["trainer"]["args"]["devices"] == 1


def test_online_teacher_loader_can_extract_uss_and_sc_from_joint_checkpoint(tmp_path):
    uss = torch.nn.Linear(2, 2, bias=False)
    sc = torch.nn.Linear(2, 2, bias=False)
    uss_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sc_weight = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    ckpt_path = tmp_path / "joint.ckpt"
    torch.save(
        {
            "state_dict": {
                "uss_model.weight": uss_weight,
                "sc_model.weight": sc_weight,
            }
        },
        ckpt_path,
    )

    _load_teacher_checkpoint(uss, str(ckpt_path), strict=True, name="uss")
    _load_teacher_checkpoint(sc, str(ckpt_path), strict=True, name="sc")

    assert torch.equal(uss.weight, uss_weight)
    assert torch.equal(sc.weight, sc_weight)


def test_stage_loader_can_select_sc_prefix_from_joint_checkpoint(tmp_path):
    sc = torch.nn.Linear(2, 2, bias=False)
    uss_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sc_weight = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    ckpt_path = tmp_path / "joint.ckpt"
    torch.save(
        {
            "state_dict": {
                "uss_model.weight": uss_weight,
                "sc_model.weight": sc_weight,
            }
        },
        ckpt_path,
    )

    _load_stage_checkpoint(sc, str(ckpt_path), preferred_prefixes=("sc_model.",))

    assert torch.equal(sc.weight, sc_weight)


def test_s5_loader_can_select_uss_prefix_from_joint_checkpoint(tmp_path):
    uss = torch.nn.Linear(2, 2, bias=False)
    uss_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sc_weight = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    ckpt_path = tmp_path / "joint.ckpt"
    torch.save(
        {
            "state_dict": {
                "sc_model.weight": sc_weight,
                "uss_model.weight": uss_weight,
            }
        },
        ckpt_path,
    )

    loader = object.__new__(Kwon2025S5)
    Kwon2025S5._load_ckpt(loader, str(ckpt_path), uss, preferred_prefixes=("uss_model.",))

    assert torch.equal(uss.weight, uss_weight)
