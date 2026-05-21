import yaml
import pytest
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


def test_joint_slot_targets_can_keep_bad_matches_as_low_weight_silence():
    sep = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
    batch = {
        "foreground_waveform": sep.clone(),
        "class_index": torch.tensor([[7]]),
        "is_silence": torch.tensor([[False]]),
    }

    module = _minimal_joint_module(
        min_match_score=100.0,
        min_energy_db=-80.0,
        bad_match_silence_weight=0.05,
    )
    class_idx, is_silence, sample_weight, _, quality_code = module._build_slot_targets(sep, batch)

    assert class_idx.item() == 0
    assert is_silence.item()
    assert sample_weight.item() == pytest.approx(0.05)
    assert quality_code.item() == 3


def test_joint_clean_source_mix_replaces_training_rows_with_oracle_sources():
    module = _minimal_joint_module(clean_source_mix_prob=1.0, clean_source_mix_weight=1.0)
    sep = torch.zeros(1, 2, 1, 4)
    class_idx = torch.zeros(1, 2, dtype=torch.long)
    is_silence = torch.ones(1, 2, dtype=torch.bool)
    sample_weight = torch.zeros(1, 2)
    span_sec = torch.full((1, 2, 2), -1.0)
    ref_index = torch.tensor([[1, -1]])
    batch = {
        "foreground_waveform": torch.tensor([[[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]]]),
        "class_index": torch.tensor([[3, 7]]),
        "is_silence": torch.tensor([[False, False]]),
        "foreground_span_sec": torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]),
    }

    mixed, class_idx, is_silence, sample_weight, span_sec, clean_mask = module._maybe_mix_clean_sources(
        sep,
        batch,
        class_idx,
        is_silence,
        sample_weight,
        span_sec,
        ref_index,
        is_training=True,
    )

    assert torch.equal(mixed[0, 0], batch["foreground_waveform"][0, 1])
    assert torch.equal(mixed[0, 1], sep[0, 1])
    assert class_idx.tolist() == [[7, 0]]
    assert is_silence.tolist() == [[False, True]]
    assert sample_weight.tolist() == [[1.0, 0.0]]
    assert torch.equal(span_sec[0, 0], batch["foreground_span_sec"][0, 1])
    assert clean_mask.tolist() == [[True, False]]


def test_joint_clean_source_mix_disabled_for_validation():
    module = _minimal_joint_module(clean_source_mix_prob=1.0, clean_source_mix_weight=1.0)
    sep = torch.zeros(1, 1, 1, 4)
    class_idx = torch.zeros(1, 1, dtype=torch.long)
    is_silence = torch.ones(1, 1, dtype=torch.bool)
    sample_weight = torch.zeros(1, 1)
    ref_index = torch.tensor([[0]])
    batch = {
        "foreground_waveform": torch.ones(1, 1, 1, 4),
        "class_index": torch.tensor([[3]]),
        "is_silence": torch.tensor([[False]]),
    }

    mixed, class_idx, is_silence, sample_weight, span_sec, clean_mask = module._maybe_mix_clean_sources(
        sep,
        batch,
        class_idx,
        is_silence,
        sample_weight,
        None,
        ref_index,
        is_training=False,
    )

    assert torch.equal(mixed, sep)
    assert class_idx.item() == 0
    assert is_silence.item()
    assert sample_weight.item() == 0.0
    assert span_sec is None
    assert not clean_mask.any()


def test_joint_clean_silence_mix_replaces_training_rows_with_oracle_silence():
    module = _minimal_joint_module(clean_silence_mix_prob=1.0, clean_silence_mix_weight=1.0)
    sep = torch.tensor([
        [[[9.0, 9.0, 9.0, 9.0]], [[8.0, 8.0, 8.0, 8.0]], [[7.0, 7.0, 7.0, 7.0]]]
    ])
    class_idx = torch.tensor([[4, 0, 0]])
    is_silence = torch.tensor([[False, True, True]])
    sample_weight = torch.tensor([[1.0, 0.05, 0.0]])
    span_sec = torch.tensor([[[0.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]]])
    batch = {
        "foreground_waveform": torch.tensor([
            [[[1.0, 2.0, 3.0, 4.0]], [[0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]]]
        ]),
        "class_index": torch.tensor([[4, 0, 0]]),
        "is_silence": torch.tensor([[False, True, True]]),
        "foreground_span_sec": torch.tensor([[[0.0, 1.0], [-1.0, -1.0], [-1.0, -1.0]]]),
    }

    mixed, class_idx, is_silence, sample_weight, span_sec, clean_silence_mask = module._maybe_mix_clean_silence_sources(
        sep,
        batch,
        class_idx,
        is_silence,
        sample_weight,
        span_sec,
        is_training=True,
    )

    assert torch.equal(mixed[0, 0], sep[0, 0])
    assert torch.equal(mixed[0, 1], sep[0, 1])
    assert torch.equal(mixed[0, 2], batch["foreground_waveform"][0, 1])
    assert class_idx.tolist() == [[4, 0, 0]]
    assert is_silence.tolist() == [[False, True, True]]
    assert sample_weight.tolist()[0][0] == pytest.approx(1.0)
    assert sample_weight.tolist()[0][1] == pytest.approx(0.05)
    assert sample_weight.tolist()[0][2] == pytest.approx(1.0)
    assert torch.equal(span_sec[0, 2], batch["foreground_span_sec"][0, 1])
    assert clean_silence_mask.tolist() == [[False, False, True]]


def test_joint_clean_silence_mix_disabled_for_validation():
    module = _minimal_joint_module(clean_silence_mix_prob=1.0, clean_silence_mix_weight=1.0)
    sep = torch.ones(1, 1, 1, 4)
    class_idx = torch.zeros(1, 1, dtype=torch.long)
    is_silence = torch.ones(1, 1, dtype=torch.bool)
    sample_weight = torch.zeros(1, 1)
    batch = {
        "foreground_waveform": torch.zeros(1, 1, 1, 4),
        "class_index": torch.tensor([[0]]),
        "is_silence": torch.tensor([[True]]),
    }

    mixed, class_idx, is_silence, sample_weight, span_sec, clean_silence_mask = module._maybe_mix_clean_silence_sources(
        sep,
        batch,
        class_idx,
        is_silence,
        sample_weight,
        None,
        is_training=False,
    )

    assert torch.equal(mixed, sep)
    assert sample_weight.item() == 0.0
    assert span_sec is None
    assert not clean_silence_mask.any()


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


def test_online_teacher_loader_can_extract_tse_from_pipeline_checkpoint(tmp_path):
    tse = torch.nn.Linear(2, 2, bias=False)
    tse_weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    uss_weight = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    sc_weight = torch.tensor([[9.0, 10.0], [11.0, 12.0]])
    ckpt_path = tmp_path / "pipeline.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.weight": tse_weight,
                "uss_model.weight": uss_weight,
                "sc_model.weight": sc_weight,
            }
        },
        ckpt_path,
    )

    _load_teacher_checkpoint(tse, str(ckpt_path), strict=True, name="tse")

    assert torch.equal(tse.weight, tse_weight)


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
