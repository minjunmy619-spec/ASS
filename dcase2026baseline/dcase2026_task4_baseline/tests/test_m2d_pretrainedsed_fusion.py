import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.m2dat.m2d_sc import (
    _MultiBranchPretrainedSEDFusion,
    _checkpoint_name_for_pretrainedsed,
    _canonical_pretrainedsed_model_name,
    _drop_pretrainedsed_prediction_heads,
    _load_pretrainedsed_feature_checkpoint,
    _remap_pretrainedsed_state_dict,
    M2DPretrainedSEDFusionClassifier,
    M2DSingleClassifierStrong,
)


def test_pretrainedsed_model_aliases_follow_official_release_branches():
    assert _canonical_pretrainedsed_model_name("beats") == "BEATs"
    assert _canonical_pretrainedsed_model_name("ATST") == "ATST-F"
    assert _canonical_pretrainedsed_model_name("AST") == "fpasst"
    assert _checkpoint_name_for_pretrainedsed("AST", "strong_1") == "fpasst_strong_1"


def test_beats_key_remap_matches_prediction_wrapper():
    raw = {
        "model.model.encoder.weight": torch.randn(8, 8),
        "strong_head.weight": torch.randn(447, 768),
        "weak_head.weight": torch.randn(447, 768),
    }

    remapped = _remap_pretrainedsed_state_dict("BEATs_strong_1", raw)
    feature_only = _drop_pretrainedsed_prediction_heads(remapped)

    assert "model.beats.encoder.weight" in feature_only
    assert "model.model.encoder.weight" not in feature_only
    assert "strong_head.weight" not in feature_only
    assert "weak_head.weight" not in feature_only


def test_fpasst_key_remap_matches_prediction_wrapper():
    raw = {"model.net.weight": torch.randn(8, 8)}
    remapped = _remap_pretrainedsed_state_dict("fpasst_strong_1", raw)

    assert "model.fpasst.net.weight" in remapped
    assert "model.net.weight" not in remapped


class ToyPretrainedSEDWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.beats = torch.nn.Module()
        self.model.beats.encoder = torch.nn.Linear(4, 3, bias=False)


def test_feature_checkpoint_loader_accepts_official_beats_key_pattern(tmp_path):
    model = ToyPretrainedSEDWrapper()
    checkpoint = {
        "model.model.encoder.weight": torch.randn(3, 4),
        "strong_head.weight": torch.randn(447, 768),
        "strong_head.bias": torch.randn(447),
        "weak_head.weight": torch.randn(447, 768),
        "weak_head.bias": torch.randn(447),
    }
    checkpoint_path = tmp_path / "BEATs_strong_1.pt"
    torch.save(checkpoint, checkpoint_path)

    missing, unexpected = _load_pretrainedsed_feature_checkpoint(model, "BEATs_strong_1", checkpoint_path)

    assert missing == []
    assert unexpected == []


def test_multibranch_weighted_fusion_keeps_single_embedding_contract():
    fusion = _MultiBranchPretrainedSEDFusion(
        branch_dims={"m2d": 512, "BEATs": 768, "ATST-F": 768, "fpasst": 768},
        output_dim=512,
        fusion_strategy="weighted_avg",
    )
    branches = {
        "m2d": torch.randn(2, 512),
        "BEATs": torch.randn(2, 768),
        "ATST-F": torch.randn(2, 768),
        "fpasst": torch.randn(2, 768),
    }

    projected = fusion.project(branches)
    embedding, weights = fusion.fuse(projected)

    assert embedding.shape == (2, 512)
    assert weights.shape == (4,)
    assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)


def test_pretrainedsed_classifier_defines_own_predict_method():
    assert "predict" in M2DPretrainedSEDFusionClassifier.__dict__
    assert M2DPretrainedSEDFusionClassifier.predict is not M2DSingleClassifierStrong.predict
