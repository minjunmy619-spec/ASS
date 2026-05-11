import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.m2dat.m2d_sc_temporal import (
    M2DPretrainedSEDTemporalFusionClassifier,
    M2DTemporalPretrainedSEDFusionClassifier,
    _TemporalConvRefiner,
)


def test_temporal_fusion_alias_points_to_temporal_classifier():
    assert M2DPretrainedSEDTemporalFusionClassifier is M2DTemporalPretrainedSEDFusionClassifier


def test_temporal_refiner_preserves_frame_embedding_shape():
    refiner = _TemporalConvRefiner(dim=32, num_layers=2, kernel_size=5, dropout=0.0)
    x = torch.randn(3, 17, 32)
    y = refiner(x)

    assert y.shape == x.shape


def test_temporal_classifier_exposes_temporal_forward_contract():
    assert "forward" in M2DTemporalPretrainedSEDFusionClassifier.__dict__
    assert "_forward_pretrainedsed" in M2DTemporalPretrainedSEDFusionClassifier.__dict__
    assert "_pool_frames" in M2DTemporalPretrainedSEDFusionClassifier.__dict__
