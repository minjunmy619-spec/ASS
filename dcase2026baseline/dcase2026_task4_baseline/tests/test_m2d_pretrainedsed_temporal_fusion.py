import sys
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.m2dat.m2d_sc_temporal import (
    M2DPretrainedSEDTemporalFusionClassifier,
    M2DTemporalPretrainedSEDFusionClassifier,
    _TemporalConvRefiner,
)
from src.training.loss.temporal import temporal_activity_loss


def test_temporal_fusion_alias_points_to_temporal_classifier():
    assert M2DPretrainedSEDTemporalFusionClassifier is M2DTemporalPretrainedSEDFusionClassifier


def test_temporal_refiner_preserves_frame_embedding_shape():
    refiner = _TemporalConvRefiner(dim=32, num_layers=2, kernel_size=5, dropout=0.0)
    x = torch.randn(3, 17, 32)
    y = refiner(x)

    assert y.shape == x.shape


def test_temporal_classifier_exposes_training_and_inference_contracts():
    assert "forward" in M2DTemporalPretrainedSEDFusionClassifier.__dict__
    assert "predict" in M2DTemporalPretrainedSEDFusionClassifier.__dict__
    assert "_forward_pretrainedsed" in M2DTemporalPretrainedSEDFusionClassifier.__dict__
    assert "_pool_frames" in M2DTemporalPretrainedSEDFusionClassifier.__dict__


def test_temporal_predict_contract_from_mocked_forward():
    class ToyTemporalClassifier(M2DTemporalPretrainedSEDFusionClassifier):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.num_classes = 3
            self.energy_thresholds = {"min_probability": 0.0, "min_activity": 0.0}

        def forward(self, inputs):
            plain_logits = torch.tensor([[0.1, 2.0, -1.0], [3.0, 0.2, -0.5]])
            return {
                "logits": plain_logits,
                "plain_logits": plain_logits,
                "energy": torch.logsumexp(plain_logits, dim=-1),
                "embedding": torch.randn(2, 8),
                "activity_logits": torch.tensor([[4.0, -4.0, 4.0], [-4.0, -4.0, -4.0]]),
            }

    out = ToyTemporalClassifier().predict({"waveform": torch.randn(2, 16000)})

    assert out["label_vector"].shape == (2, 3)
    assert out["raw_label_vector"].shape == (2, 3)
    assert out["probabilities"].shape == (2,)
    assert out["activity_probabilities"].shape == (2, 3)
    assert torch.equal(out["class_index"], torch.tensor([1, 0]))


def test_pretrainedsed_temporal_branch_forces_fp32_before_external_frontend():
    class RecordingBranch(torch.nn.Module):
        output_dim = 4

        def __init__(self):
            super().__init__()
            self.seen_dtype = None

        def forward(self, waveform):
            self.seen_dtype = waveform.dtype
            return waveform.new_ones(waveform.shape[0], self.output_dim)

    class ToyFusion(torch.nn.Module):
        def project(self, branch_embeddings):
            return branch_embeddings

        def fuse(self, projected):
            return projected["BEATs"], None

    class ToyTemporalClassifier(M2DTemporalPretrainedSEDFusionClassifier):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.input_sample_rate = 16000
            self.pretrainedsed_sample_rate = 16000
            self.branch = RecordingBranch()
            self.pretrainedsed_branches = torch.nn.ModuleDict({"BEATs": self.branch})
            self.pretrainedsed_fusion = ToyFusion()

    model = ToyTemporalClassifier()
    waveform = torch.randn(2, 16000, dtype=torch.bfloat16)
    fused, branch_embeddings, _ = model._forward_pretrainedsed(waveform)

    assert model.branch.seen_dtype == torch.float32
    assert branch_embeddings["BEATs"].dtype == torch.float32
    assert fused.dtype == torch.float32


def test_temporal_activity_loss_supervises_silence_as_zero_activity():
    output = {
        "activity_logits": torch.tensor(
            [[4.0, 4.0, 4.0], [-4.0, 4.0, -4.0]],
            requires_grad=True,
        ),
        "duration_sec": torch.tensor([1.0, 1.0]),
    }
    target = {
        "span_sec": torch.tensor([[-1.0, -1.0], [0.25, 0.75]]),
    }

    loss = temporal_activity_loss(output, target, pos_weight=1.0)

    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    assert output["activity_logits"].grad is not None
    assert output["activity_logits"].grad[0].abs().sum() > 0.0
