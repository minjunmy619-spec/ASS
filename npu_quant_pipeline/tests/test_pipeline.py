import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from npu_quant import NPUQuantizer, QuantConfig, compare_models, quantization_report


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(37)
        self.model = nn.Sequential(nn.Conv1d(2, 4, 1), nn.BatchNorm1d(4),
                                   nn.ReLU(), nn.Conv1d(4, 2, 1)).eval()
        self.calibration = [(torch.randn(2, 2, 12),), (torch.randn(1, 2, 17),)]
        self.validation = [(torch.randn(1, 2, 19),)]

    def test_end_to_end_and_preserve_original(self):
        original = copy.deepcopy(self.model.state_dict())
        result = NPUQuantizer(self.model).optimize(
            iter(self.calibration), iter(self.validation),
            adaround_iterations=3, bias_correction=True)
        self.assertLessEqual(result.report["selected"]["nmse"], result.report["baseline"]["nmse"])
        self.assertEqual(result.report["calibration_batches"], 2)
        self.assertTrue(any(stage["stage"] == "adaround" for stage in result.report["stages"]))
        for name, tensor in self.model.state_dict().items():
            torch.testing.assert_close(tensor, original[name])
        self.assertFalse(result.model.training)
        self.assertFalse(result.simulation.training)

    def test_custom_gate_can_reject_all_changes(self):
        scores = iter([10., 5., 2.])
        result = NPUQuantizer(self.model).optimize(
            self.calibration, self.validation, adaround_iterations=1,
            score_fn=lambda model: next(scores))
        self.assertTrue(all(not stage["accepted"] for stage in result.report["stages"]))
        self.assertEqual(result.report["selected_score"], 10.)
        self.assertEqual(compare_models(self.model, result.model, self.validation)["mse"], 0)

    def test_smoothing_and_bits(self):
        model = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 3)).eval()
        config = QuantConfig(overrides={"1": (4, 8)})
        result = NPUQuantizer(model, config).optimize(
            [(torch.randn(2, 4),)], [(torch.randn(3, 4),)],
            smooth=True, adaround_iterations=2)
        self.assertEqual(result.simulation.get_submodule("1").weight_quantizer.bits, 4)
        rounding = next(s for s in result.report["stages"] if s["stage"] == "adaround")
        self.assertEqual(rounding["operations"][0]["bits"], 4)

    def test_dynamic_model_can_skip_graph(self):
        class Dynamic(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                return self.fc(x) if x.sum() > 0 else -self.fc(x)

        model = Dynamic()
        result = NPUQuantizer(model).optimize(
            [(torch.ones(1, 4),)], [(torch.ones(2, 4),)], fold_bn=False, equalize=False)
        self.assertEqual(result.report["stages"], [])
        self.assertTrue(model.training)

    def test_validation_errors(self):
        tool = NPUQuantizer(self.model)
        with self.assertRaises(ValueError):
            tool.optimize([], self.validation)
        with self.assertRaises(ValueError):
            tool.optimize(self.calibration, [])
        with self.assertRaises(ValueError):
            tool.optimize(self.calibration, self.validation, bias_correction=True)
        with self.assertRaises(ValueError):
            tool.optimize(self.calibration, self.validation, score_fn=lambda m: float("nan"))

    def test_root_keeps_quantization_coverage(self):
        result = NPUQuantizer(nn.Linear(4, 3), QuantConfig(weight_bits=2)).optimize(
            [(torch.randn(2, 4),)], [(torch.randn(3, 4),)])
        self.assertEqual(set(quantization_report(result.simulation)), {""})
        self.assertGreater(result.report["selected"]["nmse"], 0)

    def test_original_aliases_rejected_before_copy(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        model[1].weight = nn.Parameter(model[0].weight.detach())
        with self.assertRaisesRegex(ValueError, "alias"):
            NPUQuantizer(model)
        layer = nn.Linear(4, 4)
        layer.register_buffer("view", layer.weight.detach())
        with self.assertRaisesRegex(ValueError, "alias"):
            NPUQuantizer(layer)

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "optional ONNX unavailable")
    def test_accepted_rounding_export_preserves_weights_and_scales(self):
        import onnx
        from onnx import numpy_helper

        model = nn.Linear(4, 3).eval()
        example = (torch.randn(2, 4),)
        result = NPUQuantizer(model, QuantConfig(weight_bits=4)).optimize(
            [example], [(torch.randn(3, 4),)], adaround_iterations=2,
            score_fn=lambda m: 0.)
        encoding = result.report["weight_encodings"][""]
        self.assertEqual(encoding["bits"], 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result.export_onnx(example, path / "model.onnx")
            result.save_encodings(path / "encodings.json")
            graph = onnx.load(path / "model.onnx")
            weights = {v.name: torch.from_numpy(numpy_helper.to_array(v).copy())
                       for v in graph.graph.initializer}
            torch.testing.assert_close(weights["weight"], result.model.weight)
            with (path / "encodings.json").open() as stream:
                saved = json.load(stream)
            self.assertIn("npu_quant.encodings.v1", str(saved))
            torch.testing.assert_close(result.simulation.weight_quantizer.scale,
                                       encoding["scales"])

    @unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime"),
                         "optional ONNX Runtime unavailable")
    def test_optimized_onnx_runtime_dynamic_shapes(self):
        import onnxruntime

        result = NPUQuantizer(self.model).optimize(self.calibration, self.validation)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            result.export_onnx(
                self.calibration[0], path, input_names=["audio"], output_names=["stems"],
                dynamic_axes={"audio": {0: "batch", 2: "time"},
                              "stems": {0: "batch", 2: "time"}})
            session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            for batch, length in ((1, 23), (3, 31)):
                audio = torch.randn(batch, 2, length)
                actual = torch.from_numpy(session.run(None, {"audio": audio.numpy()})[0])
                with torch.no_grad():
                    torch.testing.assert_close(actual, result.model(audio), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
