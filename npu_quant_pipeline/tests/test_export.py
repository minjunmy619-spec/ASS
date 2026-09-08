import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from npu_quant.export import export_onnx, save_encodings
from npu_quant.graph import optimize_graph
from npu_quant.quantization import (
    QuantConfig, QuantizedLayer, calibrate_simulation, quantization_report,
)


class ExportTests(unittest.TestCase):
    def test_missing_dependency_is_upfront(self):
        with patch.dict(sys.modules, {"onnx": None}), \
                patch("npu_quant.export.deepcopy") as clone, \
                patch("torch.onnx.export") as exporter:
            with self.assertRaisesRegex(ImportError, "pip install onnx"):
                export_onnx(nn.Linear(2, 2), (torch.ones(1, 2),), "unused.onnx",
                            validate=False)
            clone.assert_not_called()
            exporter.assert_not_called()

    def test_rejects_root_nested_and_disabled_simulations(self):
        wrapper = QuantizedLayer(nn.Linear(2, 2))
        for enabled in (True, False):
            wrapper.enabled = enabled
            for model in (wrapper, nn.Sequential(nn.ReLU(), wrapper)):
                with self.subTest(enabled=enabled, root=model is wrapper), \
                        patch.dict(sys.modules, {"onnx": Mock()}), \
                        patch("npu_quant.export.deepcopy") as clone:
                    with self.assertRaisesRegex(ValueError, "float models only.*QuantizedLayer"):
                        export_onnx(model, (torch.ones(1, 2),), "unused.onnx")
                    clone.assert_not_called()

    def test_requires_tuple(self):
        with patch.dict(sys.modules, {"onnx": Mock()}):
            with self.assertRaisesRegex(TypeError, "tuple"):
                export_onnx(nn.Identity(), torch.ones(2), "unused.onnx")

    def test_copy_options_validation_and_failure(self):
        model = nn.Sequential(nn.Linear(3, 2), nn.ReLU())
        model[1].eval()
        modes = [m.training for m in model.modules()]
        before = copy.deepcopy(model.state_dict())
        args = (torch.randn(2, 3),)
        axes = {"x": {0: "batch"}, "y": {0: "batch"}}

        def export(copied, inputs, path, **options):
            self.assertIsNot(copied, model)
            self.assertFalse(any(m.training for m in copied.modules()))
            self.assertIs(inputs, args)
            self.assertEqual(options, dict(input_names=["x"], output_names=["y"],
                                          dynamic_axes=axes, opset_version=16, dynamo=False))
            self.assertFalse(torch.is_grad_enabled())
            copied[0].weight.zero_()
            if failure:
                raise RuntimeError("export failed")

        for validate, failure in ((True, False), (False, False), (True, True)):
            onnx = Mock()
            with patch.dict(sys.modules, {"onnx": onnx}), \
                    patch("torch.onnx.export", side_effect=export):
                options = dict(input_names=["x"], output_names=["y"],
                               dynamic_axes=axes, opset_version=16, validate=validate)
                if failure:
                    with self.assertRaisesRegex(RuntimeError, "export failed"):
                        export_onnx(model, args, Path("unused.onnx"), **options)
                else:
                    export_onnx(model, args, Path("unused.onnx"), **options)
            if validate and not failure:
                onnx.checker.check_model.assert_called_once_with("unused.onnx")
            else:
                onnx.checker.check_model.assert_not_called()
            self.assertEqual([m.training for m in model.modules()], modes)
            torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "optional onnx not installed")
    def test_real_float_export(self):
        import onnx

        x = torch.randn(2, 3)
        sequential = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
        optimized, _ = optimize_graph(copy.deepcopy(sequential).eval(), (x,))
        for model in (nn.Linear(3, 2), nn.ReLU(), sequential, optimized):
            model.train()
            if isinstance(model, nn.Sequential):
                model[1].eval()
            before = copy.deepcopy(model.state_dict())
            modes = [m.training for m in model.modules()]
            with self.subTest(model=type(model).__name__), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "model.onnx"
                with patch("onnx.checker.check_model", wraps=onnx.checker.check_model) as check:
                    export_onnx(model, (x,), path, input_names=["x"], output_names=["y"],
                                dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}})
                    check.assert_called_once_with(str(path))
                graph = onnx.load(str(path))
                self.assertEqual(graph.opset_import[0].version, 17)
                for value in (graph.graph.input[0], graph.graph.output[0]):
                    self.assertEqual(value.type.tensor_type.shape.dim[0].dim_param, "batch")
                    self.assertEqual(value.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
                self.assertFalse({n.op_type for n in graph.graph.node} &
                                 {"QuantizeLinear", "DequantizeLinear", "QLinearMatMul", "MatMulInteger"})
                if importlib.util.find_spec("onnxruntime"):
                    import onnxruntime

                    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                    for batch in (1, 5):
                        sample = torch.randn(batch, 3)
                        actual = session.run(None, {"x": sample.numpy()})[0]
                        torch.testing.assert_close(torch.from_numpy(actual), model(sample))
            self.assertEqual([m.training for m in model.modules()], modes)
            torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


class EncodingTests(unittest.TestCase):
    def test_report_config_and_recursive_metadata(self):
        config = QuantConfig(exclude=("2",), overrides={"0": (8, 6)})
        model = calibrate_simulation(
            nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 2)),
            [(torch.randn(4, 3),)], config,
        )
        before = copy.deepcopy(model.state_dict())
        metadata = {"config": config, "nested": [{"tensor": torch.tensor([1., 2.], requires_grad=True)},
                                                 (torch.tensor(3),)]}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"onnx": None}):
            path = Path(tmp) / "encodings.json"
            save_encodings(model, path, metadata=metadata)
            with path.open(encoding="utf-8") as stream:
                document = json.load(stream)
        self.assertEqual(document["format"], "npu_quant.encodings.v1")
        self.assertEqual(document["encodings"], quantization_report(model))
        self.assertEqual(set(document["encodings"]), {"0"})
        weight = document["encodings"]["0"]["weight"]
        self.assertEqual((weight["qmin"], weight["qmax"], weight["channel_axis"]), (-127, 127, 0))
        self.assertIsNone(document["encodings"]["0"]["input"]["channel_axis"])
        self.assertEqual(document["metadata"]["config"]["overrides"], {"0": [8, 6]})
        self.assertEqual(document["metadata"]["config"]["exclude"], ["2"])
        self.assertEqual(document["metadata"]["nested"], [{"tensor": [1., 2.]}, [3]])
        for disclaimer in ("Module names", "not an ONNX mapping", "vendor config", "[-127, 127]",
                           "not guaranteed by any backend"):
            self.assertIn(disclaimer, document["scope"])
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)

    def test_root_and_float_reports(self):
        root = calibrate_simulation(nn.Linear(2, 2), [(torch.ones(1, 2),)])
        for model, names in ((root, {""}), (nn.Linear(2, 2), set())):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "encodings.json"
                save_encodings(model, path)
                with path.open(encoding="utf-8") as stream:
                    document = json.load(stream)
                self.assertEqual(set(document["encodings"]), names)
                self.assertIsNone(document["metadata"])

    def test_invalid_json_does_not_truncate_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encodings.json"
            model = nn.Linear(2, 2)
            save_encodings(model, path)
            before = path.read_bytes()
            for value in (float("nan"), float("inf"), -float("inf"),
                          torch.tensor([float("nan")]), object(), {1, 2}):
                with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                    save_encodings(model, path, metadata={"nested": [value]})
                self.assertEqual(path.read_bytes(), before)
            with patch("npu_quant.export.quantization_report", return_value={"bad": float("nan")}):
                with self.assertRaises(ValueError):
                    save_encodings(model, path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
