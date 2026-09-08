import array
import io
import json
import unittest
from unittest.mock import patch

import torch
from torch import nn

from npu_quant.quantization import (
    FakeQuantizer, QuantConfig, QuantizedLayer, calibrate_simulation,
    quantization_report,
)


class FakeQuantizerTests(unittest.TestCase):
    def test_validation_and_lifecycle(self):
        for bits in (1, 17, True, 2.5):
            with self.assertRaises(ValueError):
                FakeQuantizer(bits)
            with self.assertRaises(ValueError):
                QuantConfig(activation_bits=bits)
            with self.assertRaises(ValueError):
                QuantConfig(overrides={"x": (bits, 8)})
        for bits in (2, 16):
            self.assertEqual(FakeQuantizer(bits).bits, bits)
        q = FakeQuantizer()
        with self.assertRaises(RuntimeError):
            q.freeze()
        q.observe(torch.ones(2))
        with self.assertRaises(RuntimeError):
            q(torch.ones(2))
        q.freeze()
        with self.assertRaises(RuntimeError):
            q.observe(torch.ones(2))
        self.assertEqual(q.qmin, -127)
        self.assertEqual(q.qmax, 127)

    def test_multi_batch_and_constants(self):
        q = FakeQuantizer(symmetric=False)
        q.observe(torch.tensor([-3., 2.]))
        q.observe(torch.tensor([-1., 7.]))
        q.freeze()
        self.assertAlmostEqual(q.scale.item(), 10 / 255)
        self.assertEqual(q.min_val.item(), -3)
        self.assertEqual(q.max_val.item(), 7)
        for constant in (-5., 0., 5.):
            for symmetric in (True, False):
                q = FakeQuantizer(symmetric=symmetric)
                x = torch.full((4,), constant)
                q.observe(x)
                q.freeze()
                self.assertGreater(q.scale.item(), 0)
                torch.testing.assert_close(q(x), x)
                if not symmetric:
                    self.assertEqual(q.min_val.item(), min(0, constant))
                    self.assertEqual(q.max_val.item(), max(0, constant))

    def test_negative_axis_and_zero_channel(self):
        q = FakeQuantizer(channel_axis=-1)
        x = torch.tensor([[0., -2., 4.], [0., 1., -8.]])
        q.observe(x)
        q.freeze()
        torch.testing.assert_close(q.scale, torch.tensor([1., 2 / 127, 8 / 127]))
        self.assertEqual(q(x).shape, x.shape)
        self.assertTrue(torch.isfinite(q(x)).all())
        with self.assertRaises(ValueError):
            q(torch.ones(2, 4))

    def test_reject_invalid_tensors(self):
        for x in (torch.empty(0), torch.tensor([float("nan")]),
                  torch.tensor([float("inf")]), torch.tensor([-float("inf")])):
            q = FakeQuantizer()
            with self.assertRaises(ValueError):
                q.observe(x)
            q.observe(torch.ones(1))
            q.freeze()
            with self.assertRaises(ValueError):
                q(x)
        with self.assertRaises(ValueError):
            FakeQuantizer(channel_axis=-3).observe(torch.ones(2, 3))

    def test_serialization_resizes_buffers(self):
        q = FakeQuantizer(channel_axis=-1)
        q.observe(torch.randn(4, 3))
        q.freeze()
        stream = io.BytesIO()
        torch.save(q.state_dict(), stream)
        stream.seek(0)
        restored = FakeQuantizer(channel_axis=-1)
        restored.load_state_dict(torch.load(stream, weights_only=True))
        x = torch.randn(2, 3)
        torch.testing.assert_close(restored(x), q(x))
        self.assertEqual(restored.scale.dtype, torch.float32)


class SimulationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)

    def test_root_copy_accumulation_toggle_and_serialization(self):
        model = nn.Linear(2, 2)
        original = {k: v.clone() for k, v in model.state_dict().items()}
        batches = [(torch.tensor([[-8., 1.]]),), (torch.tensor([[2., 9.]]),)]
        q = calibrate_simulation(model, batches)
        self.assertIsInstance(q, QuantizedLayer)
        self.assertIsNot(q.layer, model)
        self.assertNotEqual(q.layer.weight.data_ptr(), model.weight.data_ptr())
        self.assertTrue(q.training)
        self.assertTrue(model.training)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, original[name])
        self.assertEqual(q.input_quantizer.min_val.item(), -8)
        self.assertEqual(q.input_quantizer.max_val.item(), 9)
        outputs = torch.cat([model(*args) for args in batches])
        self.assertAlmostEqual(q.output_quantizer.min_val.item(), min(0, outputs.min().item()))
        self.assertAlmostEqual(q.output_quantizer.max_val.item(), max(0, outputs.max().item()))
        self.assertFalse(q._forward_hooks)
        x = torch.randn(3, 2)
        before = q.input_quantizer.min_val.clone()
        q(x)
        torch.testing.assert_close(before, q.input_quantizer.min_val)
        q.enabled = False
        torch.testing.assert_close(q(x), model(x))
        q.enabled = True
        restored = QuantizedLayer(nn.Linear(2, 2))
        restored.load_state_dict(q.state_dict())
        torch.testing.assert_close(restored(x), q(x))
        report = quantization_report(q)
        json.dumps(report, allow_nan=False)
        self.assertEqual(report[""]["weight"]["bits"], 8)
        self.assertEqual(len(report[""]["weight"]["scales"]), 2)

    def test_exclusions_overrides_and_unsupported(self):
        class CustomLinear(nn.Linear):
            pass

        model = nn.Sequential(nn.Linear(3, 3), nn.ReLU(),
                              nn.Sequential(nn.Linear(3, 3)), CustomLinear(3, 3))
        config = QuantConfig(exclude=("2",), overrides={"0": (4, 6)})
        q = calibrate_simulation(model, [(torch.randn(2, 3),)], config)
        self.assertIsInstance(q[0], QuantizedLayer)
        self.assertIs(type(q[1]), nn.ReLU)
        self.assertIs(type(q[2][0]), nn.Linear)
        self.assertIs(type(q[3]), CustomLinear)
        self.assertEqual(q[0].weight_quantizer.bits, 4)
        self.assertEqual(q[0].input_quantizer.bits, 6)
        self.assertEqual(q[0].output_quantizer.bits, 6)
        self.assertEqual(set(quantization_report(q)), {"0"})

    def test_convolutions_preserve_padding_and_bias(self):
        for conv, shape in ((nn.Conv1d, (2, 2, 7)), (nn.Conv2d, (2, 2, 7, 7)),
                            (nn.Conv3d, (2, 2, 5, 5, 5))):
            model = conv(2, 2, 3, padding=1, padding_mode="reflect", groups=2)
            x = torch.randn(shape)
            q = calibrate_simulation(model, [(x,)])
            expected = torch.func.functional_call(
                model, {"weight": q.weight_quantizer(model.weight)},
                (q.input_quantizer(x),))
            torch.testing.assert_close(q(x), q.output_quantizer(expected))
            torch.testing.assert_close(q.layer.bias, model.bias)

    def test_float_representative_calibration_and_positional_args(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(2, 2)
                self.b = nn.Linear(2, 2)

            def forward(self, x, y):
                return self.b(input=self.a(x) + y)

        model = Model()
        batches = [(torch.randn(3, 2) * magnitude, torch.randn(3, 2)) for magnitude in (1, 12)]
        q = calibrate_simulation(model, iter(batches), QuantConfig(weight_bits=2, activation_bits=2))
        values = torch.cat([model.a(x) + y for x, y in batches])
        self.assertAlmostEqual(q.b.input_quantizer.min_val.item(), min(0, values.min().item()))
        self.assertAlmostEqual(q.b.input_quantizer.max_val.item(), max(0, values.max().item()))

    def test_weight_encodings_preserve_scales_and_rounding(self):
        for model, name, shape in ((nn.Linear(2, 2), "", (3, 2)),
                                   (nn.Sequential(nn.Conv2d(2, 2, 1)), "0", (1, 2, 3, 3))):
            with self.subTest(name=name):
                scale = torch.tensor([0.03125, 0.09375])
                encoding = {"bits": 4, "qmin": -7, "qmax": 7,
                            "channel_axis": 0, "scales": scale}
                config = QuantConfig(overrides={name: (4, 6)})
                batches = [(torch.randn(shape),), (torch.randn(shape) * 3,)]
                q = calibrate_simulation(model, batches, config, weight_encodings={name: encoding})
                wrapper = q if name == "" else q[0]
                quantizer = wrapper.weight_quantizer
                self.assertTrue(torch.equal(quantizer.scale, scale))
                self.assertNotEqual(quantizer.scale.data_ptr(), scale.data_ptr())
                self.assertTrue(torch.equal(quantizer.zero_point, torch.zeros(2)))
                broadcast = scale.reshape(2, *([1] * (wrapper.layer.weight.ndim - 1)))
                expected = (wrapper.layer.weight / broadcast).round().clamp(-7, 7) * broadcast
                torch.testing.assert_close(quantizer(wrapper.layer.weight), expected, rtol=0, atol=0)
                report = quantization_report(q)[name]["weight"]
                self.assertEqual(report["scales"], scale.tolist())
                self.assertEqual(report["channel_axis"], 0)
                restored = QuantizedLayer(nn.Linear(2, 2) if name == "" else nn.Conv2d(2, 2, 1), 4, 6)
                restored.load_state_dict(wrapper.state_dict())
                self.assertTrue(torch.equal(restored.weight_quantizer.scale, scale))
                again = calibrate_simulation(model, batches, config, weight_encodings={name: report})
                self.assertEqual(quantization_report(again)[name]["weight"], report)
                scale.fill_(1)
                self.assertEqual(quantizer.scale[0].item(), 0.03125)

    def test_invalid_weight_encodings(self):
        encoding = {"bits": 8, "qmin": -127, "qmax": 127,
                    "channel_axis": 0, "scales": [0.1, 0.2]}
        invalid = [{**encoding, field: value} for field, value in (
            ("bits", 4), ("bits", 8.0), ("qmin", -128), ("qmax", 255),
            ("channel_axis", 1), ("channel_axis", False),
            ("scales", [0.1]), ("scales", [[0.1], [0.2]]), ("scales", 0.1),
            ("scales", [0., 0.1]), ("scales", [-1., 0.1]),
            ("scales", [float("nan"), 1.]), ("scales", [float("inf"), 1.]),
            ("scales", [1e-100, 1.]), ("scales", [1e100, 1.]),
            ("zero_points", [1, 0]), ("zero_points", [0]),
        )]
        invalid += [{k: v for k, v in encoding.items() if k != field} for field in encoding]
        invalid += [None, []]
        for entry in invalid:
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                calibrate_simulation(nn.Linear(2, 2), [], weight_encodings={"": entry})
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        for name, config, message in (("typo", None, "unknown"), ("1", None, "unsupported"),
                                      ("0", QuantConfig(exclude=("",)), "excluded")):
            with self.assertRaisesRegex(ValueError, message):
                calibrate_simulation(model, [], config, weight_encodings={name: encoding})
        with self.assertRaisesRegex(ValueError, "weight_encodings must"):
            calibrate_simulation(model, [], weight_encodings=[])

    def test_unknown_config_names_and_valid_root(self):
        for config in (QuantConfig(exclude=("typo",)), QuantConfig(overrides={"typo": (4, 4)})):
            with patch("npu_quant.quantization.deepcopy") as copy:
                with self.assertRaisesRegex(ValueError, "unknown module name"):
                    calibrate_simulation(nn.Linear(2, 2), [], config)
                copy.assert_not_called()
        q = calibrate_simulation(nn.Linear(2, 2), [(torch.ones(1, 2),)], QuantConfig(exclude=("",)))
        self.assertIs(type(q), nn.Linear)

    def test_storage_aliases_rejected_before_copy(self):
        for kind in ("parameter", "buffer", "buffer_pair"):
            model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            if kind == "parameter":
                model[1].bias = nn.Parameter(model[0].weight.detach().flatten()[1:3])
            elif kind == "buffer":
                model.register_buffer("alias", model[0].weight.detach().flatten()[1:])
            else:
                storage = torch.ones(6)
                model.register_buffer("first", storage[:2])
                model.register_buffer("second", storage[3:])
            with self.subTest(kind=kind), patch("npu_quant.quantization.deepcopy") as copy:
                with self.assertRaisesRegex(ValueError, "storage aliases"):
                    calibrate_simulation(model, [])
                copy.assert_not_called()

    def test_separate_storage_intervals(self):
        for kind in ("parameter", "buffer", "buffer_pair"):
            for offset in (4, 16):
                with self.subTest(kind=kind, offset=offset):
                    backing = array.array("f", range(1, 9))
                    first = torch.frombuffer(backing, dtype=torch.float32, count=4).reshape(2, 2)
                    second = torch.frombuffer(backing, dtype=torch.float32,
                                              count=4, offset=offset).reshape(2, 2)
                    self.assertNotEqual(first.untyped_storage().data_ptr(),
                                        second.untyped_storage().data_ptr())
                    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
                    if kind == "buffer_pair":
                        model.register_buffer("first", first)
                    else:
                        model[0].weight = nn.Parameter(first)
                    if kind == "parameter":
                        model[1].weight = nn.Parameter(second)
                    else:
                        model.register_buffer("second", second)
                    batches = [(torch.ones(1, 2),)]
                    if offset == 4:
                        with patch("npu_quant.quantization.deepcopy") as copy:
                            with self.assertRaisesRegex(ValueError, "storage aliases"):
                                calibrate_simulation(model, batches)
                            copy.assert_not_called()
                    else:
                        q = calibrate_simulation(model, batches)
                        self.assertIsInstance(q[0], QuantizedLayer)

    def test_reject_already_quantized_root_and_nested(self):
        wrapped = QuantizedLayer(nn.Linear(2, 2))
        for model in (wrapped, nn.Sequential(nn.ReLU(), wrapped)):
            with patch("npu_quant.quantization.deepcopy") as copy:
                with self.assertRaisesRegex(ValueError, "double quantization"):
                    calibrate_simulation(model, [], QuantConfig(exclude=("",)))
                copy.assert_not_called()

    def test_bad_data_and_aliases(self):
        for data, error in (([], ValueError), ([torch.ones(1, 2)], TypeError),
                            ([(torch.tensor([[float("nan"), 0.]]),)], ValueError)):
            with self.assertRaises(error):
                calibrate_simulation(nn.Linear(2, 2), data)
        layer = nn.Linear(2, 2)
        with self.assertRaisesRegex(ValueError, "shared module aliases"):
            calibrate_simulation(nn.Sequential(layer, layer), [(torch.ones(1, 2),)])
        other = nn.Linear(2, 2)
        other.weight = layer.weight
        with self.assertRaisesRegex(ValueError, "shared parameter aliases"):
            calibrate_simulation(nn.Sequential(layer, other), [(torch.ones(1, 2),)])


if __name__ == "__main__":
    unittest.main()
