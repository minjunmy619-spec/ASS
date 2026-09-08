import copy
import math
import sys
import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from npu_quant.evaluation import bss_metrics, compare_models, layer_sensitivity, waveform_metrics
from npu_quant.quantization import QuantConfig, QuantizedLayer, calibrate_simulation


class Function(nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, *args):
        return self.function(*args)


class EvaluationTests(unittest.TestCase):
    def test_element_weighting_nested_outputs_and_double(self):
        reference = Function(lambda x: {"a": [x], "b": (x * 2, None)})
        candidate = Function(lambda x: {"b": (x, None), "a": [x * 0]})
        result = compare_models(reference, candidate,
                                ((x,) for x in (torch.ones(1), torch.full((3,), 3.))))
        self.assertEqual(result["mse"], 7)
        self.assertAlmostEqual(result["nmse"], 0.4)
        self.assertEqual(result["max_abs_error"], 3)
        self.assertAlmostEqual(result["sqnr_db"], 10 * math.log10(2.5))
        x = torch.tensor([1e20])
        self.assertTrue(math.isfinite(compare_models(nn.Identity(), Function(lambda x: x * 0), [(x,)])["mse"]))

    def test_recursive_input_clones_inference_and_modes(self):
        def forward(x, nested):
            self.assertTrue(torch.is_inference_mode_enabled())
            x.add_(2)
            nested[0]["v"][0].mul_(3)
            return x + nested[0]["v"][0]

        reference, candidate = Function(forward), Function(forward)
        reference.child = nn.Dropout().eval()
        candidate.eval()
        modes = [[m.training for m in model.modules()] for model in (reference, candidate)]
        x, y = torch.ones(2), torch.ones(2)
        result = compare_models(reference, candidate, [(x, [{"v": (y,)}])])
        self.assertEqual(result["mse"], 0)
        self.assertEqual(result["sqnr_db"], 120)
        torch.testing.assert_close(x, torch.ones(2))
        torch.testing.assert_close(y, torch.ones(2))
        self.assertEqual(modes, [[m.training for m in model.modules()] for model in (reference, candidate)])

    def test_invalid_outputs_and_exception_mode_restore(self):
        x = torch.ones(2)
        for output in ([x], torch.ones(1, 2), torch.tensor([float("nan"), 0]),
                       torch.tensor([float("inf"), 0]), torch.empty(0)):
            with self.subTest(output=output), self.assertRaises(ValueError):
                compare_models(nn.Identity(), Function(lambda x: output), [(x,)])
        for left, right in (([x], (x,)), ({"a": x}, {"b": x}),
                            ([x], [x, x]), (None, None), ([], [])):
            with self.assertRaises(ValueError):
                compare_models(Function(lambda x: left), Function(lambda x: right), [(x,)])
        with self.assertRaises(ValueError):
            compare_models(nn.Identity(), nn.Identity(), [])
        with self.assertRaises(TypeError):
            compare_models(nn.Identity(), nn.Identity(), [x])
        reference = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout()).train()
        reference[1].eval()
        state = copy.deepcopy(reference.state_dict())
        modes = [m.training for m in reference.modules()]

        def fail(x):
            raise RuntimeError("forward failed")

        candidate = Function(fail).train()
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            compare_models(reference, candidate, [(torch.ones(3, 2),)])
        self.assertEqual(modes, [m.training for m in reference.modules()])
        self.assertTrue(candidate.training)
        torch.testing.assert_close(reference.state_dict(), state)

    def test_zero_reference_and_perfect_are_finite(self):
        for candidate in (nn.Identity(), Function(lambda x: x + 1)):
            result = compare_models(nn.Identity(), candidate, [(torch.zeros(2),)])
            self.assertTrue(all(math.isfinite(v) for v in result.values()))

    def test_sensitivity_ranking_generator_and_restore(self):
        torch.manual_seed(3)
        reference = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1)).train()
        data = [(torch.randn(4, 2),), (torch.randn(1, 2),)]
        simulation = calibrate_simulation(reference, data, QuantConfig(weight_bits=2, activation_bits=2))
        # Make the second layer's output clipping dominate the error.
        simulation[1].output_quantizer.scale.fill_(1e-4)
        state = copy.deepcopy(simulation.state_dict())
        modes = [m.training for m in simulation.modules()]
        baseline = compare_models(reference, simulation, data)["nmse"]
        rows = layer_sensitivity(reference, simulation, (args for args in data))
        self.assertEqual(rows[0]["layer"], "1")
        self.assertGreater(rows[0]["improvement"], 0)
        for row in rows:
            layer = simulation.get_submodule(row["layer"])
            layer.enabled = False
            expected = compare_models(reference, simulation, data)["nmse"]
            layer.enabled = True
            self.assertAlmostEqual(row["nmse"], expected)
            self.assertAlmostEqual(row["improvement"], baseline - expected)
        torch.testing.assert_close(simulation.state_dict(), state)
        self.assertEqual(modes, [m.training for m in simulation.modules()])
        simulation[0].enabled = False
        rows = layer_sensitivity(reference, simulation, iter(data))
        self.assertEqual(next(r for r in rows if r["layer"] == "0")["improvement"], 0)
        self.assertFalse(simulation[0].enabled)
        self.assertTrue(simulation[1].enabled)

        def fail_on_bypass(module, args):
            if not module.enabled:
                raise RuntimeError("bypass failure")

        handle = simulation[1].register_forward_pre_hook(fail_on_bypass)
        try:
            with self.assertRaisesRegex(RuntimeError, "bypass failure"):
                layer_sensitivity(reference, simulation, iter(data))
        finally:
            handle.remove()
        self.assertFalse(simulation[0].enabled)
        self.assertTrue(simulation[1].enabled)
        self.assertEqual(modes, [m.training for m in simulation.modules()])

    def test_bypass_skips_all_quantizers(self):
        layer = QuantizedLayer(nn.Linear(2, 1))
        layer.enabled = False
        with patch.object(layer.weight_quantizer, "forward", side_effect=AssertionError), \
                patch.object(layer.input_quantizer, "forward", side_effect=AssertionError), \
                patch.object(layer.output_quantizer, "forward", side_effect=AssertionError):
            rows = layer_sensitivity(layer.layer, layer, [(torch.ones(1, 2),)])
        self.assertEqual(rows, [{"layer": "", "nmse": 0., "improvement": 0.}])
        self.assertFalse(layer.enabled)

    def test_waveform_gain_and_per_signal_metrics(self):
        reference = torch.tensor([[1., -1., 1., -1.], [2., -2., 2., -2.]])
        estimate = reference * torch.tensor([[0.1], [1.]])
        result = waveform_metrics(reference, estimate)
        self.assertEqual(set(result), {"snr_db", "si_sdr_db", "gain_db", "dc_error"})
        self.assertTrue(all(v.shape == (2,) for v in result.values()))
        self.assertAlmostEqual(result["gain_db"][0].item(), -20, places=5)
        self.assertGreater(result["si_sdr_db"][0].item(), 50)
        self.assertLess(result["snr_db"][0].item(), 1)
        self.assertEqual(result["gain_db"][1].item(), 0)
        shifted = waveform_metrics(reference, reference + 0.25)
        torch.testing.assert_close(shifted["dc_error"], torch.full((2,), 0.25, dtype=torch.float64))
        self.assertEqual(waveform_metrics(reference[0], estimate[0])["gain_db"].shape, ())
        self.assertTrue(all(torch.isfinite(v).all() for v in waveform_metrics(reference, reference * 0).values()))

    def test_waveform_validation(self):
        for left, right in ((torch.zeros(2, 4), torch.ones(2, 4)),
                            (torch.tensor([[1., 1.], [0., 0.]]), torch.ones(2, 2)),
                            (torch.ones(2), torch.ones(1, 2)),
                            (torch.empty(0), torch.empty(0)),
                            (torch.tensor(1.), torch.tensor(1.)),
                            (torch.ones(2).cfloat(), torch.ones(2)),
                            (torch.ones(2), torch.tensor([float("nan"), 1.]))):
            with self.assertRaises(ValueError):
                waveform_metrics(left, right)
        for eps in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                waveform_metrics(torch.ones(2), torch.ones(2), eps=eps)

    def test_bss_optional_missing_and_fixed_order(self):
        x = torch.ones(2, 8, requires_grad=True)
        with patch.dict(sys.modules, {"mir_eval": None, "mir_eval.separation": None}):
            with self.assertRaisesRegex(ImportError, "mir_eval and numpy"):
                bss_metrics(x, x)
        separation = types.ModuleType("mir_eval.separation")

        def backend(left, right, *, compute_permutation):
            self.assertFalse(compute_permutation)
            self.assertEqual(left.shape, (2, 8))
            self.assertEqual(right.shape, (2, 8))
            return torch.tensor([1., 2.]), torch.tensor([3., 4.]), torch.tensor([5., 6.]), None

        separation.bss_eval_sources = backend

        def numpy_stub(tensor):
            self.assertFalse(tensor.requires_grad)
            self.assertEqual(tensor.device.type, "cpu")
            return types.SimpleNamespace(shape=tuple(tensor.shape))

        # Exercise the optional adapter without making NumPy a test dependency.
        with patch.dict(sys.modules, {"mir_eval.separation": separation}), \
                patch.object(torch.Tensor, "numpy", numpy_stub):
            self.assertEqual(bss_metrics(x, x), {"SDR": [1., 2.], "SIR": [3., 4.], "SAR": [5., 6.]})
        for left, right in ((x[0], x[0]), (x, torch.zeros_like(x)), (torch.zeros_like(x), x)):
            with self.assertRaises(ValueError):
                bss_metrics(left, right)

    def test_bss_conversion_failure_is_not_reported_as_missing_dependency(self):
        x = torch.ones(2, 8)
        separation = types.ModuleType("mir_eval.separation")
        separation.bss_eval_sources = lambda *a, **k: self.fail("backend must not run")

        def broken_numpy(tensor):
            raise RuntimeError("Numpy is not available")

        with patch.dict(sys.modules, {"mir_eval.separation": separation}), \
                patch.object(torch.Tensor, "numpy", broken_numpy):
            with self.assertRaises(RuntimeError) as caught:
                bss_metrics(x, x)
        self.assertNotIsInstance(caught.exception, ImportError)
        self.assertIn("Numpy is not available", str(caught.exception))

    def test_non_module_inputs_rejected(self):
        for reference, candidate in ((nn.Identity(), object()), (object(), nn.Identity())):
            with self.assertRaises(TypeError):
                compare_models(reference, candidate, [(torch.ones(2),)])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_automatic_device_movement(self):
        reference = nn.Linear(2, 2)
        candidate = copy.deepcopy(reference).cuda()
        result = compare_models(reference, candidate, [(torch.ones(3, 2),)])
        self.assertLess(result["mse"], 1e-12)
        candidate.register_buffer("wrong_device", torch.ones(1))
        with self.assertRaisesRegex(ValueError, "single device"):
            compare_models(reference, candidate, [(torch.ones(3, 2),)])


if __name__ == "__main__":
    unittest.main()
