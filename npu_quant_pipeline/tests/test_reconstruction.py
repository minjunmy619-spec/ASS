import copy
import array
import math
from types import MappingProxyType
import unittest

import torch
from torch import nn

from npu_quant import reconstruction
from npu_quant.reconstruction import _capture, _samples, adaround, bias_correct


class TwoInputs(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(3, 2)

    def forward(self, x, offset):
        return self.layer(x + offset)


class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def assert_unchanged(self, model, state, modes):
        torch.testing.assert_close(model.state_dict(), state, rtol=0, atol=0)
        self.assertEqual([m.training for m in model.modules()], modes)

    def test_small_weights_keep_quantization_resolution(self):
        model = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1e-9, 2e-9, -3e-9], [0., 0., 0.]]))
        rounded, report = adaround(model, [(torch.ones(1, 3),)], iterations=0)
        self.assertLess(report[0]["scales"][0], 1e-10)
        self.assertEqual(report[0]["scales"][1], 1.)
        self.assertGreater(rounded.weight[0].abs().max(), 0.)

    def test_adaround_gradients_encoding_and_all_sample_acceptance(self):
        model = nn.Linear(5, 3)
        batches = [(torch.randn(n, 4, 5) + 1.5,) for n in (1, 5, 2)]
        result, report = adaround(model, batches, bits=3, iterations=40)
        row = report[0]
        self.assertGreater(row["max_abs_alpha_gradient"], 1e-8)
        self.assertLessEqual(row["accepted_mse"], row["baseline_mse"])
        self.assertFalse(torch.equal(result.weight, model.weight))
        scales = row["scales"].reshape(-1, 1)
        torch.testing.assert_close(result.weight / scales, (result.weight / scales).round())
        squared, count = 0, 0
        for (x,) in batches:
            squared += (result(x) - model(x)).square().sum().item()
            count += model(x).numel()
        self.assertAlmostEqual(squared / count, row["accepted_mse"], places=7)
        self.assertEqual(row["local_invocations"], 3)

    def test_copies_eval_and_explicit_sequence_inputs(self):
        model = TwoInputs().train()
        model.layer.eval()
        state = copy.deepcopy(model.state_dict())
        modes = [m.training for m in model.modules()]
        x, offset = torch.randn(2, 7, 3), torch.randn(2, 7, 3)
        before = x.clone(), offset.clone()
        rounded, _ = adaround(model, [(x, offset)], iterations=3)
        corrected, _ = bias_correct(model, rounded, [(x, offset)])
        self.assertEqual(corrected(x, offset).shape, (2, 7, 2))
        self.assert_unchanged(model, state, modes)
        torch.testing.assert_close((x, offset), before, rtol=0, atol=0)
        for result in (rounded, corrected):
            self.assertTrue(all(not m.training for m in result.modules()))
            self.assertNotEqual(result.layer.weight.data_ptr(), model.layer.weight.data_ptr())

    def test_adaround_sequential_local_targets_use_supplied_model(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
        # Emulate a model already transformed by another optimization pass.
        with torch.no_grad():
            model[0].weight.mul_(5)
            model[2].weight.div_(5)
        batches = [(torch.randn(n, 3) + 2,) for n in (1, 6)]
        result, report = adaround(model, batches, bits=2, iterations=5)
        baseline_sum, accepted_sum, count = 0.0, 0.0, 0
        scale = report[1]["scales"].reshape(-1, 1)
        nearest = (model[2].weight / scale).round().clamp(-1, 1) * scale
        for (x,) in batches:
            target = model(x)
            local_input = result[1](result[0](x))
            baseline = torch.func.functional_call(model[2], {"weight": nearest}, (local_input,))
            baseline_sum += (baseline - target).square().sum().item()
            accepted_sum += (result(x) - target).square().sum().item()
            count += target.numel()
        self.assertAlmostEqual(report[1]["baseline_mse"], baseline_sum / count, places=7)
        self.assertAlmostEqual(report[1]["accepted_mse"], accepted_sum / count, places=7)
        self.assertLessEqual(accepted_sum, baseline_sum + 1e-6)

    def test_all_convolutions_reflect_padding(self):
        for dim, kind in enumerate((nn.Conv1d, nn.Conv2d, nn.Conv3d), 1):
            with self.subTest(dim=dim):
                model = kind(2, 2, 3, padding=1, padding_mode="reflect", groups=2)
                x = torch.randn(2, 2, *([4] * dim))
                result, report = adaround(model, [(x,)], bits=4, iterations=2)
                mse = (result(x) - model(x)).square().mean().item()
                self.assertAlmostEqual(mse, report[0]["accepted_mse"], places=7)
                self.assertLessEqual(mse, report[0]["baseline_mse"] + 1e-7)
                candidate = copy.deepcopy(model)
                with torch.no_grad():
                    candidate.bias.add_(0.7)
                corrected, rows = bias_correct(model, candidate, [(x,)])
                torch.testing.assert_close(corrected(x), model(x))
                torch.testing.assert_close(rows[0]["correction"], torch.full((2,), -0.7))

    def test_multilayer_bias_uses_current_candidate_inputs(self):
        reference = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
        candidate = copy.deepcopy(reference)
        with torch.no_grad():
            candidate[0].bias.add_(2)
            candidate[2].bias.sub_(3)
        original = copy.deepcopy(candidate.state_dict())
        modes = [m.training for m in candidate.modules()]
        batches = [(torch.randn(n, 4, 2),) for n in (1, 7, 2)]
        result, report = bias_correct(reference, candidate, batches)
        torch.testing.assert_close(report[0]["correction"], torch.full((2,), -2.0))
        torch.testing.assert_close(report[1]["correction"], torch.tensor([3.0]))
        self.assertEqual(report[0]["elements_per_channel"], 40)
        for (x,) in batches:
            torch.testing.assert_close(result(x), reference(x))
        for name, parameter in result.named_parameters():
            if name.endswith("weight"):
                torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
        self.assert_unchanged(candidate, original, modes)

    def test_variable_batches_weighted_by_elements_and_create_bias(self):
        reference = nn.Linear(1, 1, bias=False)
        candidate = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            reference.weight.fill_(2)
            candidate.weight.fill_(1)
        result, report = bias_correct(reference, candidate,
                                      [(torch.zeros(1, 1),), (torch.full((3, 1), 4.0),)])
        torch.testing.assert_close(result.bias, torch.tensor([3.0]))
        torch.testing.assert_close(result.weight, candidate.weight, rtol=0, atol=0)
        self.assertIsNone(candidate.bias)
        self.assertEqual(report[0]["elements_per_channel"], 4)

    def test_bounded_generator_and_invalid_batches(self):
        def batches():
            yield (torch.randn(2, 3),)
            yield (torch.randn(1, 3),)
            raise AssertionError("calibration iterator consumed beyond bound")

        model = nn.Linear(3, 2)
        _, rows = adaround(model, batches(), iterations=0, max_cached_batches=2)
        self.assertEqual(rows[0]["local_invocations"], 2)
        bias_correct(model, model, batches(), max_cached_batches=2)
        for operation in (lambda data: adaround(model, data, iterations=0),
                          lambda data: bias_correct(model, model, data)):
            with self.assertRaises(ValueError):
                operation([])
            with self.assertRaises(TypeError):
                operation([torch.randn(2, 3)])
        for kwargs in ({"bits": 1}, {"bits": 17}, {"iterations": -1},
                       {"learning_rate": 0}, {"regularization": -1},
                       {"max_cached_batches": 0}):
            with self.assertRaises(ValueError):
                adaround(model, [(torch.randn(2, 3),)], **kwargs)

    def test_shared_modules_parameters_and_storage_rejected(self):
        for alias in ("module", "parameter", "storage", "buffer", "separate_storage"):
            with self.subTest(alias=alias):
                first, second = nn.Linear(2, 2), nn.Linear(2, 2)
                if alias == "module":
                    second = first
                elif alias == "parameter":
                    second.weight = first.weight
                elif alias == "storage":
                    second.weight = nn.Parameter(first.weight.detach().view_as(first.weight))
                elif alias == "buffer":
                    second.register_buffer("alias", first.weight.detach())
                else:
                    backing = array.array("f", [1, 2, 3, 4, 5])
                    first.weight = nn.Parameter(torch.frombuffer(backing, dtype=torch.float32,
                                                                count=4).reshape(2, 2))
                    second.weight = nn.Parameter(torch.frombuffer(backing, dtype=torch.float32,
                                                                 count=4, offset=4).reshape(2, 2))
                model = nn.Sequential(first, second)
                batch = [(torch.randn(2, 2),)]
                with self.assertRaisesRegex(ValueError, "shared|aliases"):
                    adaround(model, batch)
                with self.assertRaisesRegex(ValueError, "shared|aliases"):
                    bias_correct(nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)), model, batch)
                with self.assertRaisesRegex(ValueError, "shared|aliases"):
                    bias_correct(model, nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)), batch)

    def test_pairing_shape_and_hook_cleanup_on_errors(self):
        class Repeated(nn.Module):
            def __init__(self, repeats):
                super().__init__()
                self.layer = nn.Linear(2, 2)
                self.repeats = repeats

            def forward(self, x):
                for _ in range(self.repeats):
                    x = self.layer(x)
                return x

        batch = [(torch.randn(2, 2),)]
        with self.assertRaisesRegex(ValueError, "pairing"):
            bias_correct(Repeated(1), Repeated(2), batch)
        _, report = bias_correct(Repeated(2), Repeated(2), batch)
        self.assertEqual(report[0]["local_invocations"], 2)
        with self.assertRaisesRegex(ValueError, "shape"):
            bias_correct(nn.Linear(2, 3), nn.Linear(2, 2), batch)
        model = Repeated(1)
        with self.assertRaises(RuntimeError):
            adaround(model, [(torch.randn(2, 5),)])
        self.assertFalse(model.layer._forward_hooks)
        captured_model = copy.deepcopy(model).eval()
        with self.assertRaises(RuntimeError):
            _capture(captured_model, captured_model.layer, [(torch.randn(2, 5),)])
        self.assertFalse(captured_model.layer._forward_hooks)

    def test_capture_clones_outputs_before_downstream_inplace_relu(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(inplace=True)).eval()
        with torch.no_grad():
            model[0].weight.zero_()
            model[0].bias.fill_(-1)
        calls = _capture(model, model[0], [(torch.zeros(1, 2),)])
        torch.testing.assert_close(calls[0][0][2], torch.full((1, 2), -1.0))
        self.assertEqual(calls[0][0][2].device.type, "cpu")
        self.assertFalse(model[0]._forward_hooks)

    def test_only_exact_matching_supported_types(self):
        class CustomLinear(nn.Linear):
            pass

        custom = CustomLinear(2, 2)
        _, report = adaround(custom, [(torch.randn(2, 2),)])
        self.assertEqual(report, [])
        _, report = bias_correct(custom, nn.Linear(2, 2), [(torch.randn(2, 2),)])
        self.assertEqual(report, [])

    def test_layer_bits_selection_and_per_layer_encodings(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2), nn.Linear(2, 1))
        state = copy.deepcopy(model.state_dict())
        selection = MappingProxyType({"0": 2, "3": 5})
        result, report = adaround(model, [(torch.randn(3, 3),)], bits=8,
                                  layer_bits=selection, iterations=3)
        self.assertEqual([(row["layer"], row["bits"], row["qmax"]) for row in report],
                         [("0", 2, 1), ("3", 5, 15)])
        torch.testing.assert_close(result[2].state_dict(), model[2].state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(model.state_dict(), state, rtol=0, atol=0)
        for row in report:
            layer = result.get_submodule(row["layer"])
            scale = row["scales"].reshape(-1, 1)
            torch.testing.assert_close(layer.weight / scale, (layer.weight / scale).round())
            self.assertLessEqual(row["accepted_mse"], row["baseline_mse"])
            self.assertTrue(all(math.isfinite(v) for v in row.values() if isinstance(v, float)))
        _, root_report = adaround(nn.Linear(3, 2), [(torch.randn(1, 3),)],
                                   layer_bits={"": 4}, iterations=0)
        self.assertEqual(root_report[0]["bits"], 4)

    def test_selection_validation_and_empty_selections(self):
        class CustomLinear(nn.Linear):
            pass

        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), CustomLinear(2, 2))
        batch = [(torch.randn(2, 2),)]
        for selection in ({"missing": 4}, {"1": 4}, {"2": 4}, {1: 4},
                          {"0": 1}, {"0": 17}, {"0": True}, {"0": 3.5}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                adaround(model, batch, layer_bits=selection, iterations=0)
        with self.assertRaises(TypeError):
            adaround(model, batch, layer_bits=[("0", 4)])
        for selection in (("missing",), ("1",), ("2",), (1,), ("0", "0")):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                bias_correct(model, model, batch, layers=selection)
        with self.assertRaises(TypeError):
            bias_correct(model, model, batch, layers=["0"])
        with self.assertRaisesRegex(ValueError, "matching"):
            bias_correct(CustomLinear(2, 2), nn.Linear(2, 2), batch, layers=("",))
        for operation in (lambda data: adaround(model, data, layer_bits={}),
                          lambda data: bias_correct(model, model, data, layers=())):
            result, report = operation(batch)
            self.assertEqual(report, [])
            self.assertIsNot(result, model)
            self.assertTrue(all(not m.training for m in result.modules()))
            torch.testing.assert_close(result.state_dict(), model.state_dict(), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                operation([])
            with self.assertRaises(TypeError):
                operation([batch[0][0]])
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                operation([(torch.full((1, 2), float("nan")),)])

    def test_bias_selection_leaves_unselected_biases_and_all_weights(self):
        reference = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        candidate = copy.deepcopy(reference)
        with torch.no_grad():
            candidate[0].bias.add_(2)
            candidate[1].bias.add_(3)
        state = copy.deepcopy(candidate.state_dict())
        result, report = bias_correct(reference, candidate, [(torch.randn(4, 2),)], layers=("1",))
        self.assertEqual([row["layer"] for row in report], ["1"])
        torch.testing.assert_close(result[0].state_dict(), candidate[0].state_dict(), rtol=0, atol=0)
        expected = -3 - reference[1].weight.detach().sum(dim=1) * 2
        torch.testing.assert_close(report[0]["correction"], expected)
        for name, parameter in result.named_parameters():
            if name.endswith("weight"):
                torch.testing.assert_close(parameter, state[name], rtol=0, atol=0)
        torch.testing.assert_close(candidate.state_dict(), state, rtol=0, atol=0)

    def test_nonfinite_calibration_and_captured_tensors_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            for operation in (lambda model, data: adaround(model, data, iterations=1),
                              lambda model, data: bias_correct(nn.Linear(2, 2), model, data),
                              lambda model, data: bias_correct(model, nn.Linear(2, 2), data)):
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "nonfinite"):
                    operation(nn.Linear(2, 2), [(torch.full((1, 2), bad),)])
                model = nn.Linear(2, 2)
                with torch.no_grad():
                    model.bias.fill_(bad)
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    operation(model, [(torch.ones(1, 2),)])

        class BadInput(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = nn.Linear(2, 2)
                self.layer.register_forward_hook(lambda module, args, output: torch.nan_to_num(output))

            def forward(self, x):
                return self.layer(input=x * float("nan"))

        model = BadInput()
        for operation in (lambda: adaround(model, [(torch.ones(1, 2),)], iterations=0),
                          lambda: bias_correct(model, model, [(torch.ones(1, 2),)])):
            with self.assertRaisesRegex(ValueError, "captured inputs"):
                operation()
        with self.assertRaisesRegex(ValueError, "captured inputs"):
            _capture(model, model.layer, [(torch.ones(1, 2),)])
        self.assertEqual(len(model.layer._forward_hooks), 1)

    def test_finite_captures_but_overflowing_errors_rejected(self):
        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.3, 1.0]]))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            adaround(model, [(torch.tensor([[1e30, 0.0]]),)], bits=2, iterations=0)
        reference, candidate = nn.Linear(1, 1).double(), nn.Linear(1, 1).double()
        with torch.no_grad():
            reference.weight.zero_()
            candidate.weight.zero_()
            reference.bias.fill_(1e200)
            candidate.bias.fill_(-1e200)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            bias_correct(reference, candidate, [(torch.zeros(1, 1, dtype=torch.float64),)])
    def test_buffer_to_buffer_storage_alias_rejected(self):
        for alias in ("identical", "view", "overlapping"):
            with self.subTest(alias=alias):
                model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
                shared = torch.randn(2, 2)
                if alias == "identical":
                    model[0].register_buffer("running", shared)
                    model[1].register_buffer("running", shared)
                elif alias == "view":
                    model[0].register_buffer("running", shared)
                    model[1].register_buffer("running", shared.view(4))
                else:
                    backing = array.array("f", [1, 2, 3, 4, 5])
                    model[0].register_buffer("running", torch.frombuffer(
                        backing, dtype=torch.float32, count=4))
                    model[1].register_buffer("running", torch.frombuffer(
                        backing, dtype=torch.float32, count=4, offset=4))
                batch = [(torch.randn(2, 2),)]
                plain = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
                with self.assertRaisesRegex(ValueError, "shared|aliases"):
                    adaround(model, batch)
                with self.assertRaisesRegex(ValueError, "shared|aliases"):
                    bias_correct(plain, model, batch)
                with self.assertRaisesRegex(ValueError, "shared|aliases"):
                    bias_correct(model, plain, batch)

    def test_sparse_and_meta_weights_rejected_with_value_error(self):
        def sparse_model():
            model = nn.Linear(2, 2)
            model.weight = nn.Parameter(torch.eye(2).to_sparse())
            return model

        def meta_model():
            return nn.Linear(2, 2, device="meta")

        for factory, pattern in ((sparse_model, "dense"), (meta_model, "storage")):
            with self.subTest(kind=pattern):
                model, batch = factory(), [(torch.randn(2, 2),)]
                # Storage probing must not leak a RuntimeError from _check_model.
                for operation in (lambda: adaround(model, batch, iterations=0),
                                  lambda: bias_correct(model, nn.Linear(2, 2), batch),
                                  lambda: bias_correct(nn.Linear(2, 2), model, batch)):
                    with self.assertRaisesRegex(ValueError, pattern):
                        operation()

    def test_meta_parameters_are_not_reported_as_aliases(self):
        model = nn.Sequential(nn.Linear(2, 2, device="meta"), nn.Linear(2, 2, device="meta"))
        model[0].register_buffer("extra", torch.zeros(2, device="meta"))
        with self.assertRaisesRegex(ValueError, "storage"):
            adaround(model, [(torch.randn(2, 2),)], iterations=0)

    def test_adaround_does_not_retain_candidate_outputs(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
        batches = [(torch.randn(n, 3),) for n in (2, 3)]
        original, seen = reconstruction._samples, []

        def spy(*args, **kwargs):
            samples = original(*args, **kwargs)
            seen.append(samples)
            return samples

        reconstruction._samples = spy
        try:
            adaround(model, batches, iterations=2)
            bias_correct(model, model, batches)
        finally:
            reconstruction._samples = original
        adaround_samples, bias_samples = seen[:2], seen[2:]
        self.assertTrue(adaround_samples and bias_samples)
        for samples in adaround_samples:
            self.assertTrue(all(output is None for _, _, _, output in samples))
        for samples in bias_samples:
            self.assertTrue(all(isinstance(output, torch.Tensor)
                                for _, _, _, output in samples))
        # The unretained output is still validated through its shape.
        calls = _capture(model.eval(), model[0], batches, retain_output=False)
        self.assertIsNone(calls[0][0][2])
        self.assertEqual(calls[0][0][3], (2, 4))
        samples = _samples(model, model, "0", batches, capture_outputs=False)
        self.assertEqual([s[3] for s in samples], [None, None])
        with self.assertRaisesRegex(ValueError, "shape"):
            _samples(nn.Linear(2, 3).eval(), nn.Linear(2, 2).eval(), "",
                     [(torch.randn(2, 2),)], capture_outputs=False)

    def test_adaround_numerics_and_report_are_stable(self):
        def run():
            torch.manual_seed(1234)
            model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))
            batches = [(torch.randn(n, 4) + 0.5,) for n in (2, 3)]
            result, report = adaround(model, batches, bits=4, iterations=12)
            return model, batches, result, report

        model, batches, first, first_report = run()
        _, _, second, second_report = run()
        torch.testing.assert_close(first.state_dict(), second.state_dict(), rtol=0, atol=0)
        self.assertEqual([row["layer"] for row in first_report], ["0", "2"])
        for row, other in zip(first_report, second_report):
            self.assertEqual(row["accepted"], other["accepted"])
            for key in ("baseline_mse", "candidate_mse", "accepted_mse",
                        "max_abs_alpha_gradient"):
                self.assertEqual(row[key], other[key])
        # Reported local error still matches an independent recomputation.
        squared, count = 0.0, 0
        for (x,) in batches:
            target = model(x)
            local = torch.func.functional_call(
                first[2], {"weight": first[2].weight.detach(),
                           "bias": first[2].bias.detach()}, (first[1](first[0](x)),))
            squared += (local - target).square().sum().item()
            count += target.numel()
        self.assertAlmostEqual(first_report[1]["accepted_mse"], squared / count, places=7)


if __name__ == "__main__":
    unittest.main()
