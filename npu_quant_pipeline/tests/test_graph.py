import copy
import unittest

import torch
from torch import nn

from npu_quant.graph import optimize_graph


class Pair(nn.Module):
    def __init__(self, first, second, middle=None, branch=False, reuse=False):
        super().__init__()
        self.first = first
        self.middle = middle if middle is not None else nn.Identity()
        self.second = second
        self.branch = branch
        self.reuse = reuse

    def forward(self, x):
        a = self.first(x)
        b = self.second(self.middle(a))
        if self.reuse:
            b = b + self.second(self.middle(a))
        if self.branch:
            return {"result": b, "branch": (a, [a + 1])}
        return b


class GraphTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def check(self, model, x, **kwargs):
        model.eval()
        before = copy.deepcopy(model.state_dict())
        args = copy.deepcopy(x)
        result, report = optimize_graph(model, (x,), **kwargs)
        torch.testing.assert_close(result(x), model(x))
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
        torch.testing.assert_close(x, args, rtol=0, atol=0)
        self.assertIsNot(result, model)
        self.assertTrue(all(not module.training for module in result.modules()))
        original_ids = {id(p) for p in model.parameters()}
        self.assertFalse(original_ids & {id(p) for p in result.parameters()})
        return result, report

    def test_fold_all_dimensions_and_affine_modes(self):
        for dim, conv, bn in [(1, nn.Conv1d, nn.BatchNorm1d),
                              (2, nn.Conv2d, nn.BatchNorm2d),
                              (3, nn.Conv3d, nn.BatchNorm3d)]:
            for affine in (True, False):
                with self.subTest(dim=dim, affine=affine):
                    model = nn.Sequential(conv(2, 4, 3, bias=False), bn(4, affine=affine))
                    model[1].running_mean.copy_(torch.randn(4))
                    model[1].running_var.copy_(torch.rand(4) + 0.5)
                    _, report = self.check(model, torch.randn(2, 2, *([5] * dim)))
                    self.assertEqual([r["op"] for r in report], ["fold_bn"])

    def test_supported_root_layers_preserve_type_and_parameters(self):
        for kind in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d):
            with self.subTest(kind=kind):
                if kind is nn.Linear:
                    model, x = kind(3, 4), torch.randn(2, 3)
                else:
                    dim = (nn.Conv1d, nn.Conv2d, nn.Conv3d).index(kind) + 1
                    model, x = kind(3, 4, 1), torch.randn(2, 3, *([3] * dim))
                result, report = self.check(model, x, smooth=True)
                self.assertIs(type(result), kind)
                self.assertEqual(report, [])
                torch.testing.assert_close(result.state_dict(), model.state_dict(), rtol=0, atol=0)
                for name, p in result.named_parameters():
                    self.assertNotEqual(p.data_ptr(), model.get_parameter(name).data_ptr())

    def test_root_layer_keeps_hooks_and_validation(self):
        model = nn.Linear(2, 2)
        model.register_forward_pre_hook(lambda module, args: (args[0] + 1,))
        model.register_forward_hook(lambda module, args, output: output.square())
        result, report = self.check(model, torch.randn(2, 2))
        self.assertIs(type(result), nn.Linear)
        self.assertEqual(report, [])
        self.assertEqual(len(result._forward_hooks), 1)
        self.assertEqual(len(result._forward_pre_hooks), 1)
        with self.assertRaises(TypeError):
            optimize_graph(model, [torch.ones(2, 2)])
        with self.assertRaises(TypeError):
            optimize_graph(model, (torch.ones(2, 2),), activation_max=[])
        model.train()
        with self.assertRaisesRegex(ValueError, "all modules in eval mode"):
            optimize_graph(model, (torch.ones(2, 2),), fold_bn=False, equalize=False)

    def test_unbatched_conv1d_folding_zero_example_counterexample(self):
        class Unbatched(nn.Module):
            def __init__(self, squeeze):
                super().__init__()
                self.conv = nn.Conv1d(2, 2, 1, bias=False)
                self.bn = nn.BatchNorm1d(2)
                self.squeeze = squeeze
                with torch.no_grad():
                    self.conv.weight.fill_(1)
                    self.bn.weight.copy_(torch.tensor([1., 4.]))

            def forward(self, x):
                return self.bn(self.conv(x.squeeze(0) if self.squeeze else x))

        for squeeze in (False, True):
            with self.subTest(squeeze=squeeze):
                model = Unbatched(squeeze).eval()
                zero = torch.zeros(1, 2, 2) if squeeze else torch.zeros(2, 2)
                result, report = self.check(model, zero)
                self.assertEqual(report, [])
                probe = torch.arange(1., 5.).reshape(zero.shape)
                torch.testing.assert_close(result(probe), model(probe))
                incorrectly_fused = torch.nn.utils.fusion.fuse_conv_bn_eval(model.conv, model.bn)
                conv_input = probe.squeeze(0) if squeeze else probe
                torch.testing.assert_close(incorrectly_fused(torch.zeros(2, 2)), model(zero))
                self.assertFalse(torch.allclose(incorrectly_fused(conv_input), model(probe)))

    def test_cle_direct_and_relu_all_kinds(self):
        for kind in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d):
            for relu in (False, True):
                with self.subTest(kind=kind, relu=relu):
                    if kind is nn.Linear:
                        first, second = kind(3, 4), kind(4, 2)
                        x = torch.randn(2, 5, 3)
                    else:
                        dim = (nn.Conv1d, nn.Conv2d, nn.Conv3d).index(kind) + 1
                        first, second = kind(3, 4, 1), kind(4, 2, 1)
                        x = torch.randn(2, 3, *([4] * dim))
                    with torch.no_grad():
                        first.weight.mul_(10)
                    model = nn.Sequential(first, nn.ReLU(), second) if relu else nn.Sequential(first, second)
                    result, report = self.check(model, x)
                    self.assertEqual([r["op"] for r in report], ["equalize"])
                    self.assertFalse(torch.equal(result.get_submodule("0").weight, first.weight))

    def test_excluded_nonlinearities_and_identity(self):
        for middle in (nn.GELU(), nn.Sigmoid(), nn.ReLU6(), nn.Identity()):
            with self.subTest(middle=middle):
                _, report = self.check(Pair(nn.Linear(4, 4), nn.Linear(4, 4), middle), torch.randn(2, 4))
                self.assertEqual(report, [])

    def test_branches_and_reused_modules(self):
        for branch, reuse in ((True, False), (False, True)):
            model = Pair(nn.Linear(4, 4), nn.Linear(4, 4), nn.ReLU(), branch, reuse)
            _, report = self.check(model, torch.randn(2, 4))
            self.assertEqual(report, [])

    def test_fold_branch_excluded(self):
        class Branched(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(2, 2, 1)
                self.bn = nn.BatchNorm1d(2)

            def forward(self, x):
                y = self.conv(x)
                return self.bn(y), y

        _, report = self.check(Branched(), torch.randn(2, 2, 3))
        self.assertEqual(report, [])

    def test_shared_parameters_including_unused_alias(self):
        for unused in (False, True):
            model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
            if unused:
                model.register_parameter("alias", model[0].weight)
            else:
                model[1].weight = model[0].weight
            _, report = self.check(model, torch.randn(2, 4))
            self.assertEqual(report, [])

    def test_grouped_convolutions_excluded(self):
        model = nn.Sequential(nn.Conv2d(4, 4, 1, groups=2), nn.Conv2d(4, 4, 1))
        _, report = self.check(model, torch.randn(2, 4, 3, 3))
        self.assertEqual(report, [])

    def test_storage_alias_excluded(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        model[1].weight = nn.Parameter(model[0].weight.detach())
        _, report = self.check(model, torch.randn(2, 4))
        self.assertEqual(report, [])

    def test_overlapping_frombuffer_storage_with_distinct_pointers(self):
        for buffer_alias in (False, True):
            with self.subTest(buffer_alias=buffer_alias):
                allocation = bytearray(17 * 4)
                first = torch.frombuffer(allocation, dtype=torch.float32, count=16)
                second = torch.frombuffer(allocation, dtype=torch.float32, count=16, offset=4)
                first.fill_(1)
                second.fill_(2)
                self.assertNotEqual(first.untyped_storage().data_ptr(),
                                    second.untyped_storage().data_ptr())
                model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
                model[0].weight = nn.Parameter(first.reshape(4, 4))
                if buffer_alias:
                    model.register_buffer("alias", second)
                else:
                    model[1].weight = nn.Parameter(second.reshape(4, 4))
                result, report = self.check(model, torch.zeros(2, 4))
                self.assertEqual(report, [])
                probe = torch.randn(2, 4)
                torch.testing.assert_close(result(probe), model(probe))

    def test_overlapping_storage_does_not_block_unaffected_pair(self):
        allocation = bytearray(17 * 4)
        first = torch.frombuffer(allocation, dtype=torch.float32, count=16)
        second = torch.frombuffer(allocation, dtype=torch.float32, count=16, offset=4)
        first.fill_(1)
        second.fill_(2)
        model = nn.Sequential(*(nn.Linear(4, 4) for _ in range(4)))
        model[0].weight = nn.Parameter(first.reshape(4, 4))
        model[1].weight = nn.Parameter(second.reshape(4, 4))
        _, report = self.check(model, torch.randn(2, 4))
        self.assertEqual(report, [{"op": "equalize", "producer": "2", "consumer": "3"}])

    def test_hooked_cle_zero_example_counterexamples(self):
        def pre_hook(module, args):
            x = args[0]
            return (x * module.weight.sum() if hasattr(module, "weight") else x.square(),)

        def post_hook(module, args, output):
            return output * module.weight.sum() if hasattr(module, "weight") else output.square()

        for target in (0, 1, 2):
            for pre in (False, True):
                with self.subTest(target=target, pre=pre):
                    model = nn.Sequential(nn.Linear(1, 1, bias=False), nn.ReLU(),
                                          nn.Linear(1, 1, bias=False)).eval()
                    with torch.no_grad():
                        model[0].weight.fill_(4)
                        model[2].weight.fill_(1)
                    if pre:
                        model[target].register_forward_pre_hook(pre_hook)
                    else:
                        model[target].register_forward_hook(post_hook)
                    result, report = self.check(model, torch.zeros(1, 1))
                    self.assertEqual(report, [])
                    probe = torch.ones(1, 1)
                    torch.testing.assert_close(result(probe), model(probe))
                    incorrectly_equalized = copy.deepcopy(model)
                    with torch.no_grad():
                        incorrectly_equalized[0].weight.mul_(0.5)
                        incorrectly_equalized[2].weight.div_(0.5)
                    torch.testing.assert_close(incorrectly_equalized(torch.zeros(1, 1)),
                                               model(torch.zeros(1, 1)))
                    self.assertFalse(torch.allclose(incorrectly_equalized(probe), model(probe)))

    def test_hooked_fold_and_smoothing_modules_excluded(self):
        for fold in (False, True):
            for target in (0, 1):
                for pre in (False, True):
                    with self.subTest(fold=fold, target=target, pre=pre):
                        model = (nn.Sequential(nn.Conv1d(2, 2, 1), nn.BatchNorm1d(2)) if fold
                                 else nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 2)))
                        if pre:
                            model[target].register_forward_pre_hook(lambda module, args: args)
                        else:
                            model[target].register_forward_hook(lambda module, args, output: output)
                        x = torch.randn(2, 2, 3) if fold else torch.randn(2, 2)
                        _, report = self.check(model, x, smooth=True,
                                               activation_max={"0": torch.full((2,), 100.)})
                        self.assertEqual(report, [])

    def test_traced_through_hooks_rejected_upfront(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))

            def forward(self, x):
                raise AssertionError("must reject hooks before tracing or execution")

        def never_run(*args):
            raise AssertionError("must not execute hooks")

        for root in (False, True):
            for pre in (False, True):
                with self.subTest(root=root, pre=pre):
                    block = Block()
                    model = block if root else nn.Sequential(block)
                    model.eval()
                    if pre:
                        block.register_forward_pre_hook(never_run)
                    else:
                        block.register_forward_hook(never_run)
                    with self.assertRaisesRegex(ValueError, "hooks on traced-through module"):
                        optimize_graph(model, (torch.zeros(2, 2),))

    def test_hooked_leaf_does_not_block_unaffected_pair(self):
        class Independent(nn.Module):
            def __init__(self):
                super().__init__()
                self.safe = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
                self.hooked = nn.Linear(2, 2)
                self.hooked.register_forward_hook(lambda module, args, output: output.square())

            def forward(self, x):
                return self.safe(x), self.hooked(x)

        _, report = self.check(Independent(), torch.randn(2, 2))
        self.assertEqual(report, [{"op": "equalize", "producer": "safe.0", "consumer": "safe.1"}])

    def test_direct_parameter_access_excluded(self):
        class Exposed(nn.Module):
            def __init__(self, target):
                super().__init__()
                self.first = nn.Linear(4, 4)
                self.second = nn.Linear(4, 4)
                self.target = target

            def forward(self, x):
                return self.second(self.first(x)), getattr(self, self.target).weight + 1

        for target in ("first", "second"):
            with self.subTest(target=target):
                _, report = self.check(Exposed(target), torch.randn(2, 4))
                self.assertEqual(report, [])

    def test_parameter_buffer_aliases_excluded(self):
        for operation in ("fold_bn", "equalize", "smooth"):
            for view in (False, True):
                with self.subTest(operation=operation, view=view):
                    if operation == "fold_bn":
                        model = nn.Sequential(nn.Conv1d(4, 4, 1), nn.BatchNorm1d(4))
                        x = torch.randn(2, 4, 3)
                    elif operation == "smooth":
                        model = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4))
                        x = torch.randn(2, 4)
                    else:
                        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
                        x = torch.randn(2, 4)
                    weight = model[0].weight
                    alias = weight.detach().flatten()[1::2] if view else weight
                    model.register_buffer("alias", alias)
                    _, report = self.check(model, x, smooth=True,
                                           activation_max={"0": torch.full((4,), 100.)})
                    self.assertEqual(report, [])

    def test_batchnorm_buffer_alias_excluded(self):
        for view in (False, True):
            model = nn.Sequential(nn.Conv1d(4, 4, 1), nn.BatchNorm1d(4))
            buffer = model[1].running_mean
            model.register_buffer("alias", buffer[1:] if view else buffer)
            _, report = self.check(model, torch.randn(2, 4, 3))
            self.assertEqual(report, [])

    def test_norm_and_batchnorm_get_attr_excluded(self):
        class Exposed(nn.Module):
            def __init__(self, fold, attribute):
                super().__init__()
                self.first = nn.Conv1d(4, 4, 1) if fold else nn.LayerNorm(4)
                self.second = nn.BatchNorm1d(4) if fold else nn.Linear(4, 4)
                self.fold = fold
                self.attribute = attribute

            def forward(self, x):
                module = self.second if self.fold else self.first
                return self.second(self.first(x)), getattr(module, self.attribute)

        for fold, attribute in ((True, "weight"), (True, "running_mean"),
                                (False, "weight"), (False, "bias")):
            with self.subTest(fold=fold, attribute=attribute):
                x = torch.randn(2, 4, 3) if fold else torch.randn(2, 4)
                _, report = self.check(Exposed(fold, attribute), x, smooth=True,
                                       activation_max={"first": torch.full((4,), 100.)})
                self.assertEqual(report, [])

    def test_intermediate_branch_excluded(self):
        class Branched(nn.Module):
            def __init__(self):
                super().__init__()
                self.first = nn.Linear(4, 4)
                self.relu = nn.ReLU()
                self.second = nn.Linear(4, 4)

            def forward(self, x):
                y = self.relu(self.first(x))
                return self.second(y), y

        _, report = self.check(Branched(), torch.randn(2, 4))
        self.assertEqual(report, [])

    def test_fold_then_equalize(self):
        model = nn.Sequential(nn.Conv1d(2, 4, 1), nn.BatchNorm1d(4),
                              nn.ReLU(), nn.Conv1d(4, 2, 1))
        _, report = self.check(model, torch.randn(2, 2, 5))
        self.assertEqual([r["op"] for r in report], ["fold_bn", "equalize"])

    def test_norm_branch_and_reuse_excluded(self):
        class NormPath(nn.Module):
            def __init__(self, reuse):
                super().__init__()
                self.norm = nn.LayerNorm(4)
                self.linear = nn.Linear(4, 4)
                self.reuse = reuse

            def forward(self, x):
                y = self.norm(x)
                return self.linear(y), self.norm(x + 1) if self.reuse else y

        for reuse in (False, True):
            _, report = self.check(NormPath(reuse), torch.randn(2, 4),
                                   smooth=True, activation_max={"norm": torch.ones(4) * 100})
            self.assertEqual(report, [])

    def test_smoothing_is_post_norm_and_uses_formula(self):
        norms = [nn.LayerNorm(4)]
        if hasattr(nn, "RMSNorm"):
            norms.append(nn.RMSNorm(4))
        for norm in norms:
            with self.subTest(norm=type(norm)):
                model = nn.Sequential(nn.Linear(4, 4), norm, nn.Linear(4, 2))
                with torch.no_grad():
                    model[2].weight.fill_(1)
                    norm.weight.fill_(2)
                    if getattr(norm, "bias", None) is not None:
                        norm.bias.fill_(0.5)
                a = torch.tensor([0., 4., 16., 1e8])
                result, report = self.check(model, torch.randn(3, 4), smooth=True,
                                            activation_max={"1": a})
                self.assertEqual([r["op"] for r in report], ["smooth"])
                scale = torch.tensor([1., 2., 4., 100.])
                torch.testing.assert_close(result.get_submodule("1").weight, norm.weight / scale)
                torch.testing.assert_close(result.get_submodule("2").weight, model[2].weight * scale)
                torch.testing.assert_close(result.get_submodule("0").weight, model[0].weight, rtol=0, atol=0)

    def test_smoothing_exclusions_and_opt_in(self):
        x = torch.randn(2, 4)
        stats = {"0": torch.ones(4), "first": torch.ones(4)}
        for name, model, options, expected in [
            ("disabled", nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
             {"activation_max": stats}, []),
            ("enabled", nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
             {"smooth": True, "activation_max": stats}, ["smooth"]),
            ("no_stats", nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
             {"smooth": True}, []),
            ("missing_target", nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4)),
             {"smooth": True, "activation_max": {"other": torch.ones(4)}}, []),
            ("non_affine", nn.Sequential(nn.LayerNorm(4, elementwise_affine=False), nn.Linear(4, 4)),
             {"smooth": True, "activation_max": stats}, []),
            ("relu", nn.Sequential(nn.LayerNorm(4), nn.ReLU(), nn.Linear(4, 4)),
             {"smooth": True, "activation_max": stats}, []),
            ("branch", Pair(nn.LayerNorm(4), nn.Linear(4, 4), nn.ReLU(), branch=True),
             {"smooth": True, "activation_max": stats}, []),
        ]:
            with self.subTest(case=name):
                _, report = self.check(model, x, **options)
                self.assertEqual([r["op"] for r in report], expected)

    def test_invalid_statistics(self):
        for stats in (torch.ones(3), torch.tensor([1., -1., 1., 1.]),
                      torch.full((4,), float("nan"))):
            with self.assertRaises(ValueError):
                optimize_graph(nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 2)).eval(),
                               (torch.randn(2, 4),), smooth=True, activation_max={"0": stats})

    def test_disabled_and_zero_channels(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        with torch.no_grad():
            model[0].weight[0].zero_()
            model[1].weight[:, 1].zero_()
        _, report = self.check(model, torch.randn(2, 4))
        self.assertEqual(len(report), 1)
        _, report = self.check(model, torch.randn(2, 4), fold_bn=False, equalize=False)
        self.assertEqual(report, [])

    def test_trace_failure_is_not_a_fallback(self):
        class Dynamic(nn.Module):
            def forward(self, x):
                return x if x.sum() > 0 else -x

        with self.assertRaisesRegex(RuntimeError, "FX tracing failed"):
            optimize_graph(Dynamic().eval(), (torch.ones(2),))

    def test_equivalence_failure_is_closed(self):
        class TraceDependent(nn.Module):
            def forward(self, x):
                return x + (1 if isinstance(x, torch.fx.Proxy) else 2)

        with self.assertRaisesRegex(RuntimeError, "equivalence failed"):
            optimize_graph(TraceDependent().eval(), (torch.ones(2),))

    def test_training_modes_rejected_upfront(self):
        class NeverRun(nn.Module):
            def forward(self, x):
                raise AssertionError("training models must not be traced or executed")

        for mode in ("root", "child", "unused_child"):
            for disabled in (False, True):
                with self.subTest(mode=mode, disabled=disabled):
                    model = nn.Sequential(NeverRun(), nn.Linear(4, 4)).eval()
                    if mode == "root":
                        model.training = True
                    elif mode == "child":
                        model[1].train()
                    else:
                        model[0].register_module("unused", nn.Dropout())
                    before = copy.deepcopy(model.state_dict())
                    modes = [module.training for module in model.modules()]
                    with self.assertRaisesRegex(ValueError, "all modules in eval mode"):
                        optimize_graph(model, (torch.randn(2, 4),),
                                       fold_bn=not disabled, equalize=not disabled)
                    self.assertEqual([module.training for module in model.modules()], modes)
                    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)

    def test_sparse_and_meta_tensors_skip_rewriting_without_error(self):
        cases = {"sparse": lambda: torch.eye(3).to_sparse(),
                 "meta": lambda: torch.zeros(3, device="meta")}
        for kind, make in cases.items():
            for target in ("conv", "bn"):
                with self.subTest(kind=kind, target=target):
                    model = nn.Sequential(nn.Conv1d(4, 4, 1), nn.BatchNorm1d(4))
                    model[0 if target == "conv" else 1].register_buffer("extra", make())
                    result, report = self.check(model, torch.randn(2, 4, 3))
                    # Aliasing is undecidable, so the pair must be left alone.
                    self.assertEqual(report, [])
                    self.assertIn(f"{0 if target == 'conv' else 1}.extra",
                                  dict(result.named_buffers()))

    def test_folded_batchnorm_submodule_is_removed(self):
        model = nn.Sequential(nn.Conv1d(2, 4, 3, bias=False), nn.BatchNorm1d(4))
        with torch.no_grad():
            model[1].running_mean.copy_(torch.randn(4))
            model[1].running_var.copy_(torch.rand(4) + 0.5)
        x = torch.randn(2, 2, 5)
        result, report = self.check(model, x)
        self.assertEqual([r["op"] for r in report], ["fold_bn"])
        self.assertEqual(report[0]["bn"], "1")
        names = dict(result.named_modules())
        self.assertNotIn("1", names)
        self.assertIn("0", names)
        self.assertFalse([k for k in result.state_dict() if k.startswith("1.")])
        self.assertFalse([k for k in result.state_dict() if "running_" in k])
        torch.testing.assert_close(result(x), model(x))

    def test_requires_tuple(self):
        with self.assertRaises(TypeError):
            optimize_graph(nn.Identity(), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
