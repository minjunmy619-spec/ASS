"""Conservative, inference-oriented graph optimizations using only PyTorch."""

import copy
from collections import Counter

import torch
from torch import fx, nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


_CONVS = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
_BNS = {nn.Conv1d: nn.BatchNorm1d, nn.Conv2d: nn.BatchNorm2d,
        nn.Conv3d: nn.BatchNorm3d}
_NORMS = (nn.LayerNorm,) + ((nn.RMSNorm,) if hasattr(nn, "RMSNorm") else ())


def _storage_overlap(left, right):
    if left.device != right.device:
        return False
    a, b = left.untyped_storage(), right.untyped_storage()
    # Separate frombuffer storages can overlap despite different base pointers.
    return (a._cdata == b._cdata or
            (a.data_ptr() < b.data_ptr() + b.nbytes()
             and b.data_ptr() < a.data_ptr() + a.nbytes()))


def optimize_graph(model, example_args: tuple, *, fold_bn=True, equalize=True,
                   smooth=False, activation_max=None):
    """Return an independent model and a list of applied-operation dicts.

    A standard root Linear or Conv1/2/3d is copied unchanged with an empty
    report, preserving its layer type. Other models return an FX GraphModule.

    Requires every module to be in eval mode, even when rewrites are disabled;
    call ``model.eval()`` before using this inference-only API. Rewrites only
    standard, uniquely called modules with unaliased parameters and buffers
    and no forward/pre-forward hooks. Hooks on traced-through modules (including
    the graph root) are rejected because FX cannot preserve them reliably.
    BN folding requires a batched Conv output rank observed on the example;
    callers must retain that batched/unbatched input convention after rewriting.
    CLE supports same-kind Conv1/2/3d (groups=1) or Linear pairs, optionally
    separated by a module ReLU. Smoothing divides an affine norm's *output*
    parameters and compensates its sole Linear consumer, never its input.
    ``activation_max`` maps FX norm targets to nonnegative channel maxima.
    Missing statistics and unsupported paths are skipped. Invalid statistics,
    tracing failures, and example-output mismatches raise exceptions.

    Equivalence is checked recursively with torch.testing.assert_close's dtype
    tolerances on independent copies of the supplied example. This is an
    inference utility: stochastic/stateful models may fail that check.
    """
    if not isinstance(example_args, tuple):
        raise TypeError("example_args must be a tuple")
    if activation_max is not None and not isinstance(activation_max, dict):
        raise TypeError("activation_max must be a dict or None")
    if any(module.training for module in model.modules()):
        raise ValueError("optimize_graph requires all modules in eval mode; call model.eval()")

    def check_example(candidate):
        with torch.no_grad():
            expected = copy.deepcopy(copy.deepcopy(model)(*copy.deepcopy(example_args)))
            actual = candidate(*copy.deepcopy(example_args))
            try:
                torch.testing.assert_close(actual, expected)
            except (AssertionError, TypeError, ValueError) as exc:
                raise RuntimeError("optimize_graph: example-output equivalence failed") from exc

    source = copy.deepcopy(model)
    if type(source) in _CONVS + (nn.Linear,):
        check_example(source)
        return source, []

    unsafe = set()
    tracer = fx.Tracer()
    # FX can prune unused aliases, and deepcopy can sever storage sharing
    # between distinct Parameter wrappers or a parameter and a buffer view.
    for inspected in (model, source):
        tensors = (list(inspected.named_parameters(remove_duplicate=False))
                   + list(inspected.named_buffers(remove_duplicate=False)))
        aliased = set()
        for index, (_, tensor) in enumerate(tensors):
            for _, other in tensors[:index]:
                if tensor is other or _storage_overlap(tensor, other):
                    aliased.update((id(tensor), id(other)))
        for name, module in inspected.named_modules():
            if module._forward_hooks or module._forward_pre_hooks:
                if not name or not tracer.is_leaf_module(module, name):
                    raise ValueError("optimize_graph: forward/pre-forward hooks on "
                                     "traced-through module %r are unsupported" % (name or "<root>"))
                unsafe.add(id(source.get_submodule(name)))
            if any(id(t) in aliased for t in (*module.parameters(), *module.buffers())):
                unsafe.add(id(source.get_submodule(name)))

    try:
        graph = fx.symbolic_trace(source)
    except Exception as exc:
        raise RuntimeError("optimize_graph: FX tracing failed") from exc
    # FX synthesizes intermediate containers with training=True by default.
    graph.eval()

    exposed = []
    for node in graph.graph.nodes:
        if node.op == "get_attr":
            value = graph
            for part in node.target.split("."):
                value = getattr(value, part)
            if isinstance(value, torch.Tensor):
                exposed.append(value)
    for module in graph.modules():
        if any(_storage_overlap(t, value)
               for t in (*module.parameters(), *module.buffers()) for value in exposed):
            unsafe.add(id(module))

    calls = Counter(id(graph.get_submodule(n.target))
                    for n in graph.graph.nodes if n.op == "call_module")
    report = []

    def module_at(node):
        if not isinstance(node, fx.Node) or node.op != "call_module":
            return None
        module = graph.get_submodule(node.target)
        if calls[id(module)] != 1 or id(module) in unsafe:
            return None
        return module

    def unary_input(node):
        # Reject keyword and extra-argument paths rather than guessing semantics.
        if len(node.args) == 1 and not node.kwargs:
            return node.args[0]
        return None

    with torch.no_grad():
        if fold_bn:
            ranks = {}

            class ConvRanks(fx.Interpreter):
                def run_node(self, node):
                    output = super().run_node(node)
                    if (node.op == "call_module"
                            and type(self.fetch_attr(node.target)) in _CONVS
                            and isinstance(output, torch.Tensor)):
                        ranks[node.name] = output.ndim
                    return output

            if any(n.op == "call_module" and type(graph.get_submodule(n.target)) in _CONVS
                   for n in graph.graph.nodes):
                # Execute real operations, not shape tracing, on an isolated copy.
                ConvRanks(copy.deepcopy(graph)).run(*copy.deepcopy(example_args))
            for node in list(graph.graph.nodes):
                bn = module_at(node)
                if type(bn) not in _BNS.values():
                    continue
                producer = unary_input(node)
                conv = module_at(producer)
                if (type(conv) not in _CONVS or type(bn) is not _BNS[type(conv)]
                        or len(producer.users) != 1
                        or ranks.get(producer.name) != _CONVS.index(type(conv)) + 3
                        or bn.running_mean is None or bn.running_var is None):
                    continue
                fused = fuse_conv_bn_eval(conv, bn)
                graph.set_submodule(producer.target, fused)
                calls[id(fused)] = 1
                node.replace_all_uses_with(producer)
                graph.graph.erase_node(node)
                report.append({"op": "fold_bn", "conv": producer.target,
                               "bn": node.target})

        if equalize:
            for node in list(graph.graph.nodes):
                second = module_at(node)
                if type(second) not in _CONVS + (nn.Linear,):
                    continue
                middle = unary_input(node)
                first_node = middle
                if type(module_at(middle)) is nn.ReLU:
                    if len(middle.users) != 1:
                        continue
                    first_node = unary_input(middle)
                first = module_at(first_node)
                if (type(first) is not type(second) or first is None
                        or len(first_node.users) != 1):
                    continue
                if type(first) in _CONVS and (first.groups != 1 or second.groups != 1):
                    continue
                w1, w2 = first.weight, second.weight
                if w1.shape[0] != w2.shape[1]:
                    continue
                a = w1.detach().double().abs().flatten(1).amax(1)
                b = w2.detach().double().abs().movedim(1, 0).flatten(1).amax(1)
                if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
                    continue
                scale = torch.ones_like(a)
                valid = (a > 0) & (b > 0)
                scale[valid] = (b[valid] / a[valid]).sqrt().clamp(0.01, 100)
                w1.mul_(scale.to(w1).reshape(-1, *([1] * (w1.ndim - 1))))
                if first.bias is not None:
                    first.bias.mul_(scale.to(first.bias))
                w2.div_(scale.to(w2).reshape(1, -1, *([1] * (w2.ndim - 2))))
                report.append({"op": "equalize", "producer": first_node.target,
                               "consumer": node.target})

        if smooth:
            stats = activation_max if activation_max is not None else {}
            for node in graph.graph.nodes:
                linear = module_at(node)
                if type(linear) is not nn.Linear:
                    continue
                norm_node = unary_input(node)
                norm = module_at(norm_node)
                if (type(norm) not in _NORMS or len(norm_node.users) != 1
                        or tuple(norm.normalized_shape) != (linear.in_features,)
                        or norm.weight is None or norm_node.target not in stats):
                    continue
                a = torch.as_tensor(stats[norm_node.target], device=linear.weight.device,
                                    dtype=torch.float64)
                if (a.shape != (linear.in_features,) or not torch.isfinite(a).all()
                        or (a < 0).any()):
                    raise ValueError("activation_max[%r] must be finite nonnegative "
                                     "per-channel maxima" % norm_node.target)
                b = linear.weight.detach().double().abs().amax(0)
                if not torch.isfinite(b).all():
                    continue
                scale = (a / b.clamp_min(torch.finfo(b.dtype).tiny)).sqrt()
                scale = scale.clamp(min=1, max=100)
                norm.weight.div_(scale.to(norm.weight))
                if getattr(norm, "bias", None) is not None:
                    norm.bias.div_(scale.to(norm.bias))
                linear.weight.mul_(scale.to(linear.weight).unsqueeze(0))
                report.append({"op": "smooth", "norm": norm_node.target,
                               "consumer": node.target})

        graph.graph.lint()
        graph.recompile()
    check_example(graph)
    return graph, report
