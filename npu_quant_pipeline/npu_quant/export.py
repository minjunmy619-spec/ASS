"""Float ONNX export and module-level simulation encoding reports.

These utilities do not produce a QDQ or integer graph or a vendor configuration.
"""

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json
import os

import torch

from .quantization import QuantizedLayer, quantization_report


def export_onnx(model, example_args: tuple, path, *, input_names=None,
                output_names=None, dynamic_axes=None, opset_version=17,
                validate=True) -> None:
    """Export a float model (including an optimized float model) using legacy ONNX.

    Requires the optional ``onnx`` package, not ``onnxscript``. Optimization is
    the caller's responsibility. All QuantizedLayer simulations are rejected,
    even disabled ones. Evaluation and tracing operate on a deep copy.
    """
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "export_onnx requires the optional 'onnx' package; install it with "
            "`python -m pip install onnx`."
        ) from exc

    if not isinstance(example_args, tuple):
        raise TypeError("example_args must be a tuple of positional model arguments")
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLayer):
            raise ValueError(
                f"export_onnx supports float models only; QuantizedLayer at {name!r} "
                "is a simulation, not a QDQ or integer graph"
            )
    exported = deepcopy(model).eval()
    path = os.fspath(path)
    with torch.no_grad():
        torch.onnx.export(
            exported, example_args, path, input_names=input_names,
            output_names=output_names, dynamic_axes=dynamic_axes,
            opset_version=opset_version, dynamo=False,
        )
    if validate:
        onnx.checker.check_model(path)


def save_encodings(model, path, *, metadata=None) -> None:
    """Write simulation parameters and optional metadata as strict JSON.

    Metadata may contain tensors and dataclass instances (such as QuantConfig).
    Unsupported values and nonfinite numbers are rejected before opening path.
    """
    def encode(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    document = {
        "format": "npu_quant.encodings.v1",
        "scope": (
            "Module names identify PyTorch simulation modules, not an ONNX mapping "
            "or vendor config. This is not a QDQ or integer graph. Symmetric signed "
            "quantization uses a narrow-symmetric range: 8-bit [-127, 127], not "
            "[-128, 127]. These semantics are not guaranteed by any backend."
        ),
        "encodings": quantization_report(model),
        "metadata": metadata,
    }
    text = json.dumps(document, default=encode, allow_nan=False, indent=2)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(text + "\n")
