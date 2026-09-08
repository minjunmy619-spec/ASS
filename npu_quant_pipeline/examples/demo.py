"""Synthetic smoke test. Install the project with pip install -e . first."""

import argparse

import torch
from torch import nn

from npu_quant import NPUQuantizer, QuantConfig, layer_sensitivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", help="Optional destination for optimized float ONNX")
    args = parser.parse_args()
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Conv1d(2, 8, 3, padding=1), nn.BatchNorm1d(8),
        nn.ReLU(), nn.Conv1d(8, 4, 1),
    ).eval()
    with torch.no_grad():
        model[0].weight[0].mul_(12)
        model[3].weight[:, 0].div_(12)
    calibration = [(torch.randn(2, 2, 128),) for _ in range(3)]
    validation = [(torch.randn(1, 2, 160),) for _ in range(2)]
    result = NPUQuantizer(model, QuantConfig()).optimize(
        calibration, validation, adaround_iterations=10, bias_correction=True)
    print("Synthetic outputs, not measured separator quality")
    print("Baseline:", result.report["baseline"])
    print("Selected:", result.report["selected"])
    for stage in result.report["stages"]:
        print(stage["stage"], "accepted:", stage["accepted"])
    print("Sensitivity:", layer_sensitivity(model, result.simulation, validation))
    if args.export:
        result.export_onnx(validation[0], args.export,
                           input_names=["mixture"], output_names=["features"])
        result.save_encodings(args.export + ".encodings.json")
        print("Exported float ONNX and diagnostic encodings:", args.export)


if __name__ == "__main__":
    main()
