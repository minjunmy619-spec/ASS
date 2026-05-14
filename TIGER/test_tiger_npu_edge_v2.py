"""
Validation for ``TIGERNPUEdgeV2`` (NPU edge V2 preset).

Runs:
- NPU rule compliance checks (rules 1-14 from AGENT.md)
- conv receptive-field constraints
- fp32 streaming state byte budget (192 KiB target)
- no PReLU / no Dropout / no Conv1d / no AdaptivePool
- sequence vs single-frame cell numerical consistency
- ONNX export + onnx checker
- ONNX graph analysis (node count, memory ops reduction)
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TIGER.npu_edge_utils import (
    collect_conv_constraint_violations,
    streaming_state_bytes_fp32,
)
from TIGER.streaming_io import build_causal_ri_sequence
from TIGER.tiger_npu_edge_v2 import TIGERNPUEdgeV2, export_tiger_npu_edge_v2_onnx


def _assert_no_unsupported_modules(root: nn.Module) -> None:
    for name, m in root.named_modules():
        if isinstance(m, nn.PReLU):
            raise AssertionError(f"PReLU at {name}")
        if isinstance(m, nn.Dropout):
            raise AssertionError(f"Dropout at {name}")
        if isinstance(m, nn.Conv1d):
            raise AssertionError(f"Conv1d at {name} (rule2: avoid 1D ops)")
        if isinstance(m, nn.AdaptiveAvgPool1d):
            raise AssertionError(f"AdaptiveAvgPool1d at {name} (rule7)")
        if isinstance(m, nn.AdaptiveAvgPool2d):
            raise AssertionError(f"AdaptiveAvgPool2d at {name} (rule7)")


def run_sequence_vs_cell(model: TIGERNPUEdgeV2, subband_spec_ri_seq: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        seq_out, *_ = model.forward_sequence(subband_spec_ri_seq)

        past_kvs, past_valid_mask, time_ctx = model.init_streaming_state(
            batch_size=subband_spec_ri_seq.shape[0],
            device=subband_spec_ri_seq.device,
            dtype=subband_spec_ri_seq.dtype,
        )
        outs = []
        for t in range(subband_spec_ri_seq.shape[-1]):
            cell_out, past_kvs, past_valid_mask, time_ctx = model.forward_cell(
                subband_spec_ri_seq[..., t:t + 1],
                past_kvs=past_kvs,
                past_valid_mask=past_valid_mask,
                time_ctx=time_ctx,
            )
            outs.append(cell_out)
        cell_out = torch.cat(outs, dim=-1)

    return (seq_out - cell_out).abs().max().item()


def run_chunk_sizes_parity(
    model: TIGERNPUEdgeV2, subband_spec_ri_seq: torch.Tensor,
) -> Tuple[float, float]:
    """Compare chunk_size=1 (true frame-by-frame) against chunk_size in {4, 8}.

    chunk_size=1 uses ``forward_cell`` so it is the reference.  The larger
    chunk sizes go through ``_forward_chunk`` and must match within fp32
    accumulation noise.
    """
    model.eval()
    with torch.no_grad():
        ref_out, *_ = model.forward_sequence(subband_spec_ri_seq, chunk_size=1)
        chunk4_out, *_ = model.forward_sequence(subband_spec_ri_seq, chunk_size=4)
        chunk8_out, *_ = model.forward_sequence(subband_spec_ri_seq, chunk_size=8)
    return (
        (ref_out - chunk4_out).abs().max().item(),
        (ref_out - chunk8_out).abs().max().item(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate TIGERNPUEdgeV2")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--onnx-out", type=Path,
                        default=REPO_ROOT / "TIGER" / "artifacts" / "tiger_npu_edge_v2.onnx")
    parser.add_argument("--skip-onnx", action="store_true")
    args = parser.parse_args()

    print("[build] TIGERNPUEdgeV2")
    model = TIGERNPUEdgeV2()
    model.eval()

    print("[check] no unsupported modules (PReLU/Dropout/Conv1d/AdaptivePool)")
    _assert_no_unsupported_modules(model)

    print("[check] conv receptive field (k-1)*d < 14")
    bad = collect_conv_constraint_violations(model, limit=14)
    if bad:
        for v in bad:
            print(f"  VIOLATION: {v.module_name} axis={v.axis} eff={v.effective}")
        raise AssertionError("Conv constraint violations found")

    states = model.init_streaming_state(batch_size=1)
    nbytes = streaming_state_bytes_fp32(states)
    budget = 192 * 1024
    print(f"[budget] streaming state: {nbytes} bytes (limit {budget})")
    if nbytes > budget:
        raise AssertionError(f"State {nbytes} exceeds {budget} bytes")

    # Tensor dim check
    print("[check] all I/O tensors are 4D")
    dummy = torch.randn(1, 1, model.enc_dim * 2, 1)
    past_kvs, past_valid_mask, time_ctx = states
    with torch.no_grad():
        out, nk, nv, nc = model.forward_cell(dummy, past_kvs, past_valid_mask, time_ctx)
    for name, t in [("out", out), ("kv", nk), ("valid", nv), ("ctx", nc)]:
        assert t.dim() == 4, f"{name} has {t.dim()} dims, expected 4"

    # Sequence vs cell
    waveform = torch.randn(1, 1, 256 + (args.frames - 1) * model.stride)
    subband = build_causal_ri_sequence(waveform, win=model.win, hop=model.stride, startup_packet=256)
    print(f"[data] RI shape={tuple(subband.shape)}")

    max_diff = run_sequence_vs_cell(model, subband)
    print(f"[consistency] max |seq - cell| = {max_diff:.6e}")
    if max_diff > 1e-4:
        raise AssertionError(f"Mismatch too large: {max_diff}")

    # Chunked training parity: chunk_size=4 and chunk_size=8 must match
    # frame-by-frame (chunk_size=1) within fp32 tolerance.  This is what
    # lets training skip the per-frame Python loop without drifting from
    # the deployment (T=1) path.
    chunk4_diff, chunk8_diff = run_chunk_sizes_parity(model, subband)
    print(f"[chunk-parity] max |c=1 - c=4| = {chunk4_diff:.6e}")
    print(f"[chunk-parity] max |c=1 - c=8| = {chunk8_diff:.6e}")
    if chunk4_diff > 1e-4 or chunk8_diff > 1e-4:
        raise AssertionError(
            f"Chunked training parity broken: c4={chunk4_diff}, c8={chunk8_diff}"
        )

    # ONNX export
    if not args.skip_onnx:
        args.onnx_out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[onnx] export -> {args.onnx_out}")
        export_tiger_npu_edge_v2_onnx(model, args.onnx_out)

        import onnx
        onnx_model = onnx.load(str(args.onnx_out))
        onnx.checker.check_model(onnx_model)
        print("[onnx] checker passed")

        op_counts = Counter(node.op_type for node in onnx_model.graph.node)
        total_nodes = sum(op_counts.values())
        mem_ops = sum(op_counts.get(k, 0)
                      for k in ['Slice', 'Transpose', 'Concat', 'Gather', 'Reshape', 'Unsqueeze'])
        # Reference counts: V1 was ~5500 nodes with LayerNorm; the earlier
        # BatchNorm-V2 hit ~600 nodes but trained badly under T=1.
        # RMSNorm-V2 emits ~3 extra ops per norm site (~120 sites in the
        # default config), so the ceiling has been raised accordingly.
        print(f"[onnx] total nodes: {total_nodes} (V1 was 5524; BN-V2 was ~600)")
        print(f"[onnx] memory ops: {mem_ops} (V1 was 998)")
        print(f"[onnx] ReduceMean: {op_counts.get('ReduceMean', 0)} (V1 was 342)")
        print(f"[onnx] Tile: {op_counts.get('Tile', 0)} (V1 was 32)")

        forbidden_ops = {"Tile", "Expand", "ConstantOfShape"}
        found_forbidden = sorted(op for op in forbidden_ops if op_counts.get(op, 0))
        if found_forbidden:
            details = ", ".join(f"{op}={op_counts[op]}" for op in found_forbidden)
            raise AssertionError(f"Forbidden ONNX ops remain: {details}")

        # Rationale for the 1200 ceiling: BN-V2 exported at ~600 nodes.  Each
        # of the ~120 norm sites now emits ~3 extra ops, adding ~360 nodes in
        # the worst case, plus margin for compiler-version drift.
        if total_nodes > 1200:
            raise AssertionError(f"ONNX graph too large: {total_nodes} nodes")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[params] {total_params:,}")
    print("[ok] all V2 NPU edge checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
