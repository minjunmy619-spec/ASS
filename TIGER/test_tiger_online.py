import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TIGER import (
    TIGERCtx,
    TIGERCtxDeployable,
    TIGERNPULargeCtx,
    TIGERNPULargeDeployable,
    TIGERCtxStreamingTrainingWrapper,
    TIGERCtxTigerLikeApprox,
    TIGERDeployable,
    TIGERTigerLikeApprox,
    TIGERStreamingTrainingWrapper,
    build_causal_ri_sequence,
    invert_causal_ri_sequence,
)


def build_test_window(kind: str, win: int, device=None, dtype=None) -> torch.Tensor | None:
    if kind == "none":
        return None
    if kind == "hann":
        return torch.hann_window(win, device=device, dtype=dtype)
    if kind == "sqrt_hann":
        return torch.sqrt(torch.hann_window(win, device=device, dtype=dtype).clamp_min(1e-8))
    raise ValueError(f"Unsupported window kind: {kind}")


class StreamingExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, subband_spec_RIs, past_kvs, past_valid_mask, prev_states_0, prev_states_1, prev_states_2, prev_global_states):
        return self.model.forward_cell(
            subband_spec_RIs,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            prev_states_0=prev_states_0,
            prev_states_1=prev_states_1,
            prev_states_2=prev_states_2,
            prev_global_states=prev_global_states,
        )


class StreamingCtxExportWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, subband_spec_RIs, past_kvs, past_valid_mask, time_ctx):
        return self.model.forward_cell(
            subband_spec_RIs,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            time_ctx=time_ctx,
        )



def run_sequence_consistency_test(model: torch.nn.Module, subband_spec_ri_seq: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        if isinstance(model, TIGERCtx):
            seq_out, seq_kv, seq_valid, seq_ctx = model.forward_sequence(subband_spec_ri_seq)

            cell_kv, cell_valid, cell_ctx = model.init_streaming_state(
                batch_size=subband_spec_ri_seq.shape[0],
                device=subband_spec_ri_seq.device,
                dtype=subband_spec_ri_seq.dtype,
            )
            cell_outputs = []
            for t in range(subband_spec_ri_seq.shape[-1]):
                cell_out, cell_kv, cell_valid, cell_ctx = model.forward_cell(
                    subband_spec_ri_seq[..., t:t + 1],
                    past_kvs=cell_kv,
                    past_valid_mask=cell_valid,
                    time_ctx=cell_ctx,
                )
                cell_outputs.append(cell_out)
            cell_out = torch.cat(cell_outputs, dim=-1)
            print(f"[consistency] final kv shape: {tuple(seq_kv.shape)}, valid mask shape: {tuple(seq_valid.shape)}")
            print(f"[consistency] final ctx: {tuple(seq_ctx.shape)}")
        else:
            seq_out, seq_kv, seq_valid, seq_s0, seq_s1, seq_s2, seq_sg = model.forward_sequence(subband_spec_ri_seq)

            cell_kv, cell_valid, cell_s0, cell_s1, cell_s2, cell_sg = model.init_streaming_state(
                batch_size=subband_spec_ri_seq.shape[0],
                device=subband_spec_ri_seq.device,
                dtype=subband_spec_ri_seq.dtype,
            )
            cell_outputs = []
            for t in range(subband_spec_ri_seq.shape[-1]):
                cell_out, cell_kv, cell_valid, cell_s0, cell_s1, cell_s2, cell_sg = model.forward_cell(
                    subband_spec_ri_seq[..., t:t + 1],
                    past_kvs=cell_kv,
                    past_valid_mask=cell_valid,
                    prev_states_0=cell_s0,
                    prev_states_1=cell_s1,
                    prev_states_2=cell_s2,
                    prev_global_states=cell_sg,
                )
                cell_outputs.append(cell_out)
            cell_out = torch.cat(cell_outputs, dim=-1)
            print(f"[consistency] final kv shape: {tuple(seq_kv.shape)}, valid mask shape: {tuple(seq_valid.shape)}")
            print(f"[consistency] final states: {tuple(seq_s0.shape)}, {tuple(seq_s1.shape)}, {tuple(seq_s2.shape)}, {tuple(seq_sg.shape)}")

        max_diff = (seq_out - cell_out).abs().max().item()
        print(f"[consistency] max sequence-vs-cell diff: {max_diff:.8f}")


def export_streaming_onnx(model: torch.nn.Module, export_path: Path) -> None:
    model.eval()
    with torch.no_grad():
        dummy_ri = torch.randn(1, 1, model.enc_dim * 2, 1, dtype=next(model.parameters()).dtype)
        if isinstance(model, TIGERCtx):
            past_kvs, past_valid_mask, time_ctx = model.init_streaming_state(batch_size=1)
            export_model = StreamingCtxExportWrapper(model)
            torch.onnx.export(
                export_model,
                (dummy_ri, past_kvs, past_valid_mask, time_ctx),
                str(export_path),
                export_params=True,
                opset_version=14,
                input_names=[
                    "subband_spec_RIs",
                    "past_kvs",
                    "past_valid_mask",
                    "time_ctx",
                ],
                output_names=[
                    "band_masked_output",
                    "new_kv",
                    "new_valid_mask",
                    "new_time_ctx",
                ],
            )
        else:
            past_kvs, past_valid_mask, s0, s1, s2, sg = model.init_streaming_state(batch_size=1)
            export_model = StreamingExportWrapper(model)
            torch.onnx.export(
                export_model,
                (dummy_ri, past_kvs, past_valid_mask, s0, s1, s2, sg),
                str(export_path),
                export_params=True,
                opset_version=14,
                input_names=[
                    "subband_spec_RIs",
                    "past_kvs",
                    "past_valid_mask",
                    "prev_states_0",
                    "prev_states_1",
                    "prev_states_2",
                    "prev_global_states",
                ],
                output_names=[
                    "band_masked_output",
                    "new_kv",
                    "new_valid_mask",
                    "new_states_0",
                    "new_states_1",
                    "new_states_2",
                    "new_global_states",
                ],
            )
        print(f"[onnx] exported to {export_path}")


def print_state_budget(model: torch.nn.Module, state_tuple) -> None:
    total_elements = 0
    for idx, state in enumerate(state_tuple):
        elems = state.numel()
        total_elements += elems
        print(f"[budget] state[{idx}] shape={tuple(state.shape)} elems={elems}")

    print(f"[budget] total elements={total_elements}")
    print(f"[budget] int8 bytes={total_elements}")
    print(f"[budget] int16 bytes={total_elements * 2}")
    print(f"[budget] fp32 bytes={total_elements * 4}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke tests for the refactored online TIGER models")
    parser.add_argument("--variant", choices=["deployable", "tiger_like", "ctx_deployable", "ctx_tiger_like", "npu_large"], default="deployable")
    parser.add_argument("--frames", type=int, default=6, help="Number of RI frames for the sequence test")
    parser.add_argument("--window", choices=["none", "hann", "sqrt_hann"], default="none")
    parser.add_argument("--roundtrip", action="store_true", help="Run waveform -> RI -> waveform sanity check")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--onnx-path", type=Path, default=REPO_ROOT / "TIGER" / "RefactoredTigerStreaming.onnx")
    args = parser.parse_args()

    model_map = {
        "deployable": (TIGERDeployable, TIGERStreamingTrainingWrapper),
        "tiger_like": (TIGERTigerLikeApprox, TIGERStreamingTrainingWrapper),
        "ctx_deployable": (TIGERCtxDeployable, TIGERCtxStreamingTrainingWrapper),
        "ctx_tiger_like": (TIGERCtxTigerLikeApprox, TIGERCtxStreamingTrainingWrapper),
        "npu_large": (TIGERNPULargeDeployable, TIGERCtxStreamingTrainingWrapper),
    }
    model_cls, wrapper_cls = model_map[args.variant]
    common_kwargs = dict(
        win=2048,
        stride=512,
        num_sources=3,
        sample_rate=44100,
        need_streaming=True,
    )
    if args.variant == "npu_large":
        model = model_cls(
            out_channels=192,
            in_channels=1024,
            upsampling_depth=5,
            att_n_head=4,
            att_hid_chan=8,
            num_stages=2,
            **common_kwargs,
        )
    else:
        model = model_cls(
            out_channels=132,
            in_channels=256,
            num_blocks=4,
            upsampling_depth=5,
            att_n_head=4,
            att_hid_chan=4,
            att_kernel_size=8,
            att_stride=1,
            **common_kwargs,
        )

    training_wrapper = wrapper_cls(model)
    waveform = torch.randn(1, 1, 256 + (args.frames - 1) * 512)
    analysis_window = build_test_window(args.window, model.win, device=waveform.device, dtype=waveform.dtype)
    synthesis_window = analysis_window
    subband_spec_ri_seq = build_causal_ri_sequence(
        waveform,
        win=model.win,
        hop=model.stride,
        startup_packet=256,
        analysis_window=analysis_window,
    )
    print(f"[input] waveform shape={tuple(waveform.shape)}")
    print(f"[input] RI sequence shape={tuple(subband_spec_ri_seq.shape)}")
    print(f"[input] window={args.window}")

    if args.roundtrip:
        reconstructed = invert_causal_ri_sequence(
            subband_spec_ri_seq,
            win=model.win,
            hop=model.stride,
            startup_packet=256,
            num_samples=waveform.shape[-1],
            analysis_window=analysis_window,
            synthesis_window=synthesis_window,
        )
        rt_err = (waveform - reconstructed).abs().max().item()
        print(f"[roundtrip] waveform max err={rt_err:.8f}")

    state_tuple = model.init_streaming_state(batch_size=1)
    print_state_budget(model, state_tuple)

    run_sequence_consistency_test(model, subband_spec_ri_seq)

    with torch.no_grad():
        wrapped_out, *_ = training_wrapper(subband_spec_ri_seq)
        print(f"[wrapper] output shape={tuple(wrapped_out.shape)}")

    if args.export_onnx:
        export_streaming_onnx(model, args.onnx_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
