"""GRU/LSTM-compatible recurrent modules implemented with Conv2D gates.

These classes are intended for NPU-friendly exports where ONNX ``GRU`` and
``LSTM`` operators are undesirable.  They keep the same forward signatures and
state shapes as ``torch.nn.GRU`` and ``torch.nn.LSTM`` for the supported causal
case, while implementing each gate as 1x1 ``Conv2d`` plus basic elementwise ops.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence


def _check_positive_int(value: int, *, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _check_dropout(value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"dropout must be in [0, 1], got {value}")
    return value


def _prepare_sequence(input: torch.Tensor, *, input_size: int, batch_first: bool) -> tuple[torch.Tensor, bool]:
    if isinstance(input, PackedSequence):
        raise NotImplementedError("PackedSequence input is not supported by Conv2dGRU/Conv2dLSTM.")
    if input.ndim == 2:
        if input.shape[-1] != input_size:
            raise ValueError(f"Expected input_size={input_size}, got last dim {input.shape[-1]}")
        return input.unsqueeze(1), True
    if input.ndim != 3:
        raise ValueError(f"Expected 2D or 3D input, got shape {tuple(input.shape)}")
    if input.shape[-1] != input_size:
        raise ValueError(f"Expected input_size={input_size}, got last dim {input.shape[-1]}")
    if batch_first:
        return input.transpose(0, 1), False
    return input, False


def _restore_output(output: torch.Tensor, *, unbatched: bool, batch_first: bool) -> torch.Tensor:
    if unbatched:
        return output.squeeze(1)
    if batch_first:
        return output.transpose(0, 1)
    return output


def _init_hidden(
    *,
    num_layers: int,
    batch_size: int,
    hidden_size: int,
    ref: torch.Tensor,
) -> torch.Tensor:
    return ref.new_zeros(num_layers, batch_size, hidden_size)


def _prepare_hidden(
    hidden: torch.Tensor | None,
    *,
    num_layers: int,
    batch_size: int,
    hidden_size: int,
    unbatched: bool,
    ref: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if hidden is None:
        return _init_hidden(num_layers=num_layers, batch_size=batch_size, hidden_size=hidden_size, ref=ref)
    if unbatched:
        expected = (num_layers, hidden_size)
        if tuple(hidden.shape) != expected:
            raise ValueError(f"Expected {name} shape {expected} for unbatched input, got {tuple(hidden.shape)}")
        return hidden.unsqueeze(1)
    expected = (num_layers, batch_size, hidden_size)
    if tuple(hidden.shape) != expected:
        raise ValueError(f"Expected {name} shape {expected}, got {tuple(hidden.shape)}")
    return hidden


def _restore_hidden(hidden: torch.Tensor, *, unbatched: bool) -> torch.Tensor:
    return hidden.squeeze(1) if unbatched else hidden


class _Conv2dGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, *, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.input_proj = nn.Conv2d(input_size, 3 * hidden_size, kernel_size=1, bias=bias, **factory_kwargs)
        self.hidden_proj = nn.Conv2d(hidden_size, 3 * hidden_size, kernel_size=1, bias=bias, **factory_kwargs)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        x_gates = self.input_proj(x)
        h_gates = self.hidden_proj(hidden)
        xr, xz, xn = torch.split(x_gates, self.hidden_size, dim=1)
        hr, hz, hn = torch.split(h_gates, self.hidden_size, dim=1)
        reset = torch.sigmoid(xr + hr)
        update = torch.sigmoid(xz + hz)
        candidate = torch.tanh(xn + reset * hn)
        return (1.0 - update) * candidate + update * hidden


class _Conv2dLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, *, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.input_proj = nn.Conv2d(input_size, 4 * hidden_size, kernel_size=1, bias=bias, **factory_kwargs)
        self.hidden_proj = nn.Conv2d(hidden_size, 4 * hidden_size, kernel_size=1, bias=bias, **factory_kwargs)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_gates = self.input_proj(x)
        h_gates = self.hidden_proj(hidden)
        xi, xf, xg, xo = torch.split(x_gates, self.hidden_size, dim=1)
        hi, hf, hg, ho = torch.split(h_gates, self.hidden_size, dim=1)
        input_gate = torch.sigmoid(xi + hi)
        forget_gate = torch.sigmoid(xf + hf)
        candidate = torch.tanh(xg + hg)
        output_gate = torch.sigmoid(xo + ho)
        new_cell = forget_gate * cell + input_gate * candidate
        new_hidden = output_gate * torch.tanh(new_cell)
        return new_hidden, new_cell


class Conv2dGRU(nn.Module):
    """Drop-in GRU-style module using Conv2D gates.

    Supported ``torch.nn.GRU`` features: batched/unbatched input, ``batch_first``,
    multiple layers, bias, dropout between layers, and explicit initial state.
    Unsupported by design: bidirectional recurrence and ``PackedSequence``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if bidirectional:
            raise NotImplementedError("Conv2dGRU is causal and does not support bidirectional=True.")
        self.input_size = _check_positive_int(input_size, name="input_size")
        self.hidden_size = _check_positive_int(hidden_size, name="hidden_size")
        self.num_layers = _check_positive_int(num_layers, name="num_layers")
        self.bias = bool(bias)
        self.batch_first = bool(batch_first)
        self.dropout = _check_dropout(dropout)
        self.bidirectional = False

        cells = []
        for layer_idx in range(self.num_layers):
            layer_input_size = self.input_size if layer_idx == 0 else self.hidden_size
            cells.append(
                _Conv2dGRUCell(
                    layer_input_size,
                    self.hidden_size,
                    bias=self.bias,
                    device=device,
                    dtype=dtype,
                )
            )
        self.cells = nn.ModuleList(cells)

    @property
    def all_weights(self) -> list[list[torch.nn.Parameter]]:
        weights = []
        for cell in self.cells:
            layer_weights = [cell.input_proj.weight, cell.hidden_proj.weight]
            if self.bias:
                layer_weights.extend([cell.input_proj.bias, cell.hidden_proj.bias])
            weights.append(layer_weights)
        return weights

    def forward(self, input: torch.Tensor, h_0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        seq, unbatched = _prepare_sequence(input, input_size=self.input_size, batch_first=self.batch_first)
        seq_len, batch_size, _ = seq.shape
        hidden = _prepare_hidden(
            h_0,
            num_layers=self.num_layers,
            batch_size=batch_size,
            hidden_size=self.hidden_size,
            unbatched=unbatched,
            ref=seq,
            name="h_0",
        )

        layer_input = seq
        final_hidden = []
        for layer_idx, cell in enumerate(self.cells):
            h_t = hidden[layer_idx].reshape(batch_size, self.hidden_size, 1, 1)
            outputs = []
            for frame_idx in range(seq_len):
                x_t = layer_input[frame_idx].reshape(batch_size, cell.input_size, 1, 1)
                h_t = cell(x_t, h_t)
                outputs.append(h_t.reshape(batch_size, self.hidden_size))
            layer_output = torch.stack(outputs, dim=0)
            if self.dropout > 0.0 and layer_idx < self.num_layers - 1:
                layer_output = F.dropout(layer_output, p=self.dropout, training=self.training)
            layer_input = layer_output
            final_hidden.append(h_t.reshape(batch_size, self.hidden_size))

        output = _restore_output(layer_input, unbatched=unbatched, batch_first=self.batch_first)
        h_n = _restore_hidden(torch.stack(final_hidden, dim=0), unbatched=unbatched)
        return output, h_n

    @torch.no_grad()
    def copy_from_torch(self, module: nn.GRU) -> Conv2dGRU:
        _copy_gru_weights(module, self)
        return self

    @classmethod
    def from_torch(cls, module: nn.GRU) -> Conv2dGRU:
        out = cls(
            module.input_size,
            module.hidden_size,
            num_layers=module.num_layers,
            bias=module.bias,
            batch_first=module.batch_first,
            dropout=module.dropout,
            bidirectional=module.bidirectional,
            device=next(module.parameters()).device,
            dtype=next(module.parameters()).dtype,
        )
        return out.copy_from_torch(module)


class Conv2dLSTM(nn.Module):
    """Drop-in LSTM-style module using Conv2D gates.

    Supported ``torch.nn.LSTM`` features: batched/unbatched input,
    ``batch_first``, multiple layers, bias, dropout between layers, and explicit
    ``(h_0, c_0)`` state. Unsupported by design: bidirectional recurrence,
    projected LSTM, and ``PackedSequence``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if bidirectional:
            raise NotImplementedError("Conv2dLSTM is causal and does not support bidirectional=True.")
        if int(proj_size) != 0:
            raise NotImplementedError("Conv2dLSTM does not support proj_size.")
        self.input_size = _check_positive_int(input_size, name="input_size")
        self.hidden_size = _check_positive_int(hidden_size, name="hidden_size")
        self.num_layers = _check_positive_int(num_layers, name="num_layers")
        self.bias = bool(bias)
        self.batch_first = bool(batch_first)
        self.dropout = _check_dropout(dropout)
        self.bidirectional = False
        self.proj_size = 0

        cells = []
        for layer_idx in range(self.num_layers):
            layer_input_size = self.input_size if layer_idx == 0 else self.hidden_size
            cells.append(
                _Conv2dLSTMCell(
                    layer_input_size,
                    self.hidden_size,
                    bias=self.bias,
                    device=device,
                    dtype=dtype,
                )
            )
        self.cells = nn.ModuleList(cells)

    @property
    def all_weights(self) -> list[list[torch.nn.Parameter]]:
        weights = []
        for cell in self.cells:
            layer_weights = [cell.input_proj.weight, cell.hidden_proj.weight]
            if self.bias:
                layer_weights.extend([cell.input_proj.bias, cell.hidden_proj.bias])
            weights.append(layer_weights)
        return weights

    def forward(
        self,
        input: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        seq, unbatched = _prepare_sequence(input, input_size=self.input_size, batch_first=self.batch_first)
        seq_len, batch_size, _ = seq.shape
        if hx is None:
            hidden = _init_hidden(
                num_layers=self.num_layers,
                batch_size=batch_size,
                hidden_size=self.hidden_size,
                ref=seq,
            )
            cell_state = torch.zeros_like(hidden)
        else:
            hidden = _prepare_hidden(
                hx[0],
                num_layers=self.num_layers,
                batch_size=batch_size,
                hidden_size=self.hidden_size,
                unbatched=unbatched,
                ref=seq,
                name="h_0",
            )
            cell_state = _prepare_hidden(
                hx[1],
                num_layers=self.num_layers,
                batch_size=batch_size,
                hidden_size=self.hidden_size,
                unbatched=unbatched,
                ref=seq,
                name="c_0",
            )

        layer_input = seq
        final_hidden = []
        final_cell = []
        for layer_idx, cell in enumerate(self.cells):
            h_t = hidden[layer_idx].reshape(batch_size, self.hidden_size, 1, 1)
            c_t = cell_state[layer_idx].reshape(batch_size, self.hidden_size, 1, 1)
            outputs = []
            for frame_idx in range(seq_len):
                x_t = layer_input[frame_idx].reshape(batch_size, cell.input_size, 1, 1)
                h_t, c_t = cell(x_t, h_t, c_t)
                outputs.append(h_t.reshape(batch_size, self.hidden_size))
            layer_output = torch.stack(outputs, dim=0)
            if self.dropout > 0.0 and layer_idx < self.num_layers - 1:
                layer_output = F.dropout(layer_output, p=self.dropout, training=self.training)
            layer_input = layer_output
            final_hidden.append(h_t.reshape(batch_size, self.hidden_size))
            final_cell.append(c_t.reshape(batch_size, self.hidden_size))

        output = _restore_output(layer_input, unbatched=unbatched, batch_first=self.batch_first)
        h_n = _restore_hidden(torch.stack(final_hidden, dim=0), unbatched=unbatched)
        c_n = _restore_hidden(torch.stack(final_cell, dim=0), unbatched=unbatched)
        return output, (h_n, c_n)

    @torch.no_grad()
    def copy_from_torch(self, module: nn.LSTM) -> Conv2dLSTM:
        _copy_lstm_weights(module, self)
        return self

    @classmethod
    def from_torch(cls, module: nn.LSTM) -> Conv2dLSTM:
        out = cls(
            module.input_size,
            module.hidden_size,
            num_layers=module.num_layers,
            bias=module.bias,
            batch_first=module.batch_first,
            dropout=module.dropout,
            bidirectional=module.bidirectional,
            proj_size=module.proj_size,
            device=next(module.parameters()).device,
            dtype=next(module.parameters()).dtype,
        )
        return out.copy_from_torch(module)


def _copy_rnn_layer(
    *,
    src: nn.Module,
    dst_cell: nn.Module,
    layer_idx: int,
    gate_count: int,
    expected_bias: bool,
    reshape_weight: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    dst_cell.input_proj.weight.copy_(reshape_weight(getattr(src, f"weight_ih_l{layer_idx}")))
    dst_cell.hidden_proj.weight.copy_(reshape_weight(getattr(src, f"weight_hh_l{layer_idx}")))
    if expected_bias:
        dst_cell.input_proj.bias.copy_(getattr(src, f"bias_ih_l{layer_idx}"))
        dst_cell.hidden_proj.bias.copy_(getattr(src, f"bias_hh_l{layer_idx}"))
    del gate_count


def _validate_torch_rnn(src: nn.Module, dst: nn.Module, *, kind: str) -> None:
    if bool(src.bidirectional):
        raise NotImplementedError(f"Cannot copy bidirectional {kind}.")
    if src.input_size != dst.input_size or src.hidden_size != dst.hidden_size or src.num_layers != dst.num_layers:
        raise ValueError("Source and destination RNN dimensions do not match.")
    if bool(src.bias) != bool(dst.bias):
        raise ValueError("Source and destination bias settings do not match.")
    if bool(src.batch_first) != bool(dst.batch_first):
        raise ValueError("Source and destination batch_first settings do not match.")


def _copy_gru_weights(src: nn.GRU, dst: Conv2dGRU) -> None:
    _validate_torch_rnn(src, dst, kind="GRU")
    for layer_idx, cell in enumerate(dst.cells):
        _copy_rnn_layer(
            src=src,
            dst_cell=cell,
            layer_idx=layer_idx,
            gate_count=3,
            expected_bias=dst.bias,
            reshape_weight=lambda w: w.reshape(w.shape[0], w.shape[1], 1, 1),
        )


def _copy_lstm_weights(src: nn.LSTM, dst: Conv2dLSTM) -> None:
    if int(src.proj_size) != 0:
        raise NotImplementedError("Cannot copy projected LSTM weights.")
    _validate_torch_rnn(src, dst, kind="LSTM")
    for layer_idx, cell in enumerate(dst.cells):
        _copy_rnn_layer(
            src=src,
            dst_cell=cell,
            layer_idx=layer_idx,
            gate_count=4,
            expected_bias=dst.bias,
            reshape_weight=lambda w: w.reshape(w.shape[0], w.shape[1], 1, 1),
        )


__all__ = ["Conv2dGRU", "Conv2dLSTM"]
