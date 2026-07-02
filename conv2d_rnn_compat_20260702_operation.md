# Conv2D GRU/LSTM Compatibility Module - 2026-07-02

## Purpose

Add GRU/LSTM-style recurrent modules whose public input/output signatures match
`torch.nn.GRU` and `torch.nn.LSTM`, while the gate math uses only 1x1 `Conv2d`
and basic elementwise ops. This avoids ONNX `GRU` / `LSTM` operators for NPU
export experiments.

## Files

- `spectral_feature_compression/core/model/conv2d_rnn_compat.py`
  - `Conv2dGRU`
  - `Conv2dLSTM`
- `tests/test_conv2d_rnn_compat.py`
- `spectral_feature_compression/__init__.py`
  - lazy exports for `Conv2dGRU` and `Conv2dLSTM`

## API

GRU:

```python
from spectral_feature_compression.core.model.conv2d_rnn_compat import Conv2dGRU

gru = Conv2dGRU(
    input_size=128,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
)
y, h_n = gru(x, h_0)
```

LSTM:

```python
from spectral_feature_compression.core.model.conv2d_rnn_compat import Conv2dLSTM

lstm = Conv2dLSTM(
    input_size=128,
    hidden_size=128,
    num_layers=2,
    batch_first=True,
)
y, (h_n, c_n) = lstm(x, (h_0, c_0))
```

Supported input/state forms follow PyTorch:

- batched, time-first input: `(L, N, input_size)`
- batched, batch-first input: `(N, L, input_size)`
- unbatched input: `(L, input_size)`
- GRU state: `(num_layers, N, hidden_size)` or `(num_layers, hidden_size)`
- LSTM state pair with the same hidden/cell shapes

Unsupported by design:

- `bidirectional=True`
- `PackedSequence`
- LSTM `proj_size`

## Weight Copy Helper

The modules can be initialized from a native PyTorch recurrent module:

```python
native = torch.nn.GRU(128, 128, num_layers=2, batch_first=True)
conv = Conv2dGRU.from_torch(native)
```

The gate order matches PyTorch:

- GRU: reset, update, new
- LSTM: input, forget, cell, output

## Validation

Commands:

```bash
.venv/bin/python -m pytest tests/test_conv2d_rnn_compat.py -q

.venv/bin/python -m ruff check \
  spectral_feature_compression/core/model/conv2d_rnn_compat.py \
  tests/test_conv2d_rnn_compat.py \
  spectral_feature_compression/__init__.py

PYTHONPYCACHEPREFIX=/tmp/ass_pycache .venv/bin/python -m py_compile \
  spectral_feature_compression/core/model/conv2d_rnn_compat.py \
  tests/test_conv2d_rnn_compat.py
```

Results:

```text
pytest: 5 passed
ruff: All checks passed
py_compile: pass
```

The ONNX smoke test verifies that `Conv2dGRU` export contains `Conv` and does
not contain ONNX `GRU` or `LSTM` operators.

## NPU Efficiency Notes

The implementation is NPU-friendly at the operator level:

- no ONNX `GRU` or `LSTM` op
- no rank > 4 tensors inside the cell math
- gate projections are 1x1 `Conv2d`
- gate slicing uses fixed-size `Split`
- remaining ops are elementwise add/mul/sigmoid/tanh/reshape/concat

However, the PyTorch-compatible sequence API necessarily loops over sequence
frames in Python. ONNX export unrolls that loop for the fixed exported sequence
length, so node count grows linearly with `seq_len * num_layers`.

Measured with `input_size=64`, `hidden_size=64`, `num_layers=1`,
`batch_first=True`:

```text
Conv2dGRU  seq=1   35 nodes, 2 Conv
Conv2dGRU  seq=4  113 nodes, 8 Conv
Conv2dGRU  seq=16 425 nodes, 32 Conv

Conv2dLSTM seq=1   44 nodes, 2 Conv
Conv2dLSTM seq=4  125 nodes, 8 Conv
Conv2dLSTM seq=16 449 nodes, 32 Conv
```

For TV NPU deployment, export the recurrent module as a single streaming step
(`seq_len=1`) with hidden/cell state as explicit inputs/outputs. Exporting long
sequences is useful for parity tests, but it is not the small-node graph form.
