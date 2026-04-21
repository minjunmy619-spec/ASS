# Audio Processor Frequency Compression

`audio_processor.py` provides a small frequency-axis compressor for STFT-like
tensors. It is meant for tensors whose frequency bins are stored on the last
dimension:

```text
[..., frequency]
```

For example, a tensor may be shaped like:

```text
[batch, channels, frames, freq_bins]
```

## What `conv_freq` Does

`conv_freq` converts a dense linear STFT frequency axis into a smaller hybrid
axis:

1. Low-frequency bins are copied directly.
2. Higher-frequency bins are sampled with logarithmic spacing.

The default call is:

```python
compressed = conv_freq(stft)
```

By default this assumes:

```python
sample_rate = 48000
n_fft = 2048
converted_bins = 512
```

For a 48 kHz signal with `n_fft=2048`, the STFT bin resolution is:

```text
48000 / 2048 = 23.4375 Hz
```

The old implementation hardcoded this value as `23.4375` and used the
expression `1.0104 ** (y + 463)` for the high-frequency mapping. The updated
implementation derives the mapping from `sample_rate`, `n_fft`, the number of
source bins, and the requested number of compressed bins.

## Sample-Rate-Aware Mapping

The frequency resolution is computed as:

```python
freq_res = sample_rate / n_fft
```

The default linear cutoff scales with Nyquist:

```python
linear_cutoff_hz = (sample_rate / 2) * (10000 / 24000)
```

This preserves the old 48 kHz behavior:

```text
48000 Hz sample rate -> Nyquist 24000 Hz -> cutoff 10000 Hz
```

For other common sample rates, the default cutoff becomes:

```text
32000 Hz sample rate -> Nyquist 16000 Hz -> cutoff 6666.67 Hz
44100 Hz sample rate -> Nyquist 22050 Hz -> cutoff 9187.50 Hz
48000 Hz sample rate -> Nyquist 24000 Hz -> cutoff 10000.00 Hz
```

This keeps approximately the same proportion of the spectrum linear before the
log-frequency compression begins.

## Usage

For 48 kHz audio:

```python
compressed = conv_freq(stft, sample_rate=48000, n_fft=2048)
restored = inverse_conv_freq(compressed, sample_rate=48000, n_fft=2048)
```

For 44.1 kHz audio:

```python
compressed = conv_freq(stft, sample_rate=44100, n_fft=2048)
restored = inverse_conv_freq(compressed, sample_rate=44100, n_fft=2048)
```

For 32 kHz audio:

```python
compressed = conv_freq(stft, sample_rate=32000, n_fft=2048)
restored = inverse_conv_freq(compressed, sample_rate=32000, n_fft=2048)
```

For a custom fixed cutoff:

```python
compressed = conv_freq(
    stft,
    sample_rate=48000,
    n_fft=2048,
    converted_bins=512,
    linear_cutoff_hz=8000,
)
```

Use a lower `linear_cutoff_hz` when the sample rate is lower or when
`converted_bins` is small. If the cutoff consumes all compressed bins, the
function raises a `ValueError` instead of silently producing a bad mapping.

## Usage Scenarios

These helpers are useful when a model should operate on fewer frequency bins
than the original STFT while still preserving detailed low-frequency structure.

The common shape change is:

```text
original STFT:   [..., 1024]
compressed STFT: [..., 512]
```

Use `conv_freq` before a neural model:

```text
waveform -> STFT -> conv_freq -> neural model
```

This is useful because low frequencies usually need finer resolution, while high
frequencies can often be represented more coarsely. The compressed tensor is
cheaper for the model, and the high-frequency layout is closer to a perceptual
log-frequency scale than plain uniform downsampling.

Use `conv_freq_indices` when you need to inspect or reuse the exact mapping:

```python
indices = conv_freq_indices(sample_rate=48000, n_fft=2048)
```

Typical reasons to inspect the mapping:

1. Verify that 32 kHz, 44.1 kHz, and 48 kHz use the intended bins.
2. Plot the compressed frequency scale.
3. Ensure training and inference use identical mappings.
4. Debug mismatched model output shapes.

Use `inverse_conv_freq` for analysis or debugging when you want to see exactly
which original bins were retained:

```python
restored_sparse = inverse_conv_freq(
    compressed,
    sample_rate=48000,
    n_fft=2048,
    original_bins=1024,
)
```

This produces a sparse tensor. Sampled bins are restored, and unsampled bins are
zero.

Use `deconv_freq` when a downstream stage expects a dense STFT-like frequency
axis:

```python
restored_dense = deconv_freq(
    compressed,
    sample_rate=48000,
    n_fft=2048,
    original_bins=1024,
)
```

This is usually more practical before inverse STFT or when expanding a
compressed mask back onto the original mixture STFT.

A typical source-separation or masking pipeline is:

```python
stft = compute_stft(audio)

compressed_mix = conv_freq(
    stft,
    sample_rate=48000,
    n_fft=2048,
)

compressed_mask = model(compressed_mix)

dense_mask = deconv_freq(
    compressed_mask,
    origin_stft=stft,
    sample_rate=48000,
    n_fft=2048,
)

estimated_stft = stft * dense_mask
audio_hat = inverse_stft(estimated_stft)
```

The short rule is:

```text
conv_freq         -> compress original STFT for model input
conv_freq_indices -> inspect the source-bin mapping
inverse_conv_freq -> sparse restore; unsampled bins are zero
deconv_freq       -> dense restore; unsampled bins are filled by band
```

## Inspecting the Mapping

Use `conv_freq_indices` to see exactly which original frequency bins are used:

```python
indices = conv_freq_indices(sample_rate=48000, n_fft=2048)
print(indices[:10])
print(indices[-10:])
```

The returned list contains zero-based source-bin indices. Its length is equal to
`converted_bins`.

## Inverse Mapping

There are two inverse-style helpers.

`inverse_conv_freq` performs a sparse scatter back to the sampled source bins:

```python
restored = inverse_conv_freq(compressed, sample_rate=48000, n_fft=2048)
```

This operation is intentionally sparse. It places the available compressed bins
back into their source-bin locations and leaves unsampled high-frequency bins as
zero.

`deconv_freq` performs a dense piecewise-constant expansion:

```python
restored = deconv_freq(compressed, sample_rate=48000, n_fft=2048)
```

It uses the same source-bin mapping as `conv_freq`, but fills each gap between
two sampled high-frequency bins with the current compressed value. This is often
more convenient if a downstream step expects a dense spectrum:

```text
inverse_conv_freq -> sampled bins restored, unsampled bins are zero
deconv_freq       -> sampled bins restored, unsampled bins are filled by band
```

For compatibility with older call sites, `deconv_freq` also accepts an
`origin_stft` tensor:

```python
restored = deconv_freq(compressed, origin_stft)
```

When `origin_stft` is provided, its last dimension determines the restored
number of frequency bins.

If your model needs a dense reconstructed spectrum, add an interpolation or
learned reconstruction step after `inverse_conv_freq`, or use `deconv_freq` as a
simple nearest-band baseline.

## Important Parameters

`sample_rate`

Audio sample rate in Hz.

`n_fft`

FFT size used to produce the STFT. The bin resolution is `sample_rate / n_fft`.

`converted_bins`

Number of bins in the compressed frequency axis. The default is `512`.

`original_bins`

Number of bins in the original frequency axis. `conv_freq` reads this from
`stft.shape[-1]`. `inverse_conv_freq` defaults to `1024`, matching the original
48 kHz / 2048-point setup used by this file.

`linear_cutoff_hz`

Frequency below which bins are copied directly. If omitted, it scales with
Nyquist using the same proportion as the old 48 kHz mapping.

## Notes

The functions assume frequency is the last tensor dimension.

The mapping should use the same `sample_rate`, `n_fft`, `converted_bins`, and
`linear_cutoff_hz` during compression and inverse expansion.

For one-sided STFTs that include the Nyquist bin, `original_bins` may be
`n_fft // 2 + 1`. For setups that drop the Nyquist bin, `original_bins` is often
`n_fft // 2`.
