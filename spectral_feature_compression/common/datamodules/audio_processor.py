import math

import torch

DEFAULT_SAMPLE_RATE = 48000
DEFAULT_N_FFT = 2048
DEFAULT_CONVERTED_FREQ_BINS = 512
DEFAULT_ORIGINAL_FREQ_BINS = 1024
DEFAULT_LINEAR_CUTOFF_RATIO = 10000.0 / 24000.0


def _default_linear_cutoff_hz(sample_rate):
    return (sample_rate / 2.0) * DEFAULT_LINEAR_CUTOFF_RATIO


def conv_freq_indices(
    sample_rate=DEFAULT_SAMPLE_RATE,
    n_fft=DEFAULT_N_FFT,
    converted_bins=DEFAULT_CONVERTED_FREQ_BINS,
    original_bins=DEFAULT_ORIGINAL_FREQ_BINS,
    linear_cutoff_hz=None,
):
    """Return source frequency-bin indices used by conv_freq.

    The returned list has length ``converted_bins``. Each value is a zero-based
    index into the original STFT frequency axis.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if n_fft <= 0:
        raise ValueError("n_fft must be positive")
    if converted_bins <= 0:
        raise ValueError("converted_bins must be positive")
    if original_bins <= 0:
        raise ValueError("original_bins must be positive")
    if converted_bins > original_bins:
        raise ValueError("converted_bins must be less than or equal to original_bins")

    freq_res = float(sample_rate) / float(n_fft)
    max_source_hz = (original_bins - 1) * freq_res
    if linear_cutoff_hz is None:
        linear_cutoff_hz = _default_linear_cutoff_hz(sample_rate)
    if linear_cutoff_hz <= 0:
        raise ValueError("linear_cutoff_hz must be positive")
    if linear_cutoff_hz >= max_source_hz:
        raise ValueError("linear_cutoff_hz must be lower than the highest source bin")

    linear_bins = int(linear_cutoff_hz / freq_res) + 1
    if linear_bins >= converted_bins:
        raise ValueError(
            "linear_cutoff_hz leaves no room for log-frequency bins; "
            "lower linear_cutoff_hz or increase converted_bins"
        )

    indices = list(range(linear_bins))
    log_bins = converted_bins - linear_bins
    start_hz = linear_bins * freq_res

    if log_bins == 1:
        indices.append(original_bins - 1)
        return indices

    log_base = max_source_hz / start_hz
    previous_idx = linear_bins - 1
    for i in range(log_bins):
        alpha = i / float(log_bins - 1)
        target_hz = start_hz * math.pow(log_base, alpha)
        source_idx = int(round(target_hz / freq_res))
        source_idx = max(source_idx, previous_idx + 1)
        source_idx = min(source_idx, original_bins - 1)
        indices.append(source_idx)
        previous_idx = source_idx

    return indices


def conv_freq(
    stft,
    sample_rate=DEFAULT_SAMPLE_RATE,
    n_fft=DEFAULT_N_FFT,
    converted_bins=DEFAULT_CONVERTED_FREQ_BINS,
    linear_cutoff_hz=None,
):
    original_bins = stft.shape[-1]
    indices = conv_freq_indices(
        sample_rate=sample_rate,
        n_fft=n_fft,
        converted_bins=converted_bins,
        original_bins=original_bins,
        linear_cutoff_hz=linear_cutoff_hz,
    )
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=stft.device)
    return stft.index_select(-1, index_tensor)


def inverse_conv_freq(
    converted_stft,
    sample_rate=DEFAULT_SAMPLE_RATE,
    n_fft=DEFAULT_N_FFT,
    original_bins=DEFAULT_ORIGINAL_FREQ_BINS,
    linear_cutoff_hz=None,
):
    converted_bins = converted_stft.shape[-1]
    indices = conv_freq_indices(
        sample_rate=sample_rate,
        n_fft=n_fft,
        converted_bins=converted_bins,
        original_bins=original_bins,
        linear_cutoff_hz=linear_cutoff_hz,
    )
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=converted_stft.device)
    stft = converted_stft.new_zeros((*converted_stft.shape[:-1], original_bins))
    return stft.index_copy(-1, index_tensor, converted_stft)


def deconv_freq(
    converted_stft,
    origin_stft=None,
    sample_rate=DEFAULT_SAMPLE_RATE,
    n_fft=DEFAULT_N_FFT,
    original_bins=None,
    linear_cutoff_hz=None,
):
    """Expand compressed bins into dense piecewise-constant frequency bands.

    Unlike ``inverse_conv_freq``, this fills the unsampled high-frequency bins
    between two selected source bins with the current compressed value.
    """
    if original_bins is None:
        if origin_stft is not None:
            original_bins = origin_stft.shape[-1]
        else:
            original_bins = DEFAULT_ORIGINAL_FREQ_BINS

    converted_bins = converted_stft.shape[-1]
    indices = conv_freq_indices(
        sample_rate=sample_rate,
        n_fft=n_fft,
        converted_bins=converted_bins,
        original_bins=original_bins,
        linear_cutoff_hz=linear_cutoff_hz,
    )

    if origin_stft is not None:
        reconstructed_stft = torch.zeros_like(origin_stft[..., :original_bins])
    else:
        reconstructed_stft = converted_stft.new_zeros(
            (*converted_stft.shape[:-1], original_bins)
        )

    previous_idx = -1
    for converted_idx, source_idx in enumerate(indices):
        start_idx = previous_idx + 1
        end_idx = source_idx + 1
        reconstructed_stft[..., start_idx:end_idx] = converted_stft[
            ..., converted_idx : converted_idx + 1
        ]
        previous_idx = source_idx

    return reconstructed_stft


#Below just for refer
"""
def conv_freq(stft):
    converted_stft=torch.zeros_like(stft[:, :, :, :512])
    converted_stft[:,:,:,:427]=stft[:,:,:,:427]
    
    freq_res=23.4375
    for y in range(428, 513):
        x=(1.0104**(y+463))
        z=int(x/freq_res)
        
        if z>=1024:
            z=1024
        converted_stft[:, :, :, (y-1)]=stft[:, :, :, (z-1)]
    return converted_stft

def deconv_freq(converted_stft, origin_stft):
    reconstructed_stft = torch.zeros_like(origin_stft[:, :, :, :1024])
    reconstructed_stft[:, :, :, :427]=converted_stft[:, :, :, :427]
    
    freq_res=23.4375
    
    for y in range(428, 513):
        x = (1.0104**(y+463))
        x_pri=(1.0104**((y-1)+463))
        z= int(x/freq_res)
        
        if z==428:
            z_pri=427
        else:
            z_pri=int(x_pri/freq_res)
        if z>=1024:
            z=1024
        for w in range(z-z_pri):
            reconstructed_stft[:, :, :, (z-w-1)]=converted_stft[:,:,:, (y-1)]
    return reconstructed_stft
"""