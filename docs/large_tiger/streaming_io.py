import torch
import torch.nn.functional as F


def _prepare_window(window, win: int, device, dtype, name: str) -> torch.Tensor:
    if window is None:
        return torch.ones(win, device=device, dtype=dtype)

    if not torch.is_tensor(window):
        window = torch.as_tensor(window, device=device, dtype=dtype)
    else:
        window = window.to(device=device, dtype=dtype)

    if window.dim() != 1 or window.numel() != win:
        raise ValueError(f"{name} must be a 1-D tensor of length {win}, got shape {tuple(window.shape)}")

    return window.contiguous()


def build_causal_ri_sequence(
    waveform: torch.Tensor,
    win: int = 2048,
    hop: int = 512,
    startup_packet: int = 256,
    analysis_window: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Build the RI sequence expected by the online TIGER training wrapper.

    The sequence follows the 256-in / 512-infer schedule:
    - first frame uses 256 real samples + zero history
    - subsequent frames advance by 512 samples

    Args:
        waveform: [B, 1, num_samples]
        analysis_window: optional 1-D analysis window of length `win`.
            If omitted, a rectangular window is used.

    Returns:
        Tensor of shape [B, 1, 2 * (win // 2 + 1), T]
    """
    batch, channels, num_samples = waveform.shape
    assert channels == 1, "This helper expects mono waveforms"
    assert 0 < startup_packet <= win, "startup_packet must be in (0, win]"

    enc_dim = win // 2 + 1
    if num_samples < startup_packet:
        return waveform.new_zeros(batch, 1, 2 * enc_dim, 1)

    left_pad = win - startup_packet
    mono = waveform[:, 0, :]
    padded = F.pad(mono, (left_pad, 0))

    frames = padded.unfold(-1, win, hop).contiguous()  # [B, T, win]
    window = _prepare_window(analysis_window, win, frames.device, frames.dtype, "analysis_window")
    frames = frames * window.view(1, 1, win)
    spec = torch.fft.rfft(frames, n=win, dim=-1)  # [B, T, enc_dim]
    spec_ri = torch.cat([spec.real, spec.imag], dim=-1)  # [B, T, 2*enc_dim]
    return spec_ri.permute(0, 2, 1).unsqueeze(1).contiguous()


def invert_causal_ri_sequence(
    subband_spec_ri_seq: torch.Tensor,
    win: int = 2048,
    hop: int = 512,
    startup_packet: int = 256,
    num_samples: int | None = None,
    eps: float = 1e-8,
    analysis_window: torch.Tensor | None = None,
    synthesis_window: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Inverse of `build_causal_ri_sequence` for postprocessing.

    Args:
        subband_spec_ri_seq: [B, C, 2 * (win // 2 + 1), T]
        num_samples: optional target waveform length after removing the causal
            left padding. Defaults to the natural covered length:
            `startup_packet + (T - 1) * hop`.
        analysis_window: optional analysis window used in preprocessing.
        synthesis_window: optional synthesis window applied before overlap-add.
            For windowed processing, use a reconstruction-compatible pair.
            In practice, `sqrt(hann)` for both analysis and synthesis is a
            safer default than plain `hann` under overlap-add.

    Returns:
        Tensor of shape [B, C, num_samples]

    Notes:
        `build_causal_ri_sequence()` emits frames ending at:
        `startup_packet - 1 + k * hop`.
        So the naturally covered waveform length is:
        `startup_packet + (T - 1) * hop`.
        If the original waveform length is not aligned to that schedule, the
        uncovered tail is not represented in the RI sequence and therefore
        cannot be reconstructed here unless the input was padded beforehand.
    """
    batch, channels, ri_dim, num_frames = subband_spec_ri_seq.shape
    enc_dim = win // 2 + 1
    assert ri_dim == 2 * enc_dim, f"expected RI dim {2 * enc_dim}, got {ri_dim}"
    assert 0 < startup_packet <= win, "startup_packet must be in (0, win]"

    if num_frames == 0:
        if num_samples is None:
            num_samples = 0
        return subband_spec_ri_seq.new_zeros(batch, channels, num_samples)

    if num_samples is None:
        num_samples = startup_packet + (num_frames - 1) * hop

    left_pad = win - startup_packet
    padded_length = win + (num_frames - 1) * hop
    max_supported_samples = padded_length - left_pad
    assert num_samples <= max_supported_samples, (
        f"num_samples={num_samples} exceeds the supported reconstructed length "
        f"{max_supported_samples} for T={num_frames}, win={win}, hop={hop}, startup_packet={startup_packet}"
    )

    real = subband_spec_ri_seq[:, :, :enc_dim, :]
    imag = subband_spec_ri_seq[:, :, enc_dim:, :]
    complex_spec = torch.complex(real, imag).permute(0, 1, 3, 2).contiguous()

    frames = torch.fft.irfft(complex_spec, n=win, dim=-1)
    analysis = _prepare_window(analysis_window, win, frames.device, frames.real.dtype, "analysis_window")
    synthesis = _prepare_window(synthesis_window, win, frames.device, frames.real.dtype, "synthesis_window")
    frames = frames * synthesis.view(1, 1, 1, win)
    frames = frames.reshape(batch * channels, num_frames, win).transpose(1, 2).contiguous()

    ola = F.fold(
        frames,
        output_size=(1, padded_length),
        kernel_size=(1, win),
        stride=(1, hop),
    ).squeeze(2).squeeze(1)

    overlap_window = (analysis * synthesis).view(1, 1, win, 1).expand(batch * channels, 1, win, num_frames)
    overlap_frames = overlap_window.reshape(batch * channels, win, num_frames).contiguous()
    overlap = F.fold(
        overlap_frames,
        output_size=(1, padded_length),
        kernel_size=(1, win),
        stride=(1, hop),
    ).squeeze(2).squeeze(1)

    waveform = (ola / overlap.clamp_min(eps)).reshape(batch, channels, padded_length)
    return waveform[:, :, left_pad:left_pad + num_samples].contiguous()
