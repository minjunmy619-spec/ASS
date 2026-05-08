import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase


def _stack_complex(real, imag):
    return torch.cat([real, imag], dim=1)


def _temporal_film_from_conditioning(
    temporal_conditioner,
    temporal_conditioning,
    batch_size,
    n_queries,
    time_steps,
    device,
    dtype,
):
    if temporal_conditioning is None:
        return None, None
    cond = temporal_conditioning.to(device=device, dtype=dtype)
    if cond.dim() == 2:
        cond = cond.unsqueeze(1).expand(-1, n_queries, -1)
    if cond.dim() != 3:
        raise ValueError("temporal_conditioning must have shape [B, T] or [B, Q, T]")
    if cond.shape[0] != batch_size or cond.shape[1] != n_queries:
        raise ValueError("temporal_conditioning batch/query dimensions do not match TSE input")
    cond = F.interpolate(
        cond.reshape(batch_size * n_queries, 1, cond.shape[-1]),
        size=time_steps,
        mode="linear",
        align_corners=False,
    )
    beta_gamma = temporal_conditioner(cond)
    beta, gamma = beta_gamma.chunk(2, dim=1)
    return beta.unsqueeze(-1), gamma.unsqueeze(-1)


def _match_query_condition_shape(query_condition, label_vector, condition_dim):
    query_condition = query_condition.to(device=label_vector.device, dtype=label_vector.dtype)
    if label_vector.dim() == 2:
        if query_condition.dim() == 3:
            query_condition = query_condition[:, 0]
        if query_condition.dim() != 2:
            raise ValueError("query_condition must have shape [B,D] or [B,Q,D]")
        if query_condition.shape[0] != label_vector.shape[0]:
            raise ValueError("query_condition batch dimension does not match TSE input")
    elif label_vector.dim() == 3:
        if query_condition.dim() == 2:
            query_condition = query_condition.unsqueeze(1).expand(-1, label_vector.shape[1], -1)
        if query_condition.dim() != 3:
            raise ValueError("query_condition must have shape [B,D] or [B,Q,D]")
        if query_condition.shape[0] != label_vector.shape[0]:
            raise ValueError("query_condition batch dimension does not match TSE input")
        if query_condition.shape[1] < label_vector.shape[1]:
            pad = query_condition.new_zeros(
                query_condition.shape[0],
                label_vector.shape[1] - query_condition.shape[1],
                query_condition.shape[-1],
            )
            query_condition = torch.cat([query_condition, pad], dim=1)
        elif query_condition.shape[1] > label_vector.shape[1]:
            query_condition = query_condition[:, : label_vector.shape[1]]
    else:
        raise ValueError(f"Unsupported label_vector shape: {tuple(label_vector.shape)}")

    if query_condition.shape[-1] < condition_dim:
        pad = query_condition.new_zeros(*query_condition.shape[:-1], condition_dim - query_condition.shape[-1])
        query_condition = torch.cat([query_condition, pad], dim=-1)
    elif query_condition.shape[-1] > condition_dim:
        query_condition = query_condition[..., :condition_dim]
    return query_condition


class ResFiLM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x, beta=None, gamma=None):
        residual = x
        out = self.bn1(self.conv1(x))
        if beta is not None:
            out = out + beta
        if gamma is not None:
            out = out * (1.0 + gamma)
        out = F.leaky_relu(out, negative_slope=0.01)
        out = self.bn2(self.conv2(out))
        return residual + out


class DeFTBlock(nn.Module):
    """Modified DeFT block without Mamba, following the 2025 challenge report."""

    def __init__(self, channels, n_heads=4):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.freq_attn = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dim_feedforward=channels * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.time_attn = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dim_feedforward=channels * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.resfilm = ResFiLM(channels)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x, beta=None, gamma=None):
        batch_size, channels, time_steps, freq_bins = x.shape
        x = x + self.local(x)

        x_f = x.permute(0, 2, 3, 1).reshape(batch_size * time_steps, freq_bins, channels)
        x_f = self.freq_attn(x_f)
        x_f = x_f.reshape(batch_size, time_steps, freq_bins, channels).permute(0, 3, 1, 2)

        x_t = x.permute(0, 3, 2, 1).reshape(batch_size * freq_bins, time_steps, channels)
        x_t = self.time_attn(x_t)
        x_t = x_t.reshape(batch_size, freq_bins, time_steps, channels).permute(0, 3, 2, 1)

        x = self.norm(x + x_f + x_t)
        return self.resfilm(x, beta=beta, gamma=gamma)


class MemoryEfficientDeFTBlock(nn.Module):
    """DeFT block with local time attention and grouped frequency attention."""

    def __init__(
        self,
        channels,
        n_heads=4,
        time_window_size=128,
        freq_group_size=64,
        time_shift=0,
        freq_shift=0,
    ):
        super().__init__()
        self.time_window_size = int(time_window_size)
        self.freq_group_size = int(freq_group_size)
        self.time_shift = int(time_shift)
        self.freq_shift = int(freq_shift)
        if self.time_window_size <= 0 or self.freq_group_size <= 0:
            raise ValueError("time_window_size and freq_group_size must be positive")

        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.freq_attn = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dim_feedforward=channels * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.time_attn = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dim_feedforward=channels * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.resfilm = ResFiLM(channels)
        self.norm = nn.BatchNorm2d(channels)

    def _pad_to_multiple(self, length, block_size):
        return (block_size - length % block_size) % block_size

    def _grouped_freq_attention(self, x):
        batch_size, channels, time_steps, freq_bins = x.shape
        group_size = min(self.freq_group_size, freq_bins)
        shift = self.freq_shift if freq_bins > group_size else 0

        if shift:
            x = torch.roll(x, shifts=-shift, dims=3)
        pad = self._pad_to_multiple(freq_bins, group_size)
        if pad:
            x = F.pad(x, (0, pad))

        padded_freq = x.shape[-1]
        n_groups = padded_freq // group_size
        x_f = x.permute(0, 2, 3, 1).reshape(
            batch_size, time_steps, n_groups, group_size, channels
        )
        x_f = x_f.reshape(batch_size * time_steps * n_groups, group_size, channels)
        x_f = self.freq_attn(x_f)
        x_f = x_f.reshape(batch_size, time_steps, n_groups, group_size, channels)
        x_f = x_f.reshape(batch_size, time_steps, padded_freq, channels)
        x_f = x_f[:, :, :freq_bins].permute(0, 3, 1, 2)

        if shift:
            x_f = torch.roll(x_f, shifts=shift, dims=3)
        return x_f

    def _windowed_time_attention(self, x):
        batch_size, channels, time_steps, freq_bins = x.shape
        window_size = min(self.time_window_size, time_steps)
        shift = self.time_shift if time_steps > window_size else 0

        if shift:
            x = torch.roll(x, shifts=-shift, dims=2)
        pad = self._pad_to_multiple(time_steps, window_size)
        if pad:
            x = F.pad(x, (0, 0, 0, pad))

        padded_time = x.shape[-2]
        n_windows = padded_time // window_size
        x_t = x.permute(0, 3, 2, 1).reshape(
            batch_size, freq_bins, n_windows, window_size, channels
        )
        x_t = x_t.reshape(batch_size * freq_bins * n_windows, window_size, channels)
        x_t = self.time_attn(x_t)
        x_t = x_t.reshape(batch_size, freq_bins, n_windows, window_size, channels)
        x_t = x_t.reshape(batch_size, freq_bins, padded_time, channels)
        x_t = x_t[:, :, :time_steps].permute(0, 3, 2, 1)

        if shift:
            x_t = torch.roll(x_t, shifts=shift, dims=2)
        return x_t

    def forward(self, x, beta=None, gamma=None):
        x = x + self.local(x)
        x_f = self._grouped_freq_attention(x)
        x_t = self._windowed_time_attention(x)
        x = self.norm(x + x_f + x_t)
        return self.resfilm(x, beta=beta, gamma=gamma)


class ClassConditioner(nn.Module):
    def __init__(self, label_dim, channels):
        super().__init__()
        self.beta = nn.Linear(label_dim, channels)
        self.gamma = nn.Linear(label_dim, channels)

    def forward(self, label_onehot):
        beta = self.beta(label_onehot)[:, :, None, None]
        gamma = self.gamma(label_onehot)[:, :, None, None]
        return beta, gamma


class _BaseSpectralModel(nn.Module):
    def __init__(self, window_size=1024, hop_size=320):
        super().__init__()
        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )
        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window="hann",
            center=True,
            pad_mode="reflect",
            freeze_parameters=True,
        )

    def waveform_to_complex(self, waveform):
        real, imag = self.stft(waveform)
        return real, imag

    def complex_to_waveform(self, real, imag, length):
        return self.istft(real, imag, length)


class ModifiedDeFTUSS(_BaseSpectralModel):
    def __init__(
        self,
        input_channels=4,
        hidden_channels=96,
        n_deft_blocks=6,
        n_heads=4,
        n_foreground=3,
        n_interference=2,
        n_classes=18,
        window_size=1024,
        hop_size=320,
    ):
        super().__init__(window_size=window_size, hop_size=hop_size)
        self.input_channels = input_channels
        self.n_foreground = n_foreground
        self.n_interference = n_interference
        self.n_noise = 1
        self.n_objects = n_foreground + n_interference + self.n_noise
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [DeFTBlock(hidden_channels, n_heads=n_heads) for _ in range(n_deft_blocks)]
        )
        self.object_conv = nn.Conv2d(hidden_channels, hidden_channels * self.n_objects, kernel_size=1)
        self.audio_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, n_classes),
        )
        self.silence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, input_dict):
        mixture = input_dict["mixture"]
        batch_size, _, samples = mixture.shape
        real, imag = self.waveform_to_complex(mixture.reshape(-1, samples))
        _, _, time_steps, freq_bins = real.shape
        real = real.view(batch_size, self.input_channels, time_steps, freq_bins)
        imag = imag.view(batch_size, self.input_channels, time_steps, freq_bins)

        x = self.encoder(_stack_complex(real, imag))
        for block in self.blocks:
            x = block(x)
        x = self.object_conv(x)
        x = x.view(batch_size, self.n_objects, -1, time_steps, freq_bins)

        audio_mask = self.audio_head(x.reshape(batch_size * self.n_objects, -1, time_steps, freq_bins))
        audio_mask = torch.tanh(audio_mask)
        audio_mask = audio_mask.view(batch_size, self.n_objects, 2, time_steps, freq_bins)

        ref_real = real[:, :1].expand(-1, self.n_objects, -1, -1)
        ref_imag = imag[:, :1].expand(-1, self.n_objects, -1, -1)
        est_real = audio_mask[:, :, 0] * ref_real - audio_mask[:, :, 1] * ref_imag
        est_imag = audio_mask[:, :, 0] * ref_imag + audio_mask[:, :, 1] * ref_real

        waveform = self.complex_to_waveform(
            est_real.reshape(batch_size * self.n_objects, 1, time_steps, freq_bins),
            est_imag.reshape(batch_size * self.n_objects, 1, time_steps, freq_bins),
            samples,
        ).view(batch_size, self.n_objects, 1, samples)

        fg_features = x[:, : self.n_foreground]
        class_logits = self.class_head(fg_features.reshape(batch_size * self.n_foreground, -1, time_steps, freq_bins))
        class_logits = class_logits.view(batch_size, self.n_foreground, self.n_classes)
        silence_logits = self.silence_head(fg_features.reshape(batch_size * self.n_foreground, -1, time_steps, freq_bins))
        silence_logits = silence_logits.view(batch_size, self.n_foreground)

        return {
            "waveform": waveform,
            "foreground_waveform": waveform[:, : self.n_foreground],
            "interference_waveform": waveform[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_waveform": waveform[:, -1:],
            "class_logits": class_logits,
            "silence_logits": silence_logits,
        }


class ModifiedDeFTUSSSpatial(ModifiedDeFTUSS):
    """USS variant with spatial multi-channel masking and phase-aware synthesis.

    Compared with ``ModifiedDeFTUSS``, this predicts one magnitude mask plus a
    unit-normalized phase rotation for every input channel and object, then
    projects the multi-channel complex estimate to the mono output waveform.
    The public output dict is unchanged, so the existing USS loss/lightning code
    can consume this model directly.
    """

    def __init__(
        self,
        input_channels=4,
        output_channels=1,
        hidden_channels=96,
        n_deft_blocks=6,
        n_heads=4,
        n_foreground=3,
        n_interference=2,
        n_classes=18,
        window_size=1024,
        hop_size=320,
    ):
        super().__init__(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            n_deft_blocks=n_deft_blocks,
            n_heads=n_heads,
            n_foreground=n_foreground,
            n_interference=n_interference,
            n_classes=n_classes,
            window_size=window_size,
            hop_size=hop_size,
        )
        self.output_channels = output_channels
        self.mask_components = 3
        self.audio_head = nn.Conv2d(
            hidden_channels,
            input_channels * self.mask_components,
            kernel_size=1,
        )
        self.out_conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=1,
        )

    def _spatial_mask_to_waveform(self, object_features, real, imag, samples):
        batch_size, n_objects, _, time_steps, freq_bins = object_features.shape

        mask = self.audio_head(
            object_features.reshape(batch_size * n_objects, -1, time_steps, freq_bins)
        )
        mask = mask.view(
            batch_size,
            n_objects,
            self.input_channels,
            self.mask_components,
            time_steps,
            freq_bins,
        )

        mask_mag = torch.sigmoid(mask[:, :, :, 0])
        mask_real = torch.tanh(mask[:, :, :, 1])
        mask_imag = torch.tanh(mask[:, :, :, 2])
        _, mask_cos, mask_sin = magphase(mask_real, mask_imag)

        mixture_mag, mixture_cos, mixture_sin = magphase(real, imag)
        out_mag = F.relu(mixture_mag[:, None] * mask_mag)
        out_cos = mixture_cos[:, None] * mask_cos - mixture_sin[:, None] * mask_sin
        out_sin = mixture_sin[:, None] * mask_cos + mixture_cos[:, None] * mask_sin

        est_real = out_mag * out_cos
        est_imag = out_mag * out_sin

        est_real = est_real.reshape(
            batch_size * n_objects,
            self.input_channels,
            time_steps,
            freq_bins,
        )
        est_imag = est_imag.reshape(
            batch_size * n_objects,
            self.input_channels,
            time_steps,
            freq_bins,
        )

        est_real = self.out_conv(est_real)
        est_imag = self.out_conv(est_imag)

        waveform = self.complex_to_waveform(
            est_real.reshape(batch_size * n_objects * self.output_channels, 1, time_steps, freq_bins),
            est_imag.reshape(batch_size * n_objects * self.output_channels, 1, time_steps, freq_bins),
            samples,
        )
        return waveform.view(batch_size, n_objects, self.output_channels, samples)

    def forward(self, input_dict):
        mixture = input_dict["mixture"]
        batch_size, _, samples = mixture.shape
        real, imag = self.waveform_to_complex(mixture.reshape(-1, samples))
        _, _, time_steps, freq_bins = real.shape
        real = real.view(batch_size, self.input_channels, time_steps, freq_bins)
        imag = imag.view(batch_size, self.input_channels, time_steps, freq_bins)

        x = self.encoder(_stack_complex(real, imag))
        for block in self.blocks:
            x = block(x)
        x = self.object_conv(x)
        x = x.view(batch_size, self.n_objects, -1, time_steps, freq_bins)

        waveform = self._spatial_mask_to_waveform(x, real, imag, samples)

        fg_features = x[:, : self.n_foreground]
        class_logits = self.class_head(fg_features.reshape(batch_size * self.n_foreground, -1, time_steps, freq_bins))
        class_logits = class_logits.view(batch_size, self.n_foreground, self.n_classes)
        silence_logits = self.silence_head(fg_features.reshape(batch_size * self.n_foreground, -1, time_steps, freq_bins))
        silence_logits = silence_logits.view(batch_size, self.n_foreground)

        return {
            "waveform": waveform,
            "foreground_waveform": waveform[:, : self.n_foreground],
            "interference_waveform": waveform[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_waveform": waveform[:, -1:],
            "class_logits": class_logits,
            "silence_logits": silence_logits,
        }


class ModifiedDeFTUSSTemporal(ModifiedDeFTUSS):
    """Original modified DeFT USS with per-object temporal activity heads."""

    def __init__(self, *args, hidden_channels=96, sample_rate=32000, **kwargs):
        super().__init__(*args, hidden_channels=hidden_channels, **kwargs)
        self.sample_rate = sample_rate
        self.activity_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def _activity_logits(self, object_features):
        batch_size, n_objects, channels, time_steps, freq_bins = object_features.shape
        logits = self.activity_head(
            object_features.reshape(batch_size * n_objects, channels, time_steps, freq_bins)
        )
        logits = logits.mean(dim=-1).squeeze(1)
        return logits.view(batch_size, n_objects, time_steps)

    def forward(self, input_dict):
        mixture = input_dict["mixture"]
        batch_size, _, samples = mixture.shape
        real, imag = self.waveform_to_complex(mixture.reshape(-1, samples))
        _, _, time_steps, freq_bins = real.shape
        real = real.view(batch_size, self.input_channels, time_steps, freq_bins)
        imag = imag.view(batch_size, self.input_channels, time_steps, freq_bins)

        x = self.encoder(_stack_complex(real, imag))
        for block in self.blocks:
            x = block(x)
        x = self.object_conv(x)
        x = x.view(batch_size, self.n_objects, -1, time_steps, freq_bins)
        activity_logits = self._activity_logits(x)

        audio_mask = self.audio_head(x.reshape(batch_size * self.n_objects, -1, time_steps, freq_bins))
        audio_mask = torch.tanh(audio_mask)
        audio_mask = audio_mask.view(batch_size, self.n_objects, 2, time_steps, freq_bins)

        ref_real = real[:, :1].expand(-1, self.n_objects, -1, -1)
        ref_imag = imag[:, :1].expand(-1, self.n_objects, -1, -1)
        est_real = audio_mask[:, :, 0] * ref_real - audio_mask[:, :, 1] * ref_imag
        est_imag = audio_mask[:, :, 0] * ref_imag + audio_mask[:, :, 1] * ref_real

        waveform = self.complex_to_waveform(
            est_real.reshape(batch_size * self.n_objects, 1, time_steps, freq_bins),
            est_imag.reshape(batch_size * self.n_objects, 1, time_steps, freq_bins),
            samples,
        ).view(batch_size, self.n_objects, 1, samples)

        fg_features = x[:, : self.n_foreground]
        class_logits = self.class_head(fg_features.reshape(batch_size * self.n_foreground, -1, time_steps, freq_bins))
        class_logits = class_logits.view(batch_size, self.n_foreground, self.n_classes)
        silence_logits = self.silence_head(fg_features.reshape(batch_size * self.n_foreground, -1, time_steps, freq_bins))
        silence_logits = silence_logits.view(batch_size, self.n_foreground)
        duration_sec = mixture.new_full((batch_size,), float(samples) / float(self.sample_rate))

        return {
            "waveform": waveform,
            "foreground_waveform": waveform[:, : self.n_foreground],
            "interference_waveform": waveform[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_waveform": waveform[:, -1:],
            "class_logits": class_logits,
            "silence_logits": silence_logits,
            "foreground_activity_logits": activity_logits[:, : self.n_foreground],
            "interference_activity_logits": activity_logits[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_activity_logits": activity_logits[:, -1:],
            "duration_sec": duration_sec,
        }


class ModifiedDeFTUSSSpatialTemporal(ModifiedDeFTUSSSpatial):
    """Spatial modified DeFT USS with per-object temporal activity heads."""

    def __init__(self, *args, hidden_channels=96, sample_rate=32000, **kwargs):
        super().__init__(*args, hidden_channels=hidden_channels, **kwargs)
        self.sample_rate = sample_rate
        self.activity_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def _activity_logits(self, object_features):
        batch_size, n_objects, channels, time_steps, freq_bins = object_features.shape
        logits = self.activity_head(
            object_features.reshape(batch_size * n_objects, channels, time_steps, freq_bins)
        )
        logits = logits.mean(dim=-1).squeeze(1)
        return logits.view(batch_size, n_objects, time_steps)

    def _forward_full(self, input_dict):
        mixture = input_dict["mixture"]
        batch_size, _, samples = mixture.shape
        real, imag = self.waveform_to_complex(mixture.reshape(-1, samples))
        _, _, time_steps, freq_bins = real.shape
        real = real.view(batch_size, self.input_channels, time_steps, freq_bins)
        imag = imag.view(batch_size, self.input_channels, time_steps, freq_bins)

        x = self.encoder(_stack_complex(real, imag))
        for block in self.blocks:
            x = block(x)
        x = self.object_conv(x)
        x = x.view(batch_size, self.n_objects, -1, time_steps, freq_bins)

        waveform = self._spatial_mask_to_waveform(x, real, imag, samples)
        activity_logits = self._activity_logits(x)

        fg_features = x[:, : self.n_foreground]
        class_logits = self.class_head(fg_features.reshape(batch_size * self.n_foreground, -1, time_steps, freq_bins))
        class_logits = class_logits.view(batch_size, self.n_foreground, self.n_classes)
        silence_logits = self.silence_head(fg_features.reshape(batch_size * self.n_foreground, -1, time_steps, freq_bins))
        silence_logits = silence_logits.view(batch_size, self.n_foreground)
        duration_sec = mixture.new_full((batch_size,), float(samples) / float(self.sample_rate))

        return {
            "waveform": waveform,
            "foreground_waveform": waveform[:, : self.n_foreground],
            "interference_waveform": waveform[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_waveform": waveform[:, -1:],
            "class_logits": class_logits,
            "silence_logits": silence_logits,
            "foreground_activity_logits": activity_logits[:, : self.n_foreground],
            "interference_activity_logits": activity_logits[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_activity_logits": activity_logits[:, -1:],
            "duration_sec": duration_sec,
        }

    def forward(self, input_dict):
        return self._forward_full(input_dict)


class ChunkedModifiedDeFTUSSSpatial(ModifiedDeFTUSSSpatial):
    """Spatial USS with eval-time chunking for long fixed-length mixtures.

    Training uses the regular full forward pass. In eval mode, inputs longer
    than ``inference_chunk_seconds`` are split into overlapping chunks and
    waveform estimates are overlap-added back to the original length.
    """

    def __init__(
        self,
        *args,
        inference_chunk_seconds=6.0,
        inference_chunk_hop_seconds=5.0,
        sample_rate=32000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.inference_chunk_seconds = inference_chunk_seconds
        self.inference_chunk_hop_seconds = inference_chunk_hop_seconds
        self.sample_rate = sample_rate

    def _iter_chunk_starts(self, samples, chunk_samples, hop_samples):
        if samples <= chunk_samples:
            return [0]
        starts = list(range(0, max(samples - chunk_samples, 0) + 1, hop_samples))
        last_start = samples - chunk_samples
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _chunk_weight(self, chunk_samples, device, dtype):
        if chunk_samples <= 1:
            return torch.ones(1, device=device, dtype=dtype)
        weight = torch.hann_window(chunk_samples, periodic=False, device=device, dtype=dtype)
        return torch.clamp(weight, min=1e-3)

    def _chunked_forward(self, input_dict):
        mixture = input_dict["mixture"]
        batch_size, _, samples = mixture.shape
        chunk_samples = int(round(float(self.inference_chunk_seconds) * self.sample_rate))
        hop_samples = int(round(float(self.inference_chunk_hop_seconds) * self.sample_rate))
        if chunk_samples <= 0 or hop_samples <= 0:
            raise ValueError("inference_chunk_seconds and inference_chunk_hop_seconds must be positive")
        if samples <= chunk_samples:
            return super().forward(input_dict)

        starts = self._iter_chunk_starts(samples, chunk_samples, hop_samples)
        weight = self._chunk_weight(chunk_samples, mixture.device, mixture.dtype)
        weight = weight.view(1, 1, 1, chunk_samples)

        waveform_sum = None
        weight_sum = mixture.new_zeros(1, 1, 1, samples)
        class_logits = []
        silence_logits = []

        for start in starts:
            end = start + chunk_samples
            chunk = mixture[..., start:end]
            if chunk.shape[-1] < chunk_samples:
                chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))

            out = super().forward({"mixture": chunk})
            chunk_waveform = out["waveform"] * weight
            if waveform_sum is None:
                waveform_sum = mixture.new_zeros(
                    batch_size,
                    self.n_objects,
                    self.output_channels,
                    samples,
                )

            valid = min(chunk_samples, samples - start)
            waveform_sum[..., start : start + valid] += chunk_waveform[..., :valid]
            weight_sum[..., start : start + valid] += weight[..., :valid]
            class_logits.append(out["class_logits"])
            silence_logits.append(out["silence_logits"])

        waveform = waveform_sum / torch.clamp(weight_sum, min=1e-6)
        class_logits = torch.stack(class_logits, dim=0).mean(dim=0)
        silence_logits = torch.stack(silence_logits, dim=0).mean(dim=0)

        return {
            "waveform": waveform,
            "foreground_waveform": waveform[:, : self.n_foreground],
            "interference_waveform": waveform[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_waveform": waveform[:, -1:],
            "class_logits": class_logits,
            "silence_logits": silence_logits,
        }

    def forward(self, input_dict):
        if self.training or self.inference_chunk_seconds is None:
            return super().forward(input_dict)
        return self._chunked_forward(input_dict)


class ChunkedModifiedDeFTUSSSpatialTemporal(ModifiedDeFTUSSSpatialTemporal):
    """Chunked spatial USS with per-object temporal activity heads."""

    def __init__(
        self,
        *args,
        inference_chunk_seconds=6.0,
        inference_chunk_hop_seconds=5.0,
        sample_rate=32000,
        **kwargs,
    ):
        super().__init__(*args, sample_rate=sample_rate, **kwargs)
        self.inference_chunk_seconds = inference_chunk_seconds
        self.inference_chunk_hop_seconds = inference_chunk_hop_seconds
        self.sample_rate = sample_rate

    def _iter_chunk_starts(self, samples, chunk_samples, hop_samples):
        return ChunkedModifiedDeFTUSSSpatial._iter_chunk_starts(self, samples, chunk_samples, hop_samples)

    def _chunk_weight(self, chunk_samples, device, dtype):
        return ChunkedModifiedDeFTUSSSpatial._chunk_weight(self, chunk_samples, device, dtype)

    def _chunked_forward(self, input_dict):
        mixture = input_dict["mixture"]
        batch_size, _, samples = mixture.shape
        chunk_samples = int(round(float(self.inference_chunk_seconds) * self.sample_rate))
        hop_samples = int(round(float(self.inference_chunk_hop_seconds) * self.sample_rate))
        if chunk_samples <= 0 or hop_samples <= 0:
            raise ValueError("inference_chunk_seconds and inference_chunk_hop_seconds must be positive")
        if samples <= chunk_samples:
            return self._forward_full(input_dict)

        starts = self._iter_chunk_starts(samples, chunk_samples, hop_samples)
        weight = self._chunk_weight(chunk_samples, mixture.device, mixture.dtype)
        weight = weight.view(1, 1, 1, chunk_samples)

        waveform_sum = mixture.new_zeros(batch_size, self.n_objects, self.output_channels, samples)
        weight_sum = mixture.new_zeros(1, 1, 1, samples)
        class_logits = []
        silence_logits = []
        fg_activity = []
        int_activity = []
        noise_activity = []

        for start in starts:
            end = start + chunk_samples
            chunk = mixture[..., start:end]
            if chunk.shape[-1] < chunk_samples:
                chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))

            out = self._forward_full({"mixture": chunk})
            valid = min(chunk_samples, samples - start)
            waveform_sum[..., start : start + valid] += out["waveform"][..., :valid] * weight[..., :valid]
            weight_sum[..., start : start + valid] += weight[..., :valid]
            class_logits.append(out["class_logits"])
            silence_logits.append(out["silence_logits"])
            fg_activity.append(out["foreground_activity_logits"])
            int_activity.append(out["interference_activity_logits"])
            noise_activity.append(out["noise_activity_logits"])

        waveform = waveform_sum / torch.clamp(weight_sum, min=1e-6)
        duration_sec = mixture.new_full((batch_size,), float(samples) / float(self.sample_rate))
        return {
            "waveform": waveform,
            "foreground_waveform": waveform[:, : self.n_foreground],
            "interference_waveform": waveform[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_waveform": waveform[:, -1:],
            "class_logits": torch.stack(class_logits, dim=0).mean(dim=0),
            "silence_logits": torch.stack(silence_logits, dim=0).mean(dim=0),
            "foreground_activity_logits": torch.cat(fg_activity, dim=-1),
            "interference_activity_logits": torch.cat(int_activity, dim=-1),
            "noise_activity_logits": torch.cat(noise_activity, dim=-1),
            "duration_sec": duration_sec,
        }

    def forward(self, input_dict):
        if self.training or self.inference_chunk_seconds is None:
            return self._forward_full(input_dict)
        return self._chunked_forward(input_dict)


class ModifiedDeFTUSSSpatialLite(ChunkedModifiedDeFTUSSSpatial):
    """Memory-reduced spatial USS using local DeFT attention windows.

    This keeps the same spatial mask/reconstruction head as
    ``ModifiedDeFTUSSSpatial`` but replaces each global DeFT block with
    ``MemoryEfficientDeFTBlock``. The attention memory scales with the selected
    time window and frequency group sizes instead of the full clip length.
    """

    def __init__(
        self,
        input_channels=4,
        output_channels=1,
        hidden_channels=96,
        n_deft_blocks=6,
        n_heads=4,
        n_foreground=3,
        n_interference=2,
        n_classes=18,
        window_size=1024,
        hop_size=320,
        time_window_size=128,
        freq_group_size=64,
        shift_windows=True,
        inference_chunk_seconds=10.0,
        inference_chunk_hop_seconds=8.0,
        sample_rate=32000,
    ):
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=hidden_channels,
            n_deft_blocks=n_deft_blocks,
            n_heads=n_heads,
            n_foreground=n_foreground,
            n_interference=n_interference,
            n_classes=n_classes,
            window_size=window_size,
            hop_size=hop_size,
            inference_chunk_seconds=inference_chunk_seconds,
            inference_chunk_hop_seconds=inference_chunk_hop_seconds,
            sample_rate=sample_rate,
        )
        self.time_window_size = int(time_window_size)
        self.freq_group_size = int(freq_group_size)
        self.shift_windows = bool(shift_windows)
        self.blocks = nn.ModuleList()
        for block_idx in range(n_deft_blocks):
            use_shift = self.shift_windows and block_idx % 2 == 1
            self.blocks.append(
                MemoryEfficientDeFTBlock(
                    hidden_channels,
                    n_heads=n_heads,
                    time_window_size=self.time_window_size,
                    freq_group_size=self.freq_group_size,
                    time_shift=self.time_window_size // 2 if use_shift else 0,
                    freq_shift=self.freq_group_size // 2 if use_shift else 0,
                )
            )


class ModifiedDeFTUSSMemoryEfficient(ModifiedDeFTUSSSpatialLite):
    """Alias class for the memory-efficient spatial USS recipe."""


class ModifiedDeFTUSSSpatialLiteTemporal(ChunkedModifiedDeFTUSSSpatialTemporal):
    """Temporal counterpart of ``ModifiedDeFTUSSSpatialLite``."""

    def __init__(
        self,
        input_channels=4,
        output_channels=1,
        hidden_channels=96,
        n_deft_blocks=6,
        n_heads=4,
        n_foreground=3,
        n_interference=2,
        n_classes=18,
        window_size=1024,
        hop_size=320,
        time_window_size=128,
        freq_group_size=64,
        shift_windows=True,
        inference_chunk_seconds=10.0,
        inference_chunk_hop_seconds=8.0,
        sample_rate=32000,
    ):
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            hidden_channels=hidden_channels,
            n_deft_blocks=n_deft_blocks,
            n_heads=n_heads,
            n_foreground=n_foreground,
            n_interference=n_interference,
            n_classes=n_classes,
            window_size=window_size,
            hop_size=hop_size,
            inference_chunk_seconds=inference_chunk_seconds,
            inference_chunk_hop_seconds=inference_chunk_hop_seconds,
            sample_rate=sample_rate,
        )
        self.time_window_size = int(time_window_size)
        self.freq_group_size = int(freq_group_size)
        self.shift_windows = bool(shift_windows)
        self.blocks = nn.ModuleList()
        for block_idx in range(n_deft_blocks):
            use_shift = self.shift_windows and block_idx % 2 == 1
            self.blocks.append(
                MemoryEfficientDeFTBlock(
                    hidden_channels,
                    n_heads=n_heads,
                    time_window_size=self.time_window_size,
                    freq_group_size=self.freq_group_size,
                    time_shift=self.time_window_size // 2 if use_shift else 0,
                    freq_shift=self.freq_group_size // 2 if use_shift else 0,
                )
            )


class ModifiedDeFTUSSMemoryEfficientTemporal(ModifiedDeFTUSSMemoryEfficient):
    """Memory-efficient USS with per-object temporal activity heads."""

    def __init__(self, *args, hidden_channels=96, n_foreground=3, n_interference=2, **kwargs):
        super().__init__(
            *args,
            hidden_channels=hidden_channels,
            n_foreground=n_foreground,
            n_interference=n_interference,
            **kwargs,
        )
        self.activity_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def _activity_logits(self, object_features):
        batch_size, n_objects, channels, time_steps, freq_bins = object_features.shape
        logits = self.activity_head(
            object_features.reshape(batch_size * n_objects, channels, time_steps, freq_bins)
        )
        logits = logits.mean(dim=-1).squeeze(1)
        return logits.view(batch_size, n_objects, time_steps)

    def _forward_full(self, input_dict):
        mixture = input_dict["mixture"]
        batch_size, _, samples = mixture.shape
        real, imag = self.waveform_to_complex(mixture.reshape(-1, samples))
        _, _, time_steps, freq_bins = real.shape
        real = real.view(batch_size, self.input_channels, time_steps, freq_bins)
        imag = imag.view(batch_size, self.input_channels, time_steps, freq_bins)

        x = self.encoder(_stack_complex(real, imag))
        for block in self.blocks:
            x = block(x)
        x = self.object_conv(x)
        x = x.view(batch_size, self.n_objects, -1, time_steps, freq_bins)

        waveform = self._spatial_mask_to_waveform(x, real, imag, samples)
        activity_logits = self._activity_logits(x)

        fg_features = x[:, : self.n_foreground]
        class_logits = self.class_head(fg_features.reshape(batch_size * self.n_foreground, -1, time_steps, freq_bins))
        class_logits = class_logits.view(batch_size, self.n_foreground, self.n_classes)
        silence_logits = self.silence_head(fg_features.reshape(batch_size * self.n_foreground, -1, time_steps, freq_bins))
        silence_logits = silence_logits.view(batch_size, self.n_foreground)
        duration_sec = mixture.new_full((batch_size,), float(samples) / float(self.sample_rate))

        return {
            "waveform": waveform,
            "foreground_waveform": waveform[:, : self.n_foreground],
            "interference_waveform": waveform[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_waveform": waveform[:, -1:],
            "class_logits": class_logits,
            "silence_logits": silence_logits,
            "foreground_activity_logits": activity_logits[:, : self.n_foreground],
            "interference_activity_logits": activity_logits[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_activity_logits": activity_logits[:, -1:],
            "duration_sec": duration_sec,
        }

    def _chunked_forward(self, input_dict):
        mixture = input_dict["mixture"]
        batch_size, _, samples = mixture.shape
        chunk_samples = int(round(float(self.inference_chunk_seconds) * self.sample_rate))
        hop_samples = int(round(float(self.inference_chunk_hop_seconds) * self.sample_rate))
        if chunk_samples <= 0 or hop_samples <= 0:
            raise ValueError("inference_chunk_seconds and inference_chunk_hop_seconds must be positive")
        if samples <= chunk_samples:
            return self._forward_full(input_dict)

        starts = self._iter_chunk_starts(samples, chunk_samples, hop_samples)
        weight = self._chunk_weight(chunk_samples, mixture.device, mixture.dtype)
        weight = weight.view(1, 1, 1, chunk_samples)

        waveform_sum = mixture.new_zeros(batch_size, self.n_objects, self.output_channels, samples)
        weight_sum = mixture.new_zeros(1, 1, 1, samples)
        class_logits = []
        silence_logits = []
        fg_activity = []
        int_activity = []
        noise_activity = []

        for start in starts:
            end = start + chunk_samples
            chunk = mixture[..., start:end]
            if chunk.shape[-1] < chunk_samples:
                chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))

            out = self._forward_full({"mixture": chunk})
            valid = min(chunk_samples, samples - start)
            waveform_sum[..., start : start + valid] += out["waveform"][..., :valid] * weight[..., :valid]
            weight_sum[..., start : start + valid] += weight[..., :valid]
            class_logits.append(out["class_logits"])
            silence_logits.append(out["silence_logits"])
            fg_activity.append(out["foreground_activity_logits"])
            int_activity.append(out["interference_activity_logits"])
            noise_activity.append(out["noise_activity_logits"])

        waveform = waveform_sum / torch.clamp(weight_sum, min=1e-6)
        duration_sec = mixture.new_full((batch_size,), float(samples) / float(self.sample_rate))
        return {
            "waveform": waveform,
            "foreground_waveform": waveform[:, : self.n_foreground],
            "interference_waveform": waveform[:, self.n_foreground : self.n_foreground + self.n_interference],
            "noise_waveform": waveform[:, -1:],
            "class_logits": torch.stack(class_logits, dim=0).mean(dim=0),
            "silence_logits": torch.stack(silence_logits, dim=0).mean(dim=0),
            "foreground_activity_logits": torch.cat(fg_activity, dim=-1),
            "interference_activity_logits": torch.cat(int_activity, dim=-1),
            "noise_activity_logits": torch.cat(noise_activity, dim=-1),
            "duration_sec": duration_sec,
        }

    def forward(self, input_dict):
        if self.training or self.inference_chunk_seconds is None:
            return self._forward_full(input_dict)
        return self._chunked_forward(input_dict)


class ModifiedDeFTTSE(_BaseSpectralModel):
    def __init__(
        self,
        mixture_channels=4,
        enrollment_channels=1,
        hidden_channels=96,
        n_deft_blocks=6,
        n_heads=4,
        label_dim=18,
        window_size=1024,
        hop_size=320,
        query_condition_dim=0,
        query_condition_hidden_dim=256,
        query_condition_scale=1.0,
        query_condition_dropout=0.0,
        require_query_condition=False,
    ):
        super().__init__(window_size=window_size, hop_size=hop_size)
        self.mixture_channels = mixture_channels
        self.enrollment_channels = enrollment_channels
        self.label_dim = label_dim
        self.hidden_channels = hidden_channels
        self.query_condition_dim = int(query_condition_dim or 0)
        self.query_condition_scale = float(query_condition_scale)
        self.require_query_condition = bool(require_query_condition)

        self.encoder = nn.Sequential(
            nn.Conv2d((mixture_channels + enrollment_channels) * 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [DeFTBlock(hidden_channels, n_heads=n_heads) for _ in range(n_deft_blocks)]
        )
        self.class_conditioner = ClassConditioner(label_dim, hidden_channels)
        self.audio_head = nn.Conv2d(hidden_channels, 2, kernel_size=1)
        if self.query_condition_dim > 0:
            self.query_conditioner = nn.Sequential(
                nn.LayerNorm(self.query_condition_dim),
                nn.Linear(self.query_condition_dim, int(query_condition_hidden_dim)),
                nn.GELU(),
                nn.Dropout(float(query_condition_dropout)),
                nn.Linear(int(query_condition_hidden_dim), hidden_channels * 2),
            )
        else:
            self.query_conditioner = None

    def _get_query_condition(self, input_dict):
        for key in ("query_condition", "tse_condition", "bridge_condition", "proposal_condition"):
            if key in input_dict:
                return input_dict[key]
        if self.require_query_condition:
            raise ValueError("query_condition is required")
        return None

    def _query_film(self, input_dict, label_vector, batch_size, n_queries, device, dtype):
        query_condition = self._get_query_condition(input_dict)
        if self.query_conditioner is None:
            return None, None
        if query_condition is None:
            return None, None
        condition = _match_query_condition_shape(query_condition, label_vector, self.query_condition_dim)
        condition = condition.to(device=device, dtype=dtype)
        beta_gamma = self.query_conditioner(condition.reshape(batch_size * n_queries, -1))
        beta, gamma = beta_gamma.chunk(2, dim=-1)
        scale = self.query_condition_scale
        return beta[:, :, None, None] * scale, gamma[:, :, None, None] * scale

    def forward(self, input_dict):
        mixture = input_dict["mixture"]
        enroll = input_dict["enrollment"]
        label_vector = input_dict["label_vector"]
        batch_size, n_queries, _, samples = enroll.shape

        mix_real, mix_imag = self.waveform_to_complex(mixture.reshape(-1, samples))
        _, _, time_steps, freq_bins = mix_real.shape
        mix_real = mix_real.view(batch_size, self.mixture_channels, time_steps, freq_bins)
        mix_imag = mix_imag.view(batch_size, self.mixture_channels, time_steps, freq_bins)

        enr_real, enr_imag = self.waveform_to_complex(enroll.reshape(-1, samples))
        enr_real = enr_real.view(batch_size, n_queries, self.enrollment_channels, time_steps, freq_bins)
        enr_imag = enr_imag.view(batch_size, n_queries, self.enrollment_channels, time_steps, freq_bins)

        mix_real = mix_real[:, None].expand(-1, n_queries, -1, -1, -1)
        mix_imag = mix_imag[:, None].expand(-1, n_queries, -1, -1, -1)
        joint_real = torch.cat([mix_real, enr_real], dim=2)
        joint_imag = torch.cat([mix_imag, enr_imag], dim=2)
        features = _stack_complex(
            joint_real.reshape(batch_size * n_queries, -1, time_steps, freq_bins),
            joint_imag.reshape(batch_size * n_queries, -1, time_steps, freq_bins),
        )

        x = self.encoder(features)
        beta, gamma = self.class_conditioner(label_vector.reshape(batch_size * n_queries, -1))
        query_beta, query_gamma = self._query_film(input_dict, label_vector, batch_size, n_queries, x.device, x.dtype)
        if query_beta is not None:
            beta = beta + query_beta
            gamma = gamma + query_gamma
        for block in self.blocks:
            x = block(x, beta=beta, gamma=gamma)

        mask = torch.tanh(self.audio_head(x))
        mix_ref_real = mix_real[:, :, 0].reshape(batch_size * n_queries, 1, time_steps, freq_bins)
        mix_ref_imag = mix_imag[:, :, 0].reshape(batch_size * n_queries, 1, time_steps, freq_bins)
        est_real = mask[:, 0:1] * mix_ref_real - mask[:, 1:2] * mix_ref_imag
        est_imag = mask[:, 0:1] * mix_ref_imag + mask[:, 1:2] * mix_ref_real

        waveform = self.complex_to_waveform(est_real, est_imag, samples)
        waveform = waveform.view(batch_size, n_queries, 1, samples)
        return {"waveform": waveform}


class ModifiedDeFTTSETemporal(ModifiedDeFTTSE):
    """Original modified DeFT TSE with per-query temporal activity prediction."""

    def __init__(self, *args, hidden_channels=96, sample_rate=32000, **kwargs):
        super().__init__(*args, hidden_channels=hidden_channels, **kwargs)
        self.sample_rate = sample_rate
        self.activity_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.temporal_conditioner = nn.Conv1d(1, hidden_channels * 2, kernel_size=3, padding=1)

    def forward(self, input_dict):
        mixture = input_dict["mixture"]
        enroll = input_dict["enrollment"]
        label_vector = input_dict["label_vector"]
        temporal_conditioning = input_dict.get("temporal_conditioning")
        batch_size, n_queries, _, samples = enroll.shape

        mix_real, mix_imag = self.waveform_to_complex(mixture.reshape(-1, samples))
        _, _, time_steps, freq_bins = mix_real.shape
        mix_real = mix_real.view(batch_size, self.mixture_channels, time_steps, freq_bins)
        mix_imag = mix_imag.view(batch_size, self.mixture_channels, time_steps, freq_bins)

        enr_real, enr_imag = self.waveform_to_complex(enroll.reshape(-1, samples))
        enr_real = enr_real.view(batch_size, n_queries, self.enrollment_channels, time_steps, freq_bins)
        enr_imag = enr_imag.view(batch_size, n_queries, self.enrollment_channels, time_steps, freq_bins)

        mix_real = mix_real[:, None].expand(-1, n_queries, -1, -1, -1)
        mix_imag = mix_imag[:, None].expand(-1, n_queries, -1, -1, -1)
        joint_real = torch.cat([mix_real, enr_real], dim=2)
        joint_imag = torch.cat([mix_imag, enr_imag], dim=2)
        features = _stack_complex(
            joint_real.reshape(batch_size * n_queries, -1, time_steps, freq_bins),
            joint_imag.reshape(batch_size * n_queries, -1, time_steps, freq_bins),
        )

        x = self.encoder(features)
        beta, gamma = self.class_conditioner(label_vector.reshape(batch_size * n_queries, -1))
        query_beta, query_gamma = self._query_film(input_dict, label_vector, batch_size, n_queries, x.device, x.dtype)
        if query_beta is not None:
            beta = beta + query_beta
            gamma = gamma + query_gamma
        time_beta, time_gamma = _temporal_film_from_conditioning(
            self.temporal_conditioner,
            temporal_conditioning,
            batch_size,
            n_queries,
            time_steps,
            x.device,
            x.dtype,
        )
        if time_beta is not None:
            beta = beta + time_beta
            gamma = gamma + time_gamma
        for block in self.blocks:
            x = block(x, beta=beta, gamma=gamma)

        mask = torch.tanh(self.audio_head(x))
        mix_ref_real = mix_real[:, :, 0].reshape(batch_size * n_queries, 1, time_steps, freq_bins)
        mix_ref_imag = mix_imag[:, :, 0].reshape(batch_size * n_queries, 1, time_steps, freq_bins)
        est_real = mask[:, 0:1] * mix_ref_real - mask[:, 1:2] * mix_ref_imag
        est_imag = mask[:, 0:1] * mix_ref_imag + mask[:, 1:2] * mix_ref_real

        waveform = self.complex_to_waveform(est_real, est_imag, samples)
        waveform = waveform.view(batch_size, n_queries, 1, samples)
        activity_logits = self.activity_head(x).mean(dim=-1).squeeze(1)
        activity_logits = activity_logits.view(batch_size, n_queries, time_steps)
        duration_sec = mixture.new_full((batch_size,), float(samples) / float(self.sample_rate))
        return {
            "waveform": waveform,
            "activity_logits": activity_logits,
            "duration_sec": duration_sec,
        }


class ModifiedDeFTTSEMemoryEfficient(ModifiedDeFTTSE):
    """Memory-efficient TSE with local DeFT attention and eval chunking."""

    def __init__(
        self,
        mixture_channels=4,
        enrollment_channels=1,
        hidden_channels=96,
        n_deft_blocks=6,
        n_heads=4,
        label_dim=18,
        window_size=1024,
        hop_size=320,
        time_window_size=128,
        freq_group_size=64,
        shift_windows=True,
        inference_chunk_seconds=10.0,
        inference_chunk_hop_seconds=8.0,
        sample_rate=32000,
        query_condition_dim=0,
        query_condition_hidden_dim=256,
        query_condition_scale=1.0,
        query_condition_dropout=0.0,
        require_query_condition=False,
    ):
        super().__init__(
            mixture_channels=mixture_channels,
            enrollment_channels=enrollment_channels,
            hidden_channels=hidden_channels,
            n_deft_blocks=n_deft_blocks,
            n_heads=n_heads,
            label_dim=label_dim,
            window_size=window_size,
            hop_size=hop_size,
            query_condition_dim=query_condition_dim,
            query_condition_hidden_dim=query_condition_hidden_dim,
            query_condition_scale=query_condition_scale,
            query_condition_dropout=query_condition_dropout,
            require_query_condition=require_query_condition,
        )
        self.time_window_size = int(time_window_size)
        self.freq_group_size = int(freq_group_size)
        self.shift_windows = bool(shift_windows)
        self.inference_chunk_seconds = inference_chunk_seconds
        self.inference_chunk_hop_seconds = inference_chunk_hop_seconds
        self.sample_rate = sample_rate

        self.blocks = nn.ModuleList()
        for block_idx in range(n_deft_blocks):
            use_shift = self.shift_windows and block_idx % 2 == 1
            self.blocks.append(
                MemoryEfficientDeFTBlock(
                    hidden_channels,
                    n_heads=n_heads,
                    time_window_size=self.time_window_size,
                    freq_group_size=self.freq_group_size,
                    time_shift=self.time_window_size // 2 if use_shift else 0,
                    freq_shift=self.freq_group_size // 2 if use_shift else 0,
                )
            )

    def _iter_chunk_starts(self, samples, chunk_samples, hop_samples):
        if samples <= chunk_samples:
            return [0]
        starts = list(range(0, max(samples - chunk_samples, 0) + 1, hop_samples))
        last_start = samples - chunk_samples
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    def _chunk_weight(self, chunk_samples, device, dtype):
        if chunk_samples <= 1:
            return torch.ones(1, device=device, dtype=dtype)
        weight = torch.hann_window(chunk_samples, periodic=False, device=device, dtype=dtype)
        return torch.clamp(weight, min=1e-3)

    def _chunked_forward(self, input_dict):
        mixture = input_dict["mixture"]
        enrollment = input_dict["enrollment"]
        label_vector = input_dict["label_vector"]
        query_condition = self._get_query_condition(input_dict)
        batch_size, n_queries, _, samples = enrollment.shape

        chunk_samples = int(round(float(self.inference_chunk_seconds) * self.sample_rate))
        hop_samples = int(round(float(self.inference_chunk_hop_seconds) * self.sample_rate))
        if chunk_samples <= 0 or hop_samples <= 0:
            raise ValueError("inference_chunk_seconds and inference_chunk_hop_seconds must be positive")
        if samples <= chunk_samples:
            return super().forward(input_dict)

        starts = self._iter_chunk_starts(samples, chunk_samples, hop_samples)
        weight = self._chunk_weight(chunk_samples, mixture.device, mixture.dtype)
        weight = weight.view(1, 1, 1, chunk_samples)
        waveform_sum = mixture.new_zeros(batch_size, n_queries, 1, samples)
        weight_sum = mixture.new_zeros(1, 1, 1, samples)

        for start in starts:
            valid = min(chunk_samples, samples - start)
            mixture_chunk = mixture[..., start : start + valid]
            enrollment_chunk = enrollment[..., start : start + valid]
            if valid < chunk_samples:
                mixture_chunk = F.pad(mixture_chunk, (0, chunk_samples - valid))
                enrollment_chunk = F.pad(enrollment_chunk, (0, chunk_samples - valid))

            out = super().forward(
                {
                    "mixture": mixture_chunk,
                    "enrollment": enrollment_chunk,
                    "label_vector": label_vector,
                    "query_condition": query_condition,
                }
            )
            waveform_sum[..., start : start + valid] += out["waveform"][..., :valid] * weight[..., :valid]
            weight_sum[..., start : start + valid] += weight[..., :valid]

        return {"waveform": waveform_sum / torch.clamp(weight_sum, min=1e-6)}

    def forward(self, input_dict):
        if self.training or self.inference_chunk_seconds is None:
            return super().forward(input_dict)
        return self._chunked_forward(input_dict)


class ModifiedDeFTTSEMemoryEfficientTemporal(ModifiedDeFTTSEMemoryEfficient):
    """Memory-efficient TSE with per-query temporal activity prediction."""

    def __init__(self, *args, hidden_channels=96, **kwargs):
        super().__init__(*args, hidden_channels=hidden_channels, **kwargs)
        self.activity_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.temporal_conditioner = nn.Conv1d(1, hidden_channels * 2, kernel_size=3, padding=1)

    def _temporal_film(self, temporal_conditioning, batch_size, n_queries, time_steps, device, dtype):
        return _temporal_film_from_conditioning(
            self.temporal_conditioner,
            temporal_conditioning,
            batch_size,
            n_queries,
            time_steps,
            device,
            dtype,
        )

    def _forward_full(self, input_dict):
        mixture = input_dict["mixture"]
        enroll = input_dict["enrollment"]
        label_vector = input_dict["label_vector"]
        temporal_conditioning = input_dict.get("temporal_conditioning")
        batch_size, n_queries, _, samples = enroll.shape

        mix_real, mix_imag = self.waveform_to_complex(mixture.reshape(-1, samples))
        _, _, time_steps, freq_bins = mix_real.shape
        mix_real = mix_real.view(batch_size, self.mixture_channels, time_steps, freq_bins)
        mix_imag = mix_imag.view(batch_size, self.mixture_channels, time_steps, freq_bins)

        enr_real, enr_imag = self.waveform_to_complex(enroll.reshape(-1, samples))
        enr_real = enr_real.view(batch_size, n_queries, self.enrollment_channels, time_steps, freq_bins)
        enr_imag = enr_imag.view(batch_size, n_queries, self.enrollment_channels, time_steps, freq_bins)

        mix_real = mix_real[:, None].expand(-1, n_queries, -1, -1, -1)
        mix_imag = mix_imag[:, None].expand(-1, n_queries, -1, -1, -1)
        joint_real = torch.cat([mix_real, enr_real], dim=2)
        joint_imag = torch.cat([mix_imag, enr_imag], dim=2)
        features = _stack_complex(
            joint_real.reshape(batch_size * n_queries, -1, time_steps, freq_bins),
            joint_imag.reshape(batch_size * n_queries, -1, time_steps, freq_bins),
        )

        x = self.encoder(features)
        beta, gamma = self.class_conditioner(label_vector.reshape(batch_size * n_queries, -1))
        query_beta, query_gamma = self._query_film(input_dict, label_vector, batch_size, n_queries, x.device, x.dtype)
        if query_beta is not None:
            beta = beta + query_beta
            gamma = gamma + query_gamma
        time_beta, time_gamma = self._temporal_film(
            temporal_conditioning,
            batch_size,
            n_queries,
            time_steps,
            x.device,
            x.dtype,
        )
        if time_beta is not None:
            beta = beta + time_beta
            gamma = gamma + time_gamma
        for block in self.blocks:
            x = block(x, beta=beta, gamma=gamma)

        mask = torch.tanh(self.audio_head(x))
        mix_ref_real = mix_real[:, :, 0].reshape(batch_size * n_queries, 1, time_steps, freq_bins)
        mix_ref_imag = mix_imag[:, :, 0].reshape(batch_size * n_queries, 1, time_steps, freq_bins)
        est_real = mask[:, 0:1] * mix_ref_real - mask[:, 1:2] * mix_ref_imag
        est_imag = mask[:, 0:1] * mix_ref_imag + mask[:, 1:2] * mix_ref_real

        waveform = self.complex_to_waveform(est_real, est_imag, samples)
        waveform = waveform.view(batch_size, n_queries, 1, samples)
        activity_logits = self.activity_head(x).mean(dim=-1).squeeze(1)
        activity_logits = activity_logits.view(batch_size, n_queries, time_steps)
        duration_sec = mixture.new_full((batch_size,), float(samples) / float(self.sample_rate))
        return {
            "waveform": waveform,
            "activity_logits": activity_logits,
            "duration_sec": duration_sec,
        }

    def _chunked_forward(self, input_dict):
        mixture = input_dict["mixture"]
        enrollment = input_dict["enrollment"]
        label_vector = input_dict["label_vector"]
        temporal_conditioning = input_dict.get("temporal_conditioning")
        query_condition = self._get_query_condition(input_dict)
        batch_size, n_queries, _, samples = enrollment.shape

        chunk_samples = int(round(float(self.inference_chunk_seconds) * self.sample_rate))
        hop_samples = int(round(float(self.inference_chunk_hop_seconds) * self.sample_rate))
        if chunk_samples <= 0 or hop_samples <= 0:
            raise ValueError("inference_chunk_seconds and inference_chunk_hop_seconds must be positive")
        if samples <= chunk_samples:
            return self._forward_full(input_dict)

        starts = self._iter_chunk_starts(samples, chunk_samples, hop_samples)
        weight = self._chunk_weight(chunk_samples, mixture.device, mixture.dtype)
        weight = weight.view(1, 1, 1, chunk_samples)
        waveform_sum = mixture.new_zeros(batch_size, n_queries, 1, samples)
        weight_sum = mixture.new_zeros(1, 1, 1, samples)
        activity_logits = []
        temporal_conditioning_samples = self._activity_to_samples_for_chunking(
            temporal_conditioning,
            batch_size,
            n_queries,
            samples,
            mixture.device,
            mixture.dtype,
        )

        for start in starts:
            valid = min(chunk_samples, samples - start)
            mixture_chunk = mixture[..., start : start + valid]
            enrollment_chunk = enrollment[..., start : start + valid]
            temporal_conditioning_chunk = None
            if temporal_conditioning_samples is not None:
                temporal_conditioning_chunk = temporal_conditioning_samples[..., start : start + valid]
            if valid < chunk_samples:
                mixture_chunk = F.pad(mixture_chunk, (0, chunk_samples - valid))
                enrollment_chunk = F.pad(enrollment_chunk, (0, chunk_samples - valid))
                if temporal_conditioning_chunk is not None:
                    temporal_conditioning_chunk = F.pad(temporal_conditioning_chunk, (0, chunk_samples - valid))

            out = self._forward_full(
                {
                    "mixture": mixture_chunk,
                    "enrollment": enrollment_chunk,
                    "label_vector": label_vector,
                    "temporal_conditioning": temporal_conditioning_chunk,
                    "query_condition": query_condition,
                }
            )
            waveform_sum[..., start : start + valid] += out["waveform"][..., :valid] * weight[..., :valid]
            weight_sum[..., start : start + valid] += weight[..., :valid]
            activity_logits.append(out["activity_logits"])

        duration_sec = mixture.new_full((batch_size,), float(samples) / float(self.sample_rate))
        return {
            "waveform": waveform_sum / torch.clamp(weight_sum, min=1e-6),
            "activity_logits": torch.cat(activity_logits, dim=-1),
            "duration_sec": duration_sec,
        }

    def forward(self, input_dict):
        if self.training or self.inference_chunk_seconds is None:
            return self._forward_full(input_dict)
        return self._chunked_forward(input_dict)

    def _activity_to_samples_for_chunking(self, activity, batch_size, n_queries, samples, device, dtype):
        if activity is None:
            return None
        activity = activity.to(device=device, dtype=dtype)
        if activity.dim() == 2:
            activity = activity.unsqueeze(1).expand(-1, n_queries, -1)
        if activity.dim() != 3:
            raise ValueError("temporal_conditioning must have shape [B, T] or [B, Q, T]")
        if activity.shape[0] != batch_size or activity.shape[1] != n_queries:
            raise ValueError("temporal_conditioning batch/query dimensions do not match TSE input")
        return F.interpolate(
            activity.reshape(batch_size * n_queries, 1, activity.shape[-1]),
            size=samples,
            mode="linear",
            align_corners=False,
        ).view(batch_size, n_queries, samples)
