import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase


def _reshape_label_vector(label_vector, target_sources_num):
    if label_vector.dim() == 3:
        return label_vector
    batch_size = label_vector.shape[0]
    label_len = label_vector.shape[1] // target_sources_num
    return label_vector.view(batch_size, target_sources_num, label_len)


class AxisBlock(nn.Module):
    def __init__(self, channels, n_heads=4, ff_mult=2):
        super().__init__()
        self.freq_attn = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dim_feedforward=channels * ff_mult,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.time_attn = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=n_heads,
            dim_feedforward=channels * ff_mult,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pwconv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x):
        batch_size, channels, time_steps, freq_bins = x.shape

        residual = x
        x_f = x.permute(0, 2, 3, 1).reshape(batch_size * time_steps, freq_bins, channels)
        x_f = self.freq_attn(x_f)
        x_f = x_f.reshape(batch_size, time_steps, freq_bins, channels).permute(0, 3, 1, 2)

        x_t = x.permute(0, 3, 2, 1).reshape(batch_size * freq_bins, time_steps, channels)
        x_t = self.time_attn(x_t)
        x_t = x_t.reshape(batch_size, freq_bins, time_steps, channels).permute(0, 3, 2, 1)

        x = residual + x_f + x_t
        x = x + self.pwconv(self.dwconv(x))
        return self.norm(x)


class ClueEncoder(nn.Module):
    def __init__(self, label_len, channels):
        super().__init__()
        hidden = max(channels, label_len * 2)
        self.net = nn.Sequential(
            nn.Linear(label_len, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels * 2),
        )

    def forward(self, labels):
        params = self.net(labels)
        gamma, beta = torch.chunk(params, chunks=2, dim=-1)
        return gamma[:, :, :, None, None], beta[:, :, :, None, None]


class DeFTTSELike(nn.Module):
    def __init__(
        self,
        input_channels=4,
        output_channels=1,
        target_sources_num=3,
        label_len=54,
        window_size=1024,
        hop_size=320,
        subband=1,
        base_channels=96,
        n_blocks=3,
        n_heads=4,
    ):
        super().__init__()
        assert output_channels == 1
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.target_sources_num = target_sources_num
        self.label_len = label_len
        self.label_dim = label_len // target_sources_num
        self.window_size = window_size
        self.hop_size = hop_size
        self.subband = subband

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

        # 4 magnitude channels + 3 FOA intensity features.
        feature_channels = input_channels + 3
        self.input_proj = nn.Sequential(
            nn.Conv2d(feature_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [AxisBlock(base_channels, n_heads=n_heads) for _ in range(n_blocks)]
        )
        self.clue_encoder = ClueEncoder(self.label_dim, base_channels)
        self.mask_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, 2, kernel_size=1),
        )

    def _stft_features(self, mixture):
        real, imag = self.stft(mixture.view(-1, mixture.shape[-1]))
        batch_size, channels = mixture.shape[:2]
        _, time_steps, freq_bins = real.shape[-3:]
        real = real.view(batch_size, channels, time_steps, freq_bins)
        imag = imag.view(batch_size, channels, time_steps, freq_bins)

        mag = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-8)
        log_mag = torch.log1p(mag)

        w_real = real[:, 0]
        w_imag = imag[:, 0]
        denom = mag[:, 0].pow(2) + 1e-8
        intensity = []
        for ch in range(1, min(4, channels)):
            cross = w_real * real[:, ch] + w_imag * imag[:, ch]
            intensity.append(cross / denom)
        while len(intensity) < 3:
            intensity.append(torch.zeros_like(denom))
        intensity = torch.stack(intensity[:3], dim=1)

        ref_real = real[:, 0]
        ref_imag = imag[:, 0]
        return log_mag, intensity, ref_real, ref_imag

    def forward(self, input_dict):
        mixture = input_dict["mixture"]
        label_vector = _reshape_label_vector(input_dict["label_vector"], self.target_sources_num)
        batch_size, _, samples = mixture.shape

        log_mag, intensity, ref_real, ref_imag = self._stft_features(mixture)
        features = torch.cat([log_mag, intensity], dim=1)
        x = self.input_proj(features)
        for block in self.blocks:
            x = block(x)

        gamma, beta = self.clue_encoder(label_vector)
        x = x.unsqueeze(1).expand(-1, self.target_sources_num, -1, -1, -1)
        x = x * (1.0 + gamma) + beta
        x = x.reshape(batch_size * self.target_sources_num, x.shape[2], x.shape[3], x.shape[4])
        mask = self.mask_head(x)
        mask = torch.tanh(mask)
        mask = mask.view(batch_size, self.target_sources_num, 2, mask.shape[-2], mask.shape[-1])

        ref_real = ref_real[:, None, :, :]
        ref_imag = ref_imag[:, None, :, :]
        est_real = mask[:, :, 0] * ref_real - mask[:, :, 1] * ref_imag
        est_imag = mask[:, :, 0] * ref_imag + mask[:, :, 1] * ref_real

        waveform = self.istft(
            est_real.reshape(batch_size * self.target_sources_num, 1, est_real.shape[-2], est_real.shape[-1]),
            est_imag.reshape(batch_size * self.target_sources_num, 1, est_imag.shape[-2], est_imag.shape[-1]),
            samples,
        )
        waveform = waveform.view(batch_size, self.target_sources_num, 1, samples)
        return {"waveform": waveform}


class DeFTTSELikeSpatial(DeFTTSELike):
    """Spatial TSE variant with magnitude/phase masks over all mixture channels.

    The original ``DeFTTSELike`` uses multichannel magnitude/intensity features
    for conditioning, but reconstructs the final waveform from channel 0 only.
    This variant keeps the same input/output contract while predicting
    [magnitude, phase_real, phase_imag] masks for every mixture channel and
    learning a complex multi-channel projection before ISTFT.
    """

    def __init__(
        self,
        input_channels=4,
        output_channels=1,
        target_sources_num=3,
        label_len=54,
        window_size=1024,
        hop_size=320,
        subband=1,
        base_channels=96,
        n_blocks=3,
        n_heads=4,
    ):
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            target_sources_num=target_sources_num,
            label_len=label_len,
            window_size=window_size,
            hop_size=hop_size,
            subband=subband,
            base_channels=base_channels,
            n_blocks=n_blocks,
            n_heads=n_heads,
        )
        self.mask_components = 3
        self.mask_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, input_channels * self.mask_components, kernel_size=1),
        )
        self.out_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def _stft_features(self, mixture):
        real, imag = self.stft(mixture.view(-1, mixture.shape[-1]))
        batch_size, channels = mixture.shape[:2]
        _, time_steps, freq_bins = real.shape[-3:]
        real = real.view(batch_size, channels, time_steps, freq_bins)
        imag = imag.view(batch_size, channels, time_steps, freq_bins)

        mag = torch.sqrt(real.pow(2) + imag.pow(2) + 1e-8)
        log_mag = torch.log1p(mag)

        w_real = real[:, 0]
        w_imag = imag[:, 0]
        denom = mag[:, 0].pow(2) + 1e-8
        intensity = []
        for ch in range(1, min(4, channels)):
            cross = w_real * real[:, ch] + w_imag * imag[:, ch]
            intensity.append(cross / denom)
        while len(intensity) < 3:
            intensity.append(torch.zeros_like(denom))
        intensity = torch.stack(intensity[:3], dim=1)

        return log_mag, intensity, real, imag

    def _spatial_mask_to_waveform(self, mask_features, real, imag, samples):
        batch_size = real.shape[0]
        _, _, time_steps, freq_bins = mask_features.shape

        mask = self.mask_head(mask_features)
        mask = mask.view(
            batch_size,
            self.target_sources_num,
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
            batch_size * self.target_sources_num,
            self.input_channels,
            time_steps,
            freq_bins,
        )
        est_imag = est_imag.reshape(
            batch_size * self.target_sources_num,
            self.input_channels,
            time_steps,
            freq_bins,
        )

        est_real = self.out_conv(est_real)
        est_imag = self.out_conv(est_imag)

        waveform = self.istft(
            est_real.reshape(
                batch_size * self.target_sources_num * self.output_channels,
                1,
                time_steps,
                freq_bins,
            ),
            est_imag.reshape(
                batch_size * self.target_sources_num * self.output_channels,
                1,
                time_steps,
                freq_bins,
            ),
            samples,
        )
        return waveform.view(batch_size, self.target_sources_num, self.output_channels, samples)

    def forward(self, input_dict):
        mixture = input_dict["mixture"]
        label_vector = _reshape_label_vector(input_dict["label_vector"], self.target_sources_num)
        batch_size, _, samples = mixture.shape

        log_mag, intensity, real, imag = self._stft_features(mixture)
        features = torch.cat([log_mag, intensity], dim=1)
        x = self.input_proj(features)
        for block in self.blocks:
            x = block(x)

        gamma, beta = self.clue_encoder(label_vector)
        x = x.unsqueeze(1).expand(-1, self.target_sources_num, -1, -1, -1)
        x = x * (1.0 + gamma) + beta
        x = x.reshape(batch_size * self.target_sources_num, x.shape[2], x.shape[3], x.shape[4])

        return {"waveform": self._spatial_mask_to_waveform(x, real, imag, samples)}
