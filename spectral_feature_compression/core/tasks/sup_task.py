# Copyright (c) 2026 National Institute of Advanced Industrial Science and Technology (AIST), Japan
#
# SPDX-License-Identifier: MIT

from functools import partial

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from torchaudio.transforms import Spectrogram

from aiaccel.torch.lightning import OptimizerConfig, OptimizerLightningModule
from spectral_feature_compression.core.loss.snr import snr


class SupTask(OptimizerLightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        n_fft: int,
        hop_length: int,
        optimizer_config: OptimizerConfig,
        # general args
        pretrained_model_path: str | None = None,
        css_validation: bool = False,
        # ema args
        ema_weight: float | None = None,
        ema_update_freq: int | None = None,
    ):
        super().__init__(optimizer_config)

        self.model = model
        self.pretrained_model_path = pretrained_model_path
        self.load_pretrained_weight()

        self.loss = loss

        self.css_validation = css_validation

        self.stft = nn.Sequential(Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None))

        self.snr = partial(snr, negative=False)

        # NOTE: update_bn is not implemented. Needs to be if we use batchnorm
        if ema_weight is not None:
            assert 0 <= ema_weight < 1 and ema_update_freq is not None, ema_weight
            self.use_ema_model = True
            self.ema_update_freq = ema_update_freq
            self.ema_model = AveragedModel(self.model, multi_avg_fn=get_ema_multi_avg_fn(ema_weight))

            # NOTE: we need this instead of self.ema_model.requires_grad_(False)
            # since some model scripts include torch.jit.script, which clones the parameters
            for p in self.ema_model.parameters():
                p.detach_()  # make p leaf
                p.requires_grad_(False)
        else:
            self.use_ema_model = False
            self.ema_update_freq = None

    # @torch.autocast("cuda", enabled=False)
    def _step(self, wav: torch.Tensor, ref: torch.Tensor, log_prefix: str):
        # separation, est: (n_batch, n_src, n_chan, n_samples)
        model = self.ema_model.module if self.use_ema_model and log_prefix != "training" else self.model
        est = model.css(wav, ref=ref) if log_prefix != "training" and self.css_validation else model(wav)

        # loss calculation
        loss = self.loss(est.transpose(1, 2), ref.transpose(1, 2)).mean()

        # logging
        log_dict = {"step": float(self.trainer.current_epoch), f"{log_prefix}/loss": loss}
        if log_prefix == "validation":
            snr_score = self.snr(est.transpose(1, 2), ref.transpose(1, 2)).mean()
            log_dict[f"{log_prefix}/snr"] = snr_score

        self.log_dict(log_dict, prog_bar=False, on_epoch=True, on_step=False, batch_size=wav.shape[0], sync_dist=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema_model and self.global_step % self.ema_update_freq == 0:
            self.ema_model.update_parameters(self.model)

    @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def training_step(self, wav: torch.Tensor | list, batch_idx: int):
        if isinstance(wav, list):
            wav, ref = wav
        else:
            ref = None
        return self._step(wav, ref=ref, log_prefix="training")

    @torch.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def validation_step(self, wav: torch.Tensor | list, batch_idx: int):
        if isinstance(wav, list):
            wav, ref = wav
        else:
            ref = None
        return self._step(wav, ref=ref, log_prefix="validation")

    def load_pretrained_weight(self):
        """
        Although the weights specified by pretrained_model_path is loaded even in evalation,
        it's overwritten by checkpoint_path specified in inference, so there should be no problem
        """
        if self.pretrained_model_path is not None:
            if torch.cuda.is_available():
                state_dict = torch.load(self.pretrained_model_path, weights_only=False)
            else:
                state_dict = torch.load(
                    self.pretrained_model_path,
                    map_location=torch.device("cpu"),
                    weights_only=False,
                )
            try:
                state_dict = state_dict["state_dict"]
            except KeyError:
                print("No key named state_dict. Directly loading from model.")

            print(f"Load model from {self.pretrained_model_path}")
            for module in ["model"]:
                sd = {k: v for k, v in state_dict.items() if k.startswith(module)}
                sd = {".".join(k.split(".")[1:]): v for k, v in sd.items()}
                sd = {k: v for k, v in sd.items() if k != "n_averaged"}
                sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
                getattr(self, f"{module}").load_state_dict(sd, strict=True)

            # for b, module in enumerate(
            #     self.model.model.band_split_module.bandwise_decoding_module.modules()
            # ):
            #     if hasattr(module, "reset_parameters"):
            #         module.reset_parameters()

    def load_average_checkpoint(self, checkpoint_dir, **kwargs):
        """
        Load the model from the averaged checkpoint files.
        If the checkpoint_dir is a directory, load the averaged model from the directory.
        If the checkpoint_dir is a .ckpt file, load the model from that file.
        """

        from pathlib import Path

        from spectral_feature_compression.utils.average_model_params import average_model_params

        if checkpoint_dir.suffix in [".ckpt", ".pth", ".pt"]:
            checkpoint_paths = [checkpoint_dir]
        else:
            assert checkpoint_dir.is_dir(), f"{checkpoint_dir} is not a directory."
            checkpoint_paths = [
                path
                for path in Path(checkpoint_dir).iterdir()
                if path.suffix in [".ckpt", ".pth", ".pt"] and path.name != "last.ckpt"
            ]
            assert all([checkpoint_paths[0].suffix == c.suffix for c in checkpoint_paths])
        state_dict = average_model_params(checkpoint_paths)
        self.load_state_dict(state_dict, strict=True)
