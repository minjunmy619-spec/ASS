"""Temporal M2D + PretrainedSED fusion source classifier.

This module complements ``m2d_sc.M2DPretrainedSEDFusionClassifier`` with a
frame-aware variant.  The clip-level classifier keeps the existing ArcFace /
plain-logit contract used by ``m2d_sc_arcface`` while additionally exposing
``activity_logits`` and ``frame_logits`` for temporal supervision and debugging.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from .portable_m2d import PortableM2D
from .m2d_sc import (
    ArcMarginProduct,
    AttentiveStatsPool,
    PretrainedFusionHead,
    _MultiBranchPretrainedSEDFusion,
    _PretrainedSEDFeatureBranch,
    _resample_waveform,
)


class _TemporalConvRefiner(nn.Module):
    """Small residual temporal refiner over M2D frame embeddings."""

    def __init__(self, dim, num_layers=2, kernel_size=5, dropout=0.1):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("temporal_kernel_size must be a positive odd integer")

        layers = []
        for _ in range(int(num_layers)):
            layers.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(dim),
                        "dwconv": nn.Conv1d(
                            dim,
                            dim,
                            kernel_size=kernel_size,
                            padding=kernel_size // 2,
                            groups=dim,
                        ),
                        "pwconv": nn.Linear(dim, dim),
                        "dropout": nn.Dropout(dropout),
                    }
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # x: [B, T, D]
        for layer in self.layers:
            residual = x
            y = layer["norm"](x)
            y = layer["dwconv"](y.transpose(1, 2)).transpose(1, 2)
            y = torch.nn.functional.gelu(y)
            y = layer["pwconv"](y)
            y = layer["dropout"](y)
            x = residual + y
        return x


class M2DTemporalPretrainedSEDFusionClassifier(PortableM2D):
    """Frame-aware source classifier with optional BEATs/ATST-F/fPaSST fusion.

    Expected input is either a waveform tensor or a dict containing:
    ``waveform``: ``[B, T]`` or ``[B, C, T]`` audio at ``input_sample_rate``;
    ``class_index``: optional labels used by the ArcFace head.

    Returns the same clip-level keys as the existing SC models:
    ``logits``, ``plain_logits``, ``energy``, ``embedding``.
    It additionally returns:
    ``activity_logits``: ``[B, T_frame]`` binary source-activity logits;
    ``frame_logits``: ``[B, T_frame, num_classes]`` per-frame class logits;
    ``timestamps_sec`` and ``duration_sec`` for temporal target alignment.
    """

    def __init__(
        self,
        weight_file,
        num_classes=18,
        embedding_dim=512,
        m2d_embedding_dim=512,
        pooling_hidden_dim=512,
        projection_hidden_dim=1024,
        dropout=0.2,
        finetuning_layers="2_blocks",
        energy_thresholds=None,
        ref_channel=None,
        pretrainedsed_repo_root=None,
        pretrainedsed_checkpoint_dir="checkpoint/pretrainedsed",
        pretrainedsed_models=("BEATs", "ATST-F", "fpasst"),
        pretrainedsed_checkpoint_variant="strong_1",
        pretrainedsed_pooling="mean",
        pretrainedsed_seq_len=250,
        pretrainedsed_embed_dim=768,
        pretrainedsed_download_if_missing=True,
        pretrainedsed_sample_rate=16000,
        freeze_pretrainedsed=True,
        fusion_strategy="weighted_avg",
        fusion_hidden_dim=1024,
        pretrained_fusion_mode="concat_mlp",
        input_sample_rate=32000,
        temporal_layers=2,
        temporal_kernel_size=5,
        temporal_pooling="attentive_stats",
        activity_hidden_dim=256,
    ):
        super().__init__(weight_file, num_classes=None, freeze_embed=False, flat_features=None)
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.m2d_embedding_dim = int(m2d_embedding_dim)
        self.energy_thresholds = energy_thresholds or {}
        self.ref_channel = ref_channel
        self.input_sample_rate = int(input_sample_rate)
        self.pretrainedsed_sample_rate = int(pretrainedsed_sample_rate)
        self.temporal_pooling = temporal_pooling

        self.m2d_frame_projection = nn.Sequential(
            nn.LayerNorm(self.cfg.feature_d),
            nn.Linear(self.cfg.feature_d, projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, self.m2d_embedding_dim),
            nn.LayerNorm(self.m2d_embedding_dim),
        )

        pretrainedsed_models = list(pretrainedsed_models or [])
        self.pretrainedsed_branches = nn.ModuleDict()
        for model_name in pretrainedsed_models:
            branch = _PretrainedSEDFeatureBranch(
                repo_root=pretrainedsed_repo_root,
                model_name=model_name,
                checkpoint_dir=pretrainedsed_checkpoint_dir,
                checkpoint_variant=pretrainedsed_checkpoint_variant,
                pooling=pretrainedsed_pooling,
                freeze=freeze_pretrainedsed,
                seq_len=pretrainedsed_seq_len,
                embed_dim=pretrainedsed_embed_dim,
                download_if_missing=pretrainedsed_download_if_missing,
            )
            self.pretrainedsed_branches[branch.model_name] = branch

        if self.pretrainedsed_branches:
            branch_dims = OrderedDict(
                (name, branch.output_dim)
                for name, branch in self.pretrainedsed_branches.items()
            )
            self.pretrainedsed_fusion = _MultiBranchPretrainedSEDFusion(
                branch_dims=branch_dims,
                output_dim=self.m2d_embedding_dim,
                hidden_dim=fusion_hidden_dim,
                dropout=dropout,
                fusion_strategy=fusion_strategy,
            )
            self.frame_fusion = PretrainedFusionHead(
                m2d_embedding_dim=self.m2d_embedding_dim,
                aux_embedding_dim=self.m2d_embedding_dim,
                output_dim=self.embedding_dim,
                hidden_dim=fusion_hidden_dim,
                dropout=dropout,
                fusion_mode=pretrained_fusion_mode,
            )
            self.clip_fusion = PretrainedFusionHead(
                m2d_embedding_dim=self.m2d_embedding_dim,
                aux_embedding_dim=self.m2d_embedding_dim,
                output_dim=self.embedding_dim,
                hidden_dim=fusion_hidden_dim,
                dropout=dropout,
                fusion_mode=pretrained_fusion_mode,
            )
        else:
            self.pretrainedsed_fusion = None
            self.frame_fusion = None
            self.clip_fusion = nn.Sequential(
                nn.LayerNorm(self.m2d_embedding_dim),
                nn.Linear(self.m2d_embedding_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
            )

        self.temporal_refiner = _TemporalConvRefiner(
            dim=self.embedding_dim,
            num_layers=temporal_layers,
            kernel_size=temporal_kernel_size,
            dropout=dropout,
        )

        if temporal_pooling == "attentive_stats":
            self.temporal_pool = AttentiveStatsPool(
                input_dim=self.embedding_dim,
                hidden_dim=pooling_hidden_dim,
                dropout=dropout,
            )
            pooled_dim = self.temporal_pool.output_dim
        elif temporal_pooling == "mean":
            self.temporal_pool = None
            pooled_dim = self.embedding_dim
        elif temporal_pooling == "mean_max":
            self.temporal_pool = None
            pooled_dim = self.embedding_dim * 2
        else:
            raise ValueError(
                f"Unsupported temporal_pooling={temporal_pooling!r}; "
                "use attentive_stats, mean, or mean_max."
            )

        self.clip_projection = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, self.embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
        )
        self.arc_head = ArcMarginProduct(self.embedding_dim, out_features=self.num_classes)
        self.plain_head = nn.Linear(self.embedding_dim, self.num_classes)
        self.frame_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, self.num_classes),
        )
        self.activity_head = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, activity_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(activity_hidden_dim, 1),
        )

        self._apply_finetuning_policy(finetuning_layers)

    def _apply_finetuning_policy(self, finetuning_layers):
        modules = [
            self.backbone.cls_token,
            self.backbone.pos_embed,
            self.backbone.patch_embed,
            self.backbone.pos_drop,
            self.backbone.patch_drop,
            self.backbone.norm_pre,
        ]
        modules.extend(list(self.backbone.blocks))
        modules.extend(
            [
                self.backbone.norm,
                self.backbone.fc_norm,
                self.backbone.head_drop,
                self.m2d_frame_projection,
                self.temporal_refiner,
                self.clip_projection,
                self.arc_head,
                self.plain_head,
                self.frame_head,
                self.activity_head,
            ]
        )
        if self.frame_fusion is not None:
            modules.extend([self.pretrainedsed_fusion, self.frame_fusion, self.clip_fusion])

        finetuning_modules_idx = {
            "head": 6 + len(self.backbone.blocks) + 3,
            "backbone_out": 6 + len(self.backbone.blocks),
            "all": 0,
        }
        for n_blocks in range(1, len(self.backbone.blocks) + 1):
            finetuning_modules_idx[f"{n_blocks}_blocks"] = 6 + len(self.backbone.blocks) - n_blocks
        modules_idx = finetuning_modules_idx.get(finetuning_layers, finetuning_modules_idx["head"])

        for idx, module in enumerate(modules):
            requires_grad = idx >= modules_idx
            if isinstance(module, torch.nn.parameter.Parameter):
                module.requires_grad = requires_grad
            elif module is not None:
                for param in module.parameters():
                    param.requires_grad = requires_grad

    def _unpack_inputs(self, inputs):
        if isinstance(inputs, dict):
            waveform = inputs["waveform"]
            labels = inputs.get("class_index")
        else:
            waveform = inputs
            labels = None
        return waveform, labels

    def _select_mono(self, waveform):
        if waveform.dim() == 3:
            if self.ref_channel is not None:
                waveform = waveform[:, int(self.ref_channel)]
            else:
                waveform = waveform.mean(dim=1)
        if waveform.dim() != 2:
            raise ValueError(f"Expected waveform with shape [B, T] or [B, C, T], got {tuple(waveform.shape)}")
        return waveform.float()

    def _aux_waveform(self, mono_waveform):
        return _resample_waveform(
            mono_waveform,
            self.input_sample_rate,
            self.pretrainedsed_sample_rate,
        )

    def _forward_pretrainedsed(self, mono_waveform):
        if not self.pretrainedsed_branches:
            return None, OrderedDict(), None

        aux_waveform = self._aux_waveform(mono_waveform)
        branch_embeddings = OrderedDict(
            (name, branch(aux_waveform))
            for name, branch in self.pretrainedsed_branches.items()
        )
        projected = self.pretrainedsed_fusion.project(branch_embeddings)
        fused, branch_weights = self.pretrainedsed_fusion.fuse(projected)
        return fused, branch_embeddings, branch_weights

    def _pool_frames(self, frame_embedding):
        if self.temporal_pooling == "attentive_stats":
            return self.temporal_pool(frame_embedding)
        if self.temporal_pooling == "mean_max":
            return torch.cat(
                [frame_embedding.mean(dim=1), frame_embedding.amax(dim=1)],
                dim=-1,
            )
        return frame_embedding.mean(dim=1)

    def _timestamps(self, num_frames, duration_sec, device, dtype):
        unit = torch.linspace(0.0, 1.0, num_frames, device=device, dtype=dtype)
        return duration_sec[:, None].to(device=device, dtype=dtype) * unit[None, :]

    def forward(self, inputs):
        waveform, labels = self._unpack_inputs(inputs)
        mono_waveform = self._select_mono(waveform)
        duration_sec = mono_waveform.new_full(
            (mono_waveform.shape[0],),
            float(mono_waveform.shape[-1]) / float(self.input_sample_rate),
        )

        # M2D temporal features: [B, T_frame, cfg.feature_d].
        m2d_frames = self.encode(mono_waveform)
        m2d_frames = self.m2d_frame_projection(m2d_frames)
        m2d_clip = m2d_frames.mean(dim=1)

        aux_fused, aux_branch_embeddings, aux_branch_weights = self._forward_pretrainedsed(mono_waveform)
        if aux_fused is not None:
            bsz, n_frames, _ = m2d_frames.shape
            aux_frames = aux_fused[:, None, :].expand(-1, n_frames, -1)
            frame_embedding = self.frame_fusion(
                m2d_frames.reshape(bsz * n_frames, -1),
                aux_frames.reshape(bsz * n_frames, -1),
            ).reshape(bsz, n_frames, -1)
            clip_seed = self.clip_fusion(m2d_clip, aux_fused)
        else:
            frame_embedding = self.clip_fusion(m2d_frames)
            clip_seed = self.clip_fusion(m2d_clip)

        frame_embedding = self.temporal_refiner(frame_embedding)
        pooled = self._pool_frames(frame_embedding)
        clip_embedding = self.clip_projection(pooled) + clip_seed
        clip_embedding = torch.nn.functional.layer_norm(clip_embedding, (clip_embedding.shape[-1],))

        if labels is not None:
            labels = labels.to(device=clip_embedding.device, dtype=torch.long).clamp(0, self.num_classes - 1)
        logits = self.arc_head(clip_embedding, labels)
        plain_logits = self.plain_head(clip_embedding)
        frame_logits = self.frame_head(frame_embedding)
        activity_logits = self.activity_head(frame_embedding).squeeze(-1)
        energy = torch.logsumexp(plain_logits.float(), dim=-1)
        timestamps_sec = self._timestamps(
            num_frames=frame_embedding.shape[1],
            duration_sec=duration_sec,
            device=frame_embedding.device,
            dtype=frame_embedding.dtype,
        )

        return {
            "logits": logits,
            "plain_logits": plain_logits,
            "energy": energy,
            "embedding": clip_embedding,
            "frame_embedding": frame_embedding,
            "frame_logits": frame_logits,
            "plain_frame_logits": frame_logits,
            "activity_logits": activity_logits,
            "timestamps_sec": timestamps_sec,
            "duration_sec": duration_sec,
            "aux_branch_embeddings": aux_branch_embeddings,
            "aux_branch_weights": aux_branch_weights,
        }


# Alias with the wording used by the non-temporal fusion classifier name.
M2DPretrainedSEDTemporalFusionClassifier = M2DTemporalPretrainedSEDFusionClassifier
