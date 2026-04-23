import torch
import torch.nn as nn
import torch.nn.functional as F

from .portable_m2d import PortableM2D


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=32.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, x, labels=None):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        if labels is None:
            return cosine * self.s
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-8))
        phi = cosine * torch.cos(torch.tensor(self.m, device=x.device)) - sine * torch.sin(torch.tensor(self.m, device=x.device))
        one_hot = F.one_hot(labels, num_classes=cosine.shape[-1]).float()
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.s


class AttentiveStatsPool(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.output_dim = input_dim * 4
        self.attention = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        attn_mean = torch.sum(weights * x, dim=1)
        attn_var = torch.sum(weights * (x - attn_mean[:, None]).pow(2), dim=1)
        attn_std = torch.sqrt(torch.clamp(attn_var, min=1e-6))
        mean = x.mean(dim=1)
        max_pool = x.amax(dim=1)
        return torch.cat([mean, max_pool, attn_mean, attn_std], dim=-1)


class M2DSingleClassifier(PortableM2D):
    def __init__(
        self,
        weight_file,
        num_classes=18,
        embedding_dim=512,
        finetuning_layers="2_blocks",
        energy_thresholds=None,
        ref_channel=None,
    ):
        super().__init__(weight_file, num_classes=None, freeze_embed=False, flat_features=None)
        self.num_classes = num_classes
        self.ref_channel = ref_channel
        self.energy_thresholds = energy_thresholds or {}

        self.embedding = nn.Sequential(
            nn.Linear(self.cfg.feature_d, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.arc_head = ArcMarginProduct(embedding_dim, out_features=num_classes)

        modules = [self.backbone.cls_token, self.backbone.pos_embed, self.backbone.patch_embed, self.backbone.pos_drop, self.backbone.patch_drop, self.backbone.norm_pre]
        for block in self.backbone.blocks:
            modules.append(block)
        modules.extend([self.backbone.norm, self.backbone.fc_norm, self.backbone.head_drop, self.embedding, self.arc_head])

        finetuning_modules_idx = {
            "head": len(modules) - 2,
            "backbone_out": 6 + len(self.backbone.blocks),
            "all": 0,
        }
        for n_blocks in range(1, len(self.backbone.blocks) + 1):
            finetuning_modules_idx[f"{n_blocks}_blocks"] = 6 + len(self.backbone.blocks) - n_blocks
        modules_idx = finetuning_modules_idx.get(finetuning_layers, len(modules) - 2)
        for i, module in enumerate(modules):
            if isinstance(module, torch.nn.parameter.Parameter):
                module.requires_grad = i >= modules_idx
            else:
                for param in module.parameters():
                    param.requires_grad = i >= modules_idx

    def _prepare_audio(self, waveform):
        if waveform.dim() == 3:
            if waveform.shape[1] == 1:
                waveform = waveform[:, 0]
            else:
                assert self.ref_channel is not None
                waveform = waveform[:, self.ref_channel]
        return waveform

    def forward(self, input_dict):
        waveform = self._prepare_audio(input_dict["waveform"])
        features = self.encode(waveform, average_per_time_frame=False).mean(1)
        embedding = self.embedding(features)
        logits = self.arc_head(embedding, input_dict.get("class_index"))
        plain_logits = self.arc_head(embedding, None)
        energy = -torch.logsumexp(plain_logits, dim=-1)
        return {
            "embedding": embedding,
            "logits": logits,
            "plain_logits": plain_logits,
            "energy": energy,
        }

    def predict(self, input_dict):
        output = self.forward(input_dict)
        probs = torch.softmax(output["plain_logits"], dim=-1)
        values, indices = torch.max(probs, dim=-1)
        labels = F.one_hot(indices, num_classes=self.num_classes).float()

        silence = []
        for idx, energy in zip(indices.tolist(), output["energy"].tolist()):
            threshold = self.energy_thresholds.get(str(idx), self.energy_thresholds.get(idx, self.energy_thresholds.get("default", None)))
            silence.append(False if threshold is None else energy > threshold)
        silence = torch.tensor(silence, device=labels.device, dtype=torch.bool)
        labels[silence] = 0.0

        return {
            "label_vector": labels,
            "probabilities": values,
            "energy": output["energy"],
            "silence": silence,
        }


class M2DSingleClassifierStrong(PortableM2D):
    """Stronger single-label M2D classifier for separated source tagging.

    Compared with ``M2DSingleClassifier``, this keeps temporal structure until a
    learned attentive-statistics pooling layer and uses a small MLP projection
    before ArcFace. The output keys are unchanged, so existing lightning/loss/S5
    code can use this class directly.
    """

    def __init__(
        self,
        weight_file,
        num_classes=18,
        embedding_dim=512,
        finetuning_layers="2_blocks",
        pooling_hidden_dim=512,
        projection_hidden_dim=1024,
        dropout=0.2,
        energy_thresholds=None,
        ref_channel=None,
        eval_crop_seconds=None,
        eval_crop_hop_seconds=None,
    ):
        super().__init__(weight_file, num_classes=None, freeze_embed=False, flat_features=None)
        self.num_classes = num_classes
        self.ref_channel = ref_channel
        self.energy_thresholds = energy_thresholds or {}
        self.eval_crop_seconds = eval_crop_seconds
        self.eval_crop_hop_seconds = eval_crop_hop_seconds

        self.pool = AttentiveStatsPool(
            self.cfg.feature_d,
            hidden_dim=pooling_hidden_dim,
            dropout=dropout,
        )
        self.embedding = nn.Sequential(
            nn.LayerNorm(self.pool.output_dim),
            nn.Linear(self.pool.output_dim, projection_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.arc_head = ArcMarginProduct(embedding_dim, out_features=num_classes)

        modules = [
            self.backbone.cls_token,
            self.backbone.pos_embed,
            self.backbone.patch_embed,
            self.backbone.pos_drop,
            self.backbone.patch_drop,
            self.backbone.norm_pre,
        ]
        for block in self.backbone.blocks:
            modules.append(block)
        modules.extend(
            [
                self.backbone.norm,
                self.backbone.fc_norm,
                self.backbone.head_drop,
                self.pool,
                self.embedding,
                self.arc_head,
            ]
        )

        finetuning_modules_idx = {
            "head": len(modules) - 3,
            "backbone_out": 6 + len(self.backbone.blocks),
            "all": 0,
        }
        for n_blocks in range(1, len(self.backbone.blocks) + 1):
            finetuning_modules_idx[f"{n_blocks}_blocks"] = 6 + len(self.backbone.blocks) - n_blocks
        modules_idx = finetuning_modules_idx.get(finetuning_layers, len(modules) - 3)
        for i, module in enumerate(modules):
            if isinstance(module, torch.nn.parameter.Parameter):
                module.requires_grad = i >= modules_idx
            else:
                for param in module.parameters():
                    param.requires_grad = i >= modules_idx

    def _prepare_audio(self, waveform):
        if waveform.dim() == 3:
            if waveform.shape[1] == 1:
                waveform = waveform[:, 0]
            else:
                assert self.ref_channel is not None
                waveform = waveform[:, self.ref_channel]
        return waveform

    def _embed_waveform(self, waveform):
        features = self.encode(waveform, average_per_time_frame=False)
        pooled = self.pool(features)
        return self.embedding(pooled)

    def _plain_logits_from_waveform(self, waveform):
        embedding = self._embed_waveform(waveform)
        return embedding, self.arc_head(embedding, None)

    def forward(self, input_dict):
        waveform = self._prepare_audio(input_dict["waveform"])
        embedding, plain_logits = self._plain_logits_from_waveform(waveform)
        logits = self.arc_head(embedding, input_dict.get("class_index"))
        energy = -torch.logsumexp(plain_logits, dim=-1)
        return {
            "embedding": embedding,
            "logits": logits,
            "plain_logits": plain_logits,
            "energy": energy,
        }

    def _iter_eval_crops(self, waveform):
        if self.eval_crop_seconds is None:
            return [waveform]
        sample_rate = getattr(self.cfg, "sample_rate", 32000 if getattr(self.cfg, "sr", "32k") == "32k" else 16000)
        crop_samples = int(round(float(self.eval_crop_seconds) * sample_rate))
        hop_seconds = self.eval_crop_hop_seconds or self.eval_crop_seconds
        hop_samples = int(round(float(hop_seconds) * sample_rate))
        if crop_samples <= 0 or hop_samples <= 0 or waveform.shape[-1] <= crop_samples:
            return [waveform]

        starts = list(range(0, waveform.shape[-1] - crop_samples + 1, hop_samples))
        last_start = waveform.shape[-1] - crop_samples
        if starts[-1] != last_start:
            starts.append(last_start)
        return [waveform[..., start : start + crop_samples] for start in starts]

    def predict(self, input_dict):
        waveform = self._prepare_audio(input_dict["waveform"])
        plain_logits_all = []
        for crop in self._iter_eval_crops(waveform):
            _, plain_logits = self._plain_logits_from_waveform(crop)
            plain_logits_all.append(plain_logits)
        plain_logits = torch.stack(plain_logits_all, dim=0).mean(dim=0)
        energy = -torch.logsumexp(plain_logits, dim=-1)

        probs = torch.softmax(plain_logits, dim=-1)
        values, indices = torch.max(probs, dim=-1)
        labels = F.one_hot(indices, num_classes=self.num_classes).float()

        silence = []
        for idx, evalue in zip(indices.tolist(), energy.tolist()):
            threshold = self.energy_thresholds.get(str(idx), self.energy_thresholds.get(idx, self.energy_thresholds.get("default", None)))
            silence.append(False if threshold is None else evalue > threshold)
        silence = torch.tensor(silence, device=labels.device, dtype=torch.bool)
        labels[silence] = 0.0

        return {
            "label_vector": labels,
            "probabilities": values,
            "energy": energy,
            "silence": silence,
        }
