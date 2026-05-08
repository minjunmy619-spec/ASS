import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import download_url_to_file

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


def _set_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad


def _resample_waveform(waveform, input_sample_rate, target_sample_rate):
    if input_sample_rate is None or target_sample_rate is None or int(input_sample_rate) == int(target_sample_rate):
        return waveform

    try:
        import torchaudio.functional as AF
        return AF.resample(waveform, int(input_sample_rate), int(target_sample_rate))
    except Exception:
        target_length = max(1, int(round(waveform.shape[-1] * float(target_sample_rate) / float(input_sample_rate))))
        return F.interpolate(
            waveform.unsqueeze(1),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)


_PRETRAINEDSED_RELEASE_URL = "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1"
_PRETRAINEDSED_ASSETS = {
    "BEATs_strong_1": f"{_PRETRAINEDSED_RELEASE_URL}/BEATs_strong_1.pt",
    "ATST-F_strong_1": f"{_PRETRAINEDSED_RELEASE_URL}/ATST-F_strong_1.pt",
    "fpasst_strong_1": f"{_PRETRAINEDSED_RELEASE_URL}/fpasst_strong_1.pt",
    "BEATs_weak": f"{_PRETRAINEDSED_RELEASE_URL}/BEATs_weak.pt",
    "ATST-F_weak": f"{_PRETRAINEDSED_RELEASE_URL}/ATST-F_weak.pt",
    "fpasst_weak": f"{_PRETRAINEDSED_RELEASE_URL}/fpasst_weak.pt",
    "BEATs_ssl": f"{_PRETRAINEDSED_RELEASE_URL}/BEATs_ssl.pt",
    "ATST-F_ssl": f"{_PRETRAINEDSED_RELEASE_URL}/ATST-F_ssl.pt",
    "fpasst_ssl": f"{_PRETRAINEDSED_RELEASE_URL}/fpasst_ssl.pt",
}


def _canonical_pretrainedsed_model_name(name):
    aliases = {
        "beats": "BEATs",
        "BEATS": "BEATs",
        "atst": "ATST-F",
        "ATST": "ATST-F",
        "atst-f": "ATST-F",
        "ATST_Frame": "ATST-F",
        "ATST-Frame": "ATST-F",
        "ast": "fpasst",
        "AST": "fpasst",
        "fPaSST": "fpasst",
        "PaSST": "fpasst",
        "passt": "fpasst",
    }
    return aliases.get(str(name), str(name))


def _checkpoint_name_for_pretrainedsed(model_name, variant="strong_1"):
    model_name = _canonical_pretrainedsed_model_name(model_name)
    if model_name not in {"BEATs", "ATST-F", "fpasst"}:
        raise ValueError(
            f"Unsupported PretrainedSED model {model_name!r}. "
            "Use one of: BEATs, ATST-F, fpasst. AST is accepted as a fpasst alias."
        )
    if variant not in {"strong_1", "weak", "ssl"}:
        raise ValueError(f"Unsupported PretrainedSED checkpoint variant: {variant!r}")
    return f"{model_name}_{variant}"


def _resolve_pretrainedsed_checkpoint(checkpoint_dir, checkpoint_name, download_if_missing=True):
    path = Path(checkpoint_dir) / f"{checkpoint_name}.pt"
    if path.exists():
        return path
    if not download_if_missing:
        raise FileNotFoundError(
            f"Missing PretrainedSED checkpoint {path}. Download it from "
            f"{_PRETRAINEDSED_ASSETS[checkpoint_name]} or enable download_if_missing."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    download_url_to_file(_PRETRAINEDSED_ASSETS[checkpoint_name], str(path))
    return path


def _load_torch_checkpoint(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _remap_pretrainedsed_state_dict(checkpoint_name, state_dict):
    if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
        state_dict = state_dict["state_dict"]

    if "fpasst" in checkpoint_name:
        return {
            ("model.fpasst." + k[len("model."):]) if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }
    if "M2D" in checkpoint_name:
        return {
            ("model.m2d." + k[len("model."):]) if (k.startswith("model.") and not k.startswith("model.m2d.")) else k: v
            for k, v in state_dict.items()
        }
    if "BEATs" in checkpoint_name:
        return {
            ("model.beats." + k[len("model.model."):]) if k.startswith("model.model.") else k: v
            for k, v in state_dict.items()
        }
    if "ASIT" in checkpoint_name:
        return {
            ("model.asit." + k[len("model."):]) if k.startswith("model.") else k: v
            for k, v in state_dict.items()
        }
    return dict(state_dict)


def _drop_pretrainedsed_prediction_heads(state_dict):
    return {
        k: v for k, v in state_dict.items()
        if not (k.startswith("weak_head.") or k.startswith("strong_head."))
    }


def _load_pretrainedsed_feature_checkpoint(model, checkpoint_name, checkpoint_path):
    state_dict = _load_torch_checkpoint(checkpoint_path)
    state_dict = _remap_pretrainedsed_state_dict(checkpoint_name, state_dict)
    state_dict = _drop_pretrainedsed_prediction_heads(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [
        k for k in missing
        if not (k.startswith("weak_head.") or k.startswith("strong_head.") or "mel_transform" in k)
    ]
    unexpected = [k for k in unexpected if "mel_transform" not in k]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch for {checkpoint_name}: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    return missing, unexpected


class FrozenPretrainedAudioEncoder(nn.Module):
    """Optional frozen auxiliary audio encoder for source-classifier fusion.

    Supported practical paths:
    - ``official_beats``: local Microsoft BEATs code + local ``.pt`` checkpoint.
    - ``official_fpasst``: local PretrainedSED code + local ``fPaSST`` checkpoint.
    - ``transformers``: Hugging Face ``AutoModel``/``AutoModelForAudioClassification``.
    - ``torchscript``: a local TorchScript module returning an embedding tensor.

    The ``identity`` backend is intentionally tiny and is only for smoke tests
    on machines without pretrained checkpoints.
    """

    def __init__(
        self,
        aux_model="beats",
        aux_weight=None,
        aux_backend="auto",
        aux_embedding_dim=None,
        input_sample_rate=32000,
        aux_sample_rate=16000,
        pooling="mean",
        freeze=True,
        aux_input_mode="auto",
        aux_use_logits=False,
        aux_feature_extractor_weight=None,
        trust_remote_code=True,
        local_files_only=False,
        beats_source_dir=None,
        beats_use_finetuned_logits=False,
        fpasst_source_dir=None,
        fpasst_seq_len=250,
        fpasst_embed_dim=768,
    ):
        super().__init__()
        self.aux_model = aux_model
        self.aux_weight = aux_weight
        self.aux_backend = self._resolve_backend(aux_model, aux_weight, aux_backend)
        self.input_sample_rate = input_sample_rate
        self.aux_sample_rate = aux_sample_rate
        self.pooling = pooling
        self.freeze = freeze
        self.aux_input_mode = aux_input_mode
        self.aux_use_logits = aux_use_logits
        self.aux_feature_extractor_weight = aux_feature_extractor_weight or aux_weight
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.beats_source_dir = beats_source_dir
        self.beats_use_finetuned_logits = beats_use_finetuned_logits
        self.fpasst_source_dir = fpasst_source_dir
        self.fpasst_seq_len = fpasst_seq_len
        self.fpasst_embed_dim = fpasst_embed_dim
        self.model = None
        self.feature_extractor = None
        self.embedding_dim = aux_embedding_dim

        if self.aux_backend == "identity":
            self.embedding_dim = aux_embedding_dim or 16
        elif self.aux_backend == "official_beats":
            self._init_official_beats()
        elif self.aux_backend == "official_fpasst":
            self._init_official_fpasst()
        elif self.aux_backend == "transformers":
            self._init_transformers()
        elif self.aux_backend == "torchscript":
            self._init_torchscript()
        else:
            raise ValueError(f"Unsupported aux_backend: {self.aux_backend}")

        if self.embedding_dim is None:
            raise ValueError(
                "aux_embedding_dim could not be inferred. Set it explicitly in the config "
                f"for aux_model={aux_model!r}, aux_backend={self.aux_backend!r}."
            )

        if self.model is not None and freeze:
            _set_requires_grad(self.model, False)
            self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.model is not None and self.freeze:
            self.model.eval()
        return self

    def _resolve_backend(self, aux_model, aux_weight, aux_backend):
        if aux_backend != "auto":
            return aux_backend
        if aux_model in {"identity", "dummy", "none"}:
            return "identity"
        if aux_model == "fpasst":
            return "official_fpasst"
        if aux_model == "beats" and aux_weight is not None and Path(str(aux_weight)).suffix in {".pt", ".pth"}:
            return "official_beats"
        if aux_weight is not None and Path(str(aux_weight)).suffix in {".ts", ".jit", ".torchscript"}:
            return "torchscript"
        return "transformers"

    def _init_official_beats(self):
        if self.aux_weight is None:
            raise ValueError("official BEATs backend requires aux_weight pointing to a local .pt checkpoint.")
        if self.beats_source_dir is not None:
            sys.path.insert(0, str(self.beats_source_dir))
        try:
            from BEATs import BEATs, BEATsConfig
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Could not import official BEATs. Set beats_source_dir to the folder containing BEATs.py "
                "from https://github.com/microsoft/unilm/tree/master/beats, or install that code on PYTHONPATH."
            ) from exc

        checkpoint = torch.load(self.aux_weight, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        self.model = model
        if self.embedding_dim is None:
            if self.beats_use_finetuned_logits and "label_dict" in checkpoint:
                self.embedding_dim = len(checkpoint["label_dict"])
            else:
                self.embedding_dim = getattr(cfg, "encoder_embed_dim", None)

    def _load_predictions_wrapper_checkpoint(self, model, checkpoint_or_name, kind):
        checkpoint_or_name = str(checkpoint_or_name)
        if checkpoint_or_name.endswith((".pt", ".pth", ".ckpt")):
            state_dict = torch.load(checkpoint_or_name, map_location="cpu", weights_only=False)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        else:
            model.load_checkpoint(checkpoint_or_name)
            return

        if kind == "fpasst":
            state_dict = {
                ("model.fpasst." + k[len("model."):] if k.startswith("model.") else k): v
                for k, v in state_dict.items()
            }
        else:
            raise ValueError(f"Unsupported predictions-wrapper checkpoint kind: {kind}")

        head_keys = {
            "weak_head.bias",
            "weak_head.weight",
            "strong_head.bias",
            "strong_head.weight",
        }
        state_dict = {k: v for k, v in state_dict.items() if k not in head_keys}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        unexpected = [k for k in unexpected if "mel" not in k]
        missing = [
            k for k in missing
            if ("mel" not in k) and not any(k.startswith(prefix) for prefix in head_keys)
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected keys while loading {kind} checkpoint: {unexpected[:10]}")
        if missing:
            raise RuntimeError(f"Missing keys while loading {kind} checkpoint: {missing[:10]}")

    def _init_official_fpasst(self):
        if self.fpasst_source_dir is None:
            raise ValueError(
                "official fPaSST backend requires fpasst_source_dir pointing to the PretrainedSED repo root."
            )
        sys.path.insert(0, str(self.fpasst_source_dir))
        try:
            from models.frame_passt.fpasst_wrapper import FPaSSTWrapper
            from models.prediction_wrapper import PredictionsWrapper
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Could not import fPaSST from PretrainedSED. Set fpasst_source_dir to the repo root of "
                "https://github.com/fschmid56/PretrainedSED."
            ) from exc

        base_model = FPaSSTWrapper()
        model = PredictionsWrapper(
            base_model,
            checkpoint=None,
            head_type=None,
            embed_dim=self.fpasst_embed_dim,
            seq_len=self.fpasst_seq_len,
        )
        if self.aux_weight is not None:
            self._load_predictions_wrapper_checkpoint(model, self.aux_weight, kind="fpasst")
        model.eval()
        self.model = model
        if self.embedding_dim is None:
            self.embedding_dim = self.fpasst_embed_dim

    def _init_transformers(self):
        if self.aux_weight is None:
            raise ValueError("transformers backend requires aux_weight, e.g. a Hugging Face model id or local folder.")
        try:
            if self.aux_use_logits:
                from transformers import AutoModelForAudioClassification as ModelCls
            else:
                from transformers import AutoModel as ModelCls
            from transformers import AutoFeatureExtractor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "transformers is required for aux_backend='transformers'. Install requirements.txt "
                "or use aux_backend='official_beats'/'torchscript' with local checkpoints."
            ) from exc

        self.model = ModelCls.from_pretrained(
            self.aux_weight,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        if self.aux_input_mode in {"auto", "feature_extractor"}:
            try:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    self.aux_feature_extractor_weight,
                    trust_remote_code=self.trust_remote_code,
                    local_files_only=self.local_files_only,
                )
            except Exception:
                if self.aux_input_mode == "feature_extractor":
                    raise
                self.feature_extractor = None

        if self.embedding_dim is None:
            cfg = getattr(self.model, "config", None)
            if self.aux_use_logits:
                value = getattr(cfg, "num_labels", None)
                if value is not None:
                    self.embedding_dim = int(value)
            if self.embedding_dim is None:
                for attr in ("hidden_size", "d_model", "projection_dim"):
                    value = getattr(cfg, attr, None)
                    if value is not None:
                        self.embedding_dim = int(value)
                        break

    def _init_torchscript(self):
        if self.aux_weight is None:
            raise ValueError("torchscript backend requires aux_weight pointing to a local TorchScript file.")
        self.model = torch.jit.load(self.aux_weight, map_location="cpu")

    def _prepare_waveform(self, waveform):
        if waveform.dim() == 3:
            waveform = waveform.mean(dim=1)
        waveform = waveform.float()
        return _resample_waveform(waveform, self.input_sample_rate, self.aux_sample_rate)

    def _pool_sequence(self, sequence):
        if sequence.dim() == 2:
            return sequence
        if self.pooling == "cls":
            return sequence[:, 0]
        if self.pooling == "max":
            return sequence.amax(dim=1)
        return sequence.mean(dim=1)

    def _identity_embedding(self, waveform):
        mean = waveform.mean(dim=-1)
        std = waveform.std(dim=-1, unbiased=False)
        max_value = waveform.amax(dim=-1)
        min_value = waveform.amin(dim=-1)
        rms = torch.sqrt(torch.clamp((waveform ** 2).mean(dim=-1), min=1e-8))
        zero_cross = (waveform[..., 1:] * waveform[..., :-1] < 0).float().mean(dim=-1)
        base = torch.stack([mean, std, max_value, min_value, rms, zero_cross], dim=-1)
        repeat = (self.embedding_dim + base.shape[-1] - 1) // base.shape[-1]
        return base.repeat(1, repeat)[:, : self.embedding_dim]

    def _forward_official_beats(self, waveform):
        padding_mask = torch.zeros(waveform.shape, device=waveform.device, dtype=torch.bool)
        output = self.model.extract_features(waveform, padding_mask=padding_mask)[0]
        return self._pool_sequence(output)

    def _forward_transformers(self, waveform):
        if self.feature_extractor is not None and self.aux_input_mode != "waveform":
            arrays = [x.detach().cpu().numpy() for x in waveform]
            model_inputs = self.feature_extractor(
                arrays,
                sampling_rate=self.aux_sample_rate,
                return_tensors="pt",
            )
            model_inputs = {k: v.to(waveform.device) for k, v in model_inputs.items()}
        else:
            model_inputs = {"input_values": waveform}

        output = self.model(**model_inputs)
        if self.aux_use_logits and hasattr(output, "logits"):
            return output.logits
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return self._pool_sequence(output.last_hidden_state)
        if isinstance(output, dict):
            if self.aux_use_logits and "logits" in output:
                return output["logits"]
            if "pooler_output" in output and output["pooler_output"] is not None:
                return output["pooler_output"]
            if "last_hidden_state" in output:
                return self._pool_sequence(output["last_hidden_state"])
        if isinstance(output, (tuple, list)):
            return self._pool_sequence(output[0])
        return self._pool_sequence(output)

    def _forward_torchscript(self, waveform):
        output = self.model(waveform)
        if isinstance(output, dict):
            for key in ("embedding", "embeddings", "features", "logits"):
                if key in output:
                    output = output[key]
                    break
        if isinstance(output, (tuple, list)):
            output = output[0]
        return self._pool_sequence(output)

    def _forward_official_fpasst(self, waveform):
        mel = self.model.mel_forward(waveform)
        output = self.model(mel)
        return self._pool_sequence(output)

    def forward(self, waveform):
        waveform = self._prepare_waveform(waveform)
        grad_enabled = torch.is_grad_enabled() and not self.freeze
        with torch.set_grad_enabled(grad_enabled):
            if self.aux_backend == "identity":
                embedding = self._identity_embedding(waveform)
            elif self.aux_backend == "official_beats":
                embedding = self._forward_official_beats(waveform)
            elif self.aux_backend == "official_fpasst":
                embedding = self._forward_official_fpasst(waveform)
            elif self.aux_backend == "transformers":
                embedding = self._forward_transformers(waveform)
            elif self.aux_backend == "torchscript":
                embedding = self._forward_torchscript(waveform)
            else:
                raise RuntimeError(f"Unsupported aux_backend: {self.aux_backend}")
        return embedding.detach() if self.freeze else embedding


class PretrainedFusionHead(nn.Module):
    def __init__(
        self,
        m2d_embedding_dim,
        aux_embedding_dim,
        output_dim,
        hidden_dim=1024,
        dropout=0.2,
        fusion_mode="concat_mlp",
        normalize_aux=True,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.normalize_aux = normalize_aux
        self.aux_norm = nn.LayerNorm(aux_embedding_dim) if normalize_aux else nn.Identity()

        if fusion_mode == "concat_mlp":
            input_dim = m2d_embedding_dim + aux_embedding_dim
            self.fusion = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        elif fusion_mode == "gated_mlp":
            self.aux_projection = nn.Linear(aux_embedding_dim, m2d_embedding_dim)
            self.gate = nn.Sequential(
                nn.LayerNorm(m2d_embedding_dim + aux_embedding_dim),
                nn.Linear(m2d_embedding_dim + aux_embedding_dim, m2d_embedding_dim),
                nn.Sigmoid(),
            )
            self.fusion = nn.Sequential(
                nn.LayerNorm(m2d_embedding_dim),
                nn.Linear(m2d_embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

    def forward(self, m2d_embedding, aux_embedding):
        aux_embedding = self.aux_norm(aux_embedding)
        if self.fusion_mode == "concat_mlp":
            return self.fusion(torch.cat([m2d_embedding, aux_embedding], dim=-1))

        aux_projected = self.aux_projection(aux_embedding)
        gate = self.gate(torch.cat([m2d_embedding, aux_embedding], dim=-1))
        return self.fusion(m2d_embedding + gate * aux_projected)


class _PretrainedSEDFeatureBranch(nn.Module):
    def __init__(
        self,
        repo_root,
        model_name,
        checkpoint_dir="checkpoint/pretrainedsed",
        checkpoint_variant="strong_1",
        pooling="mean",
        freeze=True,
        seq_len=250,
        embed_dim=768,
        download_if_missing=True,
    ):
        super().__init__()
        self.model_name = _canonical_pretrainedsed_model_name(model_name)
        self.pooling = pooling
        self.freeze = freeze
        self.output_dim = int(embed_dim)

        repo_root = Path(repo_root).expanduser().resolve()
        if not repo_root.exists():
            raise FileNotFoundError(f"pretrainedsed_repo_root does not exist: {repo_root}")
        repo_root = str(repo_root)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        try:
            from models.atstframe.ATSTF_wrapper import ATSTWrapper
            from models.beats.BEATs_wrapper import BEATsWrapper
            from models.frame_passt.fpasst_wrapper import FPaSSTWrapper
            from models.prediction_wrapper import PredictionsWrapper
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Could not import official PretrainedSED wrappers. Set pretrainedsed_repo_root "
                "to a clone of https://github.com/fschmid56/PretrainedSED."
            ) from exc

        if self.model_name == "BEATs":
            base_model = BEATsWrapper()
        elif self.model_name == "ATST-F":
            base_model = ATSTWrapper()
        elif self.model_name == "fpasst":
            base_model = FPaSSTWrapper()
        else:
            raise ValueError(f"Unsupported PretrainedSED model: {self.model_name}")

        self.wrapper = PredictionsWrapper(
            base_model,
            checkpoint=None,
            embed_dim=self.output_dim,
            seq_len=seq_len,
            head_type=None,
        )
        checkpoint_name = _checkpoint_name_for_pretrainedsed(self.model_name, checkpoint_variant)
        checkpoint_path = _resolve_pretrainedsed_checkpoint(
            checkpoint_dir,
            checkpoint_name,
            download_if_missing=download_if_missing,
        )
        _load_pretrainedsed_feature_checkpoint(self.wrapper, checkpoint_name, checkpoint_path)

        if self.freeze:
            _set_requires_grad(self.wrapper, False)
            self.wrapper.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.wrapper.eval()
        return self

    def _pool_sequence(self, sequence):
        if sequence.dim() == 2:
            return sequence
        if self.pooling == "cls":
            return sequence[:, 0]
        if self.pooling == "max":
            return sequence.amax(dim=1)
        if self.pooling == "mean":
            return sequence.mean(dim=1)
        raise ValueError(f"Unsupported PretrainedSED pooling: {self.pooling}")

    def forward(self, waveform_16k):
        grad_enabled = torch.is_grad_enabled() and not self.freeze
        with torch.set_grad_enabled(grad_enabled):
            mel = self.wrapper.mel_forward(waveform_16k)
            sequence = self.wrapper(mel)
            if isinstance(sequence, (tuple, list)):
                raise RuntimeError("Expected sequence output from PretrainedSED head_type=None branch.")
            embedding = self._pool_sequence(sequence)
        return embedding.detach() if self.freeze else embedding


class _MultiBranchPretrainedSEDFusion(nn.Module):
    def __init__(
        self,
        branch_dims,
        output_dim,
        hidden_dim=1024,
        dropout=0.2,
        fusion_strategy="weighted_avg",
    ):
        super().__init__()
        self.branch_names = tuple(branch_dims.keys())
        self.fusion_strategy = fusion_strategy
        self.projectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, output_dim),
            )
            for name, dim in branch_dims.items()
        })
        self.branch_weight_logits = nn.Parameter(torch.zeros(len(self.branch_names)))

        if fusion_strategy == "concat_head":
            concat_dim = output_dim * len(self.branch_names)
            self.concat_head = nn.Sequential(
                nn.LayerNorm(concat_dim),
                nn.Linear(concat_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        elif fusion_strategy in {"weighted_avg", "late_fusion"}:
            self.concat_head = None
        else:
            raise ValueError(
                f"Unsupported fusion_strategy: {fusion_strategy}. "
                "Use weighted_avg, concat_head, or late_fusion."
            )

    def global_weights(self):
        return torch.softmax(self.branch_weight_logits, dim=0)

    def project(self, branch_embeddings):
        return OrderedDict(
            (name, self.projectors[name](branch_embeddings[name]))
            for name in self.branch_names
        )

    def fuse(self, projected):
        if self.fusion_strategy == "concat_head":
            return self.concat_head(torch.cat([projected[name] for name in self.branch_names], dim=-1)), None

        weights = self.global_weights()
        fused = sum(weights[i] * projected[name] for i, name in enumerate(self.branch_names))
        return fused, weights


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
        return waveform.float()

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
        raw_labels = F.one_hot(indices, num_classes=self.num_classes).float()
        labels = raw_labels.clone()

        silence = []
        for idx, energy in zip(indices.tolist(), output["energy"].tolist()):
            threshold = self.energy_thresholds.get(str(idx), self.energy_thresholds.get(idx, self.energy_thresholds.get("default", None)))
            silence.append(False if threshold is None else energy > threshold)
        silence = torch.tensor(silence, device=labels.device, dtype=torch.bool)
        labels[silence] = 0.0

        return {
            "label_vector": labels,
            "raw_label_vector": raw_labels,
            "class_indices": indices,
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
        return waveform.float()

    def _embed_waveform(self, waveform):
        features = self.encode(waveform, average_per_time_frame=False)
        pooled = self.pool(features)
        return self.embedding(pooled)

    def _plain_logits_from_waveform(self, waveform):
        embedding = self._embed_waveform(waveform)
        return embedding, self.arc_head(embedding, None)

    def forward(self, input_dict):
        waveform = self._prepare_audio(input_dict["waveform"])
        if (not self.training) and (self.eval_crop_seconds is not None):
            embeddings = []
            plain_logits_all = []
            for crop in self._iter_eval_crops(waveform):
                crop_embedding, crop_plain_logits = self._plain_logits_from_waveform(crop)
                embeddings.append(crop_embedding)
                plain_logits_all.append(crop_plain_logits)
            embedding = torch.stack(embeddings, dim=0).mean(dim=0)
            plain_logits = torch.stack(plain_logits_all, dim=0).mean(dim=0)
        else:
            embedding, plain_logits = self._plain_logits_from_waveform(waveform)
        logits = self.arc_head(embedding, input_dict.get("class_index"))
        energy = -torch.logsumexp(plain_logits, dim=-1)
        return {
            "embedding": embedding,
            "logits": logits,
            "plain_logits": plain_logits,
            "energy": energy,
        }

    def _eval_sample_rate(self):
        return getattr(self.cfg, "sample_rate", 32000 if getattr(self.cfg, "sr", "32k") == "32k" else 16000)

    def _iter_eval_crop_slices(self, waveform):
        if self.eval_crop_seconds is None:
            return [(0, waveform)]
        sample_rate = self._eval_sample_rate()
        crop_samples = int(round(float(self.eval_crop_seconds) * sample_rate))
        hop_seconds = self.eval_crop_hop_seconds or self.eval_crop_seconds
        hop_samples = int(round(float(hop_seconds) * sample_rate))
        if crop_samples <= 0 or hop_samples <= 0 or waveform.shape[-1] <= crop_samples:
            return [(0, waveform)]

        starts = list(range(0, waveform.shape[-1] - crop_samples + 1, hop_samples))
        last_start = waveform.shape[-1] - crop_samples
        if starts[-1] != last_start:
            starts.append(last_start)
        return [(start, waveform[..., start : start + crop_samples]) for start in starts]

    def _iter_eval_crops(self, waveform):
        return [crop for _, crop in self._iter_eval_crop_slices(waveform)]

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
        raw_labels = F.one_hot(indices, num_classes=self.num_classes).float()
        labels = raw_labels.clone()

        silence = []
        for idx, evalue in zip(indices.tolist(), energy.tolist()):
            threshold = self.energy_thresholds.get(str(idx), self.energy_thresholds.get(idx, self.energy_thresholds.get("default", None)))
            silence.append(False if threshold is None else evalue > threshold)
        silence = torch.tensor(silence, device=labels.device, dtype=torch.bool)
        labels[silence] = 0.0

        return {
            "label_vector": labels,
            "raw_label_vector": raw_labels,
            "class_indices": indices,
            "probabilities": values,
            "energy": energy,
            "silence": silence,
        }


class M2DSingleClassifierTemporalStrong(M2DSingleClassifierStrong):
    """Strong M2D classifier with frame-level activity supervision.

    This keeps the public SC output contract from ``M2DSingleClassifierStrong``
    and adds ``activity_logits`` for span-derived training targets.
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
        activity_hidden_dim=256,
        activity_temperature=1.0,
    ):
        super().__init__(
            weight_file=weight_file,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            finetuning_layers=finetuning_layers,
            pooling_hidden_dim=pooling_hidden_dim,
            projection_hidden_dim=projection_hidden_dim,
            dropout=dropout,
            energy_thresholds=energy_thresholds,
            ref_channel=ref_channel,
            eval_crop_seconds=eval_crop_seconds,
            eval_crop_hop_seconds=eval_crop_hop_seconds,
        )
        self.activity_head = nn.Sequential(
            nn.LayerNorm(self.cfg.feature_d),
            nn.Linear(self.cfg.feature_d, activity_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(activity_hidden_dim, 1),
        )
        self.activity_temperature = float(activity_temperature)

    def _temporal_pool(self, features, activity_logits):
        activity = torch.sigmoid(activity_logits / max(self.activity_temperature, 1e-6))
        weights = activity.clamp_min(1e-4)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weighted_mean = torch.sum(features * weights.unsqueeze(-1), dim=1)
        weighted_var = torch.sum(weights.unsqueeze(-1) * (features - weighted_mean[:, None]).pow(2), dim=1)
        weighted_std = torch.sqrt(torch.clamp(weighted_var, min=1e-6))
        return torch.cat([features.mean(dim=1), features.amax(dim=1), weighted_mean, weighted_std], dim=-1)

    def _embed_waveform_with_activity(self, waveform):
        features = self.encode(waveform, average_per_time_frame=False)
        activity_logits = self.activity_head(features).squeeze(-1)
        pooled = self._temporal_pool(features, activity_logits)
        return self.embedding(pooled), activity_logits

    def _stitch_eval_activity(self, activity_chunks, crop_starts, total_samples, crop_samples):
        if len(activity_chunks) == 1 and crop_starts[0] == 0 and crop_samples >= total_samples:
            return activity_chunks[0]
        crop_frames = activity_chunks[0].shape[-1]
        total_frames = max(1, int(round(float(crop_frames) * float(total_samples) / float(crop_samples))))
        stitched = activity_chunks[0].new_zeros(activity_chunks[0].shape[0], total_frames)
        weights = activity_chunks[0].new_zeros(1, total_frames)
        for start, activity in zip(crop_starts, activity_chunks):
            frame_start = int(round(float(start) * float(total_frames) / float(total_samples)))
            frame_end = int(round(float(start + crop_samples) * float(total_frames) / float(total_samples)))
            frame_start = max(0, min(frame_start, total_frames - 1))
            frame_end = max(frame_start + 1, min(frame_end, total_frames))
            target_frames = frame_end - frame_start
            if activity.shape[-1] != target_frames:
                activity = F.interpolate(
                    activity.unsqueeze(1),
                    size=target_frames,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)
            stitched[:, frame_start:frame_end] += activity
            weights[:, frame_start:frame_end] += 1.0
        return stitched / weights.clamp_min(1.0)

    def forward(self, input_dict):
        waveform = self._prepare_audio(input_dict["waveform"])
        if (not self.training) and (self.eval_crop_seconds is not None):
            embeddings = []
            activity_logits_all = []
            plain_logits_all = []
            crop_slices = self._iter_eval_crop_slices(waveform)
            for _, crop in crop_slices:
                crop_embedding, crop_activity_logits = self._embed_waveform_with_activity(crop)
                embeddings.append(crop_embedding)
                activity_logits_all.append(crop_activity_logits)
                plain_logits_all.append(self.arc_head(crop_embedding, None))
            embedding = torch.stack(embeddings, dim=0).mean(dim=0)
            activity_logits = self._stitch_eval_activity(
                activity_logits_all,
                [start for start, _ in crop_slices],
                waveform.shape[-1],
                crop_slices[0][1].shape[-1],
            )
            plain_logits = torch.stack(plain_logits_all, dim=0).mean(dim=0)
        else:
            embedding, activity_logits = self._embed_waveform_with_activity(waveform)
            plain_logits = self.arc_head(embedding, None)
        logits = self.arc_head(embedding, input_dict.get("class_index"))
        energy = -torch.logsumexp(plain_logits, dim=-1)
        duration_sec = waveform.new_full((waveform.shape[0],), float(waveform.shape[-1]) / float(getattr(self.cfg, "sample_rate", 32000)))
        return {
            "embedding": embedding,
            "logits": logits,
            "plain_logits": plain_logits,
            "energy": energy,
            "activity_logits": activity_logits,
            "activity_probabilities": torch.sigmoid(activity_logits),
            "duration_sec": duration_sec,
        }

    def _plain_logits_from_waveform(self, waveform):
        embedding, _ = self._embed_waveform_with_activity(waveform)
        return embedding, self.arc_head(embedding, None)

    def predict(self, input_dict):
        waveform = self._prepare_audio(input_dict["waveform"])
        plain_logits_all = []
        activity_all = []
        energy_all = []
        crop_slices = self._iter_eval_crop_slices(waveform)
        for _, crop in crop_slices:
            embedding, activity_logits = self._embed_waveform_with_activity(crop)
            plain_logits = self.arc_head(embedding, None)
            plain_logits_all.append(plain_logits)
            activity_all.append(torch.sigmoid(activity_logits))
            energy_all.append(-torch.logsumexp(plain_logits, dim=-1))
        plain_logits = torch.stack(plain_logits_all, dim=0).mean(dim=0)
        energy = torch.stack(energy_all, dim=0).mean(dim=0)

        probs = torch.softmax(plain_logits, dim=-1)
        values, indices = torch.max(probs, dim=-1)
        raw_labels = F.one_hot(indices, num_classes=self.num_classes).float()
        labels = raw_labels.clone()

        silence = []
        for idx, evalue in zip(indices.tolist(), energy.tolist()):
            threshold = self.energy_thresholds.get(str(idx), self.energy_thresholds.get(idx, self.energy_thresholds.get("default", None)))
            silence.append(False if threshold is None else evalue > threshold)
        silence = torch.tensor(silence, device=labels.device, dtype=torch.bool)
        labels[silence] = 0.0

        return {
            "label_vector": labels,
            "raw_label_vector": raw_labels,
            "class_indices": indices,
            "probabilities": values,
            "energy": energy,
            "silence": silence,
            "activity_probabilities": self._stitch_eval_activity(
                activity_all,
                [start for start, _ in crop_slices],
                waveform.shape[-1],
                crop_slices[0][1].shape[-1],
            ),
        }


class M2DPretrainedFusionClassifier(M2DSingleClassifierStrong):
    """M2D source classifier fused with a frozen pretrained audio encoder.

    This is the recommended first upgrade over ``M2DSingleClassifierStrong``:
    keep the M2D attentive embedding as the stable task-specific branch, add a
    frozen semantic/audio-event branch such as BEATs, and train only the M2D
    fine-tuning slice plus a compact fusion head by default.
    """

    def __init__(
        self,
        weight_file,
        num_classes=18,
        embedding_dim=512,
        m2d_embedding_dim=None,
        finetuning_layers="2_blocks",
        pooling_hidden_dim=512,
        projection_hidden_dim=1024,
        dropout=0.2,
        energy_thresholds=None,
        ref_channel=None,
        eval_crop_seconds=None,
        eval_crop_hop_seconds=None,
        aux_model="beats",
        aux_weight=None,
        aux_backend="auto",
        aux_embedding_dim=None,
        aux_sample_rate=16000,
        input_sample_rate=32000,
        aux_pooling="mean",
        aux_input_mode="auto",
        aux_use_logits=False,
        aux_feature_extractor_weight=None,
        freeze_aux=True,
        fusion_mode="concat_mlp",
        fusion_hidden_dim=1024,
        normalize_aux=True,
        trust_remote_code=True,
        local_files_only=False,
        beats_source_dir=None,
        beats_use_finetuned_logits=False,
        fpasst_source_dir=None,
        fpasst_seq_len=250,
        fpasst_embed_dim=768,
    ):
        self.m2d_embedding_dim = m2d_embedding_dim or embedding_dim
        super().__init__(
            weight_file=weight_file,
            num_classes=num_classes,
            embedding_dim=self.m2d_embedding_dim,
            finetuning_layers=finetuning_layers,
            pooling_hidden_dim=pooling_hidden_dim,
            projection_hidden_dim=projection_hidden_dim,
            dropout=dropout,
            energy_thresholds=energy_thresholds,
            ref_channel=ref_channel,
            eval_crop_seconds=eval_crop_seconds,
            eval_crop_hop_seconds=eval_crop_hop_seconds,
        )
        self.aux_encoder = FrozenPretrainedAudioEncoder(
            aux_model=aux_model,
            aux_weight=aux_weight,
            aux_backend=aux_backend,
            aux_embedding_dim=aux_embedding_dim,
            input_sample_rate=input_sample_rate,
            aux_sample_rate=aux_sample_rate,
            pooling=aux_pooling,
            freeze=freeze_aux,
            aux_input_mode=aux_input_mode,
            aux_use_logits=aux_use_logits,
            aux_feature_extractor_weight=aux_feature_extractor_weight,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            beats_source_dir=beats_source_dir,
            beats_use_finetuned_logits=beats_use_finetuned_logits,
            fpasst_source_dir=fpasst_source_dir,
            fpasst_seq_len=fpasst_seq_len,
            fpasst_embed_dim=fpasst_embed_dim,
        )
        self.fusion_head = PretrainedFusionHead(
            m2d_embedding_dim=self.m2d_embedding_dim,
            aux_embedding_dim=self.aux_encoder.embedding_dim,
            output_dim=embedding_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            fusion_mode=fusion_mode,
            normalize_aux=normalize_aux,
        )
        self.arc_head = ArcMarginProduct(embedding_dim, out_features=num_classes)

    def _embed_waveform(self, waveform):
        m2d_embedding = super()._embed_waveform(waveform)
        aux_embedding = self.aux_encoder(waveform)
        return self.fusion_head(m2d_embedding, aux_embedding)


class M2DPretrainedSEDFusionClassifier(M2DSingleClassifierStrong):
    """M2D source classifier fused with official PretrainedSED release branches.

    This opt-in variant keeps the same public source-classifier return contract
    as ``M2DSingleClassifierStrong`` while adding frozen BEATs, ATST-F, and
    fPaSST feature branches from PretrainedSED v0.0.1.
    """

    def __init__(
        self,
        weight_file,
        num_classes=18,
        embedding_dim=512,
        m2d_embedding_dim=None,
        finetuning_layers="2_blocks",
        pooling_hidden_dim=512,
        projection_hidden_dim=1024,
        dropout=0.2,
        energy_thresholds=None,
        ref_channel=None,
        eval_crop_seconds=None,
        eval_crop_hop_seconds=None,
        pretrainedsed_repo_root=None,
        pretrainedsed_checkpoint_dir="checkpoint/pretrainedsed",
        pretrainedsed_models=("BEATs", "ATST-F", "fpasst"),
        pretrainedsed_checkpoint_variant="strong_1",
        pretrainedsed_pooling="mean",
        pretrainedsed_seq_len=250,
        pretrainedsed_embed_dim=768,
        pretrainedsed_download_if_missing=True,
        freeze_pretrainedsed=True,
        fusion_strategy="weighted_avg",
        fusion_hidden_dim=1024,
        input_sample_rate=None,
    ):
        self.m2d_embedding_dim = m2d_embedding_dim or embedding_dim
        super().__init__(
            weight_file=weight_file,
            num_classes=num_classes,
            embedding_dim=self.m2d_embedding_dim,
            finetuning_layers=finetuning_layers,
            pooling_hidden_dim=pooling_hidden_dim,
            projection_hidden_dim=projection_hidden_dim,
            dropout=dropout,
            energy_thresholds=energy_thresholds,
            ref_channel=ref_channel,
            eval_crop_seconds=eval_crop_seconds,
            eval_crop_hop_seconds=eval_crop_hop_seconds,
        )

        if pretrainedsed_repo_root is None:
            raise ValueError("pretrainedsed_repo_root must point to a PretrainedSED clone.")

        canonical_models = tuple(_canonical_pretrainedsed_model_name(name) for name in pretrainedsed_models)
        if len(set(canonical_models)) != len(canonical_models):
            raise ValueError(f"Duplicate PretrainedSED branches after alias resolution: {canonical_models}")
        self.pretrainedsed_models = canonical_models

        self.pretrainedsed_branches = nn.ModuleDict({
            name: _PretrainedSEDFeatureBranch(
                repo_root=pretrainedsed_repo_root,
                model_name=name,
                checkpoint_dir=pretrainedsed_checkpoint_dir,
                checkpoint_variant=pretrainedsed_checkpoint_variant,
                pooling=pretrainedsed_pooling,
                freeze=freeze_pretrainedsed,
                seq_len=pretrainedsed_seq_len,
                embed_dim=pretrainedsed_embed_dim,
                download_if_missing=pretrainedsed_download_if_missing,
            )
            for name in self.pretrainedsed_models
        })

        branch_dims = OrderedDict([("m2d", self.m2d_embedding_dim)])
        for name, branch in self.pretrainedsed_branches.items():
            branch_dims[name] = branch.output_dim

        self.fusion = _MultiBranchPretrainedSEDFusion(
            branch_dims=branch_dims,
            output_dim=embedding_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=dropout,
            fusion_strategy=fusion_strategy,
        )
        self.arc_head = ArcMarginProduct(embedding_dim, out_features=num_classes)
        self.input_sample_rate = input_sample_rate or getattr(
            self.cfg,
            "sample_rate",
            32000 if getattr(self.cfg, "sr", "32k") == "32k" else 16000,
        )

    def _collect_branch_embeddings(self, waveform):
        branch_embeddings = OrderedDict()
        branch_embeddings["m2d"] = super()._embed_waveform(waveform)

        waveform_16k = _resample_waveform(waveform, self.input_sample_rate, 16000)
        for name, branch in self.pretrainedsed_branches.items():
            branch_embeddings[name] = branch(waveform_16k)
        return branch_embeddings

    def _fused_outputs(self, waveform, class_index=None):
        branch_embeddings = self._collect_branch_embeddings(waveform)
        projected = self.fusion.project(branch_embeddings)

        if self.fusion.fusion_strategy == "late_fusion":
            weights = self.fusion.global_weights()
            plain_logits = sum(
                weights[i] * self.arc_head(projected[name], None)
                for i, name in enumerate(self.fusion.branch_names)
            )
            logits = sum(
                weights[i] * self.arc_head(projected[name], class_index)
                for i, name in enumerate(self.fusion.branch_names)
            )
            embedding = sum(
                weights[i] * projected[name]
                for i, name in enumerate(self.fusion.branch_names)
            )
            return embedding, plain_logits, logits, weights

        embedding, weights = self.fusion.fuse(projected)
        plain_logits = self.arc_head(embedding, None)
        logits = self.arc_head(embedding, class_index)
        return embedding, plain_logits, logits, weights

    def _plain_logits_from_waveform(self, waveform):
        embedding, plain_logits, _, _ = self._fused_outputs(waveform, class_index=None)
        return embedding, plain_logits

    def forward(self, input_dict):
        waveform = self._prepare_audio(input_dict["waveform"])
        class_index = input_dict.get("class_index")

        if (not self.training) and (self.eval_crop_seconds is not None):
            embeddings = []
            plain_logits_all = []
            logits_all = []
            branch_weights_all = []
            for crop in self._iter_eval_crops(waveform):
                embedding, plain_logits, logits, branch_weights = self._fused_outputs(crop, class_index)
                embeddings.append(embedding)
                plain_logits_all.append(plain_logits)
                logits_all.append(logits)
                if branch_weights is not None:
                    branch_weights_all.append(branch_weights)
            embedding = torch.stack(embeddings, dim=0).mean(dim=0)
            plain_logits = torch.stack(plain_logits_all, dim=0).mean(dim=0)
            logits = torch.stack(logits_all, dim=0).mean(dim=0)
            branch_weights = torch.stack(branch_weights_all, dim=0).mean(dim=0) if branch_weights_all else None
        else:
            embedding, plain_logits, logits, branch_weights = self._fused_outputs(waveform, class_index)

        energy = -torch.logsumexp(plain_logits, dim=-1)
        output = {
            "embedding": embedding,
            "logits": logits,
            "plain_logits": plain_logits,
            "energy": energy,
        }
        if branch_weights is not None:
            output["branch_weights"] = branch_weights
        return output

    def predict(self, input_dict):
        waveform = self._prepare_audio(input_dict["waveform"])
        plain_logits_all = []
        branch_weights_all = []
        for crop in self._iter_eval_crops(waveform):
            _, plain_logits, _, branch_weights = self._fused_outputs(crop, class_index=None)
            plain_logits_all.append(plain_logits)
            if branch_weights is not None:
                branch_weights_all.append(branch_weights)
        plain_logits = torch.stack(plain_logits_all, dim=0).mean(dim=0)
        energy = -torch.logsumexp(plain_logits, dim=-1)

        probs = torch.softmax(plain_logits, dim=-1)
        values, indices = torch.max(probs, dim=-1)
        raw_labels = F.one_hot(indices, num_classes=self.num_classes).float()
        labels = raw_labels.clone()

        silence = []
        for idx, evalue in zip(indices.tolist(), energy.tolist()):
            threshold = self.energy_thresholds.get(str(idx), self.energy_thresholds.get(idx, self.energy_thresholds.get("default", None)))
            silence.append(False if threshold is None else evalue > threshold)
        silence = torch.tensor(silence, device=labels.device, dtype=torch.bool)
        labels[silence] = 0.0

        output = {
            "label_vector": labels,
            "raw_label_vector": raw_labels,
            "class_indices": indices,
            "probabilities": values,
            "energy": energy,
            "silence": silence,
        }
        if branch_weights_all:
            output["branch_weights"] = torch.stack(branch_weights_all, dim=0).mean(dim=0)
        return output
