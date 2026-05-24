import torch

from src.utils import LABELS, initialize_config

_TSE_EXTRA_CONDITION_KEYS = (
    "class_logits",
    "silence_logits",
    "pred_doa_vector",
    "doa_vector",
    "spatial_embedding",
)


class Kwon2025S5(torch.nn.Module):
    def __init__(
        self,
        label_set,
        uss_config,
        sc_config,
        tse_config,
        uss_ckpt=None,
        sc_ckpt=None,
        tse_ckpt=None,
        duplicate_recall_enabled=False,
        duplicate_recall_min_probability=0.35,
        duplicate_recall_min_waveform_rms=1e-4,
        uss_gate_enabled=False,
        uss_gate_count0_threshold=0.65,
        uss_gate_slot_active_threshold=0.45,
        uss_gate_slot_energy_threshold=1e-4,
        tse_uss_conditioning_enabled=False,
        tse_refinement_passes=2,
    ):
        super().__init__()
        self.uss = initialize_config(uss_config)
        self.sc = initialize_config(sc_config)
        self.tse = initialize_config(tse_config)
        self.label_set = label_set
        self.labels = LABELS[label_set].copy()
        self.onehots = torch.eye(len(self.labels), requires_grad=False).float()
        self.duplicate_recall_enabled = duplicate_recall_enabled
        self.duplicate_recall_min_probability = float(duplicate_recall_min_probability)
        self.duplicate_recall_min_waveform_rms = float(duplicate_recall_min_waveform_rms)
        self.uss_gate_enabled = bool(uss_gate_enabled)
        self.uss_gate_count0_threshold = float(uss_gate_count0_threshold)
        self.uss_gate_slot_active_threshold = float(uss_gate_slot_active_threshold)
        self.uss_gate_slot_energy_threshold = float(uss_gate_slot_energy_threshold)
        self.tse_uss_conditioning_enabled = bool(tse_uss_conditioning_enabled)
        self.tse_refinement_passes = self._validate_tse_refinement_passes(tse_refinement_passes)

        if uss_ckpt is not None:
            self._load_ckpt(uss_ckpt, self.uss, preferred_prefixes=("uss_model.", "model.", "module.", "net."))
        if sc_ckpt is not None:
            self._load_ckpt(sc_ckpt, self.sc, preferred_prefixes=("sc_model.", "model.", "module.", "net."))
        if tse_ckpt is not None:
            self._load_ckpt(tse_ckpt, self.tse, preferred_prefixes=("tse_model.", "model.", "module.", "net."))

        self.uss.eval()
        self.sc.eval()
        self.tse.eval()

    def _validate_tse_refinement_passes(self, passes):
        passes = int(passes)
        if passes not in (1, 2):
            raise ValueError("tse_refinement_passes must be 1 or 2")
        return passes

    def _get_tse_refinement_passes(self):
        return self._validate_tse_refinement_passes(getattr(self, "tse_refinement_passes", 2))

    def _select_model_ckpt_state(self, model_ckpt, model, preferred_prefixes=()):
        model_state = model.state_dict()
        if set(model_state.keys()) == set(model_ckpt.keys()):
            return model_ckpt
        for prefix in preferred_prefixes:
            stripped = {
                k[len(prefix) :]: v for k, v in model_ckpt.items() if isinstance(k, str) and k.startswith(prefix)
            }
            if stripped and any(k in model_state for k in stripped):
                return stripped
        one_model_key = next(iter(model_state.keys()))
        ckpt_corresponding_key = [k for k in model_ckpt.keys() if isinstance(k, str) and k.endswith(one_model_key)]
        if ckpt_corresponding_key:
            prefix = ckpt_corresponding_key[0][: -len(one_model_key)]
            return {k[len(prefix) :]: v for k, v in model_ckpt.items() if isinstance(k, str) and k.startswith(prefix)}
        return model_ckpt

    def _load_ckpt(self, path, model, preferred_prefixes=()):
        model_ckpt = torch.load(path, weights_only=False, map_location="cpu")["state_dict"]
        model_ckpt = self._select_model_ckpt_state(model_ckpt, model, preferred_prefixes=preferred_prefixes)
        model.load_state_dict(model_ckpt)

    def _vector_to_label(self, label_vector):
        labels = []
        for source_vec in label_vector:
            if source_vec.sum() == 0:
                labels.append("silence")
            else:
                labels.append(self.labels[int(torch.argmax(source_vec))])
        return labels

    def _classify_sources(self, waveforms):
        batch_size, n_sources, _, samples = waveforms.shape
        flat = waveforms[:, :, 0].reshape(batch_size * n_sources, samples)
        out = self.sc.predict({"waveform": flat})
        label_vector = out["label_vector"].view(batch_size, n_sources, -1)
        probs = out["probabilities"].view(batch_size, n_sources)
        if getattr(self, "duplicate_recall_enabled", False):
            raw_label_vector = out.get("raw_label_vector")
            if raw_label_vector is not None:
                raw_label_vector = raw_label_vector.view(batch_size, n_sources, -1)
                label_vector = self._recover_silenced_duplicates(waveforms, label_vector, raw_label_vector, probs)
        labels = [self._vector_to_label(label_vector[i]) for i in range(batch_size)]
        return labels, probs, label_vector

    def _recover_silenced_duplicates(self, waveforms, label_vector, raw_label_vector, probs):
        label_vector = label_vector.clone()
        min_probability = getattr(self, "duplicate_recall_min_probability", 0.35)
        min_waveform_rms = getattr(self, "duplicate_recall_min_waveform_rms", 1e-4)
        rms = waveforms[:, :, 0].pow(2).mean(dim=-1).sqrt()
        active = label_vector.abs().sum(dim=-1) > 0
        raw_active = raw_label_vector.abs().sum(dim=-1) > 0
        raw_class = torch.argmax(raw_label_vector, dim=-1)

        for batch_idx in range(label_vector.shape[0]):
            active_classes = set(raw_class[batch_idx, active[batch_idx]].tolist())
            if not active_classes:
                continue
            for source_idx in torch.nonzero(~active[batch_idx], as_tuple=False).flatten().tolist():
                candidate_class = int(raw_class[batch_idx, source_idx].item())
                if candidate_class not in active_classes:
                    continue
                if not bool(raw_active[batch_idx, source_idx]):
                    continue
                if probs[batch_idx, source_idx] < min_probability:
                    continue
                if rms[batch_idx, source_idx] < min_waveform_rms:
                    continue
                label_vector[batch_idx, source_idx] = raw_label_vector[batch_idx, source_idx]
        return label_vector

    def _match_condition_slots(self, tensor, n_sources, device, dtype):
        tensor = tensor.to(device=device, dtype=dtype)
        if tensor.dim() == 2:
            if tensor.shape[1] == n_sources:
                tensor = tensor.unsqueeze(-1)
            else:
                tensor = tensor.unsqueeze(1).expand(-1, n_sources, -1)
        if tensor.dim() != 3:
            raise ValueError(f"USS TSE condition tensors must be [B,S,D], got {tuple(tensor.shape)}")
        if tensor.shape[1] < n_sources:
            pad = tensor.new_zeros(tensor.shape[0], n_sources - tensor.shape[1], tensor.shape[-1])
            tensor = torch.cat([tensor, pad], dim=1)
        elif tensor.shape[1] > n_sources:
            tensor = tensor[:, :n_sources]
        return tensor

    def _build_tse_query_condition(self, uss_out, stage_waveform):
        if not getattr(self, "tse_uss_conditioning_enabled", False):
            return None
        n_sources = stage_waveform.shape[1]
        device = stage_waveform.device
        dtype = stage_waveform.dtype

        for key in ("tse_condition", "query_condition", "bridge_condition", "proposal_condition"):
            if key in uss_out:
                return self._match_condition_slots(uss_out[key], n_sources, device, dtype)

        parts = []
        if "class_logits" in uss_out:
            parts.append(self._match_condition_slots(uss_out["class_logits"].softmax(dim=-1), n_sources, device, dtype))
        if "silence_logits" in uss_out:
            parts.append(
                self._match_condition_slots(uss_out["silence_logits"].sigmoid().unsqueeze(-1), n_sources, device, dtype)
            )
        if "count_logits" in uss_out:
            count_prob = uss_out["count_logits"].softmax(dim=-1).to(device=device, dtype=dtype)
            parts.append(count_prob.unsqueeze(1).expand(-1, n_sources, -1))
        for key in ("spatial_embedding", "doa_vector", "pred_doa_vector", "used_spatial_vector"):
            if key in uss_out:
                parts.append(self._match_condition_slots(uss_out[key], n_sources, device, dtype))
        if "foreground_activity_logits" in uss_out:
            activity = uss_out["foreground_activity_logits"].sigmoid()
            activity = torch.stack([activity.mean(dim=-1), activity.amax(dim=-1)], dim=-1)
            parts.append(self._match_condition_slots(activity, n_sources, device, dtype))

        slot_rms = stage_waveform[:, :, 0].float().pow(2).mean(dim=-1).sqrt().to(dtype=dtype).unsqueeze(-1)
        parts.append(slot_rms)
        return torch.cat(parts, dim=-1) if parts else None

    def _build_tse_extra_conditions(self, uss_out, stage_waveform):
        if not getattr(self, "tse_uss_conditioning_enabled", False):
            return {}
        n_sources = stage_waveform.shape[1]
        device = stage_waveform.device
        dtype = stage_waveform.dtype
        out = {}
        for key in _TSE_EXTRA_CONDITION_KEYS:
            if key not in uss_out:
                continue
            value = uss_out[key]
            if not torch.is_tensor(value):
                continue
            out[key] = self._match_condition_slots(value, n_sources, device, dtype)
        return out

    def _run_tse(self, mixture, enroll, label_vector, query_condition=None, extra_conditions=None):
        input_dict = {
            "mixture": mixture,
            "enrollment": enroll,
            "label_vector": label_vector,
        }
        if query_condition is not None:
            input_dict["query_condition"] = query_condition
        if extra_conditions:
            input_dict.update(extra_conditions)
        return self.tse(input_dict)["waveform"]

    def _slot_silence_mask(self, label_vector):
        return label_vector.abs().sum(dim=-1) == 0

    def _force_silent_slots(self, waveforms, labels, probs, label_vector, silent_mask):
        if not silent_mask.any().item():
            return waveforms, labels, probs, label_vector
        silent_mask = silent_mask.to(device=waveforms.device, dtype=torch.bool)
        waveforms = waveforms.clone()
        probs = probs.clone()
        label_vector = label_vector.clone()
        labels = [list(batch_labels) for batch_labels in labels]
        waveforms[silent_mask] = 0.0
        probs[silent_mask] = 0.0
        label_vector[silent_mask] = 0.0
        for batch_idx, source_idx in torch.nonzero(silent_mask.cpu(), as_tuple=False).tolist():
            labels[batch_idx][source_idx] = "silence"
        return waveforms, labels, probs, label_vector

    def _apply_uss_gate_to_stage1(self, uss_out, stage1_waveform, stage1_labels, stage1_probs, stage1_vector):
        """Conservatively gate USS foreground slots before TSE.

        This gate is opt-in and uses only USS-side evidence:
            - count head probability for count=0
            - per-slot active probability from ``silence_logits``
            - per-slot waveform RMS

        It is intentionally conservative: count=0 can suppress the whole scene,
        while individual slot gating only suppresses slots with weak active logit
        or very low waveform energy.  SC/TSE still handle the remaining slots.
        """

        if not getattr(self, "uss_gate_enabled", False):
            return stage1_waveform, stage1_labels, stage1_probs, stage1_vector

        batch_size, n_sources = stage1_waveform.shape[:2]
        device = stage1_waveform.device
        silent_mask = torch.zeros(batch_size, n_sources, dtype=torch.bool, device=device)

        if "silence_logits" in uss_out:
            active_prob = uss_out["silence_logits"].float().sigmoid().to(device=device)
            silent_mask |= active_prob < self.uss_gate_slot_active_threshold

        slot_rms = stage1_waveform[:, :, 0].float().pow(2).mean(dim=-1).sqrt()
        silent_mask |= slot_rms < self.uss_gate_slot_energy_threshold

        if "count_logits" in uss_out:
            count_prob = uss_out["count_logits"].float().softmax(dim=-1).to(device=device)
            count0 = count_prob[:, 0] > self.uss_gate_count0_threshold
            silent_mask[count0] = True

        return self._force_silent_slots(stage1_waveform, stage1_labels, stage1_probs, stage1_vector, silent_mask)

    def predict_label_separate(self, mixture):
        with torch.no_grad():
            uss_out = self.uss({"mixture": mixture})
            stage1_waveform = uss_out["foreground_waveform"]
            stage1_labels, stage1_probs, stage1_vector = self._classify_sources(stage1_waveform)
            stage1_waveform, stage1_labels, stage1_probs, stage1_vector = self._apply_uss_gate_to_stage1(
                uss_out, stage1_waveform, stage1_labels, stage1_probs, stage1_vector
            )
            stage1_condition = self._build_tse_query_condition(uss_out, stage1_waveform)
            stage1_extra_conditions = self._build_tse_extra_conditions(uss_out, stage1_waveform)
            silent_slots = self._slot_silence_mask(stage1_vector)
            stage1_waveform, stage1_labels, stage1_probs, stage1_vector = self._force_silent_slots(
                stage1_waveform, stage1_labels, stage1_probs, stage1_vector, silent_slots
            )
            if silent_slots.all().item():
                zero_waveform = torch.zeros_like(stage1_waveform)
                output = {
                    "label": [["silence"] * stage1_waveform.shape[1] for _ in range(stage1_waveform.shape[0])],
                    "probabilities": torch.zeros_like(stage1_probs),
                    "waveform": zero_waveform,
                }
                if stage1_condition is not None:
                    output["query_condition"] = stage1_condition
                return output

            stage2_waveform = self._run_tse(
                mixture,
                stage1_waveform,
                stage1_vector,
                stage1_condition,
                stage1_extra_conditions,
            )
            stage2_labels, stage2_probs, stage2_vector = self._classify_sources(stage2_waveform)
            silent_slots = silent_slots | self._slot_silence_mask(stage2_vector)
            stage2_waveform, stage2_labels, stage2_probs, stage2_vector = self._force_silent_slots(
                stage2_waveform, stage2_labels, stage2_probs, stage2_vector, silent_slots
            )
            if self._get_tse_refinement_passes() == 1:
                output = {
                    "label": stage2_labels,
                    "probabilities": stage2_probs,
                    "waveform": stage2_waveform,
                }
                if stage1_condition is not None:
                    output["query_condition"] = stage1_condition
                return output

            stage3_waveform = self._run_tse(
                mixture,
                stage2_waveform,
                stage2_vector,
                stage1_condition,
                stage1_extra_conditions,
            )
            stage3_labels, stage3_probs, stage3_vector = self._classify_sources(stage3_waveform)
            silent_slots = silent_slots | self._slot_silence_mask(stage3_vector)
            stage3_waveform, stage3_labels, stage3_probs, _ = self._force_silent_slots(
                stage3_waveform, stage3_labels, stage3_probs, stage3_vector, silent_slots
            )

            output = {
                "label": stage3_labels,
                "probabilities": stage3_probs,
                "waveform": stage3_waveform,
            }
            if stage1_condition is not None:
                output["query_condition"] = stage1_condition
            return output
