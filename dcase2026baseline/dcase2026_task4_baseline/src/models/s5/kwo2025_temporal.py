import torch
import torch.nn.functional as F

from src.models.s5.kwo2025 import Kwon2025S5


class Kwon2025TemporalS5(Kwon2025S5):
    """Kwon2025 S5 assembly with temporal activity handoff between stages.

    Temporal-capable USS, SC, and TSE models expose frame-level activity
    probabilities/logits. This wrapper uses those signals at inference time to
    keep inactive frames and slots silent, and passes the current activity trace
    into temporal TSE as a time-FiLM conditioning signal when supported.
    """

    def __init__(
        self,
        *args,
        activity_threshold=0.5,
        temporal_conditioning_enabled=True,
        activity_gating_enabled=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.activity_threshold = float(activity_threshold)
        self.temporal_conditioning_enabled = bool(temporal_conditioning_enabled)
        self.activity_gating_enabled = bool(activity_gating_enabled)

    def _normalize_activity(self, activity, n_sources=None):
        if activity is None:
            return None
        if activity.dim() == 2:
            activity = activity.unsqueeze(1)
        if activity.dim() != 3:
            raise ValueError("activity must have shape [B, T] or [B, N, T]")
        if n_sources is not None and activity.shape[1] == 1 and n_sources != 1:
            activity = activity.expand(-1, n_sources, -1)
        if n_sources is not None and activity.shape[1] != n_sources:
            raise ValueError("activity source dimension does not match waveform/source count")
        return activity

    def _activity_to_samples(self, activity, samples, n_sources=None):
        activity = self._normalize_activity(activity, n_sources=n_sources)
        if activity is None:
            return None
        batch_size, n_sources, frames = activity.shape
        return F.interpolate(
            activity.reshape(batch_size * n_sources, 1, frames),
            size=samples,
            mode="linear",
            align_corners=False,
        ).view(batch_size, n_sources, 1, samples)

    def _gate_waveforms(self, waveforms, activity):
        if not self.activity_gating_enabled or activity is None:
            return waveforms
        mask = self._activity_to_samples(activity, waveforms.shape[-1], n_sources=waveforms.shape[1])
        mask = (mask >= self.activity_threshold).to(dtype=waveforms.dtype)
        return waveforms * mask

    def _activity_support(self, activity):
        activity = self._normalize_activity(activity)
        if activity is None:
            return None
        return activity.amax(dim=-1) >= self.activity_threshold

    def _combine_activity(self, *activities):
        activities = [activity for activity in activities if activity is not None]
        if not activities:
            return None
        n_sources = None
        for activity in activities:
            if activity.dim() == 3 and activity.shape[1] != 1:
                n_sources = activity.shape[1]
                break
        out = self._normalize_activity(activities[0], n_sources=n_sources)
        for activity in activities[1:]:
            activity = self._normalize_activity(activity, n_sources=out.shape[1])
            if activity.shape[-1] != out.shape[-1]:
                activity = F.interpolate(
                    activity.reshape(-1, 1, activity.shape[-1]),
                    size=out.shape[-1],
                    mode="linear",
                    align_corners=False,
                ).view(*activity.shape[:-1], out.shape[-1])
            out = torch.minimum(out, activity)
        return out

    def _recover_silenced_duplicates_with_activity(self, waveforms, label_vector, raw_label_vector, probs, support):
        if support is None:
            return self._recover_silenced_duplicates(waveforms, label_vector, raw_label_vector, probs)
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
                if not bool(support[batch_idx, source_idx]):
                    continue
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

    def _classify_sources_temporal(self, waveforms, prior_activity=None):
        batch_size, n_sources, _, samples = waveforms.shape
        flat = waveforms[:, :, 0].reshape(batch_size * n_sources, samples)
        out = self.sc.predict({"waveform": flat})
        label_vector = out["label_vector"].view(batch_size, n_sources, -1)
        probs = out["probabilities"].view(batch_size, n_sources)

        sc_activity = out.get("activity_probabilities")
        if sc_activity is not None:
            sc_activity = sc_activity.view(batch_size, n_sources, -1)
        activity = self._combine_activity(prior_activity, sc_activity)
        support = self._activity_support(activity)

        if getattr(self, "duplicate_recall_enabled", False):
            raw_label_vector = out.get("raw_label_vector")
            if raw_label_vector is not None:
                raw_label_vector = raw_label_vector.view(batch_size, n_sources, -1)
                label_vector = self._recover_silenced_duplicates_with_activity(
                    waveforms,
                    label_vector,
                    raw_label_vector,
                    probs,
                    support,
                )

        labels = [self._vector_to_label(label_vector[i]) for i in range(batch_size)]
        return labels, probs, label_vector, activity

    def _run_tse_temporal(self, mixture, enroll, label_vector, activity, query_condition=None):
        input_dict = {
            "mixture": mixture,
            "enrollment": enroll,
            "label_vector": label_vector,
        }
        if self.temporal_conditioning_enabled and activity is not None:
            input_dict["temporal_conditioning"] = activity
        if query_condition is not None:
            input_dict["query_condition"] = query_condition
        out = self.tse(input_dict)
        waveform = out["waveform"]
        tse_activity = out.get("activity_logits")
        if tse_activity is not None:
            tse_activity = torch.sigmoid(tse_activity)
        return waveform, tse_activity

    def _apply_temporal_silence(self, waveforms, labels, probs, label_vector, activity, silent_slots):
        support = self._activity_support(activity)
        if support is not None:
            silent_slots = silent_slots | (~support)
        waveforms = self._gate_waveforms(waveforms, activity)
        waveforms, labels, probs, label_vector = self._force_silent_slots(
            waveforms,
            labels,
            probs,
            label_vector,
            silent_slots,
        )
        return waveforms, labels, probs, label_vector, silent_slots

    def predict_label_separate(self, mixture):
        with torch.no_grad():
            uss_out = self.uss({"mixture": mixture})
            stage1_waveform = uss_out["foreground_waveform"]
            uss_activity = uss_out.get("foreground_activity_logits")
            if uss_activity is not None:
                uss_activity = torch.sigmoid(uss_activity)
            stage1_waveform = self._gate_waveforms(stage1_waveform, uss_activity)
            stage1_condition = self._build_tse_query_condition(uss_out, stage1_waveform)

            stage1_labels, stage1_probs, stage1_vector, stage1_activity = self._classify_sources_temporal(
                stage1_waveform,
                prior_activity=uss_activity,
            )
            silent_slots = self._slot_silence_mask(stage1_vector)
            stage1_waveform, stage1_labels, stage1_probs, stage1_vector, silent_slots = self._apply_temporal_silence(
                stage1_waveform,
                stage1_labels,
                stage1_probs,
                stage1_vector,
                stage1_activity,
                silent_slots,
            )
            if silent_slots.all().item():
                output = {
                    "label": [["silence"] * stage1_waveform.shape[1] for _ in range(stage1_waveform.shape[0])],
                    "probabilities": torch.zeros_like(stage1_probs),
                    "waveform": torch.zeros_like(stage1_waveform),
                }
                if stage1_condition is not None:
                    output["query_condition"] = stage1_condition
                return output

            stage2_waveform, tse2_activity = self._run_tse_temporal(
                mixture,
                stage1_waveform,
                stage1_vector,
                stage1_activity,
                stage1_condition,
            )
            stage2_activity_prior = self._combine_activity(stage1_activity, tse2_activity)
            stage2_waveform = self._gate_waveforms(stage2_waveform, stage2_activity_prior)
            stage2_labels, stage2_probs, stage2_vector, stage2_activity = self._classify_sources_temporal(
                stage2_waveform,
                prior_activity=stage2_activity_prior,
            )
            silent_slots = silent_slots | self._slot_silence_mask(stage2_vector)
            stage2_waveform, stage2_labels, stage2_probs, stage2_vector, silent_slots = self._apply_temporal_silence(
                stage2_waveform,
                stage2_labels,
                stage2_probs,
                stage2_vector,
                stage2_activity,
                silent_slots,
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

            stage3_waveform, tse3_activity = self._run_tse_temporal(
                mixture,
                stage2_waveform,
                stage2_vector,
                stage2_activity,
                stage1_condition,
            )
            stage3_activity_prior = self._combine_activity(stage2_activity, tse3_activity)
            stage3_waveform = self._gate_waveforms(stage3_waveform, stage3_activity_prior)
            stage3_labels, stage3_probs, stage3_vector, stage3_activity = self._classify_sources_temporal(
                stage3_waveform,
                prior_activity=stage3_activity_prior,
            )
            silent_slots = silent_slots | self._slot_silence_mask(stage3_vector)
            stage3_waveform, stage3_labels, stage3_probs, _, silent_slots = self._apply_temporal_silence(
                stage3_waveform,
                stage3_labels,
                stage3_probs,
                stage3_vector,
                stage3_activity,
                silent_slots,
            )

            output = {
                "label": stage3_labels,
                "probabilities": stage3_probs,
                "waveform": stage3_waveform,
            }
            if stage1_condition is not None:
                output["query_condition"] = stage1_condition
            return output
