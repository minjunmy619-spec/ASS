import torch
import torch.nn.functional as F

from src.temporal import spans_to_frame_targets


def temporal_activity_loss(output, target, pos_weight=1.0, silence_weight=1.0):
    """Binary frame-activity loss for source/activity heads.

    ``span_sec`` uses ``(-1, -1)`` for silence/inactive sources in the current
    SC dataset. Those examples still need all-zero activity supervision;
    otherwise the temporal head is not penalized for hallucinating activity on
    silence samples.
    """
    if "activity_logits" not in output or "span_sec" not in target:
        return None

    logits = output["activity_logits"]
    span_sec = target["span_sec"].to(device=logits.device, dtype=logits.dtype)
    duration_sec = output.get("duration_sec")
    if duration_sec is not None:
        duration_sec = duration_sec.to(device=logits.device, dtype=logits.dtype)
    frame_targets = spans_to_frame_targets(
        span_sec,
        num_frames=logits.shape[-1],
        duration_sec=duration_sec,
    )

    active = (span_sec[..., 0] >= 0.0) & (span_sec[..., 1] > span_sec[..., 0])
    silence = (span_sec[..., 0] < 0.0) & (span_sec[..., 1] < 0.0)
    supervised = active | silence
    if not supervised.any():
        return logits.sum() * 0.0

    weight = torch.ones_like(frame_targets)
    if pos_weight != 1.0:
        weight = torch.where(frame_targets > 0.5, weight * float(pos_weight), weight)
    if silence_weight != 1.0:
        silence_mask = silence.to(device=logits.device)
        while silence_mask.dim() < frame_targets.dim():
            silence_mask = silence_mask.unsqueeze(-1)
        weight = torch.where(silence_mask, weight * float(silence_weight), weight)

    loss = F.binary_cross_entropy_with_logits(logits, frame_targets, weight=weight, reduction="none")
    return loss[supervised].mean()
