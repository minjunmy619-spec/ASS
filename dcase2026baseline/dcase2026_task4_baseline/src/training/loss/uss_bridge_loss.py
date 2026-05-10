import torch
import torch.nn.functional as F

from src.training.loss.uss_loss import get_loss_func as get_base_uss_loss_func
from src.training.loss.class_aware_pit import pairwise_sa_sdr_loss, pit_from_pairwise_loss
from src.training.loss.uss_residual_loss import _mix_loss, _residual_loss


def _active_mask(target):
    return ~target["is_silence"].bool()


def _foreground_perm(output, target, lambda_class_match=1.0):
    fg_est = output["foreground_waveform"].float()
    fg_ref = target["foreground_waveform"].float()
    class_logits = output["class_logits"].float()
    class_index = target["class_index"]
    active = _active_mask(target)

    fg_pair_wave = pairwise_sa_sdr_loss(fg_est, fg_ref)
    neg_log_probs = -F.log_softmax(class_logits, dim=-1)
    batch_size, n_pred, n_classes = class_logits.shape
    n_target = class_index.shape[1]
    expanded = neg_log_probs.unsqueeze(1).expand(batch_size, n_target, n_pred, n_classes)
    gather_idx = class_index[:, :, None, None].expand(batch_size, n_target, n_pred, 1)
    fg_pair_class = expanded.gather(dim=-1, index=gather_idx).squeeze(-1)
    _, best_perm = pit_from_pairwise_loss(
        fg_pair_wave + lambda_class_match * fg_pair_class,
        active_mask=active,
    )
    return best_perm, active


def _gather_slots(x, perm):
    if x is None:
        return None
    batch_size, n_slots = perm.shape
    trailing = x.shape[2:]
    index_shape = (batch_size, n_slots) + tuple(1 for _ in trailing)
    index = perm.reshape(index_shape).expand((batch_size, n_slots) + trailing)
    return torch.gather(x, dim=1, index=index)


def _prototype_loss(output, target, best_perm, active):
    if "prototype_logits" not in output:
        return output["class_logits"].new_zeros(())
    logits = _gather_slots(output["prototype_logits"], best_perm)
    if not active.any():
        return logits.new_zeros(())
    return F.cross_entropy(logits[active], target["class_index"].to(logits.device)[active])


def _doa_loss(output, target, best_perm, active):
    if "pred_doa_vector" not in output or "spatial_vector" not in target:
        return output["class_logits"].new_zeros(())
    pred = _gather_slots(output["pred_doa_vector"], best_perm)
    target_doa = target["spatial_vector"].to(device=pred.device, dtype=pred.dtype)
    doa_mask = target.get("foreground_doa_mask", target.get("spatial_vector_mask"))
    if target_doa.shape[1] < pred.shape[1]:
        pad = target_doa.new_zeros(target_doa.shape[0], pred.shape[1] - target_doa.shape[1], target_doa.shape[-1])
        target_doa = torch.cat([target_doa, pad], dim=1)
    elif target_doa.shape[1] > pred.shape[1]:
        target_doa = target_doa[:, : pred.shape[1]]
    active = active.to(device=pred.device, dtype=torch.bool)
    if doa_mask is not None:
        doa_mask = doa_mask.to(device=pred.device, dtype=torch.bool)
        if doa_mask.shape[1] < pred.shape[1]:
            pad = torch.zeros(doa_mask.shape[0], pred.shape[1] - doa_mask.shape[1], device=pred.device, dtype=torch.bool)
            doa_mask = torch.cat([doa_mask, pad], dim=1)
        elif doa_mask.shape[1] > pred.shape[1]:
            doa_mask = doa_mask[:, : pred.shape[1]]
        active = active & doa_mask
    pred = F.normalize(pred, dim=-1)
    target_doa = F.normalize(target_doa, dim=-1)
    loss = 1.0 - (pred * target_doa).sum(dim=-1)
    active = active.to(device=loss.device, dtype=loss.dtype)
    return (loss * active).sum() / active.sum().clamp_min(1.0)


def _supervised_contrastive_loss(embeddings, labels, active, temperature=0.1):
    if embeddings is None:
        return None
    b, s, d = embeddings.shape
    z = F.normalize(embeddings.reshape(b * s, d), dim=-1)
    y = labels.reshape(b * s).to(z.device)
    mask = active.reshape(b * s).to(z.device)
    z = z[mask]
    y = y[mask]
    if z.shape[0] <= 1:
        return embeddings.new_zeros(())
    logits = z @ z.t() / temperature
    eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, -1e9)
    positive = (y[:, None] == y[None, :]) & ~eye
    has_positive = positive.any(dim=1)
    if not has_positive.any():
        return embeddings.new_zeros(())
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos_count = positive.float().sum(dim=1).clamp_min(1.0)
    loss = -((log_prob * positive.float()).sum(dim=1) / pos_count)
    return loss[has_positive].mean()


def _bidirectional_infonce(audio_emb, semantic_emb, active, temperature=0.1):
    if audio_emb is None or semantic_emb is None:
        return None
    b, s, d = semantic_emb.shape
    a = F.normalize(audio_emb.reshape(b * s, d), dim=-1)
    c = F.normalize(semantic_emb.reshape(b * s, d), dim=-1)
    mask = active.reshape(b * s).to(a.device)
    a = a[mask]
    c = c[mask]
    if a.shape[0] <= 1:
        return semantic_emb.new_zeros(())
    logits = a @ c.t() / temperature
    target = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def _embedding_norm_regularizer(output):
    vals = []
    for key in ("foreground_embedding", "foreground_audio_embedding", "tse_condition"):
        if key in output:
            vals.append((output[key].norm(dim=-1) - 1.0).abs().mean())
    if not vals:
        return output["class_logits"].new_zeros(())
    return sum(vals) / len(vals)


def get_loss_func(
    lambda_bridge_proto=0.05,
    lambda_bridge_supcon=0.02,
    lambda_bridge_infonce=0.02,
    lambda_bridge_doa=0.05,
    lambda_bridge_norm=0.001,
    lambda_residual_slot=0.0,
    lambda_mix=0.0,
    residual_stft_fft_sizes=(512, 1024, 2048),
    residual_ref_channel=0,
    bridge_temperature=0.1,
    lambda_class_match=1.0,
    **base_uss_loss_kwargs,
):
    """USS loss + opt-in semantic-acoustic bridge objectives.

    This wrapper is backward compatible: if the model does not emit bridge keys,
    all bridge losses are zero and the returned main loss equals the base USS loss.
    """
    base_loss_func = get_base_uss_loss_func(**base_uss_loss_kwargs)
    residual_stft_fft_sizes = tuple(int(x) for x in residual_stft_fft_sizes)
    residual_ref_channel = int(residual_ref_channel)

    def loss_func(output, target):
        loss_dict = base_loss_func(output, target)
        best_perm, active = _foreground_perm(output, target, lambda_class_match=lambda_class_match)

        class_index = target["class_index"].to(output["class_logits"].device)
        aligned_semantic = _gather_slots(output.get("foreground_embedding"), best_perm)
        aligned_audio = _gather_slots(output.get("foreground_audio_embedding"), best_perm)

        loss_proto = _prototype_loss(output, target, best_perm, active)
        loss_supcon = _supervised_contrastive_loss(
            aligned_semantic,
            class_index,
            active,
            temperature=bridge_temperature,
        )
        if loss_supcon is None:
            loss_supcon = loss_proto.new_zeros(())
        loss_infonce = _bidirectional_infonce(
            aligned_audio,
            aligned_semantic,
            active,
            temperature=bridge_temperature,
        )
        if loss_infonce is None:
            loss_infonce = loss_proto.new_zeros(())
        loss_doa = _doa_loss(output, target, best_perm, active)
        loss_norm = _embedding_norm_regularizer(output)

        bridge_loss = (
            lambda_bridge_proto * loss_proto
            + lambda_bridge_supcon * loss_supcon
            + lambda_bridge_infonce * loss_infonce
            + lambda_bridge_doa * loss_doa
            + lambda_bridge_norm * loss_norm
        )
        loss_residual_slot, loss_residual_slot_mae, loss_residual_slot_stft = _residual_loss(
            output,
            target,
            residual_ref_channel,
            residual_stft_fft_sizes,
        )
        loss_mix = _mix_loss(output, target, residual_ref_channel)
        residual_loss = lambda_residual_slot * loss_residual_slot + lambda_mix * loss_mix
        loss_dict["loss"] = loss_dict["loss"] + bridge_loss + residual_loss
        loss_dict.update(
            {
                "loss_bridge": bridge_loss,
                "loss_bridge_proto": loss_proto,
                "loss_bridge_supcon": loss_supcon,
                "loss_bridge_infonce": loss_infonce,
                "loss_bridge_doa": loss_doa,
                "loss_bridge_norm": loss_norm,
                "loss_residual_slot": loss_residual_slot,
                "loss_residual_slot_mae": loss_residual_slot_mae,
                "loss_residual_slot_stft": loss_residual_slot_stft,
                "loss_mix": loss_mix,
            }
        )
        return loss_dict

    return loss_func
