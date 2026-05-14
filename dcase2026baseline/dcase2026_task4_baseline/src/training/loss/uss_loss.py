import torch
import torch.nn.functional as F

from src.training.loss.class_aware_pit import (
    inactive_source_energy_loss,
    matched_pairwise_mean,
    pairwise_sa_sdr_loss,
    pit_from_pairwise_loss,
    source_activity_mask,
    unmatched_prediction_mask,
)
from src.temporal import align_spans_to_predictions
from src.training.loss.temporal import temporal_activity_loss


def _safe_energy(x):
    x = x.float()
    return torch.sum(x**2, dim=-1) + 1e-8


def _si_snr_loss_per_source(est, target):
    est = est.float()
    target = target.float()
    target_energy = torch.sum(target**2, dim=-1, keepdim=True) + 1e-8
    scale = torch.sum(est * target, dim=-1, keepdim=True) / target_energy
    target_proj = scale * target
    noise = est - target_proj
    ratio = (_safe_energy(target_proj) / (_safe_energy(noise) + 1e-8)).clamp_min(1e-8)
    return -10.0 * torch.log10(ratio)


def pairwise_sa_sdri_loss(waveform_pred, waveform_target, mixture, ref_channel: int = 0, eps: float = 1e-8):
    """Scale-aware negative SDR improvement for every target/prediction pair.

    Returns [B, S_target, S_pred]. The mixture baseline is computed on the
    ranking reference channel so the foreground assignment can follow CAPI-SDRi
    more closely than plain SDR PIT.
    """

    if waveform_pred.dim() == 4:
        pred = waveform_pred[:, :, ref_channel, :]
    else:
        pred = waveform_pred

    if waveform_target.dim() == 4:
        target = waveform_target[:, :, ref_channel, :]
    else:
        target = waveform_target

    if mixture.dim() == 3:
        mix = mixture[:, ref_channel, :]
    else:
        mix = mixture

    pred = pred.float()
    target = target.float()
    mix = mix.float()

    err_pred = pred.unsqueeze(1) - target.unsqueeze(2)
    target_power = target.pow(2).sum(dim=-1).unsqueeze(2).clamp_min(eps)
    pred_noise_power = err_pred.pow(2).sum(dim=-1).clamp_min(eps)
    sdr_pred = 10.0 * torch.log10((target_power / pred_noise_power).clamp_min(eps))

    err_mix = mix.unsqueeze(1) - target
    mix_noise_power = err_mix.pow(2).sum(dim=-1).unsqueeze(2).clamp_min(eps)
    sdr_mix = 10.0 * torch.log10((target_power / mix_noise_power).clamp_min(eps))
    return -(sdr_pred - sdr_mix)


def _class_pair_loss(class_logits, class_index):
    class_logits = class_logits.float()
    batch_size, n_pred, n_classes = class_logits.shape
    n_target = class_index.shape[1]
    neg_log_probs = -F.log_softmax(class_logits, dim=-1)
    expanded = neg_log_probs.unsqueeze(1).expand(batch_size, n_target, n_pred, n_classes)
    gather_idx = class_index[:, :, None, None].expand(batch_size, n_target, n_pred, 1)
    return expanded.gather(dim=-1, index=gather_idx).squeeze(-1)


def _target_class_probability(class_logits, class_index):
    """Return p(predicted slot class == reference class) as [B, S_ref, S_pred]."""

    probs = torch.softmax(class_logits.float(), dim=-1)
    batch_size, n_pred, n_classes = probs.shape
    n_target = class_index.shape[1]
    expanded = probs.unsqueeze(1).expand(batch_size, n_target, n_pred, n_classes)
    gather_idx = class_index[:, :, None, None].expand(batch_size, n_target, n_pred, 1)
    return expanded.gather(dim=-1, index=gather_idx).squeeze(-1)


def _class_confident_valid_pair_mask(class_logits, class_index, active_mask, confidence_threshold: float = 0.35):
    target_prob = _target_class_probability(class_logits, class_index)
    valid_pair_mask = target_prob >= confidence_threshold
    return valid_pair_mask & active_mask[:, :, None].bool()


def _capi_foreground_assignment(
    fg_est,
    fg_ref,
    mixture,
    class_logits,
    class_index,
    fg_active,
    lambda_class_pit: float = 2.0,
    confidence_threshold: float = 0.35,
    invalid_class_cost: float = 20.0,
    use_sdri: bool = True,
    use_hard_class_mask: bool = False,
    ref_channel: int = 0,
):
    if use_sdri:
        fg_pair_wave = pairwise_sa_sdri_loss(fg_est, fg_ref, mixture, ref_channel=ref_channel)
    else:
        fg_pair_wave = pairwise_sa_sdr_loss(fg_est, fg_ref)

    fg_pair_class = _class_pair_loss(class_logits, class_index)
    fg_pair_total = fg_pair_wave + lambda_class_pit * fg_pair_class

    valid_pair_mask = _class_confident_valid_pair_mask(
        class_logits=class_logits,
        class_index=class_index,
        active_mask=fg_active,
        confidence_threshold=confidence_threshold,
    )

    if use_hard_class_mask:
        loss_fg_match, fg_best_perm = pit_from_pairwise_loss(
            fg_pair_total,
            active_mask=fg_active,
            valid_pair_mask=valid_pair_mask,
            eval_func="min",
        )
    else:
        fg_pair_total = fg_pair_total + (~valid_pair_mask).float() * invalid_class_cost
        loss_fg_match, fg_best_perm = pit_from_pairwise_loss(
            fg_pair_total,
            active_mask=fg_active,
            valid_pair_mask=None,
            eval_func="min",
        )

    return loss_fg_match, fg_best_perm, fg_pair_wave, fg_pair_class, fg_pair_total, valid_pair_mask


def _safe_active_mean(values, active_mask):
    active_mask = active_mask.to(device=values.device, dtype=torch.bool)
    if active_mask.any():
        return (values * active_mask.float()).sum() / active_mask.float().sum().clamp_min(1.0)
    return values.new_zeros(())


def _residual_consistency_loss(output, target):
    if "mixture" not in target:
        return output["foreground_waveform"].new_zeros(())
    mixture_ref = target["mixture"][:, 0].float()
    recon = output["foreground_waveform"][:, :, 0].float().sum(dim=1)
    recon = recon + output["interference_waveform"][:, :, 0].float().sum(dim=1)
    recon = recon + output["noise_waveform"][:, 0, 0].float()
    return F.mse_loss(recon, mixture_ref)


def _source_activity_loss(output, target, output_key, span_key, active_mask=None, best_perm=None, pos_weight=1.0):
    if output_key not in output or span_key not in target:
        ref = output.get(output_key)
        if ref is None:
            ref = next(iter(output.values()))
        return ref.float().new_zeros(())
    activity_logits = output[output_key].float()
    span_sec = target[span_key].to(device=activity_logits.device, dtype=activity_logits.dtype)
    if best_perm is not None and active_mask is not None:
        span_sec = align_spans_to_predictions(best_perm, span_sec, active_mask, activity_logits.shape[1])
    return temporal_activity_loss(
        {"activity_logits": activity_logits, "duration_sec": output.get("duration_sec")},
        {"span_sec": span_sec},
        pos_weight=pos_weight,
    )


def _foreground_count_target(is_silence, max_count):
    """Return foreground count targets in [0, max_count]."""

    count_target = (~is_silence.bool()).long().sum(dim=1)
    return count_target.clamp(max=max_count)


def _foreground_count_loss(output, target):
    """Optional 0/1/2/3 foreground count loss."""

    if "count_logits" not in output:
        return output["foreground_waveform"].float().new_zeros(())
    count_logits = output["count_logits"].float()
    max_count = count_logits.shape[-1] - 1
    count_target = _foreground_count_target(target["is_silence"], max_count=max_count)
    count_target = count_target.to(device=count_logits.device)
    return F.cross_entropy(count_logits, count_target)


def _gather_by_reference(prediction, best_perm):
    """Gather prediction slots into reference order using PIT assignment."""

    gathered = []
    for batch_idx in range(prediction.shape[0]):
        gathered.append(prediction[batch_idx, best_perm[batch_idx]])
    return torch.stack(gathered, dim=0)


def _same_class_pair_mask(class_index, fg_active):
    same_class = class_index[:, :, None] == class_index[:, None, :]
    active_pair = fg_active[:, :, None] & fg_active[:, None, :]
    n_ref = class_index.shape[1]
    upper = torch.triu(
        torch.ones(n_ref, n_ref, device=class_index.device, dtype=torch.bool),
        diagonal=1,
    )
    return same_class & active_pair & upper[None]


def _matched_doa_loss(output, target, best_perm, fg_active):
    if "doa_vector" not in output or "foreground_doa" not in target:
        return output["foreground_waveform"].float().new_zeros(())
    pred_doa = F.normalize(output["doa_vector"].float(), dim=-1, eps=1e-8)
    ref_doa = F.normalize(target["foreground_doa"].to(device=pred_doa.device).float(), dim=-1, eps=1e-8)
    doa_mask = target.get("foreground_doa_mask")
    if doa_mask is None:
        doa_mask = torch.ones_like(fg_active, dtype=torch.bool)
    doa_mask = doa_mask.to(device=pred_doa.device, dtype=torch.bool) & fg_active.to(device=pred_doa.device, dtype=torch.bool)
    matched_pred = _gather_by_reference(pred_doa, best_perm)
    cos = (matched_pred * ref_doa).sum(dim=-1)
    if doa_mask.any():
        return ((1.0 - cos) * doa_mask.float()).sum() / doa_mask.float().sum().clamp_min(1.0)
    return pred_doa.new_zeros(())


def _spatial_diversity_loss(output, class_index, fg_active, best_perm, margin: float = 0.2):
    if "spatial_embedding" not in output:
        return output["foreground_waveform"].float().new_zeros(())
    emb = F.normalize(output["spatial_embedding"].float(), dim=-1, eps=1e-8)
    matched_emb = _gather_by_reference(emb, best_perm)
    pair_mask = _same_class_pair_mask(
        class_index.to(device=emb.device),
        fg_active.to(device=emb.device, dtype=torch.bool),
    )
    if not pair_mask.any():
        return emb.new_zeros(())
    cos = torch.matmul(matched_emb, matched_emb.transpose(1, 2))
    loss = F.relu(cos - float(margin))
    return (loss * pair_mask.float()).sum() / pair_mask.float().sum().clamp_min(1.0)


def _waveform_anticollapse_loss(fg_est, class_index, fg_active, best_perm, margin: float = 0.3):
    matched_wave = _gather_by_reference(fg_est.float(), best_perm).flatten(start_dim=2)
    matched_wave = F.normalize(matched_wave, dim=-1, eps=1e-8)
    pair_mask = _same_class_pair_mask(
        class_index.to(device=fg_est.device),
        fg_active.to(device=fg_est.device, dtype=torch.bool),
    )
    if not pair_mask.any():
        return fg_est.new_zeros(())
    corr = torch.abs(torch.matmul(matched_wave, matched_wave.transpose(1, 2)))
    loss = F.relu(corr - float(margin))
    return (loss * pair_mask.float()).sum() / pair_mask.float().sum().clamp_min(1.0)


def get_loss_func(
    lambda_non_foreground=0.01,
    # lambda_class_match=1.0,
    lambda_class_pit=0.05,
    lambda_class_ce=0.1,
    lambda_kl=1.0,
    lambda_silence=1.0,
    lambda_count=0.0,
    lambda_inactive_foreground=0.05,
    lambda_inactive_interference=0.01,
    lambda_inactive_noise=0.01,
    lambda_residual=0.0,
    lambda_activity_foreground=0.0,
    lambda_activity_interference=0.0,
    lambda_activity_noise=0.0,
    activity_pos_weight=1.0,
    active_energy_eps=1e-8,
    foreground_assignment="global_pit",
    capi_use_sdri=True,
    capi_ref_channel=0,
    capi_confidence_threshold=0.35,
    capi_invalid_class_cost=20.0,
    lambda_doa=0.0,
    lambda_spatial_diversity=0.0,
    lambda_waveform_anticollapse=0.0,
    spatial_diversity_margin=0.2,
    waveform_anticollapse_margin=0.3,
    lambda_state=None,
):
    """USS loss factory.

    All scalar ``lambda_*`` weights are stored in a mutable ``lambda_state``
    dict that is also exposed as ``loss_func.lambdas``. Each forward call
    reads the current value, so an external scheduler (see
    ``src.training.callbacks.lambda_scheduler.LambdaScheduler``) can update
    the dict in-place between training steps without rebuilding the loss.

    If ``lambda_state`` is supplied (e.g. by ``uss_bridge_loss.get_loss_func``)
    the same dict is shared and only missing keys are seeded from kwargs.
    """
    _initial_lambdas = {
        "lambda_non_foreground": float(lambda_non_foreground),
        "lambda_class_pit": float(lambda_class_pit),
        "lambda_class_ce": float(lambda_class_ce),
        "lambda_kl": float(lambda_kl),
        "lambda_silence": float(lambda_silence),
        "lambda_count": float(lambda_count),
        "lambda_inactive_foreground": float(lambda_inactive_foreground),
        "lambda_inactive_interference": float(lambda_inactive_interference),
        "lambda_inactive_noise": float(lambda_inactive_noise),
        "lambda_residual": float(lambda_residual),
        "lambda_activity_foreground": float(lambda_activity_foreground),
        "lambda_activity_interference": float(lambda_activity_interference),
        "lambda_activity_noise": float(lambda_activity_noise),
        "lambda_doa": float(lambda_doa),
        "lambda_spatial_diversity": float(lambda_spatial_diversity),
        "lambda_waveform_anticollapse": float(lambda_waveform_anticollapse),
    }
    if lambda_state is None:
        lambda_state = dict(_initial_lambdas)
    else:
        for _k, _v in _initial_lambdas.items():
            lambda_state.setdefault(_k, _v)

    def loss_func(output, target):
        # Read live lambda values from the shared dict so a scheduler can
        # mutate ``loss_func.lambdas`` and have it take effect immediately.
        lambda_non_foreground = lambda_state["lambda_non_foreground"]
        lambda_class_pit = lambda_state["lambda_class_pit"]
        lambda_class_ce = lambda_state["lambda_class_ce"]
        lambda_kl = lambda_state["lambda_kl"]
        lambda_silence = lambda_state["lambda_silence"]
        lambda_count = lambda_state["lambda_count"]
        lambda_inactive_foreground = lambda_state["lambda_inactive_foreground"]
        lambda_inactive_interference = lambda_state["lambda_inactive_interference"]
        lambda_inactive_noise = lambda_state["lambda_inactive_noise"]
        lambda_residual = lambda_state["lambda_residual"]
        lambda_activity_foreground = lambda_state["lambda_activity_foreground"]
        lambda_activity_interference = lambda_state["lambda_activity_interference"]
        lambda_activity_noise = lambda_state["lambda_activity_noise"]
        lambda_doa = lambda_state["lambda_doa"]
        lambda_spatial_diversity = lambda_state["lambda_spatial_diversity"]
        lambda_waveform_anticollapse = lambda_state["lambda_waveform_anticollapse"]
        device_type = output["foreground_waveform"].device.type
        with torch.autocast(device_type=device_type, enabled=False):
            fg_est = output["foreground_waveform"].float()
            int_est = output["interference_waveform"].float()
            noise_est = output["noise_waveform"][:, :, 0].float()

            fg_ref = target["foreground_waveform"].float()
            int_ref = target["interference_waveform"].float()
            noise_ref = target["noise_waveform"][:, :, 0].float()

            class_logits = output["class_logits"].float()
            silence_logits = output["silence_logits"].float()
            class_index = target["class_index"]
            fg_active = ~target["is_silence"].bool()

            if foreground_assignment == "global_pit":
                fg_pair_wave = pairwise_sa_sdr_loss(fg_est, fg_ref)
                fg_pair_class = _class_pair_loss(class_logits, class_index)
                fg_pair_total = fg_pair_wave + lambda_class_pit * fg_pair_class
                loss_fg_match, fg_best_perm = pit_from_pairwise_loss(
                    fg_pair_total,
                    active_mask=fg_active,
                    eval_func="min",
                )
                valid_pair_mask = torch.ones_like(fg_pair_total, dtype=torch.bool)
            elif foreground_assignment == "soft_capi":
                loss_fg_match, fg_best_perm, fg_pair_wave, fg_pair_class, fg_pair_total, valid_pair_mask = (
                    _capi_foreground_assignment(
                        fg_est=fg_est,
                        fg_ref=fg_ref,
                        mixture=target["mixture"],
                        class_logits=class_logits,
                        class_index=class_index,
                        fg_active=fg_active,
                        lambda_class_pit=lambda_class_pit,
                        confidence_threshold=capi_confidence_threshold,
                        invalid_class_cost=capi_invalid_class_cost,
                        use_sdri=capi_use_sdri,
                        use_hard_class_mask=False,
                        ref_channel=capi_ref_channel,
                    )
                )
            elif foreground_assignment == "hard_capi":
                loss_fg_match, fg_best_perm, fg_pair_wave, fg_pair_class, fg_pair_total, valid_pair_mask = (
                    _capi_foreground_assignment(
                        fg_est=fg_est,
                        fg_ref=fg_ref,
                        mixture=target["mixture"],
                        class_logits=class_logits,
                        class_index=class_index,
                        fg_active=fg_active,
                        lambda_class_pit=lambda_class_pit,
                        confidence_threshold=capi_confidence_threshold,
                        invalid_class_cost=capi_invalid_class_cost,
                        use_sdri=capi_use_sdri,
                        use_hard_class_mask=True,
                        ref_channel=capi_ref_channel,
                    )
                )
            else:
                raise ValueError(f"Unknown foreground_assignment: {foreground_assignment}")

            loss_fg_wave = matched_pairwise_mean(fg_pair_wave, fg_best_perm, fg_active)
            loss_ce = matched_pairwise_mean(fg_pair_class, fg_best_perm, fg_active)
            fg_inactive_mask = unmatched_prediction_mask(fg_best_perm, fg_active, fg_est.shape[1])
            loss_fg_inactive = inactive_source_energy_loss(fg_est, fg_inactive_mask)
            loss_doa = _matched_doa_loss(output, target, fg_best_perm, fg_active)
            loss_spatial_diversity = _spatial_diversity_loss(
                output,
                class_index,
                fg_active,
                fg_best_perm,
                margin=spatial_diversity_margin,
            )
            loss_waveform_anticollapse = _waveform_anticollapse_loss(
                fg_est,
                class_index,
                fg_active,
                fg_best_perm,
                margin=waveform_anticollapse_margin,
            )

            fg_pred_active = ~fg_inactive_mask
            # Historical name is ``silence_logits``, but the target here is an
            # active-slot indicator: high logit means keep the slot active.
            loss_silence = F.binary_cross_entropy_with_logits(silence_logits, fg_pred_active.float())
            if fg_inactive_mask.any():
                log_probs = F.log_softmax(class_logits[fg_inactive_mask], dim=-1)
                uniform = torch.full_like(log_probs, 1.0 / log_probs.shape[-1])
                loss_kl = F.kl_div(log_probs, uniform, reduction="batchmean")
            else:
                loss_kl = class_logits.new_zeros(())

            with torch.no_grad():
                target_prob = _target_class_probability(class_logits, class_index).clamp_min(1e-8)
                matched_target_class_nll = matched_pairwise_mean(-torch.log(target_prob), fg_best_perm, fg_active)
                matched_valid_pair_rate = matched_pairwise_mean(
                    valid_pair_mask.float(),
                    fg_best_perm,
                    fg_active,
                )

            loss_count = _foreground_count_loss(output, target)
            loss_fg = (
                loss_fg_wave
                + lambda_class_ce * loss_ce
                + lambda_inactive_foreground * loss_fg_inactive
                + lambda_doa * loss_doa
                + lambda_spatial_diversity * loss_spatial_diversity
                + lambda_waveform_anticollapse * loss_waveform_anticollapse
            )
            loss_fg_activity = _source_activity_loss(
                output,
                target,
                "foreground_activity_logits",
                "foreground_span_sec",
                active_mask=fg_active,
                best_perm=fg_best_perm,
                pos_weight=activity_pos_weight,
            )

            int_active = source_activity_mask(int_ref, energy_eps=active_energy_eps)
            int_pair_wave = pairwise_sa_sdr_loss(int_est, int_ref)
            loss_int_match, int_best_perm = pit_from_pairwise_loss(int_pair_wave, active_mask=int_active)
            loss_int_wave = loss_int_match.mean()
            int_inactive_mask = unmatched_prediction_mask(int_best_perm, int_active, int_est.shape[1])
            loss_int_inactive = inactive_source_energy_loss(int_est, int_inactive_mask)
            loss_int = loss_int_wave + lambda_inactive_interference * loss_int_inactive
            loss_int_activity = _source_activity_loss(
                output,
                target,
                "interference_activity_logits",
                "interference_span_sec",
                active_mask=int_active,
                best_perm=int_best_perm,
                pos_weight=activity_pos_weight,
            )

            noise_active = source_activity_mask(noise_ref, energy_eps=active_energy_eps)
            noise_loss_per_source = _si_snr_loss_per_source(noise_est, noise_ref)
            loss_noise_wave = _safe_active_mean(noise_loss_per_source, noise_active)
            loss_noise_inactive = inactive_source_energy_loss(noise_est, ~noise_active)
            loss_noise = loss_noise_wave + lambda_inactive_noise * loss_noise_inactive
            loss_noise_activity = _source_activity_loss(
                output,
                target,
                "noise_activity_logits",
                "noise_span_sec",
                active_mask=noise_active,
                pos_weight=activity_pos_weight,
            )

            loss_residual = _residual_consistency_loss(output, target)
        loss = (
            loss_fg
            + lambda_non_foreground * (loss_int + loss_noise)
            + lambda_kl * loss_kl
            + lambda_silence * loss_silence
            + lambda_count * loss_count
            + lambda_residual * loss_residual
            + lambda_activity_foreground * loss_fg_activity
            + lambda_activity_interference * loss_int_activity
            + lambda_activity_noise * loss_noise_activity
        )
        return {
            "loss": loss,
            "loss_fg": loss_fg,
            "loss_fg_match": loss_fg_match.mean(),
            "loss_fg_wave": loss_fg_wave,
            "loss_fg_inactive": loss_fg_inactive,
            "loss_doa": loss_doa,
            "loss_spatial_diversity": loss_spatial_diversity,
            "loss_waveform_anticollapse": loss_waveform_anticollapse,
            "loss_int": loss_int,
            "loss_int_wave": loss_int_wave,
            "loss_int_inactive": loss_int_inactive,
            "loss_noise": loss_noise,
            "loss_noise_wave": loss_noise_wave,
            "loss_noise_inactive": loss_noise_inactive,
            "loss_ce": loss_ce,
            "loss_matched_target_class_nll": matched_target_class_nll,
            "loss_matched_valid_pair_rate": matched_valid_pair_rate,
            "loss_kl": loss_kl,
            "loss_silence": loss_silence,
            "loss_count": loss_count,
            "loss_residual": loss_residual,
            "loss_fg_activity": loss_fg_activity,
            "loss_int_activity": loss_int_activity,
            "loss_noise_activity": loss_noise_activity,
        }

    loss_func.lambdas = lambda_state
    return loss_func
