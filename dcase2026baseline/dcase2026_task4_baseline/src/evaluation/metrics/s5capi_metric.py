from torchmetrics.functional import signal_noise_ratio as snr
import torch
import numpy as np
from itertools import combinations, permutations

class S5ClassAwareMetric():
    """Official-baseline CAPI-SDRi implementation.

    The public DCASE 2026 Task 4 baseline selects same-class permutations by
    raw SDR, then reports SDR improvement for the selected assignment. Keep this
    behavior as the default so local validation stays comparable with the
    official baseline scorer.
    """
    def __init__(self, metricfunc='sdr'):
        if metricfunc == 'sdr':
            self.metric_func = snr
            self.metric_name = 'CAPI-SDRi'
            self.min_max = 'max'
        else: raise ValueError(f"metricfunc of '{metricfunc}' is not implemented!!")
        self.metric_values = []
        self.reset()

    def update(self, batch_est_labels, batch_est_waveforms, batch_ref_labels, batch_ref_waveforms, batch_mixture):
        mvalues = self.compute_batch(batch_est_labels, batch_est_waveforms, batch_ref_labels, batch_ref_waveforms, batch_mixture)
        self.metric_values.extend(mvalues)
        return mvalues

    def _diagnostic_mean(self, key):
        values = [v[key] for v in self.diagnostic_values if key in v]
        if not values:
            return None
        return sum(values) / len(values)
        
    def compute(self, is_print=False):
        if not self.metric_values:
            return None
        non_None_metric_values = [v for v in self.metric_values if v is not None]
        if not non_None_metric_values:
            return None
        reobj = {
            'mean': sum(non_None_metric_values)/len(non_None_metric_values)
        }

        raw_sdr_mean = self._diagnostic_mean('raw_sdr')
        mix_sdr_mean = self._diagnostic_mean('mix_sdr')
        reference_mix_sdr_mean = self._diagnostic_mean('reference_mix_sdr')
        sdri_mean = self._diagnostic_mean('sdri')
        if raw_sdr_mean is not None:
            reobj['raw_sdr'] = raw_sdr_mean
        if mix_sdr_mean is not None:
            reobj['mix_sdr'] = mix_sdr_mean
        if reference_mix_sdr_mean is not None:
            reobj['reference_mix_sdr'] = reference_mix_sdr_mean
        if sdri_mean is not None:
            reobj['sdri'] = sdri_mean

        if is_print:
            print('%s: %.3f'%(self.metric_name, reobj['mean']))
            if raw_sdr_mean is not None:
                print('raw_sdr: %.3f'%raw_sdr_mean)
            if mix_sdr_mean is not None:
                print('mix_sdr: %.3f'%mix_sdr_mean)
            if reference_mix_sdr_mean is not None:
                print('reference_mix_sdr: %.3f'%reference_mix_sdr_mean)
            if sdri_mean is not None:
                print('sdri: %.3f'%sdri_mean)
        return reobj
        
    def reset(self):
        self.metric_values = []
        self.diagnostic_values = []
    
    def compute_batch(self, batch_est_labels, batch_est_waveforms, batch_ref_labels, batch_ref_waveforms, batch_mixture):
        return [self.compute_sample(est_lb, est_wf, ref_lb, ref_wf, mixture)
                for est_lb, est_wf, ref_lb, ref_wf, mixture in
                zip(batch_est_labels,  batch_est_waveforms, batch_ref_labels, batch_ref_waveforms, batch_mixture)]


    def _record_best_permutation_diagnostics(self, metrics, metrics_mixture, metrics_i, best_i):
        raw_sdr = metrics[best_i].mean().item()
        mix_sdr = metrics_mixture[best_i].mean().item()
        sdri = metrics_i[best_i].mean().item()
        self.diagnostic_values.append({
            'raw_sdr': raw_sdr,
            'mix_sdr': mix_sdr,
            'sdri': sdri,
        })
        return raw_sdr, mix_sdr, sdri

    def _record_reference_mix_sdr(self, ref_lb, ref_wf, mixture):
        active_indices = [idx for idx, label in enumerate(ref_lb) if label != 'silence']
        if not active_indices:
            return None
        active_ref_wf = ref_wf[active_indices]
        mixture_repeat = mixture.view(1, -1).expand(active_ref_wf.shape[0], -1)
        reference_mix_sdr = self.metric_func(mixture_repeat, active_ref_wf).mean().item()
        self.diagnostic_values.append({
            'reference_mix_sdr': reference_mix_sdr,
        })
        return reference_mix_sdr


    def _pi_metric(self,
                   est_wf, # nevent, wlen
                    ref_wf, # nevent, wlen
                    mixture, # 1, wlen
                  ):
        assert est_wf.shape[0] != 0 and ref_wf.shape[0] != 0
        TP = min(est_wf.shape[0], ref_wf.shape[0])
    
        # all possible permutation
        perms = []
        perm_est_wfs = []
        perm_ref_wfs = []
        for rp in combinations(range(ref_wf.shape[0]), TP):
            for ep in permutations(range(est_wf.shape[0]), TP):
                rp = list(rp)
                ep = list(ep)
                perms.append((rp, ep))
                perm_ref_wfs.append(ref_wf[rp, :])
                perm_est_wfs.append(est_wf[ep, :])
    
        perm_est_wfs_stack = torch.stack(perm_est_wfs, dim=0) # nperm, n_tp, wlen
        perm_ref_wfs_stack = torch.stack(perm_ref_wfs, dim=0) # nperm, n_tp, wlen
    
        mixture_repeat = mixture.view(1, 1, -1).expand(perm_est_wfs_stack.shape[0], perm_est_wfs_stack.shape[1], -1)
    
        # calculate metric
        metrics = self.metric_func(perm_est_wfs_stack, perm_ref_wfs_stack) # nperm, n_tp
        metrics_mixture = self.metric_func(mixture_repeat, perm_ref_wfs_stack) # nperm, n_tp
        metrics_i = metrics - metrics_mixture # metric improvement
    
        # find the best permutation
        metrics_mean = metrics.mean(dim=tuple(range(1, metrics.dim()))) # n_perm
        if self.min_max == 'max':   best_i = torch.argmax(metrics_mean).item()
        elif self.min_max == 'min': best_i = torch.argmin(metrics_mean).item()
        else: raise NotImplementedError(f"min_max '{self.min_max}' has not been implemented.")

        self._record_best_permutation_diagnostics(metrics, metrics_mixture, metrics_i, best_i)
    
        # extract the best permutation results
        best_metric = metrics[best_i] # n_tp
        best_metric_i = metrics_i[best_i] # n_tp
        best_ref_perm, best_est_perm = perms[best_i]
    
        # append TP or FP penalties of any
        if est_wf.shape[0] != ref_wf.shape[0]:
            fnfp = abs(est_wf.shape[0] - ref_wf.shape[0])
            best_metric = torch.cat((best_metric, torch.zeros(fnfp)))
            best_metric_i = torch.cat((best_metric_i, torch.zeros(fnfp)))
    
        return {
            'metric': best_metric,
            'metric_i': best_metric_i,
            'est_perm': best_est_perm, # local indices
            'ref_perm': best_ref_perm,
        }


    def compute_sample(self,
                  est_lb, # list [lb1, lb2,...]
                  est_wf, # [nevent, wlen]
                  ref_lb, # list [lb1, lb2, ...]
                  ref_wf, # [nevent, wlen]
                  mixture, # [wlen, ]
                  ):
    
        self._record_reference_mix_sdr(ref_lb, ref_wf, mixture)
        all_labels = (set(est_lb) | set(ref_lb)) - {'silence'}
        if not all_labels: return None # true silence prediction
    
        # collect waveform of the same class
        est_lists = {lb: [] for lb in all_labels}
        ref_lists = {lb: [] for lb in all_labels}
    
        for i, (lb, wf) in enumerate(zip(est_lb, est_wf)):
            if lb != 'silence':
                est_lists[lb].append(wf)
    
        for i, (lb, wf) in enumerate(zip(ref_lb, ref_wf)):
            if lb != 'silence':
                ref_lists[lb].append(wf)
    
        est_dict = {
            lb: torch.stack(est_lists[lb], dim=0) if est_lists[lb] else torch.empty(
                (0, *est_wf.shape[1:]), dtype=est_wf.dtype, device=est_wf.device
            )
            for lb in all_labels
        }
        ref_dict = {
            lb: torch.stack(ref_lists[lb], dim=0) if ref_lists[lb] else torch.empty(
                (0, *ref_wf.shape[1:]), dtype=ref_wf.dtype, device=ref_wf.device
            )
            for lb in all_labels
        }
    
        metric_i = []
        for lb in all_labels:
            est_wf_1c = est_dict[lb]
            ref_wf_1c = ref_dict[lb]
            assert est_wf_1c.shape[0] != 0 or ref_wf_1c.shape[0] != 0
            if est_wf_1c.shape[0] == 0: # all False Negative
                metric_i.append(torch.zeros(ref_wf_1c.shape[0]))
            elif ref_wf_1c.shape[0] == 0: # all False Positive
                metric_i.append(torch.zeros(est_wf_1c.shape[0]))
            else:
                output = self._pi_metric(est_wf = est_wf_1c,
                                   ref_wf = ref_wf_1c,
                                   mixture = mixture)
                metric_i.append(output['metric_i'])
        metric_i = torch.cat(metric_i)

        return  metric_i.mean().item()


class S5ClassAwareMetricSDRiAssignment(S5ClassAwareMetric):
    """Paper-definition diagnostic that selects assignments by SDR improvement.

    The DCASE task description writes the permutation objective in terms of
    SDRi. This class is intentionally separate from ``S5ClassAwareMetric`` so
    official-baseline compatibility is not changed by diagnostic experiments.
    """

    def _pi_metric(self,
                   est_wf, # nevent, wlen
                    ref_wf, # nevent, wlen
                    mixture, # 1, wlen
                  ):
        assert est_wf.shape[0] != 0 and ref_wf.shape[0] != 0
        TP = min(est_wf.shape[0], ref_wf.shape[0])

        perms = []
        perm_est_wfs = []
        perm_ref_wfs = []
        for rp in combinations(range(ref_wf.shape[0]), TP):
            for ep in permutations(range(est_wf.shape[0]), TP):
                rp = list(rp)
                ep = list(ep)
                perms.append((rp, ep))
                perm_ref_wfs.append(ref_wf[rp, :])
                perm_est_wfs.append(est_wf[ep, :])

        perm_est_wfs_stack = torch.stack(perm_est_wfs, dim=0)
        perm_ref_wfs_stack = torch.stack(perm_ref_wfs, dim=0)

        mixture_repeat = mixture.view(1, 1, -1).expand(perm_est_wfs_stack.shape[0], perm_est_wfs_stack.shape[1], -1)
        metrics = self.metric_func(perm_est_wfs_stack, perm_ref_wfs_stack)
        metrics_mixture = self.metric_func(mixture_repeat, perm_ref_wfs_stack)
        metrics_i = metrics - metrics_mixture

        metrics_i_mean = metrics_i.mean(dim=tuple(range(1, metrics_i.dim())))
        if self.min_max == 'max':   best_i = torch.argmax(metrics_i_mean).item()
        elif self.min_max == 'min': best_i = torch.argmin(metrics_i_mean).item()
        else: raise NotImplementedError(f"min_max '{self.min_max}' has not been implemented.")

        self._record_best_permutation_diagnostics(metrics, metrics_mixture, metrics_i, best_i)

        best_metric = metrics[best_i]
        best_metric_i = metrics_i[best_i]
        best_ref_perm, best_est_perm = perms[best_i]

        if est_wf.shape[0] != ref_wf.shape[0]:
            fnfp = abs(est_wf.shape[0] - ref_wf.shape[0])
            best_metric = torch.cat((best_metric, best_metric.new_zeros(fnfp)))
            best_metric_i = torch.cat((best_metric_i, best_metric_i.new_zeros(fnfp)))

        return {
            'metric': best_metric,
            'metric_i': best_metric_i,
            'est_perm': best_est_perm,
            'ref_perm': best_ref_perm,
        }


class S5ClassAwareMetricAssignmentComparison(S5ClassAwareMetric):
    """Compare official raw-SDR and paper-definition SDRi assignments.

    This diagnostic reports both CAPI-SDRi values so we can see whether
    same-class assignment choices change the score. It must not be used as the
    official ranking score unless the task organizers change the evaluator.
    """

    def __init__(self, metricfunc='sdr'):
        super().__init__(metricfunc=metricfunc)
        self.metric_name = 'CAPI-SDRi assignment comparison'

    def compute(self, is_print=False):
        if not self.metric_values:
            return None
        values = [v for v in self.metric_values if v is not None]
        if not values:
            return None
        reobj = {
            'raw_sdr_assignment_mean': sum(v['raw_sdr_assignment'] for v in values) / len(values),
            'sdri_assignment_mean': sum(v['sdri_assignment'] for v in values) / len(values),
        }
        reobj['delta_sdri_minus_raw'] = reobj['sdri_assignment_mean'] - reobj['raw_sdr_assignment_mean']
        if is_print:
            print('CAPI-SDRi official raw-SDR assignment: %.3f'%(reobj['raw_sdr_assignment_mean']))
            print('CAPI-SDRi paper SDRi assignment      : %.3f'%(reobj['sdri_assignment_mean']))
            print('CAPI-SDRi paper-minus-official delta : %.3f'%(reobj['delta_sdri_minus_raw']))
        return reobj

    def compute_sample(self,
                  est_lb,
                  est_wf,
                  ref_lb,
                  ref_wf,
                  mixture,
                  ):
        raw_metric = S5ClassAwareMetric()
        raw_metric.metric_func = self.metric_func
        raw_metric.min_max = self.min_max
        sdri_metric = S5ClassAwareMetricSDRiAssignment()
        sdri_metric.metric_func = self.metric_func
        sdri_metric.min_max = self.min_max

        raw_value = raw_metric.compute_sample(est_lb, est_wf, ref_lb, ref_wf, mixture)
        sdri_value = sdri_metric.compute_sample(est_lb, est_wf, ref_lb, ref_wf, mixture)
        if raw_value is None and sdri_value is None:
            return None
        return {
            'raw_sdr_assignment': raw_value,
            'sdri_assignment': sdri_value,
            'delta_sdri_minus_raw': sdri_value - raw_value,
        }
