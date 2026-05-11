import random
import numpy as np
import yaml
import json
from tqdm import tqdm
import torch
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr)
import os, sys;
import argparse
import soundfile as sf

from torch.utils.data import DataLoader
from src.utils import LABELS, initialize_config
from .metrics import label_metric, s5capi_metric
from .metrics.s5_validation_breakdown import S5ValidationBreakdownMetric


def build_metric_funcs(compare_assignment=False, validation_breakdown=False):
    metric_funcs = [s5capi_metric.S5ClassAwareMetric(metricfunc='sdr')]
    if compare_assignment:
        metric_funcs.append(s5capi_metric.S5ClassAwareMetricAssignmentComparison(metricfunc='sdr'))
    if validation_breakdown:
        metric_funcs.append(S5ValidationBreakdownMetric(metricfunc='sdr', prefix='valid'))
    metric_funcs.append(label_metric.LabelMetric())
    return metric_funcs


class Evaluator:
    def __init__(self,
                 config_path,
                 waveform_output_dir = '',
                 result_dir = '',
                 batch_size=2,
                 use_cpu=False,
                 compare_assignment=False,
                 validation_breakdown=False,
                 inference_only=False):
        self.config_path = config_path
        self.filename = os.path.basename(config_path)[:-5]
        self.batch_size = batch_size
        self.waveform_output_dir = os.path.join(waveform_output_dir, self.filename) if waveform_output_dir else waveform_output_dir
        self.result_dir = result_dir
        self.use_cpu = use_cpu
        self.inference_only = bool(inference_only)
        self.metric_funcs = [] if self.inference_only else build_metric_funcs(
            compare_assignment=compare_assignment,
            validation_breakdown=validation_breakdown,
        )

        if self.waveform_output_dir: os.makedirs(self.waveform_output_dir, exist_ok=True)
        

        with open(self.config_path) as f: config = yaml.safe_load(f)
        dsconfig = config['dataset']
        self.use_generated_waveform = 'estimate_target_dir' in dsconfig['args']['config'] # if estimate_target_dir is provided, generated waveforms are used to evaluate
        assert not self.use_generated_waveform or not self.waveform_output_dir, 'if estimate_target_dir is provided in the dataset, waveform will not be generated again (waveform_output_dir should not be specified)'
        if self.inference_only and self.use_generated_waveform:
            raise ValueError("--inference_only expects model inference; remove estimate_target_dir from the dataset config.")

        # load model and dataset
        dataset = initialize_config(config['dataset'], reload=True)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=dataset.collate_fn,
                                num_workers=batch_size*2)
        if not self.use_generated_waveform:
            model = initialize_config(config['model'], reload=True)
            model.eval();
            if not self.use_cpu: model = model.to('cuda')
            self.model = model

        self.dataset = dataset
        self.sr = self.dataset.sr
        self.dataloader = dataloader

    def predict(self, mixture, labels=None):
        if not self.use_cpu: mixture = mixture.to('cuda')
        if labels is not None:
            with torch.no_grad():
                batch_est_labels = labels
                output = self.model.separate(mixture, batch_est_labels)
                batch_est_waveforms = output['waveform'] # [bs, nsources, wlen]
                output['label'] = labels # bs, nsources
                output['probabilities'] = torch.ones(batch_est_waveforms.shape[:2], dtype=torch.float32)# bs, nsources
        else:
            with torch.no_grad():
                output = self.model.predict_label_separate(mixture)
                # batch_est_labels = output['label'] # bs, nsources
                # batch_probabilities = output['probablities'] # bs, nsources
                # batch_est_waveforms = output['waveform'].cpu()# [bs, nsources, wlen]
        return output

    def _jsonable(self, value):
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [self._jsonable(v) for v in value]
        return value

    def _write_prediction_waveforms(self, batch_est_labels, batch_est_waveforms, soundscapes):
        waveform_paths = [[] for _ in soundscapes]
        if not self.waveform_output_dir:
            return waveform_paths
        for sample_idx, (labels, waveforms, soundscape_name) in enumerate(zip(batch_est_labels, batch_est_waveforms, soundscapes)):
            for source_idx, (label, waveform) in enumerate(zip(labels, waveforms)):
                if label != 'silence':
                    wavpath = os.path.join(self.waveform_output_dir, soundscape_name + '_' + str(source_idx) + '_' + label + '.wav')
                    sf.write(wavpath, waveform.numpy(), self.sr)
                    waveform_paths[sample_idx].append(wavpath)
        return waveform_paths

    def _append_prediction_results(self, results, batch, batch_est_labels, batch_est_probabilities, waveform_paths):
        if results is None:
            return
        for i, soundscape_name in enumerate(batch['soundscape']):
            reobj = {
                'soundscape': soundscape_name,
                'est_labels': self._jsonable(batch_est_labels[i]),
                'probabilities': self._jsonable(batch_est_probabilities[i]),
            }
            if self.waveform_output_dir:
                reobj['waveform_files'] = waveform_paths[i]
            results.append(reobj)

    def _count_non_silence(self, batch_est_labels):
        return sum(
            1
            for sample_labels in batch_est_labels
            for label in sample_labels
            if label != 'silence'
        )

    def evaluate(self):
        results = [] if self.result_dir else None
        num_soundscapes = 0
        num_non_silence_predictions = 0
        for metric_func in self.metric_funcs: metric_func.reset()

        for batch in tqdm(self.dataloader):
            if self.use_generated_waveform:
                output = {}
                output['label'] = batch['est_label'] # bs, nsources
                output['waveform'] = batch['est_dry_sources'] # [bs, nsources, 1c, wlen]
                output['probabilities'] = torch.ones(output['waveform'].shape[:2], dtype=torch.float32) # bs, nsources
            else:
                output = self.predict(batch['mixture'])

            batch_est_waveforms = output['waveform'][:, :, 0, :].cpu() # [bs, nsources, wlen]
            batch_est_labels = output['label']
            batch_est_probabilities = output['probabilities']
            if self.inference_only:
                waveform_paths = self._write_prediction_waveforms(batch_est_labels, batch_est_waveforms, batch['soundscape'])
                num_soundscapes += len(batch['soundscape'])
                num_non_silence_predictions += self._count_non_silence(batch_est_labels)
                self._append_prediction_results(results, batch, batch_est_labels, batch_est_probabilities, waveform_paths)
                continue

            if 'dry_sources' not in batch or 'label' not in batch:
                raise KeyError(
                    "Evaluation mode requires oracle 'dry_sources' and 'label'. "
                    "For hidden-test prediction, use --inference_only with a config that omits oracle_target_dir."
                )
            self._write_prediction_waveforms(batch_est_labels, batch_est_waveforms, batch['soundscape'])
            batch_mixture = batch['mixture'][:, 0, :] # [bs, wlen]
            batch_ref_waveforms = batch['dry_sources'][:, :, 0, :] # [bs, nsources, wlen]
            batch_ref_labels = batch['label']

            metric_values = []
            for metric_func in self.metric_funcs:
                metric_value = metric_func.update(batch_est_labels=batch_est_labels,
                                  batch_est_waveforms=batch_est_waveforms,
                                  batch_ref_labels=batch_ref_labels,
                                  batch_ref_waveforms=batch_ref_waveforms,
                                  batch_mixture=batch_mixture)
                metric_values.append(metric_value)
                    # 'metric': name = getattr(metric_func, "metric_name", None),

            if results is not None:
                for i in range(len(batch_mixture)):
                    reobj = {
                        'soundscape': batch['soundscape'][i],
                        'ref_labels': batch_ref_labels[i],
                        'est_labels': batch_est_labels[i],
                        'probabilities': batch_est_probabilities[i].tolist(),
                        'metrics': []
                    }
                    for mval, mfunc in zip(metric_values, self.metric_funcs):
                        reobj['metrics'].append({
                            'metric': getattr(mfunc, "metric_name", None),
                            'value': mval[i]
                        })
                    results.append(reobj)
                    # import pdb; pdb.set_trace()

        if self.inference_only:
            summary = {
                'mode': 'inference_only',
                'num_soundscapes': num_soundscapes,
                'num_non_silence_predictions': num_non_silence_predictions,
                'waveform_output_dir': self.waveform_output_dir,
            }
        else:
            summary = {}
            for metric_func in self.metric_funcs:
                metric_summary = metric_func.compute(is_print=True)
                if isinstance(metric_summary, dict):
                    summary[getattr(metric_func, "metric_name", metric_func.__class__.__name__)] = metric_summary
        if self.result_dir:
            os.makedirs(self.result_dir, exist_ok=True)
            with open(os.path.join(self.result_dir, f"{self.filename}_results.json"), "w") as outfile:
                json.dump(results, outfile, indent=4)
            with open(os.path.join(self.result_dir, f"{self.filename}_summary.json"), "w") as outfile:
                json.dump(summary, outfile, indent=4)

def main(args):
    evalobj = Evaluator(
                 args.config,
                 args.waveform_output_dir,
                 args.result_dir,
                 args.batchsize,
                 args.cpu,
                 compare_assignment=args.compare_assignment,
                 validation_breakdown=args.validation_breakdown,
                 inference_only=args.inference_only)
    evalobj.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True,)
    parser.add_argument("--waveform_output_dir", type=str, required=False, default='')
    parser.add_argument("--result_dir", type=str, required=False, default='')
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batchsize","-b", type=int, required=False, default=2)
    parser.add_argument("--compare_assignment", action="store_true", help="Also log official raw-SDR assignment vs paper SDRi-assignment CAPI-SDRi diagnostics.")
    parser.add_argument("--validation_breakdown", action="store_true", help="Also log CAPI-SDRi, zero-target FP, silence, leakage, and scene-bucket diagnostics.")
    parser.add_argument("--inference_only", "--inference-only", action="store_true", help="Write predictions without loading oracle targets or computing validation metrics.")

    args = parser.parse_args()
    print('START')
    main(args)

# python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2dat_4c_resunetk.yaml --result_dir workspace/evaluation
# python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2dat_1c_resunetk.yaml --result_dir workspace/evaluation
