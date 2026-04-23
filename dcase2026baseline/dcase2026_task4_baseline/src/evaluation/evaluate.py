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

casdri = s5capi_metric.S5ClassAwareMetric(metricfunc = 'sdr')
label_metric = label_metric.LabelMetric()
metric_funcs = [casdri, label_metric]


class Evaluator:
    def __init__(self,
                 config_path,
                 waveform_output_dir = '',
                 result_dir = '',
                 batch_size=2,
                use_cpu=False):
        self.config_path = config_path
        self.filename = os.path.basename(config_path)[:-5]
        self.batch_size = batch_size
        self.waveform_output_dir = os.path.join(waveform_output_dir, self.filename) if waveform_output_dir else waveform_output_dir
        self.result_dir = result_dir
        self.use_cpu = use_cpu

        if self.waveform_output_dir: os.makedirs(self.waveform_output_dir, exist_ok=True)
        

        with open(self.config_path) as f: config = yaml.safe_load(f)
        dsconfig = config['dataset']
        self.use_generated_waveform = 'estimate_target_dir' in dsconfig['args']['config'] # if estimate_target_dir is provided, generated waveforms are used to evaluate
        assert not self.use_generated_waveform or not self.waveform_output_dir, 'if estimate_target_dir is provided in the dataset, waveform will not be generated again (waveform_output_dir should not be specified)'

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

    def evaluate(self):
        if self.result_dir: results = []
        for metric_func in metric_funcs: metric_func.reset()

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
            batch_mixture = batch['mixture'][:, 0, :] # [bs, wlen]
            batch_ref_waveforms = batch['dry_sources'][:, :, 0, :] # [bs, nsources, wlen]
            batch_ref_labels = batch['label']

            if self.waveform_output_dir:
                for labels, waveforms, soundscape_name in zip(batch_est_labels, batch_est_waveforms, batch['soundscape']):
                    for i, (label, waveform) in enumerate(zip(labels, waveforms)):
                        if label != 'silence':
                            wavpath = os.path.join(self.waveform_output_dir, soundscape_name + '_' + str(i) + '_' +  label + '.wav')
                            sf.write(wavpath, waveform.numpy(), self.sr)

            metric_values = []
            for metric_func in metric_funcs:
                metric_value = metric_func.update(batch_est_labels=batch_est_labels,
                                  batch_est_waveforms=batch_est_waveforms,
                                  batch_ref_labels=batch_ref_labels,
                                  batch_ref_waveforms=batch_ref_waveforms,
                                  batch_mixture=batch_mixture)
                metric_values.append(metric_value)
                    # 'metric': name = getattr(metric_func, "metric_name", None),

            if self.result_dir:
                for i in range(len(batch_mixture)):
                    reobj = {
                        'soundscape': batch['soundscape'][i],
                        'ref_labels': batch_ref_labels[i],
                        'est_labels': batch_est_labels[i],
                        'probabilities': batch_est_probabilities[i].tolist(),
                        'metrics': []
                    }
                    for mval, mfunc in zip(metric_values, metric_funcs):
                        reobj['metrics'].append({
                            'metric': getattr(mfunc, "metric_name", None),
                            'value': mval[i]
                        })
                    results.append(reobj)
                    # import pdb; pdb.set_trace()

        for metric_func in metric_funcs: metric_func.compute(is_print=True)
        if self.result_dir:
            os.makedirs(self.result_dir, exist_ok=True)
            with open(os.path.join(self.result_dir, f"{self.filename}_results.json"), "w") as outfile:
                json.dump(results, outfile, indent=4)

def main(args):
    evalobj = Evaluator(
                 args.config,
                 args.waveform_output_dir,
                 args.result_dir,
                 args.batchsize,
                 args.cpu)
    evalobj.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True,)
    parser.add_argument("--waveform_output_dir", type=str, required=False, default='')
    parser.add_argument("--result_dir", type=str, required=False, default='')
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batchsize","-b", type=int, required=False, default=2)

    args = parser.parse_args()
    print('START')
    main(args)

# python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2dat_4c_resunetk.yaml --result_dir workspace/evaluation
# python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2dat_1c_resunetk.yaml --result_dir workspace/evaluation
 
