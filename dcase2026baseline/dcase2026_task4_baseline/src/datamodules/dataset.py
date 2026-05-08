import os
import copy
import torch
import re
import numpy as np
import librosa
import random
import warnings
import json

from src.temporal import SILENCE_SPAN_SEC, event_to_span_sec, pad_spans, waveform_to_span_sec
from src.utils import LABELS

SPATIAL_SOUND_SCENE_KEYS = {
    'duration',
    'sr',
    'max_event_overlap',
    'max_event_dur',
    'ref_db',
    'foreground_dir',
    'background_dir',
    'interference_dir',
    'room_config',
    'verbose',
}

def _get_spatial_audio_synthesizer():
    try:
        from src.modules.spatial_audio_synthesizer.spatial_audio_synthesizer import SpAudSyn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "DatasetS3 generate/metadata synthesis requires "
            "src.modules.spatial_audio_synthesizer. Waveform-mode datasets do "
            "not need it; check the SpAudSyn checkout or symlink before using "
            "generate/metadata synthesis."
        ) from exc
    return SpAudSyn

def collate_fn(list_data_dict):
    data = {k: [] for k in list_data_dict[0].keys()}
    for ddict in list_data_dict:
        for k in data:
            data[k].append(ddict[k])
    for k in data.keys():
        if type(data[k][0]) is torch.Tensor:
            data[k] = torch.stack(data[k], 0)
    return data

class DatasetS3(torch.utils.data.Dataset):
    def __init__(self,
                 config,
                 n_sources,
                 label_set, # 'dcase2025t4' key of LABELS in utils
                 label_vector_mode='multihot', # multihot, concat, stack
                 silence_label_mode='zeros', # zero, onehot
                 return_meta=None, # return all the wet source, dry source, ir from spatial scaper
                 return_source=True, # if false, only return mixture and label
                ):
        super().__init__()
        self.return_meta = return_meta
        self.label_set = label_set
        self.config = config
        self.n_sources = n_sources
        self.return_source = return_source
        self.label_vector_mode = label_vector_mode
        self.silence_label_mode = silence_label_mode
        self.labels = LABELS[self.label_set].copy()

        if self.config['mode']== 'waveform':
            self.soundscape_dir = self.config['soundscape_dir']
            self.oracle_target_dir = self.config.get('oracle_target_dir', None)
            self.estimate_target_dir = self.config.get('estimate_target_dir', None)
            self.data = [{'soundscape': f[:-4],
                      'mixture_path': os.path.join(self.soundscape_dir, f)
                      } for f in os.listdir(self.soundscape_dir) if f.endswith(".wav")]
            self.data = sorted(self.data, key=lambda x: x['soundscape'])
            if self.oracle_target_dir is not None:
                self._get_data_waveform(self.data, 'ref', self.oracle_target_dir)
            if self.estimate_target_dir is not None:
                self._get_data_waveform(self.data, 'est', self.estimate_target_dir)

            self.sr = self.config['sr']
            self.dataset_length = len(self.data)
        if self.config['mode']== 'metadata':
            self.sr = self.config['sr']
            self.fg_return = self.config['fg_return']
            self.metadata_list = self.config['metadata_list']

            self.metadata_dir = os.path.dirname(self.metadata_list)
            with open(self.metadata_list) as f:
                self.data = json.load(f);
            self.dataset_length = len(self.data)
            self.shuffle_label = False

        elif self.config['mode']== 'generate':
            self.dupse_rate = self.config['dupse_rate'] # 0.5
            self.dupse_min_angle = np.deg2rad(self.config['dupse_min_angle']) # 60
            self.max_n_dupse = self.config['max_n_dupse'] # => rand(1, max_n_dupse) events will be duplcated
            self.dupse_exclusion_folder_depth = self.config['dupse_exclusion_folder_depth'] # check subfolder_level in source_file_filter function

            self.spatial_sound_scene = self.config['spatial_sound_scene']
            self.spatial_sound_scene_sources = self._build_spatial_sound_scene_sources(
                self.config.get('spatial_sound_scene_sources')
            )
            self.sr = self.config['spatial_sound_scene']['sr']
            self.snr_range = self.config['snr_range']
            self.nevent_range = self.config['nevent_range']
            self.dataset_length = self.config['dataset_length']
            self.shuffle_label = self.config['shuffle_label']
            self.fg_return = self.config['fg_return']


        print(self.labels, flush=True)
        print(len(self.labels), flush=True)
        if self.silence_label_mode == 'zeros':
            self.onehots = torch.eye(len(self.labels), requires_grad=False).to(torch.float32)
            self.label_onehots = {label: self.onehots[idx] for idx, label in enumerate(self.labels)}
            self.label_onehots['silence'] = torch.zeros(self.onehots.size(1), requires_grad=False,  dtype=torch.float32)
        elif self.silence_label_mode == 'onehot':
            self.onehots = torch.eye(len(self.labels) + 1, requires_grad=False).to(torch.float32)
            self.label_onehots = {label: self.onehots[idx] for idx, label in enumerate(self.labels)}
            self.label_onehots['silence'] = self.onehots[-1]

        self.collate_fn = collate_fn

    def get_onehot(self, label):
        return self.label_onehots[label]

    def __len__(self):
        return self.dataset_length

    def _get_label_vector(self, labels):
        label_vector_all = torch.stack([self.get_onehot(label) for label in labels]) # [nevent, nclass]
        if self.label_vector_mode == 'multihot': label_vector_all = torch.any(label_vector_all.bool(), dim=0).float() # [nclass]
        elif self.label_vector_mode == 'concat': label_vector_all = label_vector_all.flatten() # [nevent x nclass]
        elif self.label_vector_mode == 'stack': pass  # [nevent, nclass]
        else: raise NotImplementedError(f'label_vector_mode of "{self.label_vector_mode}" has not been implemented')
        return label_vector_all

    def __getitem__(self, idx):
        if self.config['mode']== 'waveform':
            soundscene = self._get_item_waveform(idx)
            soundscene['soundscape'] = self.data[idx]['soundscape']
        elif self.config['mode']== 'generate':
            soundscene = self._get_item_generate(idx)
            soundscene['soundscape'] = 'soundscape_%08d'%(idx)
        elif self.config['mode']== 'metadata':
            soundscene = self._get_item_metadata(idx)
            soundscene['soundscape'] = 'soundscape_%04d'%(idx)

        return soundscene

    #=====================================================
    # Utilizations for waveform mode
    #=====================================================
    def _get_data_waveform(self, data, est_ref, source_dir):
        if not os.path.exists(source_dir):
            raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")
        all_wav = [f for f in os.listdir(source_dir) if f.endswith(".wav")]
        for d in data:
            pattern = rf"^{re.escape(d['soundscape'])}(?:_(\d+))?_(.+)\.wav$" # d['soundscape']_number_label.wav     or    d['soundscape']_label.wav
            matched_sources = []
            for source in all_wav:
                match = re.match(pattern, source)
                if match:
                    slot = int(match.group(1)) if match.group(1) is not None else None
                    matched_sources.append((slot, source, match.group(2)))
            numbered = [slot for slot, _, _ in matched_sources if slot is not None]
            if numbered and len(numbered) != len(matched_sources):
                raise ValueError(
                    f"Mixed numbered and unnumbered source files for {d['soundscape']} in {source_dir}. "
                    "Use either '<soundscape>_<slot>_<label>.wav' for all files or no slot for all files."
                )
            if len(numbered) != len(set(numbered)):
                raise ValueError(f"Duplicate source slot ids for {d['soundscape']} in {source_dir}.")
            if numbered and (min(numbered) < 0 or max(numbered) >= self.n_sources):
                raise ValueError(
                    f"Source slot ids for {d['soundscape']} in {source_dir} must be in "
                    f"[0, {self.n_sources - 1}], got {sorted(numbered)}."
                )
            if len(matched_sources) > self.n_sources:
                files = [source for _, source, _ in sorted(matched_sources, key=lambda x: (x[0] is None, x[0] if x[0] is not None else x[1]))]
                raise ValueError(
                    f"Found {len(matched_sources)} source files for {d['soundscape']} in {source_dir}, "
                    f"but n_sources={self.n_sources}. Remove stale cache files or regenerate the cache. Files: {files}"
                )
            # if not sources: warnings.warn(f'No estimate for {d["mixture_path"]}')

            if numbered:
                labels = ['silence'] * self.n_sources
                source_paths = [None] * self.n_sources
                sources = sorted(matched_sources, key=lambda x: x[0])
                for slot, source, label in sources:
                    assert label in self.labels, f'"{source}" is not a valid filename of the estimates for {d["soundscape"]}'
                    labels[slot] = label
                    source_paths[slot] = os.path.join(source_dir, source)
            else:
                labels = []
                source_paths = []
                sources = sorted(matched_sources, key=lambda x: x[1])
                for _, source, label in sources:
                    assert label in self.labels, f'"{source}" is not a valid filename of the estimates for {d["soundscape"]}'
                    labels.append(label)
                    source_paths.append(os.path.join(source_dir, source))

            d[est_ref + '_label'] = labels
            d[est_ref + '_source_paths'] = source_paths

    def _get_label_waveform(self, info, est_ref):
        labels = list(info[est_ref + '_label'])
        if len(labels) > self.n_sources:
            raise ValueError(
                f"{est_ref}_label for {info['soundscape']} has {len(labels)} slots, "
                f"but n_sources={self.n_sources}."
            )
        if len(labels) < self.n_sources:
            labels.extend(['silence'] * (self.n_sources - len(labels)))
        return labels

    def _get_source_waveform(self, info, est_ref, wlen):
        dry_sources = []
        labels = self._get_label_waveform(info, est_ref)
        source_paths = list(info[est_ref + '_source_paths'])
        if len(source_paths) < self.n_sources:
            source_paths.extend([None] * (self.n_sources - len(source_paths)))
        for label, source_path in zip(labels, source_paths):
            if label == 'silence' or source_path is None:
                dry_sources.append(np.zeros(wlen, dtype=np.float32))
                continue
            dry_source, sr = librosa.load(source_path, sr = None)
            assert sr == self.sr, f'sr of {source_path} ({sr}) is different from the target sr ({self.sr})'
            dry_sources.append(dry_source)
        assert len(dry_sources) == self.n_sources

        return torch.from_numpy(np.stack(dry_sources))[:, None, :].to(torch.float32) # [nevents, 1, wlen]

    def _get_source_span_waveform(self, info, est_ref, wlen):
        spans = []
        labels = self._get_label_waveform(info, est_ref)
        source_paths = list(info[est_ref + '_source_paths'])
        if len(source_paths) < self.n_sources:
            source_paths.extend([None] * (self.n_sources - len(source_paths)))
        for label, source_path in zip(labels, source_paths):
            if label == 'silence' or source_path is None:
                spans.append(SILENCE_SPAN_SEC)
                continue
            dry_source, sr = librosa.load(source_path, sr=None)
            assert sr == self.sr, f'sr of {source_path} ({sr}) is different from the target sr ({self.sr})'
            spans.append(waveform_to_span_sec(dry_source, self.sr))
        return pad_spans(spans, self.n_sources)

    def _get_item_waveform(self, idx):
        info = self.data[idx]
        mixture, sr = librosa.load(info['mixture_path'], sr = None, mono=False)
        assert sr == self.sr, f'sr of {info["mixture_path"]} ({sr}) is different from the target sr ({self.sr})'
        item = {
            'mixture': torch.from_numpy(mixture).to(torch.float32), # [nch, wlen]
        }

        if self.oracle_target_dir is not None:
            item['label'] = self._get_label_waveform(info, 'ref')
            item['label_vector'] = self._get_label_vector(item['label'])
            item['span_sec'] = self._get_source_span_waveform(info, 'ref', mixture.shape[-1])
            if self.return_source: item['dry_sources'] = self._get_source_waveform(info, 'ref', mixture.shape[-1]) # nsources, 1ch, wlen

        if self.estimate_target_dir is not None:
            item['est_label'] = self._get_label_waveform(info, 'est')
            item['est_label_vector'] = self._get_label_vector(item['est_label'])
            item['est_span_sec'] = self._get_source_span_waveform(info, 'est', mixture.shape[-1])
            if self.return_source: item['est_dry_sources'] = self._get_source_waveform(info, 'est', mixture.shape[-1]) # nsources, 1ch, wlen
        if self.return_meta: item['metadata'] = info
        return item

    #=====================================================
    # Utilizations for generate mode
    #=====================================================
    def _build_spatial_sound_scene_sources(self, source_config):
        if source_config is None:
            return None

        sampling_mode = source_config.get('sampling_mode', 'scene_weighted')
        if sampling_mode != 'scene_weighted':
            raise ValueError(
                "DatasetS3 currently supports only "
                "spatial_sound_scene_sources.sampling_mode='scene_weighted'"
            )

        pools = source_config.get('pools', [])
        if not pools:
            raise ValueError("spatial_sound_scene_sources.pools must contain at least one pool")

        active_pools = []
        for idx, pool in enumerate(pools):
            name = pool.get('name', f'pool_{idx}')
            weight = float(pool.get('weight', 1.0))
            if weight < 0:
                raise ValueError(f"Source pool '{name}' has negative weight: {weight}")
            if weight == 0:
                continue

            pool_scene_config = copy.deepcopy(self.spatial_sound_scene)
            overrides = {}
            overrides.update(pool.get('spatial_sound_scene', {}))
            for key, value in pool.items():
                if key in {'name', 'weight', 'spatial_sound_scene'}:
                    continue
                overrides[key] = value

            unknown_keys = sorted(set(overrides) - SPATIAL_SOUND_SCENE_KEYS)
            if unknown_keys:
                raise ValueError(
                    f"Source pool '{name}' has unsupported spatial_sound_scene override keys: {unknown_keys}"
                )
            pool_scene_config.update(overrides)
            active_pools.append({
                'name': name,
                'weight': weight,
                'spatial_sound_scene': pool_scene_config,
            })

        if not active_pools:
            raise ValueError("spatial_sound_scene_sources must contain at least one pool with weight > 0")

        weights = [pool['weight'] for pool in active_pools]
        print(
            "DatasetS3 source pools: "
            + ", ".join(f"{pool['name']}={pool['weight']}" for pool in active_pools),
            flush=True,
        )
        return {
            'sampling_mode': sampling_mode,
            'pools': active_pools,
            'weights': weights,
        }

    def _select_spatial_sound_scene(self):
        if self.spatial_sound_scene_sources is None:
            return copy.deepcopy(self.spatial_sound_scene), None

        pool = random.choices(
            self.spatial_sound_scene_sources['pools'],
            weights=self.spatial_sound_scene_sources['weights'],
            k=1,
        )[0]
        return copy.deepcopy(pool['spatial_sound_scene']), pool['name']

    def _get_position(self,
                      ref_pos, # [3,] or [nrefpos, 3]     x,y,z
                      all_pos): # [npos, 3]
        ref_pos = np.atleast_2d(ref_pos)
        ref_unit = ref_pos / np.linalg.norm(ref_pos, axis=1, keepdims=True)
        all_unit = all_pos / np.linalg.norm(all_pos, axis=1, keepdims=True)
        cos_theta = all_unit @ ref_unit.T

        angles = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        valid_mask = np.all(angles > self.dupse_min_angle, axis=1)
        valid_indices = np.where(valid_mask)[0]

        return all_pos[random.choice(valid_indices)]
    def _generate(self, s3):
        output = s3.synthesize(fg_return=self.fg_return,
                               int_return=self.config.get('int_return', {}),
                               bg_return=self.config.get('bg_return', {}),)
        mixture = output['mixture'] # [nchan, wlen]

        if self.shuffle_label:
            random.shuffle(output['fg_events'])

        label = [fge['metadata']['label'] for fge in output['fg_events']];
        span_sec = [event_to_span_sec(fge) for fge in output['fg_events']]
        npad = self.n_sources - len(output['fg_events'])
        if npad > 0: label.extend(['silence'] * npad) # add silence to get n_sources
        return_obj = {
            'mixture': torch.from_numpy(mixture).to(torch.float32), # nchan, wlen
            'label': label, # [lb1, lb2,...]
            'label_vector': self._get_label_vector(label),
            'span_sec': pad_spans(span_sec, self.n_sources),
        }

        if self.return_source:
            source = [fge['waveform_dry'] for fge in output['fg_events']];
            if npad > 0:
                source.extend([np.zeros((1, mixture.shape[-1]), dtype=mixture.dtype) for _ in range(npad)])
            assert len(return_obj['label']) == len(source)

            return_obj['dry_sources'] = torch.from_numpy(np.stack(source)).to(torch.float32) # nsources, 1ch, wlen
        if self.return_meta:
            if 'source_pool' in s3.config:
                output['source_pool'] = s3.config['source_pool']
                output['source_pool_sampling_mode'] = s3.config.get('source_pool_sampling_mode')
            return_obj['metadata'] = output

        return return_obj


    def _get_item_generate(self, idx):
        SpAudSyn = _get_spatial_audio_synthesizer()
        spatial_sound_scene, source_pool_name = self._select_spatial_sound_scene()
        s3 = SpAudSyn(**spatial_sound_scene)
        if source_pool_name is not None:
            s3.config['source_pool'] = source_pool_name
            s3.config['source_pool_sampling_mode'] = self.spatial_sound_scene_sources['sampling_mode']
        # s3.set_room(('choose', [])) # room has been set in __init__ if room is not None

        # add events
        nevents = random.randint(self.nevent_range[0], self.nevent_range[1])
        if nevents < 2 or random.random() > self.dupse_rate:
            for i in range(nevents):
                s3.add_event(
                    label={'method': 'choose_wo_replacement'},
                    source_file={'method': 'choose'},
                    source_time={'method': 'choose'},
                    event_time={'method': 'choose'},
                    event_position={'method': 'choose', 'get_position_args': {'mode': 'point'}},
                    snr={'method': 'uniform', 'range': self.config['snr_range']},
                )
        else:
            # n se will be duplicated with the added se
            n_dupse = random.randint(1, min(nevents-1, self.max_n_dupse))
            for _ in range(nevents-n_dupse):
                s3.add_event(
                    label={'method': 'choose_wo_replacement'},
                    source_file={'method': 'choose'},
                    source_time={'method': 'choose'},
                    event_time={'method': 'choose'},
                    event_position={'method': 'choose', 'get_position_args': {'mode': 'point'}},
                    snr={'method': 'uniform', 'range': self.config['snr_range']},
                )
            selected_labels = [e['label'] for e in s3.fg_events]
            for _ in range(n_dupse):
                ref_label = random.choice(selected_labels)
                # maybe more than 1 se with same class already added to fg
                ref_positions = [e['event_position'][0] for e in s3.fg_events if e['label'] == ref_label]

                selected_position = self._get_position(ref_pos = ref_positions,
                                                all_pos = s3.room.get_all_positions())
                selected_position = [selected_position.tolist()]
                s3.add_event(
                    label={'method': 'const', 'value': ref_label},
                    source_file={'method': 'choose_wo_replacement', 'exclusion_folder_depth': self.dupse_exclusion_folder_depth},
                    source_time={'method': 'choose'},
                    event_time={'method': 'choose'},
                    event_position={'method': 'const', 'value': selected_position},
                    snr={'method': 'uniform', 'range': self.config['snr_range']},
                )
        assert self.nevent_range[0] <= len(s3.fg_events) <=self.nevent_range[1]
        assert len(s3.fg_events) == nevents
        # import pdb; pdb.set_trace()
        if spatial_sound_scene.get('interference_dir'):
            ninteferences = random.randint(self.config['ninterference_range'][0], self.config['ninterference_range'][1])
            for _ in range(ninteferences):
                s3.add_interference(
                    label={'method': 'choose'},
                    source_file={'method': 'choose'},
                    source_time={'method': 'choose'},
                    event_time={'method': 'choose'},
                    event_position={'method': 'choose', 'get_position_args': {'mode': 'point'}},
                    snr={'method': 'uniform', 'range': self.config['inteference_snr_range']},
                )
        # add background, make sure it is consistent with room
        if spatial_sound_scene.get('background_dir'): # only add noise if there is background_dir
            s3.add_background(source_file={'method': 'choose'},);
        return self._generate(s3)

    #=====================================================
    # Utilizations for metadata mode
    #=====================================================
    def _get_item_metadata(self, idx):
        SpAudSyn = _get_spatial_audio_synthesizer()
        metadata_path = os.path.join(self.metadata_dir, self.data[idx]['metadata_path'])
        s3 = SpAudSyn.from_metadata(metadata_path)
        return self._generate(s3)
