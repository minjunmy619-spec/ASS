# Spatial Semantic Segmentation of Sound Scenes

This is a baseline implementation for the [DCASE2026 Challenge Task 4: Spatial Semantic Segmentation of Sound Scenes](https://dcase.community/challenge2026/task-spatial-semantic-segmentation-of-sound-scenes).

This implementation is based on the [
*dcase2025_task4_baseline*](https://github.com/nttcslab/dcase2025_task4_baseline) from the previous [DCASE2025 Task 4](https://dcase.community/challenge2025/task-spatial-semantic-segmentation-of-sound-scenes). While the folder structure and training scripts are similar to *dcase2025_task4_baseline*, this implementation introduces several new elements, including modifications to the models, new training loss functions, and an updated evaluation metric. In addition, data synthesis is carried out using a new module, [SpAudSyn](https://github.com/nttcslab/SpAudSyn) (Spatial Audio Synthesizer).

[DCASE2026 Challenge](https://dcase.community/challenge2026/index) provides an overview of all the challenge tasks.

## Description
### Baseline System
The baseline system consists of two models, audio tagging (AT) and source separation (SS), which are trained separately.
The AT model uses a pre-trained feature extractor backbone (M2D) and a head layer.
For the AT model, we provide two variants: one that uses a single channel of the mixture as input and another that uses all four channels.

### Experimental Task 4 System Variants
This checkout also contains opt-in research variants beyond the official-style
baseline. The original `M2dAt` + `ResUNet30` configs are still available; the
newer variants are added as sibling configs so existing baseline pipelines do
not change silently.

| Area | Main files | What was added |
| --- | --- | --- |
| Stage map and commands | `recipes.md`, `docs/task4_variant_matrix.md` | Recommended `USS -> SC -> TSE -> S5` promotion path, stage isolation, and checkpoint naming. |
| Strong source classification | `src/models/m2dat/m2d_sc.py`, `config/label/m2d_sc_stage*_strong.yaml` | Attentive-statistics M2D source classifier, multi-crop prediction, ArcFace training, and energy-based silence filtering. |
| Pretrained SC fusion | `config/label/m2d_sc_stage*_beats_fusion.yaml`, `config/label/m2d_sc_stage*_fpasst_fusion.yaml`, `config/label/m2d_sc_stage1_pretrainedsed_fusion.yaml` | Single-branch BEATs/fPaSST fusion plus a multi-branch official PretrainedSED BEATs/ATST-F/fPaSST variant. |
| DeFT separation stack | `config/separation/modified_deft_*.yaml`, `src/models/deft/` | Spatial USS, oracle and estimated-enrollment TSE, memory-efficient 6s recipes, temporal variants, count heads, FOA spatial features, residual slots, and bridge variants. |
| Estimated-source adaptation | `src/evaluation/export_sc_finetune_cache.py`, `config/label/m2d_sc_stage3_estimated_*.yaml`, `config/separation/*estimated_enrollment*.yaml` | Cached USS/TSE estimates for SC adaptation and TSE fine-tuning on the same enrollment distribution used by S5 inference. |
| USS-conditioned TSE | `config/separation/*uss_conditioned.yaml`, `src/models/s5/kwo2025*.py` | Opt-in query/proposal FiLM in TSE and S5 handoff from USS outputs such as `tse_condition`, class logits, count logits, spatial embeddings, DoA vectors, and activity summaries. |
| Robust and duplicate-aware paths | `src/training/loss/m2d_sc_arcface.py`, `src/models/s5/kwo2025.py`, `src/evaluation/eval_configs/*duplicate_recall.yaml` | Soft truncation for noisy estimated-source labels and S5 duplicate recall for same-class source recovery. |
| Diagnostics and calibration | `src/evaluation/calibrate_sc_energy_thresholds.py`, `src/evaluation/evaluate_stage.py`, `src/evaluation/debug_sc_loss_flow.py` | Per-class SC energy-threshold calibration, USS/SC/TSE stage-only evaluation, and SC loss-flow debugging. |
| AudioSet-Strong source pools | `docs/audioset_strong_augmentation.md`, `src/datamodules/dataset.py` | Opt-in `spatial_sound_scene_sources` mixing for controlled official-vs-AudioSet-Strong source-pool curricula. |

For a compact overview, start from `recipes.md`. For a complete stage-aware
variant table, use `docs/task4_variant_matrix.md`.

### Dataset and folder structure
The data consists of two parts: the development dataset and the evaluation dataset.
The development dataset (provided as [DCASE2026 Task4 Dataset](https://zenodo.org/records/19328046)) is constructed based on the previous [DCASE2025 Task4 Dataset](https://zenodo.org/records/15117227) by incorporating newly recorded target events, RIRs, and background noise, using a similar recording setup.
The evaluation dataset will be released at a later stage.

The structure of the data is as follows (`data/dev_set` folder contains the development dataset):
```
data
`-- dev_set
    |-- config
    |   |-- EARS_config.json
    |   `-- FSD50K_config.json
    |-- metadata
    |   |-- valid
    |   `-- valid.json
    |-- noise
    |   |-- train
    |   `-- valid
    |-- room_ir
    |   |-- train
    |   `-- valid
    |-- sound_event
    |   |-- train
    |   `-- valid
    |-- interference
    |   |-- train
    |   `-- valid
    `-- synthesized
        `-- test
            |-- oracle_target
            `-- soundscape
```
The `config`, `metadata`, `noise`, `room_ir`, `interference`, and `sound_event` folders are used for generating the training data, including the training and validation splits.
The `synthesized/test` folder contains test data for evaluating model checkpoints, consisting of pre-mixed soundscapes in `soundscape` and oracle target sources in `oracle_target`.

The DCASE2026Task4Dataset: A Dataset for Spatial Semantic Segmentation of Sound Scenes is available on [Zenodo](https://zenodo.org/records/19328046).

### Related Repositories
Part of `src/models/resunet` originates from  https://github.com/bytedance/uss/tree/master/uss/models \
Part of `src/models/m2dat` originates from  https://github.com/nttcslab/m2d


## Data Preparation and Environment Configuration
### Setting
Clone this repository
```
git clone https://github.com/nttcslab/dcase2026_task4_baseline.git
```

Download SpAudSyn module
```
# Clone repository
git clone https://github.com/nttcslab/SpAudSyn.git

# Place the module in dcase2026_task4_baseline
cd dcase2026_task4_baseline
cp -r path/to/SpAudSyn/src src/modules/spatial_audio_synthesizer
```

Download checkpoint of M2D
```
cd dcase2026_task4_baseline
wget -P checkpoint https://github.com/nttcslab/m2d/releases/download/v0.3.0/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly.zip
unzip checkpoint/m2d_as_vit_base-80x1001p16x16p32k-240413_AS-FT_enconly.zip -d checkpoint
```

Optional pretrained source-classifier fusion branches need extra local assets:

```bash
# BEATs single-branch fusion
git clone https://github.com/microsoft/unilm.git external/unilm
ln -s unilm/beats external/BEATs

# PretrainedSED fPaSST and multi-branch BEATs/ATST-F/fPaSST fusion
git clone https://github.com/fschmid56/PretrainedSED.git external/PretrainedSED
mkdir -p checkpoint/pretrainedsed
```

`M2DPretrainedSEDFusionClassifier` can download the official PretrainedSED
v0.0.1 release checkpoints into `checkpoint/pretrainedsed` when
`pretrainedsed_download_if_missing: true` is set. To prepare them explicitly,
download `BEATs_strong_1.pt`, `ATST-F_strong_1.pt`, and `fpasst_strong_1.pt`
from the PretrainedSED v0.0.1 release into that folder.

Install environment (identical to *dcase2025_task4_baseline*)
```
cd dcase2026_task4_baseline

# Using conda
conda env create -f environment.yml
conda activate dcase2026t4

# Or using pip (python=3.11)
python -m venv dcase2026t4
source dcase2026t4/bin/activate
pip install -r requirements.txt
```

SoX may be required for the above environment installation
```
sudo apt-get update && sudo apt-get install -y gcc g++ sox libsox-dev
```

### Data Preparation
Download the development dataset and place it in the `data` folder.
```
# Download all files from https://zenodo.org/records/19328046 and unzip
wget -i dev_set_zenodo.txt
zip -s 0 DCASE2026Task4Dataset.zip --out DCASE2026Task4DatasetFull.zip
unzip DCASE2026Task4DatasetFull.zip

# Place the dev_set in dcase2026_task4_baseline/data folder
ln -s "$(pwd)/DCASE2026Task4Dataset/data/dev_set" /path/to/dcase2026_task4_baseline/data
```
Add sound events from other datasets
```
# Download the Semantic Hearing dataset
# https://github.com/vb000/SemanticHearing
wget -P data https://semantichearing.cs.washington.edu/BinauralCuratedDataset.tar

# Download EARS dataset using bash
# https://github.com/facebookresearch/ears_dataset
mkdir EARS
cd EARS
for X in $(seq -w 001 107); do
  curl -L https://github.com/facebookresearch/ears_dataset/releases/download/dataset/p${X}.zip -o p${X}.zip
  unzip p${X}.zip
  rm p${X}.zip
done

# Add data
cd dcase2026_task4_baseline
bash add_data.sh --semhear_path /path/to/BinauralCuratedDataset --ears_path /path/to/EARS
```


Verify data folder structure
```
cd dcase2026_task4_baseline
python verify.py --source_dir .
```

## Training

All the TensorBoard log and model checkpoint will be saved to `workspace`.
### Audio Tagging Model
There are two versions of audio tagging models, using one channel and using four channels of the mixture.
Each model is fine-tuned in two steps.

Train only head
```
# 1 channel
python -m src.train -c config/label/m2dat_1c.yaml -w workspace/label
# 4 channel
python -m src.train -c config/label/m2dat_4c.yaml -w workspace/label
```
Continue fine-tuning the last blocks of the M2D backbone, replace the `BEST_EPOCH_NUMBER` with the appropriate epoch number
```
# 1 channel
python -m src.train -c config/label/m2dat_1c_2blks.yaml -w workspace/label -r workspace/label/m2dat_1c/checkpoints/epoch=BEST_EPOCH_NUMBER.ckpt
# 4 channel
python -m src.train -c config/label/m2dat_4c_2blks.yaml -w workspace/label -r workspace/label/m2dat_4c/checkpoints/epoch=BEST_EPOCH_NUMBER.ckpt
```

### Separation Model
The separation model can be trained using
```
python -m src.train -c config/separation/resunetk_capisdr.yaml -w workspace/separation
```

### Extended S5 Recipes
The current experimental system is documented in `recipes.md`. A typical
memory-efficient path is:

```bash
python -m src.train -c config/separation/modified_deft_uss_lite_6s.yaml -w workspace/separation
python -m src.train -c config/label/m2d_sc_stage1_strong.yaml -w workspace/label
python -m src.train -c config/label/m2d_sc_stage2_strong.yaml -w workspace/label
python -m src.train -c config/separation/modified_deft_tse_lite_6s.yaml -w workspace/separation
```

Then export estimated slots and adapt downstream stages:

```bash
python -m src.evaluation.export_sc_finetune_cache \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_strong_sc.yaml \
  --output_root workspace/sc_finetune \
  --mode pseudo_s5 \
  --batchsize 1

python -m src.train -c config/label/m2d_sc_stage3_estimated_strong.yaml -w workspace/label
python -m src.train -c config/separation/modified_deft_tse_lite_6s_estimated_enrollment.yaml -w workspace/separation
python -m src.train -c config/separation/modified_deft_tse_lite_10s_estimated_enrollment.yaml -w workspace/separation
```

For temporal systems where USS proposal cues should guide TSE, use the opt-in
conditioned siblings after exporting bridge features or cached estimated
enrollments:

```bash
python -m src.train -c config/separation/modified_deft_tse_lite_6s_temporal_estimated_enrollment_uss_conditioned.yaml -w workspace/separation
python -m src.train -c config/separation/modified_deft_tse_lite_10s_temporal_estimated_enrollment_uss_conditioned.yaml -w workspace/separation
```

When the cache is exported from a conditioned S5 config, the exporter stores the
matching proposal tensors in `workspace/sc_finetune/uss_bridge_features/` so the
conditioned TSE fine-tune sees the same USS context that S5 will pass at
inference.

For stronger SC branches, use sibling configs such as
`m2d_sc_stage*_beats_fusion.yaml`, `m2d_sc_stage*_fpasst_fusion.yaml`, or
`m2d_sc_stage1_pretrainedsed_fusion.yaml`. Robust estimated-source configs end
in `_robust.yaml` and should be used only after a matching non-robust branch has
been measured.

### Training hyperparameters

Some hyperparameters that affect training time and performance can be set in the YAML configuration files
- `dataset_length` in `train_dataloader`: Since each training mixture is generated randomly and independently, `dataset_length` can be set arbitrarily. A higher value increases the number of training steps per epoch and may slightly speed up training by reducing the frequency of validation.
- `dupse_exclusion_folder_depth`: Specifies the folder level (counted from the label folder; e.g., `data/dev_set/sound_event/train/Speech` is level 0) at which sound sources are considered related, and those within the same folder at this depth must not be mixed together in the same mixture. This helps avoid unrealistic cases, such as the same person speaking at multiple positions simultaneously in a mixture. When set to `0`, no folder-based exclusion is applied and sources can be mixed freely, with the only constraint that each source file is used at most once.
- `batch_size` in `train_dataloader`: When using N GPUs, the effective batch size becomes `N * batch_size`. We found that a larger batch size positively impacts audio tagging model training, but it also increases the training time.
- `num_workers` in `train_dataloader`: Each training mixture loads and mixes 3 to 6 audio samples, which can be time-consuming. `num_workers` should be set based on the number of GPUs and CPU cores to optimize the dataloading process.
- `lr` in `optimizer`: The learning rate should be adjusted based on the effective batch size, which changes with the number of GPUs or the `batch_size` in `train_dataloader`.

Each baseline checkpoint was trained using 4 RTX 3090 GPUs in under 3 days.
However, we found that with appropriate hyperparameter settings, similar results can be achieved using fewer GPUs in a similar amount of time.

## Evaluating Baseline Checkpoints
There are three checkpoints for the two baseline systems, corresponding to the SS model and the two variants of the AT model described above.
These can be downloaded from the release [version e.g., v1.0.0] and placed in the `checkpoint` folder as
```
cd dcase2026_task4_baseline
wget -P checkpoint https://github.com/nttcslab/dcase2026_task4_baseline/releases/download/v1.0.0/baseline_checkpoint.zip
unzip checkpoint/baseline_checkpoint.zip -d checkpoint
```

Class-aware permutation-invariant signal-to-distortion ratio (CAPI-SDRi) and label prediction accuracy can be calculated on the `data/dev_set/test` data using the baseline checkpoints as
```
# M2DAT_1c + ResUNetK
python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2dat_1c_resunetk.yaml --result_dir workspace/evaluation
"""
CAPI-SDRi: 8.171
Accuracy (mixture): 57.143
Accuracy (source) : 67.147
"""

# M2DAT_4c + ResUNetK
python -m src.evaluation.evaluate -c src/evaluation/eval_configs/m2dat_4c_resunetk.yaml --result_dir workspace/evaluation
"""
CAPI-SDRi: 8.489
Accuracy (mixture): 60.714
Accuracy (source) : 70.394
"""
```
To evaluate other model checkpoints, specify their paths under `tagger_ckpt` and `separator_ckpt` in the corresponding config files located in `src/evaluation/eval_configs`.

The newer S5 evaluation configs live in the same folder. Common entry points:

```bash
# Main lite USS/TSE system with estimated-source-adapted SC
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc.yaml \
  --result_dir workspace/evaluation

# Duplicate-recall ablation for same-class source recovery
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc_duplicate_recall.yaml \
  --result_dir workspace/evaluation

# Temporal S5 with USS-conditioned TSE handoff
python -m src.evaluation.evaluate \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_temporal_sc_uss_conditioned_tse.yaml \
  --result_dir workspace/evaluation

# Stage-only diagnostics against the same eval config
python -m src.evaluation.evaluate_stage \
  -c src/evaluation/eval_configs/kwo2025_top1_like_lite_estimated_sc.yaml \
  --stage sc \
  --result_dir workspace/evaluation_stage
```

Before final S5 comparison, calibrate SC energy thresholds on the validation
slots for the exact SC checkpoint/config being evaluated:

```bash
python -m src.evaluation.calibrate_sc_energy_thresholds \
  -c config/label/m2d_sc_stage3_estimated_strong.yaml \
  --checkpoint checkpoint/m2d_sc_stage3_estimated_strong.ckpt \
  --output-dir workspace/calibration/sc_energy
```

## License 
This project is licensed under the terms described in [LICENSE.pdf](LICENSE.pdf).


# Citation
If you use this system, please cite the following papers:

```
@inproceedings{nguyen2026capisdr,
  title={Class-Aware Permutation-Invariant Signal-to-Distortion Ratio for Semantic Segmentation of Sound Scene with Same-Class Sources},
  author={Nguyen, Binh Thien and Yasuda, Masahiro and Takeuchi, Daiki and Niizumi, Daisuke and Harada, Noboru},
  booktitle={2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026},
  url={https://arxiv.org/pdf/2601.22504}
}

@article{yasuda2026dcase,
  title={Description and Discussion on DCASE 2026 Challenge task 4: Spatial Semantic Segmentation of Sound Scenes},
  author={Yasuda, Masahiro and Nguyen, Binh Thien and Harada, Noboru and Serizel, Romain and Mishra, Mayank and Delcroix, Marc and Carlos, Hernandez-Olivan and Araki, Shoko and Takeuchi, Daiki and Nakatani, Tomohiro and Ono, Nobutaka},
  journal={arXiv preprint arXiv:2604.00776},
  year={2026},
  url={https://arxiv.org/pdf/2604.00776}
}
```
