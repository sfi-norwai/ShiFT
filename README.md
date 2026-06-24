<h1 align="center">ShiFT</h1>
<h2 align="center">
Learning by Shifting:<br>
Temporal View Construction for Time Series Contrastive Learning
</h2>

<p align="center">
  <a href="https://arxiv.org/abs/2606.21957">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2606.21957-b31b1b.svg">
  </a>
  <img alt="License" src="https://img.shields.io/github/license/sfi-norwai/ShiFT">
  <img alt="Last Commit" src="https://img.shields.io/github/last-commit/sfi-norwai/ShiFT">
  <img alt="Stars" src="https://img.shields.io/github/stars/sfi-norwai/ShiFT?style=social">
</p>

<p align="center">
  <a href="https://ecmlpkdd.org/2026/">
    <img alt="ECML PKDD 2026 Accepted" src="https://img.shields.io/badge/Accepted%20at-ECML-PKDD%202026-blueviolet">
  </a>
</p>


This repository contains the official Pytorch implementation of the "[**Learning by Shifting (ShiFT)**]" (ECML PKDD 2026), a simple, deterministic view construction to learn strong representations for time-series classification.

![ShiFT](./visuals/Speed_Ranks.png?raw=true "Title")

## Data


To use **ShiFT** and the baseline models, you will need access to relevant time-series datasets. The following datasets are used in this repository:

For the Large benchmark datasets, we use the preprocessed dataset from the [**Series2Vec**](https://github.com/Navidfoumani/Series2Vec) repository for the PAMAP2, WISDM2, SLEEP and SKODA datasets. 

- [**HARTH**](https://archive.ics.uci.edu/dataset/779/harth): This is a human activity recognition (HAR) dataset that contains recordings from 22 participants, each wearing two 3-axial Axivity AX3 accelerometers for approximately 2 hours in a free-living setting at a sampling rate of 50Hz.

- [**ECG**](https://physionet.org/content/afdb/1.0.0/): We use the MIT-BIH Atrial Fibrillation dataset, which includes 25 long-term electrocardiogram (ECG) recordings of human subjects with atrial fibrillation, each with a duration of 10 hours.


For the UCR and UEA benchmarks, you can download them from the [**official website**](https://www.timeseriesclassification.com/)

Make sure to place the dataset in the appropriate directory (e.g., `datasets/harth`) as specified in the configuration files.


## Usage

### Unsupervised pretraining of ShiFT

To pretrain the ShiFT model for classification, use the following command:

```bash
python pretrain.py <method> <dataset> -p <configs/<dataset>config.yml> -s < > --evaluate < >

```
- `method` specifies the self-supervised method to train.
- `dataset` specifies the dataset directory.
- `-p` specifies the configuration file.
- `-s` sets the seed for reproducibility.
- `--evaluate` [optional] define the downstream task to perform after pretraining.

### Example
For example, to pretrain ShiFT on the harth dataset with a seed of 1 and evaluate on supervised classification, run:
```bash
python pretrain.py ShiFT harth -p configs/harthconfig.yml -s 1 --evaluate supervised
```
Check the scripts/ directory for complete list of training scripts for all tasks in the paper as well as the different seeds used for reproducibility.

### Running Baseline Methods
To compare ShiFT against competitive method, you can use similar commands to pretrain the baselines. For example, to pretrain the SimMTM baseline on Skoda:

```bash
python pretrain.py SimMTM Skoda -p configs/Skodaconfig.yml -s 1 --evaluate supervised
```

## Views Visualizations

View construction strategies for contrastive learning in
time serie

![Views Visualization](./visuals/ShiFT_view.png?raw=true "Title")


## Acknowledgements

This repository provides reimplementations baselines for time-series representation learning using some parts of the codes provided by the following  works:

- [**TS2Vec**](https://github.com/zhihanyue/ts2vec): Towards Universal Representation of Time Series.

- [**SimMTM**](https://github.com/thuml/SimMTM): A Simple Pre-Training Framework for Masked Time-Series Modeling.

- [**InfoTS**](https://github.com/chengw07/InfoTS): Time Series Contrastive Learning with Information-Aware Augmentations.

- [**SSL_Comparison**](https://github.com/DL4mHealth/SSL_Comparison): Self-Supervised Learning for Time Series: Contrastive or Generative?.


Please check out the original repositories for more details.

## Citations

If you use **ShiFT** in your research, please consider citing it as follows:

```bibtex
@inproceedings{shift2026,
  title     = {Learning by Shifting: Temporal View Construction for Time Series Contrastive Learning},
  author    = {Abdul-Kazeem Shamba and Kerstin Bach and Gavin Taylor},
  booktitle = {European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases},
  year      = {2026}
}