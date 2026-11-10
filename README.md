# CNN–WNN–WLSTM with Attention for Lithium-Ion Battery SOH Prediction

Implementation of the hybrid battery-health forecasting architecture presented in:

**Walid Mchara, Mohamed Abdellatif Khalfa, and Lazhar Manai.** *Improved Diagnosis of Lithium-Ion Battery Health in Electric Vehicles via a Hybrid Deep Learning Model Incorporating Wavelet Transform and Attention Mechanism.* 2024 IEEE/ACS 21st International Conference on Computer Systems and Applications (AICCSA). [DOI: 10.1109/AICCSA63423.2024.10912566](https://doi.org/10.1109/AICCSA63423.2024.10912566).

## Method

The forecasting network combines convolutional feature extraction, a trainable Morlet wavelet layer, a wavelet-activated recurrent block, and temporal attention. Its goal is to estimate battery State of Health (SOH) from prior discharge-cycle measurements.

The architecture follows the principal settings summarized in Table I of the article:

| Component | Architecture |
| --- | --- |
| CNN | 64 filters, kernel 3, ReLU, max pooling 2 |
| Dropout | 0.30 |
| Wavelet network | 300 wavelet units, Morlet activation |
| Wavelet LSTM | 100 recurrent units, Morlet-based gates |
| Attention | 100 units, sigmoid-based weighting |
| Objective | Mean-squared error |

The Morlet generator is defined in Eq. (14): `cos(1.75*x)*exp(-x²/2)`. The recurrent block follows the paper's Morlet gate transformations. Because the paper does not specify the cell-state update, the implementation uses the conventional LSTM state equation. Sigmoid attention scores are normalized across the historical window to obtain a weighted temporal representation. These implementation choices are described in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## NASA battery-aging experiments

Eight NASA PCoE cell-level CSV series are included, prepared from the source MATLAB discharge traces. The held-out evaluation protocol follows the study:

| Group | Development batteries | Independent test battery |
| --- | --- | --- |
| Ambient temperature 24 °C | B0005, B0006, B0007 | **B0018** |
| Ambient temperature 43 °C | B0029, B0031, B0032 | **B0030** |

The observations are ordered by discharge cycle. Inputs consist of historical voltage, current, temperature, ambient-temperature and discharge-duration indicators; the target is a capacity-based SOH ratio. Data preparation keeps batteries separate, fits scaling parameters on development cells only, and excludes the target cycle from each input window.

The processed files are under [`data/processed/nasa/`](data/processed/nasa/), with source checksums and record counts recorded in [`data/processed/manifest.json`](data/processed/manifest.json). Full-resolution NASA MATLAB traces are maintained separately.

## Installation and execution

```bash
pip install -r requirements.txt
pytest -q

# Independent B0018 experiment (24 °C)
python train.py --group 24C --epochs 30 --out outputs/b0018

# Independent B0030 experiment (43 °C)
python train.py --group 43C --epochs 30 --out outputs/b0030
```

A reduced-width software check uses the same measured inputs and holdout batteries, but shorter training and smaller network layers:

```bash
python train.py --group 24C --smoke --out outputs/check_24
python train.py --group 43C --smoke --out outputs/check_43
```

For comparisons with the published baseline families:

```bash
python train.py --group 24C --model CNN --epochs 30 --out outputs/cnn_b0018
python train.py --group 24C --model LSTM --epochs 30 --out outputs/lstm_b0018
python train.py --group 24C --model CNN-LSTM --epochs 30 --out outputs/cnn_lstm_b0018
```

These reference baselines are provided for controlled comparisons; their exact hyperparameters require alignment with the publication before numerical equivalence is claimed. Each run saves epoch-level losses, held-out predictions, evaluation metrics and the selected checkpoint.

## Reproducibility

The article reports low held-out errors on B0018 and B0030. This repository does not substitute software-check metrics for the published values. The current settings use a 16-cycle historical window, an 80/20 chronological development/validation split, Adam optimization and a 2.0 Ah nominal capacity; those choices are explicitly separated from the parameters fixed by the paper in [`configs/paper.json`](configs/paper.json).

For data provenance, method-specific limitations, and the published-vs-implemented distinctions, see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) and [`data/README.md`](data/README.md).
