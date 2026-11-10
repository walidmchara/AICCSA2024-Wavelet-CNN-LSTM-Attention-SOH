# Experimental methodology and implementation boundaries

## Publication correspondence

Paper: *Improved Diagnosis of Lithium-Ion Battery Health in Electric Vehicles via a Hybrid Deep Learning Model Incorporating Wavelet Transform and Attention Mechanism* — AICCSA 2024, DOI [10.1109/AICCSA63423.2024.10912566](https://doi.org/10.1109/AICCSA63423.2024.10912566).

The publication describes **AM–CNN–WNN–WLSTM**, which differs from a standard CNN–LSTM or CNN–BiLSTM model: a wavelet neural layer precedes a recurrent block whose gates use Morlet transformations (Eqs. 15–19). No bidirectionality is assumed in this implementation because it is not specified as a recurrent-layer requirement in the architecture table.

- **CNN:** one 1-D convolution with 64 filters, kernel width 3, ReLU activation, pooling of 2 and dropout 0.30.
- **WNN:** 300 learnable wavelet units using the Morlet formula from Eq. (14).
- **WLSTM:** 100 units; the paper supplies Morlet-transformed input, forget, output and candidate equations but omits an explicit state-update equation. The implementation adopts the standard LSTM state update and documents the choice rather than silently imposing a conventional sigmoid LSTM.
- **Attention:** 100 intermediate units. Sigmoid scores are normalized over time to obtain aggregation weights; this normalization is an implementation choice consistent with weighted pooling, not a verbatim reproduction of all formula notation.
- **Objective:** mean-squared error for capacity-based SOH prediction.

## Dataset protocol

The paper groups NASA battery B0005/B0006/B0007 as development cells with B0018 held out at 24 °C. The second experiment uses B0029/B0031/B0032 for development and B0030 held out at 43 °C. No samples from the test cell are used when fitting preprocessing statistics, choosing checkpoints or tuning features.

The capacity-derived label is separated from model inputs. Training windows terminate at the cycle preceding the SOH prediction target. The user-supplied original archives were converted into cycle-level numerical summaries; this may differ from the feature-extraction and sampling procedure used to produce the original publication tables.

The experiment configuration explicitly marks assumptions not fixed in the conference manuscript: 16 historical cycles, an 80/20 development/validation split per battery, Adam learning rate 0.001, 32 batch size, 30 epochs, seed 42, and the 2.0 Ah nominal-capacity denominator. The target and preprocessing choices must be reconciled with the reported SOH scale before direct comparison.

## Baselines and reporting

Separate CNN, LSTM, and CNN–LSTM reference models are provided. Their outputs should be treated as independent computational results until the reference architectures and training protocols are matched to the paper's baselines. Error measures are computed on dimensionless SOH targets; the code outputs MAE, RMSE and MAPE in percent, unlike ambiguous percent notation sometimes found in the literature.

All execution and unit tests operate on genuine NASA-derived cycle-level CSV data, although reduced-width `--smoke` runs are software checks, not the benchmark results described in Table II. No unperformed scientific experiments or paper-equivalent results are asserted by this codebase.
