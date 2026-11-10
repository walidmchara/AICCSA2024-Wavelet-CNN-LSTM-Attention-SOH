# NASA PCoE battery-aging records

The series in [`processed/nasa/`](processed/nasa/) were derived from the NASA lithium-ion battery-aging archive. The eight cells correspond to the two held-out-battery experiments in the AICCSA 2024 study.

The CSVs include discharge-cycle capacity and numerical summaries of voltage, current, temperature, and discharge duration. The target field `capacity_ah` is used **only for constructing the SOH reference**, not as a predictor at the same cycle. The evaluation input at cycle *t* uses the prior `lookback` cycles.

The data preparation module estimates missing-value medians and min–max limits using only the development-battery training segments, then applies them to subsequent validation periods and independent test cells. Windows are never constructed across battery boundaries.

The default SOH definition is `capacity_ah / 2.0`, using a 2.0 Ah nominal-capacity assumption for these NASA cells. This denominator can be changed in [`configs/paper.json`](../configs/paper.json), or `target_mode` can be switched to `first_observed` if the comparison protocol requires that convention. These target definitions must not be conflated.

The provenance record at [`processed/manifest.json`](processed/manifest.json) includes the individual battery IDs, row counts and source-derived file hashes. Original full-resolution MATLAB files are not included in this repository. The processed records are cycle-level aggregates, not complete intra-cycle time series.

Source: [NASA PCoE Battery Data Set Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).
