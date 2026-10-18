# 5G SBA Active-Probe Graph ML Experiment

This repository contains a fully reproducible experiment for a paper on anomaly detection and compromised network-function localization in a 5G Service-Based Architecture (SBA).

## Scientific hypothesis

A detector that combines:
1. local NF telemetry,
2. temporal context,
3. graph/neighborhood context, and
4. active-probe trust signals

should detect stealth compromise more reliably than passive local-only or temporal-only anomaly detectors.

## 5G SBA digital twin

Network functions: AMF, SMF, UDM, AUSF, PCF, NRF, NSSF, CHF.

Normal telemetry includes request rate, latency, errors, CPU, memory, stream concurrency and routing entropy. Active probes add probe latency, loss, staleness and report inconsistency. Zone features measure robust deviation from neighboring NFs.

Attack scenarios:
- signaling storm against AMF;
- HTTP/2-style stream exhaustion against NRF;
- NF degradation against PCF;
- stealth compromise of SMF;
- stealth compromise of AMF.

The stealth scenarios intentionally keep local reported counters close to the normal regime while changing active-probe consistency and neighboring NF behavior.

## Methods

Baselines:
- Isolation Forest;
- One-Class SVM;
- LSTM predictor;
- TemporalGNN.

Proposed:
- spatio-temporal graph predictor with an active-probe-conditioned trust gate.

Ablations:
- no zone context;
- no graph context;
- no temporal context.

All models are trained only on normal data. The anomaly threshold is the 99th percentile of validation-normal node scores.

Primary metrics: ROC-AUC, PR-AUC, F1, FPR, stealth-compromise recall, top-1 compromised-NF localization, stealth top-1 localization, and detection delay.

## Reproducibility

The GitHub Actions workflow runs three independent seeds and uploads:
- `per_seed_metrics.csv`;
- `metrics_summary.csv`;
- `scenario_recall.csv`;
- `manuscript_table.md`;
- figures for the article;
- `experiment_manifest.json`;
- `summary.md`.

Important limitation: the current CI experiment uses a controlled 5G-SBA digital-twin simulator. It does **not** claim to use real operator traffic or a production 5G core. This limitation is intentionally preserved in all generated summaries.

## Final GitHub Actions result

The five-seed experiment completed successfully in GitHub Actions. See [RESULTS.md](RESULTS.md) for the scientific interpretation and [results/manuscript_table.md](results/manuscript_table.md) for the aggregate table.

The strongest validated effect is the active-probe channel: stealth top-1 localization is 0.9489 for the proposed detector versus 0.0722 without probes. Zone context reduces FPR, while the current experiment does not independently justify claiming gains from the trust gate, graph block, or temporal block.
