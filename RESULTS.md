# Reproducible experiment results

GitHub Actions run: https://github.com/verge88/article_5G_V3/actions/runs/39119543125

Artifact: https://github.com/verge88/article_5G_V3/actions/runs/39119543125/artifacts/12134140197

The final experiment used five independent seeds: 11, 29, 47, 71, and 101.

## Headline result

The proposed active-probe-conditioned model achieved:

- ROC-AUC: **0.9969 ± 0.0025**
- PR-AUC: **0.9900 ± 0.0082**
- F1: **0.9440 ± 0.0308**
- FPR: **0.0649 ± 0.0381**
- stealth-compromise recall: **1.0000 ± 0.0000**
- top-1 compromised-NF localization: **0.9748 ± 0.0123**
- top-1 stealth-compromise localization: **0.9489 ± 0.0282**
- median detection delay: **0 time steps**

## Main ablation result: active probes

The direct no-probe ablation used the same graph-temporal family without the active-probe channel.

| Metric | Proposed | No probe | Difference |
|---|---:|---:|---:|
| ROC-AUC | 0.9969 | 0.9897 | +0.0072 |
| PR-AUC | 0.9900 | 0.9748 | +0.0152 |
| F1 | 0.9440 | 0.9331 | +0.0109 |
| Stealth recall | 1.0000 | 0.9378 | +0.0622 |
| Top-1 NF localization | 0.9748 | 0.5981 | +0.3767 |
| **Top-1 stealth localization** | **0.9489** | **0.0722** | **+0.8767** |

The strongest empirical contribution is therefore the active-control/probe information: it changes the task from detecting a network-level disturbance to correctly attributing stealth compromise to the responsible NF.

## Zone-context ablation

Removing zone features reduced mean F1 from **0.9440** to **0.9231** and increased FPR from **0.0649** to **0.0900**. This supports the use of robust neighbor/zone context primarily as a false-positive-control mechanism.

## What the current experiment does NOT support

The current digital-twin dataset does not provide evidence that the full architectural complexity is necessary:

- `Ablation_concat_probe`: F1 0.9432, stealth localization 0.9467.
- `Ablation_no_graph`: F1 0.9442, stealth localization 0.9489.
- `Ablation_no_temporal`: F1 0.9449, stealth localization 0.9644.
- `Proposed`: F1 0.9440, stealth localization 0.9489.

Therefore the paper should **not** claim, on the basis of this experiment, that the trust gate, graph block, or temporal block independently improves detection. Their performance is statistically and practically very close in this controlled setup.

A defensible novelty claim is instead:

> Active control transactions/probe-derived features, fused with local NF telemetry and zone context, enable reliable localization of stealth compromise in a 5G SBA digital twin, whereas passive graph-temporal telemetry without probes frequently detects the disturbance but fails to identify the compromised NF.

## Limitations

The generated data are controlled synthetic/digital-twin 5G SBA telemetry. They are **not** operator traces and are **not** produced by a production 5G core. The experiment is appropriate for validating the mechanism and ablations; exploitation-level or operator-level claims require a free5GC/UERANSIM or real operator validation stage.
