# Real free5GC + UERANSIM validation results

Final reproducible GitHub Actions run:

- Run: https://github.com/verge88/article_5G_V3/actions/runs/39121474105
- Artifact: https://github.com/verge88/article_5G_V3/actions/runs/39121474105/artifacts/12134227722
- Artifact SHA-256: `09983f93fb44fce472ac663ff1beb3c5011e749b39a9dd79624a791cf7de3208`
- free5GC Compose commit: `235ef1b91ceacebb07f4118384f2c4190f431098`
- free5GC image tag: `v4.3.0`
- UERANSIM observed in runtime: `v3.3.0`

## Evidence that this is a real control-plane run

The workflow does not merely start containers. It requires successful NGAP/NAS operation and independently preserves evidence.

Observed in the final run:

- UERANSIM establishes SCTP to AMF `10.100.200.16:38412`;
- NG Setup completes successfully;
- UE logs contain `Initial Registration is successful`;
- AMF logs contain `Handle Registration Complete` for the provisioned SUPI;
- SBI logs contain real service calls including:
  - AMF -> AUSF: `POST /nausf-auth/v1/ue-authentications`;
  - AUSF 5G-AKA confirmation;
  - NRF discovery for AUSF and UDM;
- packet-capture validation counted:
  - **46,838 TCP/8000 packets**;
  - **1,448 SCTP packets**.

The CI run fails if either SBI TCP/8000 or SCTP is absent from the packet capture.

## Dataset

The final run contains:

| Scenario | Events | Registration outcome |
|---|---:|---:|
| Normal | 19 | 100% successful |
| Low-amplitude UDM delay | 7 | 100% successful |
| Low-amplitude AUSF delay | 7 | 100% successful |
| AMF delay | 7 | 100% successful |
| UDM outage | 7 | 0% successful, as expected |
| Registration storm (5 concurrent UEs) | 7 | 100% successful |

Median control-plane registration time measured from the experiment harness:

- Normal: **275.6 ms**
- AUSF low-amplitude delay: **833.1 ms**
- UDM low-amplitude delay: **1969.4 ms**
- AMF delay: **2746.2 ms**
- Registration storm: **608.4 ms**

The outage scenario reaches the configured 7 s timeout.

## ML results

| Method | ROC-AUC | PR-AUC | F1 | FPR | Top-1 NF localization | Stealth top-1 localization | Stealth detection recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| ActiveProbe_IF | **1.0000** | **1.0000** | **0.9736 ± 0.0067** | **0.0000** | **0.6457 ± 0.0256** | **0.9429 ± 0.0319** | **1.0000** |
| ActiveProbe_OCSVM | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.6000 | 1.0000 | 1.0000 |
| PassiveZone_IF | 0.9814 | 0.9980 | 0.6184 ± 0.0790 | 0.0000 | 0.2914 ± 0.0128 | 0.0714 | 0.2714 ± 0.0782 |
| Passive_IF | 0.8757 | 0.9859 | 0.8497 ± 0.0061 | 0.0500 | 0.2914 ± 0.0128 | 0.1429 | 0.8571 |

For the principal Isolation-Forest ablation, adding active control/probe information to the passive zonal feature set changes:

- F1: **0.6184 -> 0.9736**;
- stealth detection recall: **0.2714 -> 1.0000**;
- stealth top-1 NF localization: **0.0714 -> 0.9429**.

This is the strongest result for the article.

## Scenario-level localization of ActiveProbe_IF

- AMF delay: **1.000**
- low-amplitude AUSF delay: **0.886**
- low-amplitude UDM delay: **1.000**
- registration storm: **0.343**
- UDM outage: **0.000**

The model is therefore particularly strong for the article's target case—low-amplitude internal NF impairment—but node attribution for a hard outage and distributed storm remains a limitation. This should be reported rather than hidden.

## Interpretation boundary

The experiment uses real free5GC network functions and real UERANSIM gNB/UE software. The low-amplitude "stealth" cases are controlled `tc netem` impairments injected into the network namespace of a running NF. They emulate a compromise-like condition in which local resource/load telemetry need not be conspicuous.

They are **not an exploit of free5GC and not evidence of a real compromised NF**.

The workflow intentionally requests no PDU session. Therefore the experiment validates the **5G Core control plane** (SCTP/NGAP, NAS/5G-AKA and SBA/SBI), not the UPF/GTP-U user plane.
