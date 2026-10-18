# Real free5GC + UERANSIM experiment v2

This redesign addresses the methodological limitations found in the first real-testbed experiment.

## Core change: no tautological root-latency experiment

The proposed active signal is no longer a GET / request to the same NF that is impaired.

A dedicated canary UE performs a real Initial Registration over:
- UERANSIM RRC/NAS;
- SCTP/NGAP to AMF;
- AMF/AUSF authentication;
- AUSF/UDM authentication-data exchange;
- UDM/UDR subscriber-data exchange.

For the four primary single-target scenarios, tc/netem matches only a legitimate response path:
- UDR -> UDM;
- UDM -> AUSF;
- AUSF -> AMF;
- AMF -> gNB/UE.

The Prometheus blackbox exporter reaches the NF from a different source, therefore the selective filter does not match its root probe.

## Explicit simple baselines

The experiment compares:
- Prometheus blackbox/SLO alert;
- robust z-score of blackbox root latency;
- robust z-score of canary UE registration time;
- robust z-score of the per-NF active business-stage latency, with argmax localization;
- passive per-NF Isolation Forest;
- passive + topology-context Isolation Forest;
- active ML Isolation Forest.

This makes it possible to determine whether machine learning adds value beyond a simple active threshold.

## Statistical design

The dataset is collected in one GitHub Actions job and one runner.

There are eight temporal collection blocks. Each contains:
- six normal events at background registration loads 0..5;
- four selective single-target service-path degradations;
- one registration storm;
- one hard UDM outage.

Normal and anomalous scenarios are reproducibly shuffled within every block.

Total target dataset: 96 events, 48 normal and 48 anomalous.

Evaluation:
- leave-one-block-out cross-validation;
- 42 normal events in each outer training set;
- Isolation Forest thresholds calibrated from cross-fitted normal scores in the training blocks;
- 95% confidence intervals from block bootstrap;
- paired block-bootstrap confidence intervals for method differences.

Random Top-1 localization is explicitly 0.25 because primary localization is defined over AMF, AUSF, UDM and UDR only. Storm and hard outage are reported separately and are not mixed into the primary single-target localization claim.

## Topology context

There is a separate one-class model per NF. Node identity is not a model feature.

Each passive feature is first normalized against the same NF's own normal history:

scale = max(1.4826 * MAD, 0.05 * abs(median), feature-specific floor).

Only after that normalization are topology-neighbor residuals compared. Raw AMF and UDR loads are never directly pooled into one median/MAD.

## Reproducibility and realism

Prometheus blackbox_exporter v0.28.0 is run as a real container and its native probe metrics are used.

All normal and anomalous events are interleaved in one run to prevent the classifier from learning differences between GitHub runners.

The active canary enters through the normal UE control-plane path instead of calling protected NF business APIs directly. The upstream free5GC compose testbed does not enable production mTLS/service-mesh termination; this remains an explicit external-validity limitation.

Selective tc/netem impairment is a caller-specific service-path degradation, not a software exploit and not proof that free5GC itself has been compromised.
