# Real free5GC + UERANSIM validation

This directory implements a real-software control-plane validation of the article hypothesis.

## What is real

- free5GC v4.3.0 network-function containers;
- UERANSIM gNB and UE software;
- SCTP/NGAP between gNB and AMF;
- NAS 5G-AKA registration;
- free5GC SBI communication between AMF/AUSF/UDM/UDR/NRF/NSSF/PCF and related functions;
- direct HTTP/2 reachability/latency probes;
- Docker CPU/memory telemetry;
- packet capture and raw NF/UE logs.

## Why the workflow does not require UPF

The experiment is designed to validate signaling/control-plane anomaly detection. Generated UE configurations use `sessions: []`, so registration and authentication are exercised without establishing a PDU session. This avoids making the experiment depend on the host-specific `gtp5g` kernel module and keeps the GitHub-hosted run reproducible.

This is therefore a real 5G Core control-plane experiment, not a user-plane throughput benchmark.

## Scenarios

- normal initial registration;
- registration storm using five concurrent UEs;
- AMF network delay;
- UDM outage;
- low-amplitude UDM delay;
- low-amplitude AUSF delay.

The low-amplitude delay cases are compromise-like **controlled impairments** produced with Linux netem inside the network namespace of the running NF. They test whether active control measurements help localize a problem that is not necessarily obvious from local CPU/request counters. They are not exploits and must not be described as proof of a free5GC compromise.

## Analysis

The ML stage is one-class / anomaly detection trained only on normal observations:

- `Passive_IF`: local NF telemetry;
- `PassiveZone_IF`: local telemetry plus zonal deviations;
- `ActiveProbe_IF`: local + zonal + active-probe signals;
- `ActiveProbe_OCSVM`: one-class SVM on the same proposed feature family.

The principal scientific comparison is `PassiveZone_IF` vs `ActiveProbe_IF`.
