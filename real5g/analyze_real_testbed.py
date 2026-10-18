#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

PASSIVE = ["cpu_pct", "mem_mib", "http_requests", "http_error_rate", "path_entropy"]
ZONE_PASSIVE = ["zone_cpu_dev", "zone_req_dev"]
ACTIVE = ["probe_latency_ms", "probe_ok", "zone_probe_dev"]

FEATURE_SETS = {
    "Passive_IF": PASSIVE,
    "PassiveZone_IF": PASSIVE + ZONE_PASSIVE,
    "ActiveProbe_IF": PASSIVE + ZONE_PASSIVE + ACTIVE,
    "ActiveProbe_OCSVM": PASSIVE + ZONE_PASSIVE + ACTIVE,
}

def split_normal_events(events):
    normal = events.loc[events.scenario == "normal", "event_id"].drop_duplicates().sort_values().tolist()
    if len(normal) < 8:
        raise RuntimeError(f"Need at least 8 normal events, got {len(normal)}")
    n = len(normal)
    a = max(4, int(round(n * 0.60)))
    b = max(a + 2, int(round(n * 0.80)))
    b = min(b, n - 2)
    return normal[:a], normal[a:b], normal[b:]

def fit_model(name, X, seed):
    if name.endswith("_IF"):
        model = make_pipeline(
            StandardScaler(),
            IsolationForest(
                n_estimators=300,
                max_samples="auto",
                contamination="auto",
                random_state=seed,
                n_jobs=2,
            ),
        )
    else:
        model = make_pipeline(StandardScaler(), OneClassSVM(kernel="rbf", gamma="scale", nu=0.02))
    model.fit(X)
    return model

def anomaly_scores(model, X):
    return -model.decision_function(X)

def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")

def evaluate_seed(nodes, events, method, features, seed, train_ids, val_ids, test_normal_ids):
    train = nodes[nodes.event_id.isin(train_ids)].copy()
    val = nodes[nodes.event_id.isin(val_ids)].copy()
    test_ids = set(test_normal_ids) | set(events.loc[events.scenario != "normal", "event_id"].tolist())
    test = nodes[nodes.event_id.isin(test_ids)].copy()

    model = fit_model(method, train[features].fillna(0.0).values, seed)
    val["score"] = anomaly_scores(model, val[features].fillna(0.0).values)
    test["score"] = anomaly_scores(model, test[features].fillna(0.0).values)

    val_event = val.groupby("event_id").score.max()
    threshold = float(np.quantile(val_event.values, 0.95)) if len(val_event) else 0.0

    event_scores = test.groupby("event_id").score.max().rename("score").reset_index()
    ev = events[events.event_id.isin(test_ids)].merge(event_scores, on="event_id", how="left")
    ev["score"] = ev["score"].fillna(0.0)
    ev["y"] = (ev.scenario != "normal").astype(int)
    ev["pred"] = (ev.score > threshold).astype(int)

    p, r, f, _ = precision_recall_fscore_support(ev.y, ev.pred, average="binary", zero_division=0)
    fpr = float(ev.loc[ev.y == 0, "pred"].mean()) if (ev.y == 0).any() else float("nan")

    anomaly_events = ev[ev.y == 1].copy()
    loc_hits = []
    stealth_hits = []
    scenario_rows = []
    for _, er in anomaly_events.iterrows():
        sub = test[test.event_id == er.event_id]
        predicted_node = sub.sort_values("score", ascending=False).iloc[0].node if len(sub) else ""
        hit = int(predicted_node == er.target_node)
        loc_hits.append(hit)
        if str(er.scenario).startswith("stealth_"):
            stealth_hits.append(hit)
        scenario_rows.append({
            "method": method,
            "seed": seed,
            "event_id": int(er.event_id),
            "scenario": er.scenario,
            "target_node": er.target_node,
            "predicted_node": predicted_node,
            "localization_hit": hit,
            "detected": int(er.pred),
            "event_score": float(er.score),
        })

    stealth_mask = ev.scenario.astype(str).str.startswith("stealth_")
    row = {
        "method": method,
        "seed": seed,
        "roc_auc": safe_auc(ev.y.values, ev.score.values),
        "pr_auc": float(average_precision_score(ev.y.values, ev.score.values)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "fpr": fpr,
        "localization_top1": float(np.mean(loc_hits)) if loc_hits else float("nan"),
        "stealth_localization_top1": float(np.mean(stealth_hits)) if stealth_hits else float("nan"),
        "stealth_detection_recall": float(ev.loc[stealth_mask, "pred"].mean()) if stealth_mask.any() else float("nan"),
        "threshold": threshold,
        "train_normal_events": len(train_ids),
        "validation_normal_events": len(val_ids),
        "test_normal_events": len(test_normal_ids),
    }
    return row, scenario_rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    nodes = pd.read_csv(args.input)
    events = pd.read_csv(args.events)
    train_ids, val_ids, test_normal_ids = split_normal_events(events)

    metrics = []
    scenario_rows = []
    for method, features in FEATURE_SETS.items():
        seeds = [11, 29, 47, 71, 101] if method.endswith("_IF") else [11]
        for seed in seeds:
            m, sr = evaluate_seed(nodes, events, method, features, seed, train_ids, val_ids, test_normal_ids)
            metrics.append(m)
            scenario_rows.extend(sr)

    mdf = pd.DataFrame(metrics)
    sdf = pd.DataFrame(scenario_rows)
    mdf.to_csv(out / "real_model_metrics_per_seed.csv", index=False)
    sdf.to_csv(out / "real_scenario_predictions.csv", index=False)

    summary = (
        mdf.groupby("method")
        .agg({
            "roc_auc": ["mean", "std"],
            "pr_auc": ["mean", "std"],
            "f1": ["mean", "std"],
            "fpr": ["mean", "std"],
            "localization_top1": ["mean", "std"],
            "stealth_localization_top1": ["mean", "std"],
            "stealth_detection_recall": ["mean", "std"],
        })
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / "real_metrics_summary.csv", index=False)

    scen = (
        sdf.groupby(["method", "scenario"])
        .agg(detection_recall=("detected", "mean"), localization_top1=("localization_hit", "mean"))
        .reset_index()
    )
    scen.to_csv(out / "real_scenario_summary.csv", index=False)

    order = [m for m in FEATURE_SETS if m in set(summary.method)]
    plotdf = summary.set_index("method").loc[order]
    ax = plotdf[["f1_mean", "localization_top1_mean", "stealth_localization_top1_mean"]].plot(
        kind="bar", figsize=(10, 5)
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Real free5GC + UERANSIM validation")
    plt.tight_layout()
    plt.savefig(out / "fig_real_metrics.png", dpi=180)
    plt.close()

    pivot = scen.pivot(index="scenario", columns="method", values="localization_top1")
    ax = pivot.plot(kind="bar", figsize=(11, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("top-1 localization")
    ax.set_title("Localization by controlled real-testbed scenario")
    plt.tight_layout()
    plt.savefig(out / "fig_real_scenario_localization.png", dpi=180)
    plt.close()

    table_cols = [
        "method",
        "roc_auc_mean",
        "pr_auc_mean",
        "f1_mean",
        "fpr_mean",
        "localization_top1_mean",
        "stealth_localization_top1_mean",
        "stealth_detection_recall_mean",
    ]
    table = summary[table_cols].copy()
    for c in table_cols[1:]:
        table[c] = table[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
    (out / "real_manuscript_table.md").write_text(table.to_markdown(index=False), encoding="utf-8")

    active = summary[summary.method == "ActiveProbe_IF"].iloc[0]
    passive = summary[summary.method == "PassiveZone_IF"].iloc[0]
    base = summary[summary.method == "Passive_IF"].iloc[0]
    event_counts = events.scenario.value_counts().to_dict()

    report = f"""# Real free5GC + UERANSIM experiment

This run uses **real free5GC network functions and real UERANSIM gNB/UE software** inside GitHub Actions. UE authentication and registration traverse real NGAP/NAS and free5GC SBI traffic.

The control-plane experiment intentionally does not request a PDU session, so it does not depend on the gtp5g kernel module. This is appropriate for the article's signaling/control-plane hypothesis, but it is not a user-plane throughput experiment.

## Dataset

- normal events: {event_counts.get("normal", 0)}
- stealth UDM delay events: {event_counts.get("stealth_udm_delay", 0)}
- stealth AUSF delay events: {event_counts.get("stealth_ausf_delay", 0)}
- AMF delay events: {event_counts.get("amf_delay", 0)}
- UDM outage events: {event_counts.get("udm_outage", 0)}
- registration-storm events: {event_counts.get("registration_storm", 0)}
- observed NFs per event: {nodes.node.nunique()}

Normal events are split by event ID into train/validation/held-out-normal sets. All anomaly scenarios are held out from training.

## Active-probe result

**ActiveProbe_IF**
- ROC-AUC: {active.roc_auc_mean:.4f}
- PR-AUC: {active.pr_auc_mean:.4f}
- F1: {active.f1_mean:.4f}
- FPR: {active.fpr_mean:.4f}
- top-1 NF localization: {active.localization_top1_mean:.4f}
- stealth top-1 localization: {active.stealth_localization_top1_mean:.4f}
- stealth detection recall: {active.stealth_detection_recall_mean:.4f}

**PassiveZone_IF**
- F1: {passive.f1_mean:.4f}
- top-1 NF localization: {passive.localization_top1_mean:.4f}
- stealth top-1 localization: {passive.stealth_localization_top1_mean:.4f}

**Passive_IF**
- F1: {base.f1_mean:.4f}
- top-1 NF localization: {base.localization_top1_mean:.4f}
- stealth top-1 localization: {base.stealth_localization_top1_mean:.4f}

## Interpretation boundary

The two "stealth" scenarios are controlled impairments injected with Linux tc netem into the network namespace of a **real running NF** while UE registrations use real free5GC/UERANSIM procedures. They emulate a compromise-like condition in which local load need not become conspicuous. They are **not evidence that free5GC itself was compromised** and should not be described as an exploit.

The artifact also contains raw NF/UE logs, container metadata, image digests, and a packet capture of SBI/NGAP traffic for independent verification.
"""
    (out / "summary.md").write_text(report, encoding="utf-8")

    meta = {
        "train_normal_event_ids": train_ids,
        "validation_normal_event_ids": val_ids,
        "heldout_normal_event_ids": test_normal_ids,
        "feature_sets": FEATURE_SETS,
        "model_seeds": [11, 29, 47, 71, 101],
        "threshold": "95th percentile of validation-normal event max anomaly scores",
        "localization": "argmax node anomaly score within each anomalous event",
    }
    (out / "analysis_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(report)

if __name__ == "__main__":
    main()
