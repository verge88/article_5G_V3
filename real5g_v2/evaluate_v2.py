#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from real5g_v2.analysis_core import CANDIDATES, FLOORS, outer_fold

def metric_values(df):
    y = df.is_anomaly.astype(int).values
    p = df.pred.astype(int).values
    s = df.score.astype(float).values
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "fpr": float(fp / max(1, fp + tn)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "mcc": float(matthews_corrcoef(y, p)),
    }

def localization_top1(df):
    d = df[(df.primary_localization == 1) & (df.pred_node.astype(str) != "")]
    if len(d) == 0:
        return float("nan")
    return float((d.pred_node == d.target_node).mean())

def bootstrap_ci(pred, method, metric, n=2000, seed=2026):
    rng = np.random.default_rng(seed)
    blocks = sorted(pred.block_id.unique())
    vals = []
    for _ in range(n):
        chosen = rng.choice(blocks, size=len(blocks), replace=True)
        d = pd.concat(
            [pred[(pred.method == method) & (pred.block_id == b)] for b in chosen],
            ignore_index=True,
        )
        try:
            v = localization_top1(d) if metric == "localization_top1" else metric_values(d)[metric]
            if not math.isnan(v):
                vals.append(v)
        except Exception:
            pass
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))

def paired_difference_ci(pred, method_a, method_b, metric, n=3000, seed=77):
    rng = np.random.default_rng(seed)
    blocks = sorted(pred.block_id.unique())
    vals = []
    for _ in range(n):
        chosen = rng.choice(blocks, size=len(blocks), replace=True)
        a = pd.concat(
            [pred[(pred.method == method_a) & (pred.block_id == b)] for b in chosen],
            ignore_index=True,
        )
        b = pd.concat(
            [pred[(pred.method == method_b) & (pred.block_id == bb)] for bb in chosen],
            ignore_index=True,
        )
        if metric == "localization_top1":
            va, vb = localization_top1(a), localization_top1(b)
        else:
            va, vb = metric_values(a)[metric], metric_values(b)[metric]
        if not math.isnan(va) and not math.isnan(vb):
            vals.append(va - vb)
    return (
        float(np.mean(vals)),
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.975)),
    )

def target_signals(nodes, events):
    rows = []
    for _, ev in events[events.primary_localization == 1].iterrows():
        r = nodes[(nodes.event_id == ev.event_id) & (nodes.node == ev.target_node)].iloc[0]
        rows.append({
            "event_id": int(ev.event_id),
            "block_id": int(ev.block_id),
            "scenario": ev.scenario,
            "target_node": ev.target_node,
            "severity_ms": float(ev.severity_ms),
            "load_ues": int(ev.load_ues),
            "blackbox_latency_ms": float(r.blackbox_latency_ms),
            "active_stage_ms": float(r.active_stage_ms) if pd.notna(r.active_stage_ms) else float("nan"),
            "canary_registration_ms": float(ev.canary_registration_ms),
        })
    return pd.DataFrame(rows)

def write_figures(out, summary, pred, events, nodes):
    order = [
        "Blackbox_SLO", "Blackbox_Z", "Canary_E2E_Z", "ActiveStage_Z",
        "Passive_IF", "PassiveTopo_IF", "ActiveML_IF",
    ]
    methods = [m for m in order if m in set(summary.method)]
    p = summary.set_index("method").loc[methods]
    ax = p[["f1", "balanced_accuracy"]].plot(kind="bar", figsize=(11, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Leave-one-block-out detection results")
    plt.tight_layout()
    plt.savefig(out / "fig_detection_v2.png", dpi=180)
    plt.close()

    loc_methods = [m for m in methods if m != "Canary_E2E_Z"]
    ax = p.loc[loc_methods, ["localization_top1"]].plot(
        kind="bar", figsize=(10, 5), legend=False
    )
    ax.axhline(0.25, linestyle="--", linewidth=1, label="random among 4 candidates")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-1")
    ax.set_title("Single-target selective scenarios")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "fig_localization_v2.png", dpi=180)
    plt.close()

    normal = events[events.scenario == "normal"]
    nn = nodes[nodes.event_id.isin(normal.event_id)]
    drift = (
        normal.groupby("block_id")
        .agg(
            normal_events=("event_id", "count"),
            canary_registration_median_ms=("canary_registration_ms", "median"),
            canary_registration_p90_ms=("canary_registration_ms", lambda x: float(np.quantile(x, 0.9))),
        )
        .reset_index()
    )
    bb = nn.groupby("block_id").blackbox_latency_ms.median().rename("blackbox_latency_median_ms").reset_index()
    drift = drift.merge(bb, on="block_id")
    drift.to_csv(out / "normal_drift_by_block.csv", index=False)

    ax = drift.plot(
        x="block_id",
        y=["canary_registration_median_ms", "blackbox_latency_median_ms"],
        marker="o",
        figsize=(10, 5),
    )
    ax.set_ylabel("ms")
    ax.set_title("Normal-state drift across collection blocks")
    plt.tight_layout()
    plt.savefig(out / "fig_normal_drift_v2.png", dpi=180)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    nodes = pd.read_csv(args.nodes)
    events = pd.read_csv(args.events)

    blocks = sorted(events.block_id.unique())
    if len(blocks) < 6:
        raise RuntimeError("at least six collection blocks are required")
    normal = events[events.scenario == "normal"]
    if (normal.canary_success != 1).any():
        raise RuntimeError("normal training/evaluation data include failed canary registrations")

    preds, nscores = [], []
    for block in blocks:
        p, n = outer_fold(nodes, events, block)
        preds.append(p)
        nscores.append(n)
        print(f"leave-one-block-out fold {block} complete")
    pred = pd.concat(preds, ignore_index=True)
    node_scores = pd.concat(nscores, ignore_index=True)
    pred.to_csv(out / "oof_predictions_v2.csv", index=False)
    node_scores.to_csv(out / "oof_node_scores_v2.csv", index=False)

    rows = []
    for method in pred.method.unique():
        d = pred[pred.method == method]
        row = {"method": method, **metric_values(d), "localization_top1": localization_top1(d)}
        for metric in [
            "roc_auc", "pr_auc", "f1", "recall", "fpr",
            "balanced_accuracy", "mcc", "localization_top1"
        ]:
            lo, hi = bootstrap_ci(pred, method, metric)
            row[f"{metric}_ci_low"] = lo
            row[f"{metric}_ci_high"] = hi
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "metrics_with_block_bootstrap_ci.csv", index=False)

    comparison_pairs = [
        ("ActiveStage_Z", "Blackbox_Z"),
        ("ActiveML_IF", "ActiveStage_Z"),
        ("ActiveML_IF", "PassiveTopo_IF"),
        ("PassiveTopo_IF", "Passive_IF"),
    ]
    diffs = []
    for a, b in comparison_pairs:
        for metric in ["f1", "balanced_accuracy", "localization_top1"]:
            mean, lo, hi = paired_difference_ci(pred, a, b, metric)
            diffs.append({
                "method_a": a,
                "method_b": b,
                "metric": metric,
                "mean_difference": mean,
                "ci_low": lo,
                "ci_high": hi,
            })
    diffdf = pd.DataFrame(diffs)
    diffdf.to_csv(out / "paired_block_bootstrap_differences.csv", index=False)

    scen = (
        pred.groupby(["method", "scenario"])
        .agg(n=("event_id", "count"), detection_recall=("pred", "mean"))
        .reset_index()
    )
    locsrc = pred[pred.pred_node.astype(str) != ""].copy()
    locsrc["hit"] = (locsrc.pred_node == locsrc.target_node).astype(int)
    loc = (
        locsrc.groupby(["method", "scenario"])
        .hit.mean()
        .rename("localization_top1")
        .reset_index()
    )
    scen = scen.merge(loc, on=["method", "scenario"], how="left")
    scen.to_csv(out / "scenario_results_v2.csv", index=False)

    target = target_signals(nodes, events)
    target.to_csv(out / "selective_target_signals.csv", index=False)
    (
        target.groupby(["scenario", "severity_ms"])
        .agg(
            blackbox_latency_ms=("blackbox_latency_ms", "median"),
            active_stage_ms=("active_stage_ms", "median"),
            canary_registration_ms=("canary_registration_ms", "median"),
        )
        .reset_index()
        .to_csv(out / "selective_effect_by_severity.csv", index=False)
    )

    fp_rows = []
    for method in pred.method.unique():
        d = pred[(pred.method == method) & (pred.scenario == "normal")]
        for load, g in d.groupby("load_ues"):
            fp_rows.append({
                "method": method,
                "load_ues": int(load),
                "n": len(g),
                "false_positive_rate": float(g.pred.mean()),
            })
    pd.DataFrame(fp_rows).to_csv(out / "normal_fpr_by_load.csv", index=False)

    write_figures(out, summary, pred, events, nodes)

    cols = [
        "method", "roc_auc", "pr_auc", "f1", "balanced_accuracy",
        "fpr", "recall", "mcc", "localization_top1"
    ]
    tbl = summary[cols].copy()
    for c in cols[1:]:
        tbl[c] = tbl[c].map(lambda x: "NA" if pd.isna(x) else f"{x:.4f}")
    (out / "manuscript_table_v2.md").write_text(tbl.to_markdown(index=False), encoding="utf-8")

    def val(method, field):
        return float(summary.loc[summary.method == method, field].iloc[0])

    dloc = diffdf[
        (diffdf.method_a == "ActiveML_IF")
        & (diffdf.method_b == "ActiveStage_Z")
        & (diffdf.metric == "localization_top1")
    ].iloc[0]
    dbb = diffdf[
        (diffdf.method_a == "ActiveStage_Z")
        & (diffdf.method_b == "Blackbox_Z")
        & (diffdf.metric == "localization_top1")
    ].iloc[0]
    ml_statement = (
        "The ML localization gain over the simple active-stage rule is statistically separated from zero."
        if dloc.ci_low > 0
        else "The 95% block-bootstrap interval for the ML-minus-simple-active localization difference includes zero; the experiment does not establish a non-zero ML contribution."
    )

    report = f"""# Real free5GC + UERANSIM experiment v2

The redesigned dataset contains {len(events)} events collected in {len(blocks)} temporal blocks on one GitHub Actions runner:
{int((events.scenario == "normal").sum())} normal and {int((events.scenario != "normal").sum())} anomalous events.
Each block contains six normal and six anomalous events in a reproducibly shuffled order.

Primary selective scenarios delay only a legitimate service response path:
UDR->UDM, UDM->AUSF, AUSF->AMF, or AMF->gNB.
The Prometheus blackbox root probe does not traverse these filtered paths.

## Out-of-fold results

Blackbox_Z:
- F1 {val("Blackbox_Z","f1"):.4f}
- balanced accuracy {val("Blackbox_Z","balanced_accuracy"):.4f}
- primary Top-1 localization {val("Blackbox_Z","localization_top1"):.4f}

Canary_E2E_Z:
- F1 {val("Canary_E2E_Z","f1"):.4f}
- balanced accuracy {val("Canary_E2E_Z","balanced_accuracy"):.4f}

ActiveStage_Z:
- F1 {val("ActiveStage_Z","f1"):.4f}
- balanced accuracy {val("ActiveStage_Z","balanced_accuracy"):.4f}
- primary Top-1 localization {val("ActiveStage_Z","localization_top1"):.4f}

ActiveML_IF:
- F1 {val("ActiveML_IF","f1"):.4f}
- balanced accuracy {val("ActiveML_IF","balanced_accuracy"):.4f}
- primary Top-1 localization {val("ActiveML_IF","localization_top1"):.4f}

Random Top-1 for the four explicitly defined localization candidates is 0.2500.

Paired block-bootstrap ActiveStage_Z - Blackbox_Z localization difference:
{dbb.mean_difference:+.4f}, 95% CI [{dbb.ci_low:+.4f}, {dbb.ci_high:+.4f}].

Paired block-bootstrap ActiveML_IF - ActiveStage_Z localization difference:
{dloc.mean_difference:+.4f}, 95% CI [{dloc.ci_low:+.4f}, {dloc.ci_high:+.4f}].

**ML contribution:** {ml_statement}

## Statistical design

Evaluation is leave-one-block-out cross-validation. Each outer training set contains 42 normal events.
Isolation Forest thresholds are calibrated from cross-fitted normal scores from the training blocks rather than four validation points.
All reported confidence intervals use block bootstrap, so the resampling unit is a collection block, not a random forest seed.

The total test stream is balanced at the experiment level rather than containing about 90% anomalies; therefore F1 and PR-AUC are no longer inflated by extreme anomaly prevalence. Balanced accuracy, MCC, FPR and threshold-free ROC-AUC are reported alongside F1.

## Zone/topology design

There is a separate one-class model for every NF and node identity is not supplied as a feature.
Raw AMF CPU/request rate is never compared directly with raw UDR/UDM values.
Every feature is first normalized against that NF's own normal history using:

scale = max(1.4826*MAD, 0.05*|median|, feature floor).

Topology context is then the disagreement between the already normalized NF residual and normalized residuals of actual SBA neighbors.
The feature floors are stored in analysis_manifest_v2.json.

## Scope

The active control is a dedicated UE Initial Registration over the normal NGAP/NAS/free5GC service chain, not an unauthenticated GET request to an NF business API.
The separate Prometheus blackbox baseline intentionally represents ordinary liveness monitoring.

The upstream free5GC compose setup used here does not enable production mTLS/service-mesh termination. This remains an external-validity limitation.
The selective impairments emulate caller-specific service-path degradation, not a software exploit or proof of compromise.
"""
    (out / "summary_v2.md").write_text(report, encoding="utf-8")

    manifest = {
        "cross_validation": "leave-one-collection-block-out",
        "bootstrap": "2000 block bootstrap replicates for metric CI; 3000 for paired differences",
        "outer_training_normal_events": 42,
        "events_per_test_block": 12,
        "class_balance": {
            "normal": int((events.scenario == "normal").sum()),
            "anomaly": int((events.scenario != "normal").sum()),
        },
        "localization_candidates": CANDIDATES,
        "random_top1": 0.25,
        "robust_scale": "max(1.4826*MAD, 0.05*abs(median), feature-specific floor)",
        "feature_floors": FLOORS,
        "simple_baselines": ["Blackbox_SLO", "Blackbox_Z", "Canary_E2E_Z", "ActiveStage_Z"],
        "ml_methods": ["Passive_IF", "PassiveTopo_IF", "ActiveML_IF"],
    }
    (out / "analysis_manifest_v2.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(report)

if __name__ == "__main__":
    main()
