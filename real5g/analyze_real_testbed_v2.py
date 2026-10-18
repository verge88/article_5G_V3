#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PASSIVE_NODES = ["AMF", "AUSF", "UDM", "UDR"]
TARGET_NODES = ["AUSF", "UDM", "UDR"]
PASSIVE = ["cpu_pct", "mem_mib", "http_requests", "http_error_rate", "path_entropy"]
ROOT = ["root_probe_duration_ms", "root_probe_success"]
BUSINESS = [
    "canary_business_requests",
    "canary_business_median_ms",
    "canary_business_p95_ms",
    "canary_business_status_5xx",
]
ADJ = {
    "AMF": ["AUSF"],
    "AUSF": ["AMF", "UDM"],
    "UDM": ["AUSF", "UDR"],
    "UDR": ["UDM"],
}
RNG = np.random.default_rng(20261018)


def robust_center_scale(x):
    x = np.asarray(x, dtype=float)
    med = float(np.nanmedian(x))
    q25, q75 = np.nanpercentile(x, [25, 75])
    iqr = float(q75 - q25)
    mad = float(np.nanmedian(np.abs(x - med)))
    eps = max(1e-6, 0.01 * max(abs(med), 1.0))
    scale = max(1.4826 * mad, iqr / 1.349 if iqr > 0 else 0.0, eps)
    return med, scale


def apply_fold_normalization(nodes, train_neg):
    out = nodes.copy()
    params = {}
    for node in PASSIVE_NODES:
        for feat in PASSIVE:
            vals = train_neg.loc[train_neg.node == node, feat].astype(float).values
            med, scale = robust_center_scale(vals)
            params[(node, feat)] = (med, scale)
            mask = out.node == node
            z = (out.loc[mask, feat].astype(float) - med) / scale
            out.loc[mask, f"z_{feat}"] = np.clip(z, -8.0, 8.0)

    for feat in PASSIVE:
        zcol = f"z_{feat}"
        rescol = f"neighbor_resid_{feat}"
        out[rescol] = 0.0
        for (rep, eid), grp in out.groupby(["replicate", "event_id"]):
            zmap = dict(zip(grp.node, grp[zcol]))
            for idx, row in grp.iterrows():
                neigh = [n for n in ADJ.get(row.node, []) if n in zmap]
                med_neigh = float(np.median([zmap[n] for n in neigh])) if neigh else 0.0
                out.at[idx, rescol] = float(np.clip(row[zcol] - med_neigh, -8.0, 8.0))
    return out, params


def load_inputs(input_dir):
    root = Path(input_dir)
    nfiles = sorted(root.glob("**/real5g_node_observations.csv"))
    efiles = sorted(root.glob("**/real5g_events.csv"))
    if not nfiles or not efiles:
        raise RuntimeError(f"No collected CSV files under {root}")
    nodes = pd.concat([pd.read_csv(p) for p in nfiles], ignore_index=True)
    events = pd.concat([pd.read_csv(p) for p in efiles], ignore_index=True)
    nodes["replicate"] = nodes.replicate.astype(int)
    events["replicate"] = events.replicate.astype(int)
    return nodes, events


def valid_primary_events(events):
    controlled = {"delay", "jitter", "error", "evasion_delay"}
    invalid = (events.label == 1) & events.fault_kind.isin(controlled) & (events.fault_effective != 1)
    return events.loc[~invalid].copy(), events.loc[invalid].copy()


def build_event_wide(nodes, events, feature_cols):
    parts = []
    key = ["replicate", "event_id"]
    for node in PASSIVE_NODES:
        sub = nodes[nodes.node == node][key + feature_cols].copy()
        sub = sub.rename(columns={c: f"{node}__{c}" for c in feature_cols})
        parts.append(sub)
    wide = parts[0]
    for p in parts[1:]:
        wide = wide.merge(p, on=key, how="outer")
    globals_ = events[key + [
        "label", "scenario", "scenario_family", "target_node", "sequence", "elapsed_s",
        "canary_registration_latency_ms", "canary_registration_success",
        "workload_registration_latency_ms", "workload_registration_success",
    ]].copy()
    wide = globals_.merge(wide, on=key, how="left")
    return wide.fillna(0.0)


def fit_if(X):
    return make_pipeline(
        StandardScaler(),
        IsolationForest(
            n_estimators=500,
            contamination=0.05,
            max_samples="auto",
            random_state=20261018,
            n_jobs=2,
        ),
    ).fit(X)


def binary_metrics(y, pred, score):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    score = np.asarray(score, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    fpr = fp / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, score) if len(np.unique(y)) > 1 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "positive_f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": specificity,
        "fpr": fpr,
    }


def node_score_normalizer(train_scores):
    med, scale = robust_center_scale(train_scores)
    return med, scale


def event_if_predictions(train_nodes, test_nodes, train_events, test_events, event_features, local_features, method):
    train_w = build_event_wide(train_nodes, train_events, event_features)
    test_w = build_event_wide(test_nodes, test_events, event_features)
    xcols = [c for c in train_w.columns if "__" in c]
    if "canary_registration_latency_ms" in event_features:
        # global columns are merged separately below; this branch is retained for clarity.
        pass
    global_cols = []
    if method == "Active_IF":
        global_cols = ["canary_registration_latency_ms", "canary_registration_success"]
    xcols += global_cols

    model = fit_if(train_w[xcols].values)
    score = -model.decision_function(test_w[xcols].values)
    pred = (model.predict(test_w[xcols].values) == -1).astype(int)

    loc_rows = []
    per_node_test = {}
    for node in TARGET_NODES:
        tr = train_nodes[train_nodes.node == node]
        te = test_nodes[test_nodes.node == node]
        lf = list(local_features)
        Xtr = tr[lf].fillna(0.0).values
        Xte = te[lf].fillna(0.0).values
        m = fit_if(Xtr)
        tr_score = -m.decision_function(Xtr)
        te_score = -m.decision_function(Xte)
        med, sc = node_score_normalizer(tr_score)
        z = np.clip((te_score - med) / sc, -20, 20)
        tmp = te[["replicate", "event_id", "scenario", "scenario_family", "target_node", "label"]].copy()
        tmp["node"] = node
        tmp["loc_score"] = z
        per_node_test[node] = tmp

    loc_all = pd.concat(per_node_test.values(), ignore_index=True)
    for (rep, eid), grp in loc_all.groupby(["replicate", "event_id"]):
        first = grp.iloc[0]
        if int(first.label) != 1 or first.target_node not in TARGET_NODES:
            continue
        ranked = grp.sort_values("loc_score", ascending=False).node.tolist()
        rank = ranked.index(first.target_node) + 1 if first.target_node in ranked else len(TARGET_NODES) + 1
        loc_rows.append({
            "replicate": rep, "event_id": eid, "method": method,
            "scenario": first.scenario, "scenario_family": first.scenario_family,
            "target_node": first.target_node, "predicted_node": ranked[0],
            "top1": int(rank == 1), "top2": int(rank <= 2), "rr": 1.0 / rank,
        })

    pred_df = test_w[[
        "replicate", "event_id", "label", "scenario", "scenario_family", "target_node", "sequence", "elapsed_s"
    ]].copy()
    pred_df["method"] = method
    pred_df["score"] = score
    pred_df["pred"] = pred
    return pred_df, pd.DataFrame(loc_rows)


def per_node_robust_scores(train_nodes, test_nodes, feature, failure_feature=None):
    scores = []
    for node in TARGET_NODES:
        tr = train_nodes[train_nodes.node == node]
        te = test_nodes[test_nodes.node == node].copy()
        med, scale = robust_center_scale(tr[feature].astype(float).values)
        z = (te[feature].astype(float).values - med) / scale
        if failure_feature is not None:
            fail = te[failure_feature].astype(float).values
            z = np.where(fail <= 0, np.maximum(z, 12.0), z)
        te["simple_score"] = np.clip(z, -20, 20)
        te["node"] = node
        scores.append(te[[
            "replicate", "event_id", "label", "scenario", "scenario_family", "target_node", "node", "simple_score"
        ]])
    return pd.concat(scores, ignore_index=True)


def simple_node_baseline(train_nodes, test_nodes, test_events, method):
    if method in ("RootProbeZ", "BlackboxSLO"):
        scored = per_node_robust_scores(train_nodes, test_nodes, "root_probe_duration_ms", "root_probe_success")
    elif method in ("BusinessLatencyZ", "BusinessSLO", "CombinedRules"):
        scored = per_node_robust_scores(train_nodes, test_nodes, "canary_business_median_ms", None)
        # An injected/observed 5xx is independently severe even when latency is short.
        err = test_nodes[test_nodes.node.isin(TARGET_NODES)][[
            "replicate", "event_id", "node", "canary_business_status_5xx"
        ]]
        scored = scored.merge(err, on=["replicate", "event_id", "node"], how="left")
        scored["simple_score"] = np.where(
            scored.canary_business_status_5xx.fillna(0).values > 0,
            np.maximum(scored.simple_score.values, 12.0),
            scored.simple_score.values,
        )
    else:
        raise ValueError(method)

    event_score = scored.groupby(["replicate", "event_id"]).simple_score.max().rename("score").reset_index()
    pred_df = test_events.merge(event_score, on=["replicate", "event_id"], how="left")
    pred_df["score"] = pred_df.score.fillna(0.0)

    if method.endswith("SLO"):
        if method == "BlackboxSLO":
            train_scored = per_node_robust_scores(train_nodes, train_nodes, "root_probe_duration_ms", "root_probe_success")
        else:
            train_scored = per_node_robust_scores(train_nodes, train_nodes, "canary_business_median_ms", None)
            terr = train_nodes[train_nodes.node.isin(TARGET_NODES)][[
                "replicate", "event_id", "node", "canary_business_status_5xx"
            ]]
            train_scored = train_scored.merge(terr, on=["replicate", "event_id", "node"], how="left")
            train_scored["simple_score"] = np.where(
                train_scored.canary_business_status_5xx.fillna(0).values > 0,
                np.maximum(train_scored.simple_score.values, 12.0),
                train_scored.simple_score.values,
            )
        train_event_score = train_scored.groupby(["replicate", "event_id"]).simple_score.max().values
        threshold = float(np.quantile(train_event_score, 0.95))
    else:
        threshold = 3.5
    pred_df["pred"] = (pred_df.score > threshold).astype(int)
    pred_df["method"] = method

    loc_rows = []
    for (rep, eid), grp in scored.groupby(["replicate", "event_id"]):
        first = grp.iloc[0]
        if int(first.label) != 1 or first.target_node not in TARGET_NODES:
            continue
        ranked = grp.sort_values("simple_score", ascending=False).node.tolist()
        rank = ranked.index(first.target_node) + 1
        loc_rows.append({
            "replicate": rep, "event_id": eid, "method": method,
            "scenario": first.scenario, "scenario_family": first.scenario_family,
            "target_node": first.target_node, "predicted_node": ranked[0],
            "top1": int(rank == 1), "top2": int(rank <= 2), "rr": 1.0 / rank,
        })
    return pred_df, pd.DataFrame(loc_rows)


def registration_slo(train_events, test_events):
    vals = train_events.canary_registration_latency_ms.astype(float).values
    threshold = float(np.quantile(vals, 0.95))
    out = test_events.copy()
    failure_score = np.where(out.canary_registration_success.astype(float).values < 1.0, threshold * 4, 0.0)
    out["score"] = np.maximum(out.canary_registration_latency_ms.astype(float).values, failure_score)
    out["pred"] = ((out.canary_registration_success < 1.0) | (out.canary_registration_latency_ms > threshold)).astype(int)
    out["method"] = "UERegistrationSLO"
    return out


def combined_rules(train_nodes, test_nodes, train_events, test_events):
    business_pred, business_loc = simple_node_baseline(train_nodes, test_nodes, test_events, "BusinessSLO")
    root_pred, _ = simple_node_baseline(train_nodes, test_nodes, test_events, "BlackboxSLO")
    reg_pred = registration_slo(train_events, test_events)
    keys = ["replicate", "event_id"]
    a = business_pred[keys + ["score", "pred"]].rename(columns={"score": "bs", "pred": "bp"})
    b = root_pred[keys + ["score", "pred"]].rename(columns={"score": "rs", "pred": "rp"})
    c = reg_pred[keys + ["score", "pred"]].rename(columns={"score": "us", "pred": "up"})
    out = test_events.merge(a, on=keys).merge(b, on=keys).merge(c, on=keys)
    out["pred"] = ((out.bp > 0) | (out.rp > 0) | (out.up > 0)).astype(int)
    # standardized max is used only for ranking/AUC; binary alert is rule-based.
    out["score"] = np.maximum.reduce([out.bs.values, out.rs.values, out.us.values / max(1.0, np.median(train_events.canary_registration_latency_ms))])
    out["method"] = "CombinedRules"
    loc = business_loc.copy()
    loc["method"] = "CombinedRules"
    return out, loc


def bootstrap_ci(preds, locs, iterations=1000):
    rows = []
    metric_names = ["roc_auc", "pr_auc", "balanced_accuracy", "mcc", "macro_f1", "positive_f1",
                    "precision", "sensitivity", "specificity", "fpr"]
    for method, df in preds.groupby("method"):
        loc = locs[locs.method == method] if len(locs) else pd.DataFrame()
        point = binary_metrics(df.label.values, df.pred.values, df.score.values)
        point["localization_top1"] = float(loc.top1.mean()) if len(loc) else np.nan
        point["localization_top2"] = float(loc.top2.mean()) if len(loc) else np.nan
        point["localization_mrr"] = float(loc.rr.mean()) if len(loc) else np.nan
        boots = {k: [] for k in metric_names + ["localization_top1", "localization_top2", "localization_mrr"]}

        groups = {(r, y): g.index.to_numpy() for (r, y), g in df.groupby(["replicate", "label"])}
        loc_groups = {r: g.index.to_numpy() for r, g in loc.groupby("replicate")} if len(loc) else {}

        for _ in range(iterations):
            idx = []
            for inds in groups.values():
                idx.extend(RNG.choice(inds, size=len(inds), replace=True).tolist())
            bdf = df.loc[idx]
            try:
                bm = binary_metrics(bdf.label.values, bdf.pred.values, bdf.score.values)
                for k in metric_names:
                    boots[k].append(bm[k])
            except Exception:
                pass
            if len(loc):
                lidx = []
                for inds in loc_groups.values():
                    lidx.extend(RNG.choice(inds, size=len(inds), replace=True).tolist())
                b = loc.loc[lidx]
                boots["localization_top1"].append(float(b.top1.mean()))
                boots["localization_top2"].append(float(b.top2.mean()))
                boots["localization_mrr"].append(float(b.rr.mean()))

        row = {"method": method}
        for k, v in point.items():
            row[k] = v
        for k, vals in boots.items():
            vals = [v for v in vals if pd.notna(v)]
            if vals:
                row[f"{k}_ci_low"] = float(np.quantile(vals, 0.025))
                row[f"{k}_ci_high"] = float(np.quantile(vals, 0.975))
            else:
                row[f"{k}_ci_low"] = np.nan
                row[f"{k}_ci_high"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_differences(preds, locs, reference="Active_IF", iterations=1000):
    out = []
    ref = preds[preds.method == reference].set_index(["replicate", "event_id"])
    ref_loc = locs[locs.method == reference].set_index(["replicate", "event_id"]) if len(locs) else pd.DataFrame()
    for method in sorted(set(preds.method) - {reference, "AlwaysAnomaly"}):
        cmp = preds[preds.method == method].set_index(["replicate", "event_id"])
        common = ref.index.intersection(cmp.index)
        if not len(common):
            continue
        d = pd.DataFrame({
            "label": ref.loc[common, "label"],
            "ref_pred": ref.loc[common, "pred"],
            "ref_score": ref.loc[common, "score"],
            "cmp_pred": cmp.loc[common, "pred"],
            "cmp_score": cmp.loc[common, "score"],
        })
        diffs_ba, diffs_mcc = [], []
        for _ in range(iterations):
            ix = RNG.choice(np.arange(len(d)), size=len(d), replace=True)
            b = d.iloc[ix]
            diffs_ba.append(
                balanced_accuracy_score(b.label, b.ref_pred) - balanced_accuracy_score(b.label, b.cmp_pred)
            )
            diffs_mcc.append(
                matthews_corrcoef(b.label, b.ref_pred) - matthews_corrcoef(b.label, b.cmp_pred)
            )
        row = {
            "reference": reference,
            "comparison": method,
            "delta_balanced_accuracy": balanced_accuracy_score(d.label, d.ref_pred) - balanced_accuracy_score(d.label, d.cmp_pred),
            "delta_balanced_accuracy_ci_low": float(np.quantile(diffs_ba, 0.025)),
            "delta_balanced_accuracy_ci_high": float(np.quantile(diffs_ba, 0.975)),
            "delta_mcc": matthews_corrcoef(d.label, d.ref_pred) - matthews_corrcoef(d.label, d.cmp_pred),
            "delta_mcc_ci_low": float(np.quantile(diffs_mcc, 0.025)),
            "delta_mcc_ci_high": float(np.quantile(diffs_mcc, 0.975)),
        }
        if len(ref_loc) and method in set(locs.method):
            cl = locs[locs.method == method].set_index(["replicate", "event_id"])
            ci = ref_loc.index.intersection(cl.index)
            if len(ci):
                delta = ref_loc.loc[ci, "top1"].values - cl.loc[ci, "top1"].values
                boot = []
                for _ in range(iterations):
                    ix = RNG.choice(np.arange(len(delta)), size=len(delta), replace=True)
                    boot.append(float(np.mean(delta[ix])))
                row["delta_localization_top1"] = float(np.mean(delta))
                row["delta_localization_top1_ci_low"] = float(np.quantile(boot, 0.025))
                row["delta_localization_top1_ci_high"] = float(np.quantile(boot, 0.975))
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nodes_raw, events_raw = load_inputs(args.input_dir)
    events, ineffective = valid_primary_events(events_raw)
    nodes_raw = nodes_raw.merge(events[["replicate", "event_id"]], on=["replicate", "event_id"], how="inner")
    reps = sorted(events.replicate.unique().tolist())
    if len(reps) < 3:
        raise RuntimeError(f"Need >=3 complete runner replicates, got {reps}")

    all_preds, all_locs, fold_metrics = [], [], []
    for heldout in reps:
        train_events = events[(events.replicate != heldout) & (events.label == 0)].copy()
        test_events = events[events.replicate == heldout].copy()
        train_nodes_raw = nodes_raw.merge(train_events[["replicate", "event_id"]], on=["replicate", "event_id"], how="inner")
        test_nodes_raw = nodes_raw.merge(test_events[["replicate", "event_id"]], on=["replicate", "event_id"], how="inner")

        combined = pd.concat([train_nodes_raw, test_nodes_raw], ignore_index=True)
        normalized, _ = apply_fold_normalization(combined, train_nodes_raw)
        train_nodes = normalized.merge(train_events[["replicate", "event_id"]], on=["replicate", "event_id"], how="inner")
        test_nodes = normalized.merge(test_events[["replicate", "event_id"]], on=["replicate", "event_id"], how="inner")

        zone_cols = [f"neighbor_resid_{f}" for f in PASSIVE]
        passive_event = PASSIVE
        passive_zone_event = PASSIVE + zone_cols
        active_event = PASSIVE + zone_cols + ROOT + BUSINESS

        p1, l1 = event_if_predictions(
            train_nodes, test_nodes, train_events, test_events,
            passive_event, PASSIVE, "Passive_IF"
        )
        p2, l2 = event_if_predictions(
            train_nodes, test_nodes, train_events, test_events,
            passive_zone_event, PASSIVE + zone_cols, "PassiveZone_IF"
        )
        p3, l3 = event_if_predictions(
            train_nodes, test_nodes, train_events, test_events,
            active_event,
            PASSIVE + zone_cols + ROOT + BUSINESS + ["canary_registration_latency_ms", "canary_registration_success"],
            "Active_IF",
        )

        # Add global canary fields to node-local active localization table.
        globals_ = test_events[["replicate", "event_id", "canary_registration_latency_ms", "canary_registration_success"]]
        train_globals = train_events[["replicate", "event_id", "canary_registration_latency_ms", "canary_registration_success"]]
        # Re-run Active localization with the global fields merged into node rows.
        train_nodes_active = train_nodes.drop(columns=["canary_registration_latency_ms", "canary_registration_success"], errors="ignore").merge(
            train_globals, on=["replicate", "event_id"], how="left"
        )
        test_nodes_active = test_nodes.drop(columns=["canary_registration_latency_ms", "canary_registration_success"], errors="ignore").merge(
            globals_, on=["replicate", "event_id"], how="left"
        )
        p3, l3 = event_if_predictions(
            train_nodes_active, test_nodes_active, train_events, test_events,
            active_event,
            PASSIVE + zone_cols + ROOT + BUSINESS + ["canary_registration_latency_ms", "canary_registration_success"],
            "Active_IF",
        )

        simple_pairs = []
        for m in ["RootProbeZ", "BlackboxSLO", "BusinessLatencyZ", "BusinessSLO"]:
            simple_pairs.append(simple_node_baseline(train_nodes, test_nodes, test_events, m))
        reg = registration_slo(train_events, test_events)
        combo_p, combo_l = combined_rules(train_nodes, test_nodes, train_events, test_events)

        always = test_events.copy()
        always["score"] = 1.0
        always["pred"] = 1
        always["method"] = "AlwaysAnomaly"

        fold_preds = [p1, p2, p3] + [x[0] for x in simple_pairs] + [reg, combo_p, always]
        fold_locs = [l1, l2, l3] + [x[1] for x in simple_pairs] + [combo_l]

        for df in fold_preds:
            keep = ["replicate", "event_id", "label", "scenario", "scenario_family", "target_node", "method", "score", "pred"]
            all_preds.append(df[keep].copy())
            mm = binary_metrics(df.label.values, df.pred.values, df.score.values)
            mm.update({"heldout_replicate": heldout, "method": df.method.iloc[0]})
            fold_metrics.append(mm)
        for df in fold_locs:
            if len(df):
                all_locs.append(df.copy())

    preds = pd.concat(all_preds, ignore_index=True)
    locs = pd.concat(all_locs, ignore_index=True) if all_locs else pd.DataFrame()
    summary = bootstrap_ci(preds, locs, iterations=args.bootstrap)
    diffs = paired_bootstrap_differences(preds, locs, "Active_IF", iterations=args.bootstrap)
    folds = pd.DataFrame(fold_metrics)

    preds.to_csv(out / "cross_runner_predictions.csv", index=False)
    locs.to_csv(out / "localization_predictions.csv", index=False)
    summary.to_csv(out / "metrics_with_ci.csv", index=False)
    diffs.to_csv(out / "paired_bootstrap_differences.csv", index=False)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    ineffective.to_csv(out / "ineffective_fault_events.csv", index=False)

    scen = preds.groupby(["method", "scenario_family"]).apply(
        lambda d: pd.Series({
            "n": len(d),
            "detection_recall_or_alert_rate": float(d.pred.mean()),
        }), include_groups=False
    ).reset_index()
    if len(locs):
        lscen = locs.groupby(["method", "scenario_family"]).agg(
            localization_top1=("top1", "mean"),
            localization_top2=("top2", "mean"),
            localization_mrr=("rr", "mean"),
            n_localization=("top1", "size"),
        ).reset_index()
        scen = scen.merge(lscen, on=["method", "scenario_family"], how="left")
    scen.to_csv(out / "scenario_metrics.csv", index=False)

    neg = preds[preds.label == 0].copy()
    neg["time_quartile"] = neg.groupby(["method", "replicate"]).cumcount()
    # Quartiles are based on event order within the original collected events.
    ev_order = events[["replicate", "event_id", "sequence", "scenario"]].copy()
    neg = neg.drop(columns=["scenario"], errors="ignore").merge(ev_order, on=["replicate", "event_id"], how="left")
    neg["time_quartile"] = neg.groupby(["method", "replicate"]).sequence.transform(
        lambda x: pd.qcut(x.rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    )
    drift = neg.groupby(["method", "scenario", "time_quartile"], observed=True).agg(
        false_positive_rate=("pred", "mean"), n=("pred", "size")
    ).reset_index()
    drift.to_csv(out / "negative_load_and_drift_fpr.csv", index=False)

    # Plots.
    order = [
        "AlwaysAnomaly", "BlackboxSLO", "RootProbeZ", "UERegistrationSLO",
        "BusinessSLO", "BusinessLatencyZ", "CombinedRules",
        "Passive_IF", "PassiveZone_IF", "Active_IF",
    ]
    ps = summary.set_index("method").reindex([m for m in order if m in set(summary.method)])
    ax = ps[["balanced_accuracy", "mcc", "macro_f1"]].plot(kind="bar", figsize=(13, 6))
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Cross-runner detection: balanced metrics")
    plt.tight_layout()
    plt.savefig(out / "fig_detection_balanced.png", dpi=180)
    plt.close()

    loc_ps = ps.dropna(subset=["localization_top1"])
    ax = loc_ps[["localization_top1", "localization_top2", "localization_mrr"]].plot(kind="bar", figsize=(12, 6))
    ax.axhline(1/3, linestyle="--", linewidth=1, label="random Top-1 = 1/3")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Localization on AUSF/UDM/UDR business-path faults")
    plt.tight_layout()
    plt.savefig(out / "fig_localization.png", dpi=180)
    plt.close()

    # Manuscript table.
    cols = [
        "method", "roc_auc", "balanced_accuracy", "mcc", "macro_f1", "sensitivity", "specificity",
        "fpr", "localization_top1", "localization_mrr"
    ]
    tab = summary[cols].copy()
    for c in cols[1:]:
        tab[c] = tab[c].map(lambda x: "NA" if pd.isna(x) else f"{x:.3f}")
    (out / "manuscript_table.md").write_text(tab.to_markdown(index=False), encoding="utf-8")

    active = summary[summary.method == "Active_IF"].iloc[0]
    business = summary[summary.method == "BusinessSLO"].iloc[0]
    reg = summary[summary.method == "UERegistrationSLO"].iloc[0]
    black = summary[summary.method == "BlackboxSLO"].iloc[0]
    passive = summary[summary.method == "PassiveZone_IF"].iloc[0]

    report = f"""# Revised real free5GC + UERANSIM experiment

## Why this experiment is different

The primary controlled faults are no longer Linux netem delays applied to the whole NF network namespace.
A dedicated h2c reverse proxy is registered in NRF for AUSF, UDM and UDR. The proxy **never delays the root /**
used by endpoint monitoring, but selectively affects non-root SBI business requests. Thus a root TCP/HTTP
probe can remain healthy while a real UE registration traverses the impaired business path.

A real UERANSIM canary UE registration is the active control transaction. Prometheus blackbox_exporter
v0.28.0 is run as a separate baseline. The experiment also includes a probe-aware evasion condition in which
the controlled fault explicitly exempts the canary SUPI while ordinary UE traffic remains impaired.

## Statistical design

- complete runner replicates: {len(reps)}
- all scenarios are randomly interleaved inside every runner;
- each held-out fold is an entire GitHub runner; models train only on legitimate events from the other runners;
- negative events include single-UE normal operation, 3-UE benign bursts and 5-UE benign heavy load;
- primary controlled-fault events that did not actually reach the configured business path are excluded and listed separately;
- Isolation Forest contamination is fixed at 0.05; no threshold is estimated from four validation points;
- confidence intervals use {args.bootstrap} stratified bootstrap resamples of held-out predictions;
- localization candidates are explicitly AUSF, UDM and UDR (random Top-1 baseline = 1/3);
- localization uses a separate one-class model per NF, with no NF identifier as an input feature.

## Zonal feature definition

Raw CPU/request values are **not compared directly across heterogeneous functions**. For every outer fold,
each NF and each passive feature is first normalized against that NF's own legitimate training distribution.
The robust scale is max(1.4826*MAD, IQR/1.349, epsilon), where
epsilon = max(1e-6, 0.01*max(|median|,1)). Robust z-scores are clipped to [-8,8].
Only after this normalization is the neighbor residual computed on the registration-path graph
AMF--AUSF--UDM--UDR. This removes the previous raw-median comparison across unlike functions.

## Headline metrics

**Active_IF**
- balanced accuracy: {active.balanced_accuracy:.3f} [{active.balanced_accuracy_ci_low:.3f}, {active.balanced_accuracy_ci_high:.3f}]
- MCC: {active.mcc:.3f} [{active.mcc_ci_low:.3f}, {active.mcc_ci_high:.3f}]
- sensitivity: {active.sensitivity:.3f}
- specificity: {active.specificity:.3f}
- FPR: {active.fpr:.3f}
- Top-1 localization: {active.localization_top1:.3f}
- localization MRR: {active.localization_mrr:.3f}

**Simple business-path SLO baseline**
- balanced accuracy: {business.balanced_accuracy:.3f}
- MCC: {business.mcc:.3f}
- Top-1 localization: {business.localization_top1:.3f}

**UE registration-time SLO**
- balanced accuracy: {reg.balanced_accuracy:.3f}
- MCC: {reg.mcc:.3f}

**Prometheus blackbox root SLO**
- balanced accuracy: {black.balanced_accuracy:.3f}
- MCC: {black.mcc:.3f}

**Passive + normalized-neighbor model**
- balanced accuracy: {passive.balanced_accuracy:.3f}
- MCC: {passive.mcc:.3f}

## Interpretation rule

If Active_IF is not materially better than BusinessSLO or CombinedRules under paired bootstrap confidence
intervals, the article must not claim an independent machine-learning contribution. In that case the supported
result is that a business-path active control transaction is valuable, while a simple rule is sufficient for
the tested faults. Conversely, an ML contribution is claimed only if the paired difference is positive with a
confidence interval that does not include zero on the pre-specified balanced metrics and/or localization.

The probe-aware evasion scenario is reported separately and bounds the independence claim: a canary that can
be identified by the adversary is not an independent source of evidence.

## Scope

The experiment is a real free5GC/UERANSIM control-plane testbed, but GitHub Actions is not a substitute for
day-scale operator drift. Cross-runner validation and benign load levels test environment and short-term load
variation; long-duration drift remains a stated limitation rather than an inferred result.
"""
    (out / "summary.md").write_text(report, encoding="utf-8")

    manifest = {
        "outer_validation": "leave-one-GitHub-runner-out",
        "replicates": reps,
        "bootstrap_iterations": args.bootstrap,
        "negative_classes": ["normal", "benign_burst", "benign_heavy_load"],
        "localization_candidates": TARGET_NODES,
        "random_localization_top1": 1/3,
        "iforest_contamination": 0.05,
        "zone_scale": "max(1.4826*MAD, IQR/1.349, epsilon); epsilon=max(1e-6,0.01*max(abs(median),1)); z clipped [-8,8]",
        "zone_graph": ADJ,
        "invalid_controlled_fault_events_excluded": len(ineffective),
        "workload_registration_latency_used_as_feature": False,
        "canary_registration_latency_used_as_feature_and_simple_baseline": True,
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(summary[["method", "balanced_accuracy", "mcc", "fpr", "localization_top1"]].sort_values("balanced_accuracy", ascending=False).to_string(index=False))
    print(report)


if __name__ == "__main__":
    main()
