import math
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

PASSIVE = ["cpu_pct", "mem_mib", "http_requests", "http_error_rate", "path_entropy"]
CANDIDATES = ["AMF", "AUSF", "UDM", "UDR"]
TOPOLOGY = {
    "AMF": ["AUSF", "UDM", "NSSF", "PCF", "SMF", "NRF"],
    "AUSF": ["AMF", "UDM", "NRF"],
    "UDM": ["AUSF", "UDR", "AMF", "NRF"],
    "UDR": ["UDM", "PCF"],
    "NSSF": ["AMF", "NRF"],
    "PCF": ["AMF", "UDR", "SMF", "NRF"],
    "SMF": ["AMF", "PCF", "UDM", "NRF"],
    "NRF": ["AMF", "AUSF", "UDM", "UDR", "NSSF", "PCF", "SMF"],
}
FLOORS = {
    "cpu_pct": 0.25,
    "mem_mib": 0.50,
    "http_requests": 1.0,
    "http_error_rate": 0.001,
    "path_entropy": 0.01,
    "blackbox_latency_ms": 0.10,
    "active_stage_ms": 0.10,
    "canary_registration_ms": 1.0,
}
ML_METHODS = ["Passive_IF", "PassiveTopo_IF", "ActiveML_IF"]

def robust_stats(values, floor):
    a = np.asarray([x for x in values if pd.notna(x)], dtype=float)
    if len(a) == 0:
        return 0.0, float(floor)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    scale = max(1.4826 * mad, 0.05 * abs(med), float(floor))
    return med, scale

def robust_z(value, stat):
    med, scale = stat
    if pd.isna(value):
        return float("nan")
    return (float(value) - med) / scale

def fit_stats(nodes, events, blocks):
    normal_ids = set(events[(events.block_id.isin(blocks)) & (events.scenario == "normal")].event_id)
    tr = nodes[nodes.event_id.isin(normal_ids)].copy()
    stats = {}
    for node in sorted(nodes.node.unique()):
        sub = tr[tr.node == node]
        stats[node] = {}
        for f in PASSIVE + ["blackbox_latency_ms", "active_stage_ms"]:
            stats[node][f] = robust_stats(sub[f].values, FLOORS[f])
    er = events[(events.block_id.isin(blocks)) & (events.scenario == "normal")]
    estats = {
        "canary_registration_ms": robust_stats(
            er.canary_registration_ms.values, FLOORS["canary_registration_ms"]
        )
    }
    return stats, estats

def transform(nodes, events, stats, estats):
    d = nodes.copy()
    for f in PASSIVE + ["blackbox_latency_ms", "active_stage_ms"]:
        d[f"z_{f}"] = [
            robust_z(v, stats[node][f]) for node, v in zip(d.node, d[f])
        ]
    d["z_blackbox_latency_ms"] = d["z_blackbox_latency_ms"].fillna(0.0)
    d["z_active_stage_ms"] = d["z_active_stage_ms"].fillna(0.0)
    d["stage_missing"] = 1 - d["active_stage_present"].astype(int)

    topo_rows = []
    for event_id, g in d.groupby("event_id", sort=False):
        by = g.set_index("node")
        for idx, r in g.iterrows():
            neigh = [n for n in TOPOLOGY[r.node] if n in by.index]
            disagreement = []
            peer_activity = []
            for f in PASSIVE:
                own = float(r[f"z_{f}"])
                peer = np.asarray([float(by.loc[n, f"z_{f}"]) for n in neigh], dtype=float)
                if len(peer):
                    pm = float(np.median(peer))
                    disagreement.append(abs(own - pm))
                    peer_activity.append(float(np.median(np.abs(peer))))
            topo_rows.append((
                idx,
                float(np.median(disagreement)) if disagreement else 0.0,
                float(np.median(peer_activity)) if peer_activity else 0.0,
            ))
    topo = pd.DataFrame(topo_rows, columns=["idx", "topo_disagreement", "topo_peer_activity"]).set_index("idx")
    d.loc[topo.index, "topo_disagreement"] = topo.topo_disagreement
    d.loc[topo.index, "topo_peer_activity"] = topo.topo_peer_activity

    emed, escale = estats["canary_registration_ms"]
    ez = events[["event_id", "canary_registration_ms", "canary_success"]].copy()
    ez["z_canary_registration"] = (ez.canary_registration_ms - emed) / escale
    fail = ez.canary_success == 0
    ez.loc[fail, "z_canary_registration"] = np.maximum(
        ez.loc[fail, "z_canary_registration"], 10.0
    )
    d = d.merge(ez[["event_id", "z_canary_registration"]], on="event_id", how="left")
    return d, ez[["event_id", "z_canary_registration"]]

def feature_columns(method):
    base = [f"z_{f}" for f in PASSIVE]
    if method == "Passive_IF":
        return base
    if method == "PassiveTopo_IF":
        return base + ["topo_disagreement", "topo_peer_activity"]
    if method == "ActiveML_IF":
        return base + [
            "topo_disagreement",
            "topo_peer_activity",
            "z_blackbox_latency_ms",
            "z_active_stage_ms",
            "stage_missing",
            "z_canary_registration",
        ]
    raise ValueError(method)

def fit_models(transformed, events, blocks, method):
    normal_ids = set(events[(events.block_id.isin(blocks)) & (events.scenario == "normal")].event_id)
    tr = transformed[transformed.event_id.isin(normal_ids)]
    cols = feature_columns(method)
    models = {}
    for node in sorted(transformed.node.unique()):
        X = tr[tr.node == node][cols].fillna(0.0).values
        if len(X) < 8:
            raise RuntimeError(f"insufficient normal rows for {node}: {len(X)}")
        models[node] = IsolationForest(
            n_estimators=120,
            max_samples=min(64, len(X)),
            contamination="auto",
            random_state=42,
            n_jobs=2,
        ).fit(X)
    return models

def score_models(transformed, models, method):
    cols = feature_columns(method)
    rows = []
    for node, model in models.items():
        sub = transformed[transformed.node == node].copy()
        X = sub[cols].fillna(0.0).values
        sub["node_score"] = -model.decision_function(X)
        rows.append(sub[[
            "event_id", "block_id", "scenario", "target_node",
            "primary_localization", "node", "node_score"
        ]])
    return pd.concat(rows, ignore_index=True)

def event_scores(node_scores):
    return node_scores.groupby("event_id").node_score.max().rename("score").reset_index()

def crossfit_calibration(nodes, events, outer_train_blocks, method):
    scores = []
    for held in outer_train_blocks:
        fit_blocks = [b for b in outer_train_blocks if b != held]
        stats, estats = fit_stats(nodes, events, fit_blocks)
        t, _ = transform(nodes, events, stats, estats)
        models = fit_models(t, events, fit_blocks, method)
        held_t = t[t.block_id == held]
        es = event_scores(score_models(held_t, models, method))
        normal_ids = set(events[(events.block_id == held) & (events.scenario == "normal")].event_id)
        scores.extend(es[es.event_id.isin(normal_ids)].score.tolist())
    return np.asarray(scores, dtype=float)

def simple_scores(test_t, event_z, test_events, stats):
    rows = []
    for event_id, g in test_t.groupby("event_id"):
        cand = g[g.node.isin(CANDIDATES)].copy()

        bb = cand.z_blackbox_latency_ms.astype(float).copy()
        bad = cand.blackbox_success.values == 0
        if bad.any():
            arr = bb.values.copy()
            arr[bad] = np.maximum(arr[bad], 10.0)
            bb = pd.Series(arr, index=bb.index)
        bb_score = float(bb.max())
        bb_node = cand.loc[bb.idxmax(), "node"]

        stage = cand.z_active_stage_ms.astype(float)
        stage_score = float(stage.max())
        stage_node = cand.loc[stage.idxmax(), "node"]

        # Standard blackbox/SLO-style alert. Each NF has its own normal baseline.
        # The alert threshold is median + max(10 ms, 4 robust scales), so a near-zero
        # MAD cannot create an unrealistically tight availability SLO.
        slo_arr = []
        for _, r in cand.iterrows():
            _, scale = stats[r.node]["blackbox_latency_ms"]
            threshold_z = max(4.0, 10.0 / scale)
            violation = float(r.z_blackbox_latency_ms) - threshold_z
            if int(r.blackbox_success) == 0:
                violation = max(violation, 10.0)
            slo_arr.append(violation)
        slo_arr = np.asarray(slo_arr, dtype=float)
        slo_score = float(np.max(slo_arr))
        slo_node = cand.iloc[int(np.argmax(slo_arr))].node

        cz = float(event_z[event_z.event_id == event_id].z_canary_registration.iloc[0])

        rows.extend([
            {"event_id": event_id, "method": "Blackbox_Z", "score": bb_score, "pred": int(bb_score > 3.5), "pred_node": bb_node},
            {"event_id": event_id, "method": "Blackbox_SLO", "score": slo_score, "pred": int(slo_score > 0.0), "pred_node": slo_node},
            {"event_id": event_id, "method": "Canary_E2E_Z", "score": cz, "pred": int(cz > 3.5), "pred_node": ""},
            {"event_id": event_id, "method": "ActiveStage_Z", "score": stage_score, "pred": int(stage_score > 3.5), "pred_node": stage_node},
        ])
    return pd.DataFrame(rows)

def outer_fold(nodes, events, test_block):
    train_blocks = [b for b in sorted(events.block_id.unique()) if b != test_block]
    stats, estats = fit_stats(nodes, events, train_blocks)
    transformed, event_z = transform(nodes, events, stats, estats)
    test_t = transformed[transformed.block_id == test_block].copy()
    test_e = events[events.block_id == test_block].copy()

    parts = [simple_scores(test_t, event_z, test_e, stats)]
    node_parts = []
    for method in ML_METHODS:
        cal = crossfit_calibration(nodes, events, train_blocks, method)
        threshold = float(np.quantile(cal, 0.95))
        models = fit_models(transformed, events, train_blocks, method)
        ns = score_models(test_t, models, method)
        es = event_scores(ns)
        es["method"] = method
        es["pred"] = (es.score > threshold).astype(int)

        cand = ns[ns.node.isin(CANDIDATES)]
        pred_node = (
            cand.sort_values(["event_id", "node_score"], ascending=[True, False])
            .groupby("event_id")
            .first()
            .node.rename("pred_node")
            .reset_index()
        )
        es = es.merge(pred_node, on="event_id", how="left")
        parts.append(es[["event_id", "method", "score", "pred", "pred_node"]])
        ns["method"] = method
        node_parts.append(ns)

    pred = pd.concat(parts, ignore_index=True)
    pred = pred.merge(
        test_e[[
            "event_id", "block_id", "scenario", "is_anomaly",
            "primary_localization", "target_node", "severity_ms", "load_ues"
        ]],
        on="event_id",
        how="left",
    )
    return pred, pd.concat(node_parts, ignore_index=True)
