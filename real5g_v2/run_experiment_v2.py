#!/usr/bin/env python3
import argparse
import collections
import concurrent.futures
import datetime as dt
import json
import math
import random
import re
import statistics
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from real5g.run_real_testbed import (
    NODES,
    PLMN,
    api_post,
    container_pid,
    docker_stats,
    get_logs_since,
    make_supi,
    make_ue_yaml,
    sh,
    start_ue,
    stop_ue,
    strip_ansi,
    subscriber_payload,
    wait_for_gnb,
    wait_http,
)

CANDIDATE_NODES = ["AMF", "AUSF", "UDM", "UDR"]
GIN_RE = re.compile(r"\|\s*(\d{3})\s*\|\s*([^|]+?)\s*\|\s*(GET|POST|PUT|PATCH|DELETE)\s*\|\s*([^|]+?)\s*\|")
ISO_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.(\d+)Z")
UE_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")

def docker_ip(container):
    p = sh(["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", container])
    return p.stdout.strip()

def write_ue(outdir, supi):
    p = outdir / "ue_configs" / f"{supi}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(make_ue_yaml(supi), encoding="utf-8")
    remote = f"/tmp/{supi}.yaml"
    sh(["docker", "cp", str(p), f"ueransim:{remote}"])
    return remote

def amf_completed_supis(start_epoch, supis):
    txt = get_logs_since("amf", start_epoch)
    done = set()
    for line in txt.splitlines():
        if "Handle Registration Complete" not in line:
            continue
        for supi in supis:
            if f"SUPI:{supi}" in line:
                done.add(supi)
    return done

def ue_log_text(log_path):
    p = sh(["docker", "exec", "ueransim", "sh", "-lc", f"cat {log_path} 2>/dev/null || true"], check=False)
    return strip_ansi(p.stdout or "")

def start_many(supis, configs):
    handles = {}
    for supi in supis:
        handles[supi] = start_ue(supi, configs[supi])
    return handles

def stop_many(handles):
    for _, (_, _, pidf) in handles.items():
        stop_ue(pidf)

def wait_many(supis, handles, start_perf, start_epoch, timeout_s):
    completion = {}
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        for supi in amf_completed_supis(start_epoch, supis):
            if supi not in completion:
                completion[supi] = (time.perf_counter() - start_perf) * 1000.0
        if len(completion) == len(supis):
            break
        time.sleep(0.08)
    logs = {s: ue_log_text(handles[s][1]) for s in supis}
    for supi, txt in logs.items():
        if supi not in completion and "Initial Registration is successful" in txt:
            completion[supi] = (time.perf_counter() - start_perf) * 1000.0
    return completion, logs

def run_background_burst(supis, configs, concurrent=False):
    if not supis:
        return {}, {}, 1.0, 0.0
    start_epoch = time.time()
    start_perf = time.perf_counter()
    handles = start_many(supis, configs)
    if concurrent:
        return handles, {}, float("nan"), float("nan")
    completion, logs = wait_many(supis, handles, start_perf, start_epoch, timeout_s=4.0)
    success = len(completion) / len(supis)
    med = float(statistics.median(completion.values())) if completion else 4000.0
    return handles, logs, success, med

def run_canary(supi, config, timeout_s=5.0):
    start_epoch = time.time()
    start_perf = time.perf_counter()
    handle = start_ue(supi, config)
    handles = {supi: handle}
    completion, logs = wait_many([supi], handles, start_perf, start_epoch, timeout_s)
    txt = logs[supi]
    success = supi in completion or "Initial Registration is successful" in txt
    latency = completion.get(supi, timeout_s * 1000.0)
    return handle, success, float(latency), txt, start_epoch

def parse_prometheus(text):
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        key, value = line.rsplit(" ", 1)
        base = key.split("{", 1)[0]
        if base in {"probe_success", "probe_duration_seconds", "probe_http_status_code"}:
            try:
                out[base] = float(value)
            except ValueError:
                pass
    return out

def blackbox_one(alias):
    target = f"http://{alias}:8000/"
    url = "http://127.0.0.1:9115/probe?" + urllib.parse.urlencode(
        {"target": target, "module": "http_liveness"}
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=2.5) as r:
            txt = r.read().decode()
        m = parse_prometheus(txt)
        return {
            "latency_ms": 1000.0 * m.get("probe_duration_seconds", time.perf_counter() - t0),
            "success": int(m.get("probe_success", 0.0) > 0.5),
            "status": int(m.get("probe_http_status_code", 0.0)),
        }
    except Exception:
        return {"latency_ms": 2500.0, "success": 0, "status": 0}

def blackbox_all():
    aliases = {node: f"{container}.free5gc.org" for node, container in NODES.items()}
    aliases["NRF"] = "nrf.free5gc.org"
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(aliases)) as ex:
        fut = {ex.submit(blackbox_one, alias): node for node, alias in aliases.items()}
        for f in concurrent.futures.as_completed(fut):
            out[fut[f]] = f.result()
    return out

def selective_tcp_response_delay(src_container, dst_container, delay_ms):
    dst_ip = docker_ip(dst_container)
    pid = container_pid(src_container)
    sh(f"sudo nsenter -t {pid} -n tc qdisc replace dev eth0 root handle 1: prio")
    sh(f"sudo nsenter -t {pid} -n tc qdisc replace dev eth0 parent 1:3 handle 30: netem delay {delay_ms}ms")
    sh(
        f"sudo nsenter -t {pid} -n tc filter replace dev eth0 protocol ip parent 1: prio 3 "
        f"u32 match ip dst {dst_ip}/32 match ip sport 8000 0xffff flowid 1:3"
    )

def selective_sctp_delay(src_container, dst_container, delay_ms):
    dst_ip = docker_ip(dst_container)
    pid = container_pid(src_container)
    sh(f"sudo nsenter -t {pid} -n tc qdisc replace dev eth0 root handle 1: prio")
    sh(f"sudo nsenter -t {pid} -n tc qdisc replace dev eth0 parent 1:3 handle 30: netem delay {delay_ms}ms")
    sh(
        f"sudo nsenter -t {pid} -n tc filter replace dev eth0 protocol ip parent 1: prio 3 "
        f"u32 match ip dst {dst_ip}/32 match ip protocol 132 0xff flowid 1:3"
    )

def clear_tc(container):
    try:
        pid = container_pid(container)
        sh(f"sudo nsenter -t {pid} -n tc qdisc del dev eth0 root", check=False)
    except Exception:
        pass

def apply_scenario(scenario, severity_ms):
    if scenario == "selective_ausf_delay":
        selective_tcp_response_delay("ausf", "amf", severity_ms)
        return ("tc", "ausf")
    if scenario == "selective_udm_delay":
        selective_tcp_response_delay("udm", "ausf", severity_ms)
        return ("tc", "udm")
    if scenario == "selective_udr_delay":
        selective_tcp_response_delay("udr", "udm", severity_ms)
        return ("tc", "udr")
    if scenario == "selective_amf_delay":
        selective_sctp_delay("amf", "ueransim", severity_ms)
        return ("tc", "amf")
    if scenario == "udm_outage":
        sh(["docker", "pause", "udm"])
        return ("pause", "udm")
    return None

def clear_scenario(handle):
    if not handle:
        return
    kind, container = handle
    if kind == "pause":
        sh(["docker", "unpause", container], check=False)
        time.sleep(0.5)
    else:
        clear_tc(container)
        time.sleep(0.15)

def parse_nf_ts(line):
    m = ISO_RE.match(line)
    if not m:
        return None
    frac = (m.group(2) + "000000")[:6]
    s = f"{m.group(1)}.{frac}+00:00"
    return dt.datetime.fromisoformat(s).timestamp() * 1000.0

def parse_ue_ts(line):
    m = UE_TS_RE.match(line)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp() * 1000.0

def last_time(lines, predicate, before=None):
    xs = []
    for line in lines:
        if not predicate(line):
            continue
        t = parse_nf_ts(line)
        if t is None:
            continue
        if before is None or t <= before:
            xs.append(t)
    return max(xs) if xs else None

def first_time(lines, predicate, after=None):
    xs = []
    for line in lines:
        if not predicate(line):
            continue
        t = parse_nf_ts(line)
        if t is None:
            continue
        if after is None or t >= after:
            xs.append(t)
    return min(xs) if xs else None

def active_stage_latencies(logs, ue_log, supi):
    suffix = supi.split("20893", 1)[-1]
    suci = f"suci-0-208-93-0000-0-0-{suffix}"
    amf = logs.get("AMF", "").splitlines()
    ausf = logs.get("AUSF", "").splitlines()
    udm = logs.get("UDM", "").splitlines()
    udr = logs.get("UDR", "").splitlines()

    # Final successful UDM authentication-data response.
    t_udm = last_time(
        udm,
        lambda l: suci in l and "generate-auth-data" in l and "[GIN]" in l,
    )
    # AUSF learns the SUCI/SUPI mapping only after receiving the UDM response.
    t_ausf_ready = last_time(
        ausf,
        lambda l: suci in l and "Add SuciSupiPair" in l,
    )
    # UDR response that immediately precedes the final UDM generate-auth-data completion.
    t_udr = last_time(
        udr,
        lambda l: ("[GIN]" in l and (supi in l or suci in l)),
        before=t_udm,
    )
    # AMF sends the authentication challenge after receiving the AUSF response.
    t_amf_auth = first_time(
        amf,
        lambda l: "Send Authentication Request" in l,
        after=t_ausf_ready,
    )
    # Registration Accept is the final AMF->UE control-plane response.
    t_amf_accept = last_time(
        amf,
        lambda l: f"SUPI:{supi}" in l and "Send Registration Accept" in l,
    )
    t_amf_complete = last_time(
        amf,
        lambda l: f"SUPI:{supi}" in l and "Handle Registration Complete" in l,
    )

    def gap(a, b):
        if a is None or b is None or b < a:
            return float("nan")
        return float(b - a)

    return {
        "UDR": gap(t_udr, t_udm),
        "UDM": gap(t_udm, t_ausf_ready),
        "AUSF": gap(t_ausf_ready, t_amf_auth),
        "AMF": gap(t_amf_accept, t_amf_complete),
    }

def summarize_business_logs(txt):
    statuses = []
    paths = []
    for line in txt.splitlines():
        m = GIN_RE.search(line)
        if not m:
            continue
        status = int(m.group(1))
        path = m.group(4).strip()
        if path == "/":
            continue
        statuses.append(status)
        paths.append(path.split("?", 1)[0])
    n = len(statuses)
    errors = sum(s >= 400 for s in statuses)
    if paths:
        c = collections.Counter(paths)
        total = sum(c.values())
        entropy = -sum((v / total) * math.log(v / total + 1e-12) for v in c.values())
    else:
        entropy = 0.0
    return {
        "http_requests": n,
        "http_error_rate": errors / n if n else 0.0,
        "path_entropy": entropy,
    }

def avg_stats(a, b, container):
    va = a.get(container, {"cpu_pct": 0.0, "mem_mib": 0.0})
    vb = b.get(container, {"cpu_pct": 0.0, "mem_mib": 0.0})
    return {
        "cpu_pct": (va["cpu_pct"] + vb["cpu_pct"]) / 2.0,
        "mem_mib": (va["mem_mib"] + vb["mem_mib"]) / 2.0,
    }

def event_specifications(block_id, severity):
    normal = [
        {"scenario": "normal", "target_node": "", "severity_ms": 0, "load_ues": k, "primary_localization": 0}
        for k in range(6)
    ]
    anomalous = [
        {"scenario": "selective_amf_delay", "target_node": "AMF", "severity_ms": severity, "load_ues": (block_id + 1) % 6, "primary_localization": 1},
        {"scenario": "selective_ausf_delay", "target_node": "AUSF", "severity_ms": severity, "load_ues": (block_id + 2) % 6, "primary_localization": 1},
        {"scenario": "selective_udm_delay", "target_node": "UDM", "severity_ms": severity, "load_ues": (block_id + 3) % 6, "primary_localization": 1},
        {"scenario": "selective_udr_delay", "target_node": "UDR", "severity_ms": severity, "load_ues": (block_id + 4) % 6, "primary_localization": 1},
        {"scenario": "registration_storm", "target_node": "AMF", "severity_ms": 0, "load_ues": 8, "primary_localization": 0},
        {"scenario": "udm_outage", "target_node": "UDM", "severity_ms": 0, "load_ues": block_id % 3, "primary_localization": 0},
    ]
    specs = normal + anomalous
    random.Random(1000 + block_id).shuffle(specs)
    return specs

def allocate_supis(pool, cursor, count):
    if cursor + count > len(pool):
        raise RuntimeError("subscriber pool exhausted")
    return pool[cursor:cursor + count], cursor + count

def run_event(event_id, block_id, order_idx, spec, supi_pool, cursor, configs, outdir):
    scenario = spec["scenario"]
    severity = int(spec["severity_ms"])
    load_ues = int(spec["load_ues"])
    concurrent_bg = scenario == "registration_storm"

    need = 1 + load_ues
    allocated, cursor = allocate_supis(supi_pool, cursor, need)
    canary = allocated[0]
    background = allocated[1:]

    raw_dir = outdir / "raw_logs" / f"block_{block_id:02d}_event_{event_id:04d}_{scenario}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    impairment = apply_scenario(scenario, severity)
    event_start = time.time()
    try:
        bb0 = blackbox_all()
        stats0 = docker_stats()

        bg_handles, bg_logs, bg_success, bg_latency = run_background_burst(background, configs, concurrent=concurrent_bg)
        if not concurrent_bg:
            stop_many(bg_handles)
            time.sleep(0.06)
            bg_handles = {}

        canary_handle, canary_success, canary_latency, canary_log, canary_start = run_canary(
            canary, configs[canary], timeout_s=5.5 if scenario == "udm_outage" else 4.5
        )

        if concurrent_bg:
            bg_start_epoch = event_start
            bg_start_perf = time.perf_counter() - min(3.5, (time.time() - event_start))
            completion, bg_logs = wait_many(background, bg_handles, bg_start_perf, bg_start_epoch, timeout_s=2.5)
            bg_success = len(completion) / len(background) if background else 1.0
            bg_latency = float(statistics.median(completion.values())) if completion else 2500.0
            stop_many(bg_handles)

        stop_ue(canary_handle[2])
        time.sleep(0.04)
        stats1 = docker_stats()
        bb1 = blackbox_all()
        event_end = time.time()

        logs = {}
        for node, container in NODES.items():
            txt = get_logs_since(container, event_start)
            logs[node] = txt
            (raw_dir / f"{container}.log").write_text(txt, encoding="utf-8")
        (raw_dir / f"{canary}_canary_ue.log").write_text(canary_log, encoding="utf-8")
        for s, txt in bg_logs.items():
            (raw_dir / f"{s}_background_ue.log").write_text(txt, encoding="utf-8")

        stages = active_stage_latencies(logs, canary_log, canary)

        node_rows = []
        for node, container in NODES.items():
            passive = summarize_business_logs(logs[node])
            ds = avg_stats(stats0, stats1, container)
            b0 = bb0.get(node, {"latency_ms": 2500.0, "success": 0, "status": 0})
            b1 = bb1.get(node, b0)
            node_rows.append({
                "event_id": event_id,
                "block_id": block_id,
                "order_in_block": order_idx,
                "scenario": scenario,
                "is_anomaly": int(scenario != "normal"),
                "primary_localization": int(spec["primary_localization"]),
                "target_node": spec["target_node"],
                "severity_ms": severity,
                "load_ues": load_ues,
                "node": node,
                "cpu_pct": ds["cpu_pct"],
                "mem_mib": ds["mem_mib"],
                **passive,
                "blackbox_latency_ms": (b0["latency_ms"] + b1["latency_ms"]) / 2.0,
                "blackbox_success": int(b0["success"] and b1["success"]),
                "blackbox_http_status": int(b1["status"]),
                "active_stage_ms": stages.get(node, float("nan")),
                "active_stage_present": int(node in CANDIDATE_NODES and not math.isnan(stages.get(node, float("nan")))),
                "canary_registration_ms": canary_latency,
                "canary_success": int(canary_success),
            })

        event_row = {
            "event_id": event_id,
            "block_id": block_id,
            "order_in_block": order_idx,
            "scenario": scenario,
            "is_anomaly": int(scenario != "normal"),
            "primary_localization": int(spec["primary_localization"]),
            "target_node": spec["target_node"],
            "severity_ms": severity,
            "load_ues": load_ues,
            "canary_supi": canary,
            "canary_success": int(canary_success),
            "canary_registration_ms": canary_latency,
            "background_success_rate": float(bg_success),
            "background_registration_ms": float(bg_latency),
            "duration_s": event_end - event_start,
            "started_at_utc": dt.datetime.fromtimestamp(event_start, tz=dt.timezone.utc).isoformat(),
        }
        return node_rows, event_row, cursor
    finally:
        clear_scenario(impairment)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--subscriber-pool", type=int, default=480)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    wait_for_gnb()

    login = wait_http("http://127.0.0.1:5000/api/login")
    token = login["access_token"]

    supis = [make_supi(1000 + i) for i in range(args.subscriber_pool)]
    print(f"Provisioning {len(supis)} subscribers for blocked/interleaved experiment")
    for i, supi in enumerate(supis):
        status, body = api_post(
            f"http://127.0.0.1:5000/api/subscriber/{supi}/{PLMN}",
            token,
            subscriber_payload(supi, f"091{i:07d}"),
        )
        if status not in (200, 201, 409):
            raise RuntimeError(f"subscriber provisioning failed: {status} {body}")

    configs = {s: write_ue(outdir, s) for s in supis}
    severities = [5, 10, 15, 20, 30, 40, 60, 80]

    all_nodes = []
    all_events = []
    cursor = 0
    event_id = 1

    for block in range(args.blocks):
        severity = severities[block % len(severities)]
        specs = event_specifications(block, severity)
        for order_idx, spec in enumerate(specs):
            rows, ev, cursor = run_event(
                event_id, block, order_idx, spec, supis, cursor, configs, outdir
            )
            all_nodes.extend(rows)
            all_events.append(ev)
            print(
                f"block={block} event={event_id} scenario={spec['scenario']} "
                f"target={spec['target_node'] or '-'} severity={spec['severity_ms']}ms "
                f"load={spec['load_ues']} canary_success={ev['canary_success']} "
                f"latency={ev['canary_registration_ms']:.1f}ms"
            )
            event_id += 1

    ndf = pd.DataFrame(all_nodes)
    edf = pd.DataFrame(all_events)
    ndf.to_csv(outdir / "nodes_v2.csv", index=False)
    edf.to_csv(outdir / "events_v2.csv", index=False)

    order = edf[
        ["event_id", "block_id", "order_in_block", "scenario", "target_node", "severity_ms", "load_ues"]
    ].copy()
    order.to_csv(outdir / "event_order_v2.csv", index=False)

    manifest = {
        "design": "8 blocked campaigns in one GitHub Actions job and one runner; 6 normal + 6 anomalous events per block; order shuffled reproducibly within each block",
        "primary_selective_scenarios": [
            "selective_amf_delay",
            "selective_ausf_delay",
            "selective_udm_delay",
            "selective_udr_delay",
        ],
        "selective_impairment": "egress-only tc/netem filter to the legitimate business caller; root HTTP liveness probe path is not matched",
        "blackbox": "Prometheus blackbox_exporter v0.28.0, HTTP liveness module accepting the free5GC root 404 as reachable",
        "active_control": "dedicated UE initial registration traversing real NGAP/NAS and the free5GC business service chain; per-NF stage gaps are reconstructed from canary-specific logs",
        "normal_loads": [0, 1, 2, 3, 4, 5],
        "severity_ms_by_block": severities[:args.blocks],
        "candidate_localization_nodes": CANDIDATE_NODES,
        "oauth_note": "the active control transaction uses the normal free5GC control-plane path; it does not directly call protected NF business APIs from the probe process",
        "mtls_note": "mTLS/service-mesh termination is not enabled by the upstream free5GC compose testbed and remains an external-validity limitation",
    }
    (outdir / "collection_manifest_v2.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Collection complete:", len(edf), "events and", len(ndf), "node observations")

if __name__ == "__main__":
    main()
