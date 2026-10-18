#!/usr/bin/env python3
import argparse
import collections
import datetime as dt
import json
import math
import random
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

PASSIVE_NODES = {
    "AMF": "amf",
    "AUSF": "ausf",
    "UDM": "udm",
    "UDR": "udr",
}
PROXY_NODES = {
    "AUSF": ("sbi-proxy-ausf", "ausf-proxy.free5gc.org"),
    "UDM": ("sbi-proxy-udm", "udm-proxy.free5gc.org"),
    "UDR": ("sbi-proxy-udr", "udr-proxy.free5gc.org"),
}
TARGET_NODES = list(PROXY_NODES)
PLMN = "20893"
KEY = "8baf473f2f8fd09487cccbd7097c6862"
OP = "8e27b6af0e692e750f32667a3b14605d"
GIN_RE = re.compile(r"\|\s*(\d{3})\s*\|.*?\|\s*(GET|POST|PUT|PATCH|DELETE)\s*\|\s*([^|]+?)\s*\|")


def sh(cmd, check=True, timeout=None):
    p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p


def http_json(url, method="GET", obj=None, timeout=5):
    data = None if obj is None else json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data, method=method)
    if obj is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_webui(timeout_s=90):
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:5000/api/login",
                data=json.dumps({"username": "admin", "password": "free5gc"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3) as r:
                if r.status == 200:
                    return json.loads(r.read().decode())
        except Exception as e:
            last = repr(e)
        time.sleep(2)
    raise RuntimeError(f"webconsole did not become ready: {last}")


def api_post(url, token, obj):
    req = urllib.request.Request(
        url,
        data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json", "Token": token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        if e.code == 409:
            return e.code, body
        raise RuntimeError(f"POST {url} -> {e.code}: {body}") from e


def subscriber_payload(supi, msisdn):
    return {
        "plmnID": PLMN,
        "ueId": supi,
        "AuthenticationSubscription": {
            "authenticationManagementField": "8000",
            "authenticationMethod": "5G_AKA",
            "milenage": {"op": {"encryptionAlgorithm": 0, "encryptionKey": 0, "opValue": OP}},
            "opc": {"encryptionAlgorithm": 0, "encryptionKey": 0, "opcValue": ""},
            "permanentKey": {
                "encryptionAlgorithm": 0,
                "encryptionKey": 0,
                "permanentKeyValue": KEY,
            },
            "sequenceNumber": "000000000000",
        },
        "AccessAndMobilitySubscriptionData": {
            "gpsis": [f"msisdn-{msisdn}"],
            "nssai": {
                "defaultSingleNssais": [{"sst": 1, "sd": "010203", "isDefault": True}],
                "singleNssais": [],
            },
            "subscribedUeAmbr": {"downlink": "2 Gbps", "uplink": "1 Gbps"},
        },
        "SessionManagementSubscriptionData": [{
            "singleNssai": {"sst": 1, "sd": "010203"},
            "dnnConfigurations": {
                "internet": {
                    "sscModes": {"defaultSscMode": "SSC_MODE_1", "allowedSscModes": ["SSC_MODE_2", "SSC_MODE_3"]},
                    "pduSessionTypes": {"defaultSessionType": "IPV4", "allowedSessionTypes": ["IPV4"]},
                    "sessionAmbr": {"uplink": "200 Mbps", "downlink": "100 Mbps"},
                    "5gQosProfile": {"5qi": 9, "arp": {"priorityLevel": 8}, "priorityLevel": 8},
                }
            },
        }],
        "SmfSelectionSubscriptionData": {
            "subscribedSnssaiInfos": {"01010203": {"dnnInfos": [{"dnn": "internet"}]}}
        },
        "AmPolicyData": {"subscCats": ["free5gc"]},
        "SmPolicyData": {
            "smPolicySnssaiData": {
                "01010203": {
                    "snssai": {"sst": 1, "sd": "010203"},
                    "smPolicyDnnData": {"internet": {"dnn": "internet"}},
                }
            }
        },
        "FlowRules": [],
        "QosFlows": [],
    }


def make_supi(replicate, index):
    return f"imsi-20893{replicate:02d}{index:08d}"


def make_ue_yaml(supi):
    return f'''supi: "{supi}"
mcc: "208"
mnc: "93"
key: "{KEY}"
op: "{OP}"
opType: "OP"
amf: "8000"
imei: "356938035643803"
imeiSv: "4370816125816151"
gnbSearchList:
  - 127.0.0.1
uacAic:
  mps: false
  mcs: false
uacAcc:
  normalClass: 0
  class11: false
  class12: false
  class13: false
  class14: false
  class15: false
sessions: []
configured-nssai:
  - sst: 0x01
    sd: 0x010203
default-nssai:
  - sst: 1
    sd: 0x010203
integrity:
  IA1: true
  IA2: true
  IA3: true
ciphering:
  EA1: true
  EA2: true
  EA3: true
integrityMaxRate:
  uplink: "full"
  downlink: "full"
'''


def strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def parse_mem_to_mib(value):
    left = value.split("/")[0].strip()
    m = re.match(r"([0-9.]+)\s*([KMGTP]?i?B)", left, re.I)
    if not m:
        return 0.0
    num = float(m.group(1))
    unit = m.group(2).lower()
    factor = {
        "kb": 1/1024, "kib": 1/1024, "mb": 1.0, "mib": 1.0,
        "gb": 1024.0, "gib": 1024.0, "tb": 1024.0*1024, "tib": 1024.0*1024,
        "b": 1/(1024*1024),
    }.get(unit, 1.0)
    return num * factor


def docker_stats():
    names = " ".join(PASSIVE_NODES.values())
    p = sh(f"docker stats --no-stream --format '{{{{json .}}}}' {names}", check=False, timeout=15)
    out = {}
    for line in p.stdout.splitlines():
        try:
            d = json.loads(line)
            out[d.get("Name", "")] = {
                "cpu_pct": float(d.get("CPUPerc", "0").strip("%") or 0),
                "mem_mib": parse_mem_to_mib(d.get("MemUsage", "0B / 0B")),
            }
        except Exception:
            continue
    return out


def get_logs_since(container, start_epoch):
    p = sh(["docker", "logs", "--since", str(int(start_epoch)), container], check=False, timeout=10)
    return strip_ansi((p.stdout or "") + "\n" + (p.stderr or ""))


def summarize_logs(txt):
    status, paths = [], []
    for line in txt.splitlines():
        m = GIN_RE.search(line)
        if m:
            status.append(int(m.group(1)))
            paths.append(m.group(3).strip())
    n = len(status)
    err = sum(s >= 400 for s in status)
    c = collections.Counter(paths)
    total = sum(c.values())
    entropy = -sum((v/total) * math.log(v/total + 1e-12) for v in c.values()) if total else 0.0
    return {
        "http_requests": n,
        "http_error_rate": (err/n) if n else 0.0,
        "path_entropy": entropy,
    }


_PROXY_IPS = {}


def container_ip(name):
    if name not in _PROXY_IPS:
        p = sh(["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name])
        _PROXY_IPS[name] = p.stdout.strip()
    return _PROXY_IPS[name]


def set_proxy(node, event_id, mode="none", delay_ms=0, jitter_ms=0,
              error_every_n=0, exclude_token="", fault_rate=0.0):
    container, _ = PROXY_NODES[node]
    ip = container_ip(container)
    return http_json(
        f"http://{ip}:9901/control",
        "POST",
        {
            "event_id": str(event_id),
            "mode": mode,
            "delay_ms": delay_ms,
            "jitter_ms": jitter_ms,
            "error_every_n": error_every_n,
            "exclude_token": exclude_token,
            "fault_rate": fault_rate,
        },
    )


def set_all_none(event_id):
    for node in TARGET_NODES:
        set_proxy(node, event_id, mode="none")


def proxy_stats(node, event_id):
    container, _ = PROXY_NODES[node]
    ip = container_ip(container)
    return http_json(f"http://{ip}:9901/stats?event_id={urllib.parse.quote(str(event_id))}")


def blackbox_probe(node):
    if node in PROXY_NODES:
        _, host = PROXY_NODES[node]
    else:
        host = f"{PASSIVE_NODES[node]}.free5gc.org"
    target = f"http://{host}:8000/"
    qs = urllib.parse.urlencode({"target": target, "module": "http_sbi_root"})
    url = f"http://127.0.0.1:9115/probe?{qs}"
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            text = r.read().decode()
    except Exception:
        return {"success": 0.0, "duration_ms": 4000.0}
    vals = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("probe_success", "probe_duration_seconds"):
            try:
                vals[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return {
        "success": vals.get("probe_success", 0.0),
        "duration_ms": 1000.0 * vals.get("probe_duration_seconds", 4.0),
    }


def all_blackbox():
    return {node: blackbox_probe(node) for node in PASSIVE_NODES}


def write_and_copy_ue(outdir, supi):
    p = outdir / "ue_configs" / f"{supi}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(make_ue_yaml(supi), encoding="utf-8")
    remote = f"/tmp/{supi}.yaml"
    sh(["docker", "cp", str(p), f"ueransim:{remote}"])
    return remote


def start_ue(supi, remote_cfg):
    safe = supi.replace("-", "_")
    log = f"/tmp/{safe}.log"
    pidf = f"/tmp/{safe}.pid"
    sh(["docker", "exec", "ueransim", "sh", "-lc",
        f"rm -f {log} {pidf}; ./nr-ue -c {remote_cfg} > {log} 2>&1 & echo $! > {pidf}"])
    return safe, log, pidf


def ue_state(log):
    p = sh(["docker", "exec", "ueransim", "sh", "-lc", f"cat {log} 2>/dev/null || true"], check=False)
    txt = strip_ansi(p.stdout or "")
    if "Initial Registration is successful" in txt:
        return "success", txt
    for marker in ["Registration reject", "PLMN selection failure", "No cells in coverage"]:
        if marker.lower() in txt.lower():
            return "failure", txt
    return "pending", txt


def amf_completed_supis(start_epoch, supis):
    txt = get_logs_since("amf", start_epoch)
    completed = set()
    for line in txt.splitlines():
        if "Handle Registration Complete" not in line:
            continue
        for supi in supis:
            if f"SUPI:{supi}" in line:
                completed.add(supi)
    return completed


def stop_ue(pidf):
    sh(["docker", "exec", "ueransim", "sh", "-lc",
        f"test -f {pidf} && kill $(cat {pidf}) 2>/dev/null || true"], check=False)


def run_ues(supis, configs, timeout_s=6.0, collect_stats=False):
    handles = []
    start = time.perf_counter()
    wall_start = time.time()
    for supi in supis:
        handles.append(start_ue(supi, configs[supi]))

    statbox = {}
    stats_thread = None
    if collect_stats:
        def take_stats():
            time.sleep(0.12)
            statbox["stats"] = docker_stats()
        stats_thread = threading.Thread(target=take_stats, daemon=True)
        stats_thread.start()

    states = {s: "pending" for s in supis}
    logs = {s: "" for s in supis}
    success_time = {}
    deadline = time.perf_counter() + timeout_s
    last_amf = 0.0
    while time.perf_counter() < deadline:
        for supi, (_, log, _) in zip(supis, handles):
            if states[supi] != "pending":
                continue
            state, txt = ue_state(log)
            logs[supi] = txt
            if state != "pending":
                states[supi] = state
                if state == "success":
                    success_time[supi] = (time.perf_counter() - start) * 1000.0
        now = time.perf_counter()
        if now - last_amf >= 0.20:
            for supi in amf_completed_supis(wall_start, supis):
                if states[supi] == "pending":
                    states[supi] = "success"
                    success_time[supi] = (now - start) * 1000.0
            last_amf = now
        if all(v != "pending" for v in states.values()):
            break
        time.sleep(0.08)

    for supi, (_, log, pidf) in zip(supis, handles):
        state, txt = ue_state(log)
        logs[supi] = txt
        if states[supi] == "pending":
            states[supi] = state
        stop_ue(pidf)

    if stats_thread:
        stats_thread.join(timeout=5)
    successes = [s for s in supis if states[s] == "success"]
    lat = [success_time[s] for s in successes if s in success_time]
    return {
        "states": states,
        "logs": logs,
        "success_rate": len(successes) / max(1, len(supis)),
        "latency_ms": float(statistics.median(lat)) if lat else timeout_s * 1000.0,
        "stats": statbox.get("stats", {}),
    }


def configure_fault(node, event_id, kind, canary_supi="", phase="canary"):
    set_all_none(event_id)
    if not node:
        return
    if kind == "evasion_delay" and phase == "canary":
        # Adversarial upper bound: the compromised NF recognizes the control
        # transaction/window and behaves normally only while the canary runs.
        set_proxy(node, event_id, mode="none")
        return
    if kind in ("delay", "evasion_delay"):
        set_proxy(node, event_id, mode="delay", delay_ms=30 if kind == "delay" else 45)
    elif kind == "jitter":
        set_proxy(node, event_id, mode="jitter", jitter_ms=45)
    elif kind == "error":
        set_proxy(node, event_id, mode="error", error_every_n=1)


def scenario_schedule(replicate):
    items = []
    def add(name, count, label, target="", kind="", workload=1, family="", post_sleep_s=0):
        for _ in range(count):
            items.append({
                "scenario": name, "label": label, "target_node": target,
                "fault_kind": kind, "workload_ues": workload, "family": family or name,
                "post_sleep_s": post_sleep_s,
            })

    add("normal", 40, 0, workload=1, family="legitimate")
    add("benign_burst", 8, 0, workload=3, family="legitimate")
    add("benign_heavy_load", 8, 0, workload=5, family="legitimate")
    # Medium-duration stability/drift samples. These are legitimate events,
    # distributed through the shuffled sequence; each inserts a 20 s gap so
    # the full run spans materially more wall-clock time and three load levels.
    add("drift_normal_1ue", 8, 0, workload=1, family="drift_legitimate", post_sleep_s=20)
    add("drift_normal_3ue", 8, 0, workload=3, family="drift_legitimate", post_sleep_s=20)
    add("drift_normal_5ue", 8, 0, workload=5, family="drift_legitimate", post_sleep_s=20)
    for node in TARGET_NODES:
        add(f"selective_delay_{node.lower()}", 5, 1, target=node, kind="delay", family="selective_delay")
    for node in ("AUSF", "UDM"):
        add(f"selective_jitter_{node.lower()}", 5, 1, target=node, kind="jitter", family="selective_jitter")
    for node in TARGET_NODES:
        add(f"business_error_{node.lower()}", 5, 1, target=node, kind="error", family="business_error")
    add("canary_evasion_udm", 6, 1, target="UDM", kind="evasion_delay", family="evasion")
    add("udm_outage", 6, 1, target="UDM", kind="outage", family="outage")
    add("registration_storm", 6, 1, target="AMF", workload=5, family="storm")

    rng = random.Random(20261018 + replicate)
    rng.shuffle(items)
    return items


def wait_for_gnb():
    deadline = time.time() + 60
    last = ""
    while time.time() < deadline:
        p = sh(["docker", "logs", "ueransim"], check=False)
        last = strip_ansi((p.stdout or "") + (p.stderr or ""))
        if "NG Setup procedure is successful" in last or "NG Setup Response received" in last:
            return
        time.sleep(2)
    raise RuntimeError("UERANSIM gNB did not complete NG setup. Last log:\n" + last[-5000:])


def wait_for_proxies():
    for node, (container, _) in PROXY_NODES.items():
        deadline = time.time() + 30
        last = ""
        while time.time() < deadline:
            try:
                ip = container_ip(container)
                with urllib.request.urlopen(f"http://{ip}:9901/healthz", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception as e:
                last = repr(e)
                time.sleep(1)
        else:
            raise RuntimeError(f"proxy {node} not ready: {last}")


def record_event(seq, desc, canary_supi, workload_supis, configs, outdir, replicate, started_at):
    scenario = desc["scenario"]
    target = desc["target_node"]
    fault = desc["fault_kind"]
    canary_event = f"r{replicate}-e{seq}-canary"
    workload_event = f"r{replicate}-e{seq}-workload"
    paused = False

    raw_dir = outdir / "raw_logs" / f"event_{seq:04d}_{scenario}"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if fault == "outage":
        set_all_none(canary_event)
        sh(["docker", "pause", "udm"])
        paused = True
    else:
        configure_fault(target, canary_event, fault, canary_supi, phase="canary")

    try:
        canary = run_ues([canary_supi], configs, timeout_s=6.0, collect_stats=False)
        for supi, txt in canary["logs"].items():
            (raw_dir / f"{supi}_canary.log").write_text(txt, encoding="utf-8")

        canary_proxy = {node: proxy_stats(node, canary_event) for node in TARGET_NODES}
        root_probe = all_blackbox()

        if fault == "outage":
            # Keep UDM paused for workload; root probe above must observe the outage.
            set_all_none(workload_event)
        else:
            configure_fault(target, workload_event, fault, canary_supi="", phase="workload")

        workload_start = time.time()
        workload = run_ues(workload_supis, configs, timeout_s=6.0, collect_stats=True)
        for supi, txt in workload["logs"].items():
            (raw_dir / f"{supi}_workload.log").write_text(txt, encoding="utf-8")
        workload_proxy = {node: proxy_stats(node, workload_event) for node in TARGET_NODES}

        node_rows = []
        for node, container in PASSIVE_NODES.items():
            txt = get_logs_since(container, workload_start)
            (raw_dir / f"{container}.log").write_text(txt, encoding="utf-8")
            ls = summarize_logs(txt)
            st = workload["stats"].get(container, {"cpu_pct": 0.0, "mem_mib": 0.0})
            bb = root_probe.get(node, {"success": 0.0, "duration_ms": 4000.0})
            cp = canary_proxy.get(node, {})
            wp = workload_proxy.get(node, {})
            node_rows.append({
                "replicate": replicate,
                "event_id": seq,
                "sequence": seq,
                "scenario": scenario,
                "scenario_family": desc["family"],
                "label": desc["label"],
                "target_node": target,
                "node": node,
                "localization_candidate": int(node in TARGET_NODES),
                "cpu_pct": st.get("cpu_pct", 0.0),
                "mem_mib": st.get("mem_mib", 0.0),
                "http_requests": ls["http_requests"],
                "http_error_rate": ls["http_error_rate"],
                "path_entropy": ls["path_entropy"],
                "root_probe_success": bb["success"],
                "root_probe_duration_ms": bb["duration_ms"],
                "canary_business_requests": cp.get("business_requests", 0),
                "canary_business_median_ms": cp.get("median_ms", 0.0),
                "canary_business_p95_ms": cp.get("p95_ms", 0.0),
                "canary_business_status_5xx": cp.get("status_5xx", 0),
                "canary_business_faulted": cp.get("faulted_requests", 0),
                "workload_business_median_ms": wp.get("median_ms", 0.0),
                "workload_business_status_5xx": wp.get("status_5xx", 0),
                "canary_registration_latency_ms": canary["latency_ms"],
                "canary_registration_success": canary["success_rate"],
                "workload_registration_latency_ms": workload["latency_ms"],
                "workload_registration_success": workload["success_rate"],
                "elapsed_s": time.time() - started_at,
            })

        target_canary = canary_proxy.get(target, {}) if target in TARGET_NODES else {}
        target_workload = workload_proxy.get(target, {}) if target in TARGET_NODES else {}
        effective = 1
        if fault in ("delay", "jitter", "error"):
            effective = int(target_canary.get("faulted_requests", 0) > 0 and target_workload.get("faulted_requests", 0) > 0)
        elif fault == "evasion_delay":
            effective = int(target_canary.get("faulted_requests", 0) == 0 and target_workload.get("faulted_requests", 0) > 0)
        elif fault == "outage":
            effective = int(root_probe["UDM"]["success"] == 0)

        event_row = {
            "replicate": replicate,
            "event_id": seq,
            "sequence": seq,
            "scenario": scenario,
            "scenario_family": desc["family"],
            "label": desc["label"],
            "target_node": target,
            "workload_ues": len(workload_supis),
            "fault_kind": fault,
            "fault_effective": effective,
            "canary_registration_latency_ms": canary["latency_ms"],
            "canary_registration_success": canary["success_rate"],
            "workload_registration_latency_ms": workload["latency_ms"],
            "workload_registration_success": workload["success_rate"],
            "elapsed_s": time.time() - started_at,
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        return node_rows, event_row
    finally:
        if paused:
            sh(["docker", "unpause", "udm"], check=False)
            time.sleep(1.0)
        for node in TARGET_NODES:
            try:
                set_proxy(node, f"cleanup-{replicate}-{seq}", mode="none")
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicate", type=int, required=True)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    wait_for_gnb()
    wait_for_proxies()
    login = wait_webui()
    token = login["access_token"]

    schedule = scenario_schedule(args.replicate)
    total_needed = 1 + sum(1 + int(d["workload_ues"]) for d in schedule) + 25
    supis = [make_supi(args.replicate, i + 1) for i in range(total_needed)]
    print(f"replicate={args.replicate}: provisioning {len(supis)} subscribers for {len(schedule)} interleaved events")

    for i, supi in enumerate(supis):
        status, body = api_post(
            f"http://127.0.0.1:5000/api/subscriber/{supi}/{PLMN}",
            token,
            subscriber_payload(supi, f"09{args.replicate:02d}{i:07d}"),
        )
        if status not in (200, 201, 409):
            raise RuntimeError(f"subscriber creation failed: {status} {body}")

    configs = {s: write_and_copy_ue(outdir, s) for s in supis}
    cursor = 0

    smoke = supis[cursor]
    cursor += 1
    set_all_none("smoke")
    smoke_result = run_ues([smoke], configs, timeout_s=8.0)
    if smoke_result["success_rate"] < 1.0:
        raise RuntimeError("smoke registration failed")
    print("smoke registration succeeded")

    all_nodes, all_events = [], []
    started_at = time.time()
    for seq, desc in enumerate(schedule, start=1):
        canary = supis[cursor]
        cursor += 1
        nwork = int(desc["workload_ues"])
        workload_supis = supis[cursor:cursor+nwork]
        cursor += nwork
        rows, event = record_event(seq, desc, canary, workload_supis, configs, outdir, args.replicate, started_at)
        all_nodes.extend(rows)
        all_events.append(event)
        print(
            f"rep={args.replicate} seq={seq}/{len(schedule)} scenario={desc['scenario']} "
            f"effective={event['fault_effective']} canary={event['canary_registration_latency_ms']:.1f}ms "
            f"workload={event['workload_registration_latency_ms']:.1f}ms"
        )
        if int(desc.get("post_sleep_s", 0)) > 0:
            time.sleep(int(desc["post_sleep_s"]))

    nodes = pd.DataFrame(all_nodes)
    events = pd.DataFrame(all_events)
    nodes.to_csv(outdir / "real5g_node_observations.csv", index=False)
    events.to_csv(outdir / "real5g_events.csv", index=False)

    manifest = {
        "replicate": args.replicate,
        "events": len(events),
        "negative_events": int((events.label == 0).sum()),
        "positive_events": int((events.label == 1).sum()),
        "all_scenarios_interleaved_on_one_runner": True,
        "target_nodes": TARGET_NODES,
        "passive_nodes": list(PASSIVE_NODES),
        "fault_semantics": "Selective business-path HTTP/2 proxy fault; root / is never delayed by selective faults.",
        "evasion_semantics": "Adversarial upper-bound: manipulation is suspended during the recognizable canary/control window and resumed for ordinary workload.",
        "blackbox": "Prometheus blackbox_exporter probes the registered root endpoint.",
        "transport_security": "Pinned free5GC compose uses h2c/HTTP for SBI. The proxy is internal and forwards request headers transparently; mTLS termination is not evaluated and remains a deployment limitation.",
        "drift_design": "24 legitimate drift/stability events per runner at 1/3/5 UE loads, each separated by 20 seconds and shuffled with other scenarios.",
        "active_control": "A real UERANSIM canary UE registration traversing NAS/NGAP and SBI business APIs.",
        "zone_features": "Not computed during collection; derived fold-wise from per-NF normal baselines to avoid leakage.",
    }
    (outdir / "collection_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    ineffective = events[(events.label == 1) & (events.fault_kind.isin(["delay", "jitter", "error", "evasion_delay"])) & (events.fault_effective == 0)]
    print(f"collection complete; ineffective controlled-fault events={len(ineffective)}")


if __name__ == "__main__":
    main()
