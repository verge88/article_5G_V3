#!/usr/bin/env python3
import argparse
import collections
import datetime as dt
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

NODES = {
    "NRF": "nrf",
    "AMF": "amf",
    "AUSF": "ausf",
    "UDM": "udm",
    "UDR": "udr",
    "NSSF": "nssf",
    "PCF": "pcf",
    "SMF": "smf",
}
ALIASES = {k: f"{v}.free5gc.org" for k, v in NODES.items()}
ALIASES["NRF"] = "nrf.free5gc.org"
PLMN = "20893"
KEY = "8baf473f2f8fd09487cccbd7097c6862"
OP = "8e27b6af0e692e750f32667a3b14605d"
GIN_RE = re.compile(r"\|\s*(\d{3})\s*\|.*?\|\s*(GET|POST|PUT|PATCH|DELETE)\s*\|\s*([^|]+?)\s*\|")

def sh(cmd, check=True, timeout=None, text=True):
    p = subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        capture_output=True,
        text=text,
        timeout=timeout,
    )
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")
    return p

def wait_http(url, timeout_s=90):
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                url,
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

def subscriber_payload(supi, msisdn):
    return {
        "plmnID": PLMN,
        "ueId": supi,
        "AuthenticationSubscription": {
            "authenticationManagementField": "8000",
            "authenticationMethod": "5G_AKA",
            "milenage": {
                "op": {
                    "encryptionAlgorithm": 0,
                    "encryptionKey": 0,
                    "opValue": OP,
                }
            },
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
        "SessionManagementSubscriptionData": [
            {
                "singleNssai": {"sst": 1, "sd": "010203"},
                "dnnConfigurations": {
                    "internet": {
                        "sscModes": {
                            "defaultSscMode": "SSC_MODE_1",
                            "allowedSscModes": ["SSC_MODE_2", "SSC_MODE_3"],
                        },
                        "pduSessionTypes": {
                            "defaultSessionType": "IPV4",
                            "allowedSessionTypes": ["IPV4"],
                        },
                        "sessionAmbr": {"uplink": "200 Mbps", "downlink": "100 Mbps"},
                        "5gQosProfile": {"5qi": 9, "arp": {"priorityLevel": 8}, "priorityLevel": 8},
                    }
                },
            }
        ],
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

def make_supi(index):
    return f"imsi-20893{index:010d}"

def make_ue_yaml(supi):
    return f'''# Generated for reproducible CI validation
supi: "{supi}"
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
        "kb": 1 / 1024,
        "kib": 1 / 1024,
        "mb": 1.0,
        "mib": 1.0,
        "gb": 1024.0,
        "gib": 1024.0,
        "tb": 1024.0 * 1024,
        "tib": 1024.0 * 1024,
        "b": 1 / (1024 * 1024),
    }.get(unit, 1.0)
    return num * factor

def docker_stats():
    names = " ".join(NODES.values())
    p = sh(f"docker stats --no-stream --format '{{{{json .}}}}' {names}", check=False, timeout=15)
    out = {}
    for line in p.stdout.splitlines():
        try:
            d = json.loads(line)
            name = d.get("Name", "")
            out[name] = {
                "cpu_pct": float(d.get("CPUPerc", "0").strip("%") or 0),
                "mem_mib": parse_mem_to_mib(d.get("MemUsage", "0B / 0B")),
            }
        except Exception:
            continue
    return out

def probe_nf(alias):
    cmd = (
        "docker exec probe-helper sh -lc "
        + json.dumps(
            "curl --http2-prior-knowledge -sS -o /dev/null "
            "--connect-timeout 0.5 --max-time 1.2 "
            "-w '%{http_code},%{time_total}' "
            f"http://{alias}:8000/ "
            "|| curl -sS -o /dev/null --connect-timeout 0.5 --max-time 1.2 "
            "-w '%{http_code},%{time_total}' "
            f"http://{alias}:8000/"
        )
    )
    t0 = time.perf_counter()
    p = sh(cmd, check=False, timeout=3)
    elapsed = (time.perf_counter() - t0) * 1000.0
    txt = (p.stdout or "").strip().splitlines()
    val = txt[-1] if txt else ""
    try:
        code_s, time_s = val.split(",", 1)
        code = int(code_s) if code_s.isdigit() else 0
        return float(time_s) * 1000.0, int(code > 0), code
    except Exception:
        return elapsed, 0, 0

def all_probes():
    return {node: probe_nf(alias) for node, alias in ALIASES.items()}

def get_logs_since(container, start_epoch):
    p = sh(["docker", "logs", "--since", str(int(start_epoch)), container], check=False, timeout=10)
    return strip_ansi((p.stdout or "") + "\n" + (p.stderr or ""))

def summarize_logs(txt):
    status = []
    paths = []
    for line in txt.splitlines():
        m = GIN_RE.search(line)
        if m:
            status.append(int(m.group(1)))
            paths.append(m.group(3).strip())
    n = len(status)
    err = sum(s >= 400 for s in status)
    if not paths:
        ent = 0.0
    else:
        c = collections.Counter(paths)
        total = sum(c.values())
        ent = -sum((v / total) * math.log(v / total + 1e-12) for v in c.values())
    return {
        "http_requests": n,
        "http_error_rate": (err / n) if n else 0.0,
        "path_entropy": ent,
    }

def container_pid(container):
    p = sh(["docker", "inspect", "-f", "{{.State.Pid}}", container], check=True)
    return int(p.stdout.strip())

def netem(container, delay_ms):
    pid = container_pid(container)
    sh(
        f"sudo nsenter -t {pid} -n tc qdisc replace dev eth0 root netem delay {delay_ms}ms",
        check=True,
    )

def clear_netem(container):
    try:
        pid = container_pid(container)
        sh(f"sudo nsenter -t {pid} -n tc qdisc del dev eth0 root", check=False)
    except Exception:
        pass

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
    script = f"rm -f {log} {pidf}; ./nr-ue -c {remote_cfg} > {log} 2>&1 & echo $! > {pidf}"
    sh(["docker", "exec", "ueransim", "sh", "-lc", script])
    return safe, log, pidf

def ue_state(log):
    p = sh(["docker", "exec", "ueransim", "sh", "-lc", f"cat {log} 2>/dev/null || true"], check=False)
    txt = strip_ansi(p.stdout or "")
    if "Initial Registration is successful" in txt:
        return "success", txt
    # Do not treat "Authentication Failure" generically as terminal:
    # UERANSIM can legitimately send a 5G-AKA synchronization failure (AUTS)
    # and then complete registration after AMF/AUSF resynchronization.
    failure_markers = [
        "Registration reject",
        "PLMN selection failure",
        "No cells in coverage",
    ]
    if any(x.lower() in txt.lower() for x in failure_markers):
        return "failure", txt
    return "pending", txt

def stop_ue(pidf):
    sh(
        ["docker", "exec", "ueransim", "sh", "-lc", f"test -f {pidf} && kill $(cat {pidf}) 2>/dev/null || true"],
        check=False,
    )

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

def run_ues(supis, configs, timeout_s=6.0):
    handles = []
    start = time.perf_counter()
    wall_start = time.time()
    for supi in supis:
        handles.append(start_ue(supi, configs[supi]))

    telemetry = {}
    def collect_telemetry():
        telemetry["stats"] = docker_stats()
        telemetry["probes"] = all_probes()

    telemetry_thread = threading.Thread(target=collect_telemetry, daemon=True)
    telemetry_thread.start()

    states = {s: "pending" for s in supis}
    logs = {s: "" for s in supis}
    success_time = {}
    deadline = time.perf_counter() + timeout_s
    last_amf_check = 0.0
    while time.perf_counter() < deadline:
        for supi, (safe, log, pidf) in zip(supis, handles):
            if states[supi] == "pending":
                state, txt = ue_state(log)
                logs[supi] = txt
                if state != "pending":
                    states[supi] = state
                    if state == "success":
                        success_time[supi] = (time.perf_counter() - start) * 1000.0

        # AMF's "Handle Registration Complete" is the authoritative control-plane
        # confirmation. It also avoids false negatives caused by buffered UE logs.
        now = time.perf_counter()
        if now - last_amf_check >= 0.20:
            for supi in amf_completed_supis(wall_start, supis):
                if states[supi] == "pending":
                    states[supi] = "success"
                    success_time[supi] = (now - start) * 1000.0
            last_amf_check = now

        if all(states[s] != "pending" for s in supis):
            break
        time.sleep(0.08)
    for supi, (_, log, pidf) in zip(supis, handles):
        state, txt = ue_state(log)
        logs[supi] = txt
        if states[supi] == "pending":
            states[supi] = state
        stop_ue(pidf)
    telemetry_thread.join(timeout=8.0)
    stats = telemetry["stats"] if "stats" in telemetry else docker_stats()
    probes = telemetry["probes"] if "probes" in telemetry else all_probes()
    successes = [s for s in supis if states[s] == "success"]
    lat = [success_time[s] for s in successes if s in success_time]
    return {
        "states": states,
        "logs": logs,
        "registration_success_rate": len(successes) / max(1, len(supis)),
        "registration_latency_ms": float(statistics.median(lat)) if lat else timeout_s * 1000.0,
        "stats": stats,
        "probes": probes,
    }

def robust_dev(values):
    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) + 1e-6
    return [(float(v) - med) / mad for v in arr]

def record_event(event_id, scenario, target, supis, configs, outdir, timeout_s=6.0):
    manipulation = None
    start_epoch = time.time()
    if scenario == "stealth_udm_delay":
        netem("udm", 40); manipulation = ("netem", "udm")
    elif scenario == "stealth_ausf_delay":
        netem("ausf", 40); manipulation = ("netem", "ausf")
    elif scenario == "amf_delay":
        netem("amf", 80); manipulation = ("netem", "amf")
    elif scenario == "udm_outage":
        sh(["docker", "pause", "udm"]); manipulation = ("pause", "udm")

    try:
        result = run_ues(supis, configs, timeout_s=timeout_s)
        end_epoch = time.time()
        node_rows = []
        raw_dir = outdir / "raw_logs" / f"event_{event_id:04d}_{scenario}"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for supi, txt in result["logs"].items():
            (raw_dir / f"{supi}_ue.log").write_text(txt, encoding="utf-8")
        for node, container in NODES.items():
            logtxt = get_logs_since(container, start_epoch)
            (raw_dir / f"{container}.log").write_text(logtxt, encoding="utf-8")
            ls = summarize_logs(logtxt)
            st = result["stats"].get(container, {"cpu_pct": 0.0, "mem_mib": 0.0})
            lat_ms, http_ok, http_code = result["probes"].get(node, (1200.0, 0, 0))
            node_rows.append({
                "event_id": event_id,
                "scenario": scenario,
                "target_node": target or "",
                "node": node,
                "anomaly": int(scenario != "normal"),
                "target_label": int(target == node),
                "cpu_pct": st["cpu_pct"],
                "mem_mib": st["mem_mib"],
                "http_requests": ls["http_requests"],
                "http_error_rate": ls["http_error_rate"],
                "path_entropy": ls["path_entropy"],
                "probe_latency_ms": lat_ms,
                "probe_ok": http_ok,
                "probe_http_code": http_code,
                "registration_latency_ms": result["registration_latency_ms"],
                "registration_success_rate": result["registration_success_rate"],
                "event_duration_s": end_epoch - start_epoch,
            })
        cpu_dev = robust_dev([r["cpu_pct"] for r in node_rows])
        req_dev = robust_dev([r["http_requests"] for r in node_rows])
        probe_dev = robust_dev([r["probe_latency_ms"] for r in node_rows])
        for r, a, b, c in zip(node_rows, cpu_dev, req_dev, probe_dev):
            r["zone_cpu_dev"] = a
            r["zone_req_dev"] = b
            r["zone_probe_dev"] = c
        event_row = {
            "event_id": event_id,
            "scenario": scenario,
            "target_node": target or "",
            "ue_count": len(supis),
            "registration_success_rate": result["registration_success_rate"],
            "registration_latency_ms": result["registration_latency_ms"],
            "duration_s": end_epoch - start_epoch,
            "started_at_utc": dt.datetime.fromtimestamp(start_epoch, tz=dt.timezone.utc).isoformat(),
        }
        return node_rows, event_row
    finally:
        if manipulation:
            kind, container = manipulation
            if kind == "pause":
                sh(["docker", "unpause", container], check=False)
                time.sleep(2.0)
            else:
                clear_netem(container)
                time.sleep(0.5)

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

def smoke_registration(supi, cfg, outdir):
    rows, event = record_event(0, "normal", "", [supi], cfg, outdir, timeout_s=8.0)
    if event["registration_success_rate"] < 1.0:
        amf = get_logs_since("amf", time.time() - 60)
        ue_log = list((outdir / "raw_logs").rglob("*_ue.log"))
        txt = ue_log[-1].read_text(errors="replace") if ue_log else ""
        raise RuntimeError("Smoke registration failed.\nUE:\n" + txt[-5000:] + "\nAMF:\n" + amf[-5000:])
    return rows, event

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compose-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--normal-runs", type=int, default=18)
    ap.add_argument("--runs-per-scenario", type=int, default=7)
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    wait_for_gnb()

    login = wait_http("http://127.0.0.1:5000/api/login")
    token = login["access_token"]

    total_ues = 1 + args.normal_runs + args.runs_per_scenario * (1 + 1 + 1 + 1 + 5) + 10
    supis = [make_supi(i + 1) for i in range(total_ues)]
    print(f"Provisioning {len(supis)} subscribers")
    for i, supi in enumerate(supis):
        status, body = api_post(
            f"http://127.0.0.1:5000/api/subscriber/{supi}/{PLMN}",
            token,
            subscriber_payload(supi, f"090{i:07d}"),
        )
        if status not in (201, 200, 409):
            raise RuntimeError(f"unexpected subscriber create status {status}: {body}")

    configs = {s: write_and_copy_ue(outdir, s) for s in supis}
    cursor = 0
    all_nodes = []
    all_events = []

    smoke_supi = supis[cursor]; cursor += 1
    nrows, erow = smoke_registration(smoke_supi, configs, outdir)
    erow["event_id"] = 0
    for r in nrows:
        r["event_id"] = 0
    all_nodes += nrows
    all_events.append(erow)
    print("Smoke registration succeeded")

    event_id = 1
    for _ in range(args.normal_runs):
        accepted = False
        for attempt in range(1, 4):
            s = supis[cursor]; cursor += 1
            rows, ev = record_event(event_id, "normal", "", [s], configs, outdir)
            if ev["registration_success_rate"] >= 1.0:
                all_nodes += rows; all_events.append(ev)
                print(f"event={event_id} normal success=1.00 attempt={attempt}")
                accepted = True
                break
            print(f"event={event_id} normal retry after failed registration attempt={attempt}")
            time.sleep(0.8)
        if not accepted:
            raise RuntimeError(f"normal event {event_id} failed after 3 attempts")
        event_id += 1

    plans = [
        ("stealth_udm_delay", "UDM", 1),
        ("stealth_ausf_delay", "AUSF", 1),
        ("amf_delay", "AMF", 1),
        ("udm_outage", "UDM", 1),
        ("registration_storm", "AMF", 5),
    ]
    for scenario, target, batch in plans:
        for _ in range(args.runs_per_scenario):
            ss = supis[cursor:cursor + batch]; cursor += batch
            timeout = 7.0 if scenario == "udm_outage" else 6.0
            rows, ev = record_event(event_id, scenario, target, ss, configs, outdir, timeout_s=timeout)
            all_nodes += rows; all_events.append(ev)
            print(
                f"event={event_id} {scenario} target={target} "
                f"success={ev['registration_success_rate']:.2f} latency={ev['registration_latency_ms']:.1f}ms"
            )
            event_id += 1

    pd.DataFrame(all_nodes).to_csv(outdir / "real5g_node_observations.csv", index=False)
    pd.DataFrame(all_events).to_csv(outdir / "real5g_events.csv", index=False)

    evidence = {
        "free5gc_software": "real Docker images from free5gc/free5gc-compose",
        "ueransim_software": "real free5gc/ueransim container",
        "control_plane": "real NGAP/NAS registration plus free5GC SBI",
        "user_plane": "not required in this experiment; UE configs request no PDU session",
        "stealth_surrogate": "tc netem delay injected into the network namespace of a real NF; this is controlled impairment, not a claim of real compromise",
        "normal_runs": args.normal_runs + 1,
        "runs_per_scenario": args.runs_per_scenario,
        "scenarios": [p[0] for p in plans],
        "nodes": list(NODES.keys()),
    }
    (outdir / "testbed_manifest.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print("Real testbed collection complete")

if __name__ == "__main__":
    main()
