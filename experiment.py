import argparse, json, math, os, random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

import torch
import torch.nn as nn

NODES = ["AMF","SMF","UDM","AUSF","PCF","NRF","NSSF","CHF"]
LOCAL_NAMES = ["req_rate","latency","error_rate","cpu","memory","streams","route_entropy"]
PROBE_NAMES = ["probe_latency","probe_loss","probe_staleness","report_inconsistency"]
ZONE_NAMES = ["zone_req_dev","zone_lat_dev","neighbor_error_dev"]
FEATURE_NAMES = LOCAL_NAMES + PROBE_NAMES + ZONE_NAMES
LOCAL_IDX = list(range(len(LOCAL_NAMES)))
PROBE_IDX = list(range(len(LOCAL_NAMES), len(LOCAL_NAMES)+len(PROBE_NAMES)))
ZONE_IDX = list(range(len(LOCAL_NAMES)+len(PROBE_NAMES), len(FEATURE_NAMES)))

SCENARIOS = {
    "signaling_storm": (1300, 1380, "AMF"),
    "http2_dos": (1500, 1580, "NRF"),
    "nf_degradation": (1750, 1830, "PCF"),
    "stealth_compromise_smf": (2050, 2150, "SMF"),
    "stealth_compromise_amf": (2250, 2330, "AMF"),
}

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)

def adjacency():
    n = len(NODES)
    A = np.eye(n, dtype=np.float32)
    edges = [
        ("AMF","AUSF"),("AUSF","UDM"),("AMF","UDM"),("AMF","SMF"),
        ("SMF","PCF"),("SMF","CHF"),("PCF","UDM"),("AMF","NSSF"),
        ("AMF","NRF"),("SMF","NRF"),("AUSF","NRF"),("UDM","NRF"),
        ("PCF","NRF"),("NSSF","NRF"),("CHF","NRF")
    ]
    idx = {n:i for i,n in enumerate(NODES)}
    for a,b in edges:
        i,j = idx[a],idx[b]
        A[i,j]=A[j,i]=1
    d = A.sum(1)
    return (A / np.sqrt(np.outer(d,d))).astype(np.float32)

def ar_noise(rng, T, scale=1.0, phi=0.85):
    x = np.zeros(T, dtype=np.float32)
    eps = rng.normal(0, scale, size=T).astype(np.float32)
    for t in range(1,T):
        x[t] = phi*x[t-1] + eps[t]
    return x

def make_data(seed=11, T=2400):
    rng = np.random.default_rng(seed)
    n = len(NODES)
    x = np.zeros((T,n,len(FEATURE_NAMES)), dtype=np.float32)
    node_labels = np.zeros((T,n), dtype=np.int64)
    scenario = np.array(["normal"]*T, dtype=object)

    base_req = np.array([115,85,55,48,60,135,35,50],dtype=np.float32)
    base_lat = np.array([8,11,7,8,10,5,6,9],dtype=np.float32)
    base_cpu = np.array([38,34,28,26,30,32,22,25],dtype=np.float32)
    base_mem = np.array([46,42,40,38,41,44,35,37],dtype=np.float32)
    base_streams = np.array([42,31,18,15,20,50,10,16],dtype=np.float32)

    t = np.arange(T)
    global_load = 1 + 0.10*np.sin(2*np.pi*t/240) + 0.05*np.sin(2*np.pi*t/61)

    for i in range(n):
        req = base_req[i]*global_load*(1 + 0.025*ar_noise(rng,T,1))
        lat = base_lat[i]*(1+0.35*(global_load-1)) + ar_noise(rng,T,0.18)
        err = np.clip(0.004 + 0.0015*global_load + 0.001*ar_noise(rng,T,0.4), 0, None)
        cpu = base_cpu[i] + 14*(global_load-1) + ar_noise(rng,T,0.8)
        mem = base_mem[i] + 3*(global_load-1) + ar_noise(rng,T,0.25)
        streams = base_streams[i]*global_load + ar_noise(rng,T,0.8)
        ent = 2.2 + 0.08*np.sin(2*np.pi*t/180+i) + ar_noise(rng,T,0.015)
        x[:,i,0] = req
        x[:,i,1] = lat
        x[:,i,2] = err
        x[:,i,3] = cpu
        x[:,i,4] = mem
        x[:,i,5] = streams
        x[:,i,6] = ent

        x[:,i,7] = lat*1.12 + rng.normal(0,0.3,T)
        x[:,i,8] = np.clip(0.002 + rng.normal(0,0.0008,T),0,None)
        x[:,i,9] = np.clip(rng.normal(0.15,0.04,T),0,None)
        x[:,i,10] = np.clip(rng.normal(0.04,0.015,T),0,None)

    idx = {name:i for i,name in enumerate(NODES)}

    s,e,target = SCENARIOS["signaling_storm"]
    i=idx[target]; sl=slice(s,e)
    x[sl,i,0] *= 2.7; x[sl,i,3] += 34; x[sl,i,5] *= 3.0; x[sl,i,2] += 0.055
    x[sl,i,7] += 22; x[sl,i,8] += 0.06; x[sl,i,10] += 0.9
    for nb in [idx["SMF"],idx["NRF"],idx["AUSF"]]:
        x[sl,nb,1] += 2.2; x[sl,nb,3] += 5

    s,e,target = SCENARIOS["http2_dos"]
    i=idx[target]; sl=slice(s,e)
    x[sl,i,0] *= 3.4; x[sl,i,1] += 18; x[sl,i,2] += 0.11; x[sl,i,3] += 42; x[sl,i,5] *= 4.0
    x[sl,i,7] += 28; x[sl,i,8] += 0.12; x[sl,i,9] += 0.8; x[sl,i,10] += 0.7

    s,e,target = SCENARIOS["nf_degradation"]
    i=idx[target]; sl=slice(s,e)
    x[sl,i,1] += 15; x[sl,i,2] += 0.075; x[sl,i,3] += 18
    x[sl,i,7] += 20; x[sl,i,8] += 0.04; x[sl,i,9] += 0.5; x[sl,i,10] += 0.5

    for key, neigh in [
        ("stealth_compromise_smf", ["AMF","PCF","CHF"]),
        ("stealth_compromise_amf", ["SMF","AUSF","UDM","NSSF"]),
    ]:
        s,e,target = SCENARIOS[key]
        i=idx[target]; sl=slice(s,e)
        x[sl,i,0] *= 1.03
        x[sl,i,1] += 0.3
        x[sl,i,3] += 0.8
        x[sl,i,7] += 2.6
        x[sl,i,8] += 0.004
        x[sl,i,9] += 0.16
        x[sl,i,10] += 0.18
        for name in neigh:
            j=idx[name]
            x[sl,j,1] += 2.0
            x[sl,j,2] += 0.005
            x[sl,j,7] += 2.4

    A = adjacency()
    rawA = (A > 0).astype(np.float32)
    np.fill_diagonal(rawA, 0)
    for ti in range(T):
        for i in range(n):
            nb = np.where(rawA[i] > 0)[0]
            if len(nb)==0:
                continue
            med_req = np.median(x[ti,nb,0]); mad_req = np.median(np.abs(x[ti,nb,0]-med_req))+1e-3
            med_lat = np.median(x[ti,nb,1]); mad_lat = np.median(np.abs(x[ti,nb,1]-med_lat))+1e-3
            med_err = np.median(x[ti,nb,2]); mad_err = np.median(np.abs(x[ti,nb,2]-med_err))+1e-4
            x[ti,i,11] = (x[ti,i,0]-med_req)/mad_req
            x[ti,i,12] = (x[ti,i,1]-med_lat)/mad_lat
            x[ti,i,13] = (x[ti,i,2]-med_err)/mad_err

    for name,(s,e,target) in SCENARIOS.items():
        scenario[s:e] = name
        node_labels[s:e, idx[target]] = 1

    return x, node_labels, scenario, A

def standardize(x, train_end=900):
    mu = x[:train_end].reshape(-1,x.shape[-1]).mean(0)
    sd = x[:train_end].reshape(-1,x.shape[-1]).std(0) + 1e-6
    return ((x-mu)/sd).astype(np.float32), mu, sd

def make_windows(x, labels, scenario, W=12):
    X=[]; Y=[]; times=[]; S=[]
    for t in range(W, len(x)):
        X.append(x[t-W:t])
        Y.append(x[t])
        times.append(t)
        S.append(scenario[t])
    return np.stack(X), np.stack(Y), np.array(times), np.array(S, dtype=object), labels[np.array(times)]

class STPredictor(nn.Module):
    def __init__(self, input_idx, target_idx, probe_idx=None, hidden=32, use_graph=True, use_temporal=True):
        super().__init__()
        self.input_idx = input_idx
        self.target_idx = target_idx
        self.probe_idx = probe_idx or []
        self.use_graph = use_graph
        self.use_temporal = use_temporal
        self.lin = nn.Linear(len(input_idx), hidden)
        if use_temporal:
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, len(target_idx))

    def forward(self, X, A):
        z = X[..., self.input_idx]
        h0 = torch.relu(self.lin(z))
        if self.use_graph:
            h = 0.75*h0 + 0.25*torch.einsum("ij,bwjh->bwih", A, h0)
        else:
            h = h0

        if self.use_temporal:
            B,W,N,H = h.shape
            hs = h.permute(0,2,1,3).reshape(B*N,W,H)
            _, hn = self.gru(hs)
            hlast = hn[-1].reshape(B,N,H)
        else:
            hlast = h[:,-1]

        if self.probe_idx:
            p = X[:,-1,:,self.probe_idx]
            bad = torch.relu(torch.abs(p) - 1.0).mean(-1, keepdim=True)
            trust = torch.exp(-0.55*bad).clamp(0.05,1.0)
            if self.use_graph:
                neigh = torch.einsum("ij,bjh->bih", A, hlast)
                hlast = trust*hlast + (1-trust)*(0.5*hlast + 0.5*neigh)
            else:
                hlast = trust*hlast

        return self.out(hlast)

def fit_deep(Xtr, Ytr, Xv, Yv, A, input_idx, target_idx, probe_idx=None,
             use_graph=True, use_temporal=True, seed=11, epochs=18):
    seed_all(seed)
    device = torch.device("cpu")
    model = STPredictor(input_idx, target_idx, probe_idx, hidden=32,
                        use_graph=use_graph, use_temporal=use_temporal).to(device)
    At = torch.tensor(A, dtype=torch.float32, device=device)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Yt = torch.tensor(Ytr[:,:,target_idx], dtype=torch.float32, device=device)
    Xval = torch.tensor(Xv, dtype=torch.float32, device=device)
    Yval = torch.tensor(Yv[:,:,target_idx], dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    best = None; best_loss = float("inf"); patience=4; stale=0
    bs=64
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for start in range(0,len(Xt),bs):
            ix = perm[start:start+bs]
            pred = model(Xt[ix], At)
            loss = ((pred-Yt[ix])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = ((model(Xval,At)-Yval)**2).mean().item()
        if vl < best_loss - 1e-5:
            best_loss=vl; stale=0
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
            if stale>=patience:
                break
    if best is not None:
        model.load_state_dict(best)
    return model, At

def deep_scores(model, At, X, Y, target_idx, batch=128):
    model.eval()
    out=[]
    with torch.no_grad():
        for s in range(0,len(X),batch):
            xb=torch.tensor(X[s:s+batch],dtype=torch.float32)
            yb=torch.tensor(Y[s:s+batch,:,target_idx],dtype=torch.float32)
            pred=model(xb,At)
            sc=((pred-yb)**2).mean(-1)
            out.append(sc.cpu().numpy())
    return np.concatenate(out,0)

def classical_scores(kind, Xtr, Xv, Xt, input_idx, seed):
    def flatten(X):
        Z=X[...,input_idx]
        return Z.transpose(0,2,1,3).reshape(-1, Z.shape[1]*len(input_idx))
    tr=flatten(Xtr); va=flatten(Xv); te=flatten(Xt)
    if kind=="IsolationForest":
        model=IsolationForest(n_estimators=180, max_samples="auto", contamination="auto",
                              random_state=seed, n_jobs=2)
    else:
        model=OneClassSVM(kernel="rbf", gamma="scale", nu=0.01)
    model.fit(tr)
    def score(Z, samples):
        s=-model.decision_function(Z)
        return s.reshape(samples,len(NODES))
    return score(va,len(Xv)), score(te,len(Xt))

def metric_row(method, seed, y, scores, val_scores, scen, times):
    thr=float(np.quantile(val_scores.reshape(-1),0.99))
    yt=y.max(axis=1)
    st=scores.max(axis=1)
    pred=(st>thr).astype(int)
    p,r,f,_=precision_recall_fscore_support(yt,pred,average="binary",zero_division=0)
    fpr=float(pred[yt==0].mean())
    auc=float(roc_auc_score(yt,st))
    ap=float(average_precision_score(yt,st))
    stealth_mask=np.array([str(s).startswith("stealth_compromise") for s in scen])
    stealth_recall=float(pred[stealth_mask].mean()) if stealth_mask.any() else float("nan")

    loc_hits=[]; stealth_loc=[]; delays=[]
    for name,(s,e,target) in SCENARIOS.items():
        mask=(times>=s)&(times<e)
        ti=NODES.index(target)
        if mask.any():
            hits=(np.argmax(scores[mask],axis=1)==ti).astype(float)
            loc_hits.extend(hits.tolist())
            if name.startswith("stealth_compromise"):
                stealth_loc.extend(hits.tolist())
            above=np.where(scores[mask,ti]>thr)[0]
            delays.append(float(above[0]) if len(above) else float(e-s))
    return {
        "method":method,"seed":seed,"roc_auc":auc,"pr_auc":ap,"precision":float(p),
        "recall":float(r),"f1":float(f),"fpr":fpr,"stealth_recall":stealth_recall,
        "localization_top1":float(np.mean(loc_hits)),
        "stealth_localization_top1":float(np.mean(stealth_loc)),
        "median_detection_delay":float(np.median(delays)),
        "threshold":thr
    }

def scenario_rows(method, seed, y, scores, val_scores, times):
    thr=float(np.quantile(val_scores.reshape(-1),0.99))
    rows=[]
    for name,(s,e,target) in SCENARIOS.items():
        mask=(times>=s)&(times<e)
        ti=NODES.index(target)
        rows.append({
            "method":method,"seed":seed,"scenario":name,"target":target,
            "recall":float((scores[mask,ti]>thr).mean()),
            "mean_score":float(scores[mask,ti].mean()),
            "localization_top1":float((np.argmax(scores[mask],axis=1)==ti).mean())
        })
    return rows

def run_seed(seed):
    x, labels, scen_raw, A = make_data(seed)
    x,mu,sd = standardize(x)
    X,Y,times,scen,y = make_windows(x,labels,scen_raw)
    train_mask=(times<900)
    val_mask=(times>=900)&(times<1200)
    test_mask=(times>=1200)
    Xtr,Ytr=X[train_mask],Y[train_mask]
    Xv,Yv=X[val_mask],Y[val_mask]
    Xt,Yt=X[test_mask],Y[test_mask]
    yt=y[test_mask]; st=scen[test_mask]; tt=times[test_mask]

    metrics=[]; scen_rows=[]; pred_cache={}
    for kind in ["IsolationForest","OneClassSVM"]:
        sv,ss = classical_scores(kind,Xtr,Xv,Xt,LOCAL_IDX,seed)
        metrics.append(metric_row(kind,seed,yt,ss,sv,st,tt))
        scen_rows += scenario_rows(kind,seed,yt,ss,sv,tt)
        pred_cache[kind]=(sv,ss)

    deep_specs = [
        ("LSTM", LOCAL_IDX, LOCAL_IDX, None, False, True),
        ("TemporalGNN", LOCAL_IDX+ZONE_IDX, LOCAL_IDX+ZONE_IDX, None, True, True),
        ("Ablation_no_probe", LOCAL_IDX+ZONE_IDX, LOCAL_IDX+ZONE_IDX, None, True, True),
        ("Ablation_concat_probe", list(range(len(FEATURE_NAMES))), list(range(len(FEATURE_NAMES))), None, True, True),
        ("Proposed", LOCAL_IDX+ZONE_IDX, list(range(len(FEATURE_NAMES))), PROBE_IDX, True, True),
        ("Ablation_no_zone", LOCAL_IDX, LOCAL_IDX+PROBE_IDX, PROBE_IDX, True, True),
        ("Ablation_no_graph", list(range(len(FEATURE_NAMES))), list(range(len(FEATURE_NAMES))), PROBE_IDX, False, True),
        ("Ablation_no_temporal", LOCAL_IDX+ZONE_IDX, list(range(len(FEATURE_NAMES))), PROBE_IDX, True, False),
    ]
    for name,input_idx,target_idx,probe_idx,use_graph,use_temporal in deep_specs:
        m,At=fit_deep(Xtr,Ytr,Xv,Yv,A,input_idx,target_idx,probe_idx,use_graph,use_temporal,seed)
        sv=deep_scores(m,At,Xv,Yv,target_idx)
        ss=deep_scores(m,At,Xt,Yt,target_idx)
        metrics.append(metric_row(name,seed,yt,ss,sv,st,tt))
        scen_rows += scenario_rows(name,seed,yt,ss,sv,tt)
        pred_cache[name]=(sv,ss)
    return metrics, scen_rows, {"times":tt,"labels":yt,"scenario":st,"pred":pred_cache}

def save_outputs(outdir, metrics, scenarios, timeline):
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(metrics)
    sdf=pd.DataFrame(scenarios)
    df.to_csv(out/"per_seed_metrics.csv",index=False)
    sdf.to_csv(out/"scenario_recall.csv",index=False)

    agg=df.groupby("method").agg({
        "roc_auc":["mean","std"],"pr_auc":["mean","std"],"f1":["mean","std"],
        "fpr":["mean","std"],"stealth_recall":["mean","std"],
        "localization_top1":["mean","std"],"stealth_localization_top1":["mean","std"],"median_detection_delay":["mean","std"]
    })
    agg.columns=["_".join(c) for c in agg.columns]
    agg=agg.reset_index()
    agg.to_csv(out/"metrics_summary.csv",index=False)

    full_order=["IsolationForest","OneClassSVM","LSTM","TemporalGNN","Ablation_no_probe",
                "Ablation_concat_probe","Proposed","Ablation_no_zone","Ablation_no_graph","Ablation_no_temporal"]
    order=[m for m in full_order if m in set(agg.method)]
    plotdf=agg.set_index("method").loc[order]
    ax=plotdf[["f1_mean","stealth_recall_mean","localization_top1_mean"]].plot(kind="bar",figsize=(11,5))
    ax.set_ylim(0,1.05); ax.set_ylabel("score"); ax.set_title("5G SBA anomaly detection and compromise localization")
    plt.tight_layout(); plt.savefig(out/"fig_metrics.png",dpi=180); plt.close()

    ab=[m for m in order if m.startswith("Ablation") or m=="Proposed"]
    ax=plotdf.loc[ab][["f1_mean","stealth_recall_mean"]].plot(kind="bar",figsize=(9,5))
    ax.set_ylim(0,1.05); ax.set_ylabel("score"); ax.set_title("Ablation study")
    plt.tight_layout(); plt.savefig(out/"fig_ablation.png",dpi=180); plt.close()

    if timeline:
        tt=timeline["times"]; yt=timeline["labels"]; pred=timeline["pred"]
        fig,ax=plt.subplots(figsize=(12,5))
        for name in ["TemporalGNN","Proposed"]:
            if name in pred:
                sc=pred[name][1].max(axis=1)
                ax.plot(tt,sc,label=name,linewidth=1.1)
        ax.fill_between(tt,0,1,where=(yt.max(axis=1)>0),alpha=0.12,transform=ax.get_xaxis_transform(),label="attack window")
        ax.set_xlabel("time step"); ax.set_ylabel("max anomaly score"); ax.set_title("Detection timeline (seed 11)")
        ax.legend(); plt.tight_layout(); plt.savefig(out/"fig_timeline.png",dpi=180); plt.close()

    cols=["method","roc_auc_mean","pr_auc_mean","f1_mean","fpr_mean","stealth_recall_mean","localization_top1_mean","stealth_localization_top1_mean","median_detection_delay_mean"]
    tbl=agg[cols].copy()
    for c in cols[1:]:
        tbl[c]=tbl[c].map(lambda v:f"{v:.4f}")
    (out/"manuscript_table.md").write_text(tbl.to_markdown(index=False),encoding="utf-8")

    best=agg.sort_values("f1_mean",ascending=False).iloc[0]
    prop=agg[agg.method=="Proposed"].iloc[0]
    tgnn=agg[agg.method=="TemporalGNN"].iloc[0]
    summary=f"""# 5G SBA active-probe graph experiment

**Best F1:** {best.method} = {best.f1_mean:.4f} ± {best.f1_std:.4f}

**Proposed model**
- ROC-AUC: {prop.roc_auc_mean:.4f} ± {prop.roc_auc_std:.4f}
- PR-AUC: {prop.pr_auc_mean:.4f} ± {prop.pr_auc_std:.4f}
- F1: {prop.f1_mean:.4f} ± {prop.f1_std:.4f}
- FPR: {prop.fpr_mean:.4f} ± {prop.fpr_std:.4f}
- Stealth-compromise recall: {prop.stealth_recall_mean:.4f} ± {prop.stealth_recall_std:.4f}
- Top-1 compromised-NF localization: {prop.localization_top1_mean:.4f} ± {prop.localization_top1_std:.4f}
- Top-1 stealth-compromise localization: {prop.stealth_localization_top1_mean:.4f} ± {prop.stealth_localization_top1_std:.4f}
- Median detection delay: {prop.median_detection_delay_mean:.2f} time steps

**Against TemporalGNN**
- ΔF1: {prop.f1_mean-tgnn.f1_mean:+.4f}
- Δstealth recall: {prop.stealth_recall_mean-tgnn.stealth_recall_mean:+.4f}
- Δlocalization top-1: {prop.localization_top1_mean-tgnn.localization_top1_mean:+.4f}

Data are generated by a controlled 5G-SBA digital-twin simulator. They are not real operator traces; this distinction must remain explicit in the article.
"""
    (out/"summary.md").write_text(summary,encoding="utf-8")
    manifest={
        "nodes":NODES,"features":FEATURE_NAMES,"window":12,
        "train_interval":[12,899],"validation_interval":[900,1199],"test_interval":[1200,2399],
        "scenarios":{k:{"start":v[0],"end":v[1],"target":v[2]} for k,v in SCENARIOS.items()},
        "methods":sorted(df.method.unique().tolist()),
        "seeds":sorted(df.seed.unique().tolist()),
        "threshold_policy":"99th percentile of validation-normal node scores",
        "primary_detection_task":"network-level attack detection; node scores are used separately for localization",
        "data_status":"controlled synthetic/digital-twin 5G SBA telemetry, not operator traffic"
    }
    (out/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="results")
    ap.add_argument("--seeds",nargs="+",type=int,default=[11,29,47])
    args=ap.parse_args()
    allm=[]; alls=[]; timeline=None
    for seed in args.seeds:
        m,s,t=run_seed(seed)
        allm+=m; alls+=s
        if timeline is None:
            timeline=t
        print(f"finished seed={seed}")
    save_outputs(args.out,allm,alls,timeline)
    print(pd.DataFrame(allm).groupby("method")[["f1","stealth_recall","localization_top1","stealth_localization_top1","fpr"]].mean().sort_values("f1",ascending=False))

if __name__=="__main__":
    main()
