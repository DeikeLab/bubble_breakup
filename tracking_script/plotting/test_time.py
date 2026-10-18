# -*- coding: utf-8 -*-
# %%
import json
import time
from pathlib import Path
try:
    import orjson                  # ~5x faster JSON parsing: pip install orjson
    _loads, _mode = orjson.loads, "rb"
except ImportError:
    _loads, _mode = json.loads, "r"
    print("orjson not found, using json (slower). Install with: pip install orjson")
import matplotlib.pyplot as plt
from fun_track_fast import clean_genealogy_from_raw_history,print_cleaning_summary, \
                           count_raw_topological_events
import numpy as np

# %%
#from def_cases import Case_mp  # Import case class from def_case.py
#import read_case  # Import read_case.py to apply updates
from param_fig import set_fig_params
from read_case import (
    cases_ReL_038_We06p0_dL1p49,
    cases_ReL_038_We15p0_dL1p49
)

# %%
ens = 8
print(cases_ReL_038_We15p0_dL1p49[ens])
cases = cases_ReL_038_We15p0_dL1p49

# %%
#
# ============================================================
# PARAMETERS
# ============================================================
#
#
#general_path = Path("/scratch/gpfs/DEIKE/ns8802/bubbles/multiphase/ReL_038_WeD15_d0l1p49_Leq10/statistics_tracking/statistics_output/");
general_path = Path("/home/nicolo-scapin/numerical_codes/3D_HIT/We15/statistics_output/")
res = cases[ens].res
is_2d = 0
lbox_v = [120,120,120]
periodic_bc = [1,1,1]
separation_veto = True
#
A = 1/np.sqrt(96);
rho_liq = cases[ens].rho_r
mu_liq = cases[ens].mu1_in
sigma = cases[ens].sigma
d_h = cases[ens].d_h
t_h = A*np.sqrt(rho_liq/sigma)*d_h**1.5
do_debug = False
#

# %%
#
# ============================================================
# WORKING DIRS
# ============================================================
#
case = f"{ens:02d}"
work_dir = general_path / f"s{case}"
raw_history_file       = work_dir / "event_history_raw.out"
initial_genealogy_file = work_dir / "initial_genealogy.json"
#

# %%
print(f"I am doing s{case}", flush=True)
print(raw_history_file)

# %%
#
# ============================================================
# LOAD INITIAL GENEALOGY
# ============================================================
#
with open(initial_genealogy_file,"r") as f:
    initial_data = json.load(f)
#
idx_to_id_initial = {
    int(k):v
    for k,v in initial_data["idx_to_id_initial"].items()
}
#
id_to_birth_time_initial = {
    k:float(v)
    for k,v in initial_data["id_to_birth_time_initial"].items()
}
#
Nb_initial = int(initial_data["Nb_initial"])
istep_initial = int(initial_data["istep_initial"])
time_initial = float(initial_data["time_initial"])
is_2d = int(initial_data["is_2d"])
#
print("Initial genealogy loaded")
print("Nb_initial =",Nb_initial)
print("istep_initial =",istep_initial)
print("time_initial =",time_initial)
#

# %%
#
# ============================================================
# LOAD RAW EVENT HISTORY
# ============================================================
#
t_start = time.time()
event_history_raw = []
#
with open(raw_history_file,_mode) as f:
    for line in f:
        if line.strip():
            rec = _loads(line)
            for k in ("ucmx0","ucmy0","ucmz0","ucmx1","ucmy1","ucmz1","summary"):
                rec.pop(k, None)          # not used by the cleaning; saves memory
            event_history_raw.append(rec)
print(f"loading took {time.time()-t_start:.1f} s", flush=True)
#
if len(event_history_raw) == 0:
    raise RuntimeError(
        f"No records found in {raw_history_file}"
    )
#
print("event_history_raw loaded:",len(event_history_raw),"records")
tend = event_history_raw[-1]["time_p1"]
print("end time =", tend)
#
# ============================================================
# SAFETY CHECK INITIAL STATE
# ============================================================
#
n0_raw = len(event_history_raw[0]["voll0"])
#
if len(idx_to_id_initial) != n0_raw:
    raise RuntimeError(
        "Initial genealogy does not match first raw-history frame.\n"
        f"Initial IDs = {len(idx_to_id_initial)}\n"
        f"First raw frame bubbles = {n0_raw}"
    )
#
if set(idx_to_id_initial.values()) != set(id_to_birth_time_initial.keys()):
    raise RuntimeError(
        "Mismatch between initial physical IDs and birth-time IDs."
    )
#
# ============================================================
# RAW TOPOLOGICAL EVENT COUNTS
# ============================================================
#
(
    total_breakups_raw,
    total_coalescences_raw,
    raw_event_series,
) = count_raw_topological_events(event_history_raw)
#
print("")
print("==============================================")
print("RAW TOPOLOGICAL EVENTS")
print("==============================================")
print("Total raw breakups     =", total_breakups_raw)
print("Total raw coalescences =", total_coalescences_raw)
print("==============================================")
#
# ============================================================
# CLEAN PHYSICAL GENEALOGY
# ============================================================
#
t_clean = time.time()
cleaned = clean_genealogy_from_raw_history(
        event_history_raw,
        idx_to_id_initial,
        id_to_birth_time_initial,
        rho_liq,
        mu_liq,
        sigma,
        is_2d,
        debug=do_debug,
        lbox_v=lbox_v,
        periodic_bc=periodic_bc,
        separation_veto=separation_veto)
#
print(f"cleaning took {time.time()-t_clean:.1f} s", flush=True)
print_cleaning_summary(cleaned,total_breakups_raw,total_coalescences_raw)
total_breakups = cleaned["total_breakups"]
total_coalescences = cleaned["total_coalescences"]
breakup_event = cleaned["breakup_event"]
diameter_tree = cleaned["diameter_tree"]
diameter_history = cleaned["diameter_history"]
physical_event_history = cleaned["physical_event_history"]
bubble_lifetime = cleaned["bubble_lifetime"]
idx_to_id_final = cleaned["idx_to_id_final"]
id_to_birth_time = cleaned["id_to_birth_time"]
genealogy = cleaned["genealogy"]
#
# ============================================================
# CLEAN TOPOLOGICAL EVENT COUNTS
# ============================================================
#
print("")
print("==============================================")
print("CLEAN TOPOLOGICAL EVENTS")
print("==============================================")
print("Total clean breakups     =", total_breakups)
print("Total clean coalescences =", total_coalescences)
print("Removed breakups         =", total_breakups_raw-total_breakups)
print("Removed coalescences     =", total_coalescences_raw-total_coalescences)
print("==============================================")
#

# %%
#stop

# %%
BREAKUP_TYPES = {"breakup"}   # add "transient_breakup" if you want those too
#
from collections import defaultdict
breakup_children = defaultdict(list)   # child_id  -> times it was born by breakup
breakup_parents  = defaultdict(list)   # parent_id -> times it died by breakup
n_parent_reused = 0                    # events where the parent ID survives as a child
#
for record in physical_event_history:
    for event in record["events"]:
        if event.get("type") not in BREAKUP_TYPES:
            continue
        t_event = event["time"]
        if event["parent"] in event["children"]:
            n_parent_reused += 1
        breakup_parents[event["parent"]].append(t_event)
        for cid in event["children"]:
            breakup_children[cid].append(t_event)
#
print("breakup births :", len(breakup_children))
print("breakup deaths :", len(breakup_parents))
print("IDs born by breakup more than once :", sum(len(v) > 1 for v in breakup_children.values()))
print("IDs dying by breakup more than once:", sum(len(v) > 1 for v in breakup_parents.values()))
print("breakups where parent ID is reused as a child:", n_parent_reused)
# All three of the numbers above should be 0. If not, a physical ID does not
# correspond to a single bubble life and the lifetime definition below breaks.

# %%
#
# ============================================================
# BUBBLE LIFETIME VS DIAMETER
# ============================================================
T_eddy = cases[ens].ted
n_eddy = 4.0
t_window_start = time_initial
t_window_end   = time_initial + n_eddy * T_eddy
tol = 1e-12 * max(1.0, abs(t_window_end))   # float-comparison safety
#
# ------------------------------------------------------------
# T(Delta, delta) as in Riviere et al., PRF 7, 083602 (2022), Fig. 3
# ------------------------------------------------------------
# For every splitting (breakup) event with t < 4 t_c(d0):
#   Delta = diameter of the parent just before the split
#   delta = diameter of the SMALLER child
#   T     = lifetime of the PARENT (from its birth to this split)
# Coalescence deaths do not define T (no split, no delta).
#
# Optional satellite filter: a split counts only if its small child survives
# at least min_child_life. Splits that fail are treated as non-events (the
# parent lives on through the large child, and the satellite re-merging does
# not end its life either). min_child_snapshots = 0 is the paper's raw definition.
min_child_snapshots = 0
dt_snap = np.median(np.diff([r["time_p1"] for r in physical_event_history]))
min_child_life = min_child_snapshots * dt_snap
#
def child_survives(cid):
    v = bubble_lifetime[cid]
    tl = v["t_life"] if v["t_death"] is not None else tend - v["t_birth"]
    return tl >= min_child_life
#
split_of = {}          # parent_id -> (counts_as_split, big_child, small_child, event)
satellites = set()     # small children of splits that do not count
for ev in diameter_tree:
    if ev["type"] != "breakup":
        continue
    (a, b), (Da, Db) = ev["children"], ev["D_children"]
    small, big = (a, b) if Da <= Db else (b, a)
    ok = child_survives(small)
    split_of[ev["parent"]] = (ok, big, small, ev)
    if not ok:
        satellites.add(small)
#
def predecessor(pid):
    """Previous ID of the same physical bubble, if pid was not a genuine birth."""
    v = bubble_lifetime[pid]
    if v["birth_event"] == "breakup":
        par = v["parents"][0]
        ok, big, small, _ = split_of[par]
        if not ok and pid == big:
            return par
    elif v["birth_event"] == "coalescence":
        others = [q for q in v["parents"] if q not in satellites]
        if len(others) == 1 and len(v["parents"]) == 2:
            return others[0]
    return None
#
def effective_birth(pid):
    cur = pid
    while (prev := predecessor(cur)) is not None:
        cur = prev
    return bubble_lifetime[cur]["t_birth"]
#
Delta_s, delta_s, T_s = [], [], []
for par, (ok, big, small, ev) in split_of.items():
    if not ok or not (t_window_start - tol <= ev["time_p1"] <= t_window_end + tol):
        continue
    Delta_s.append(ev["D_parent"])
    delta_s.append(min(ev["D_children"]))
    T_s.append(ev["time_p1"] - effective_birth(par))
#
Delta_s = np.asarray(Delta_s); delta_s = np.asarray(delta_s); T_s = np.asarray(T_s)
print("snapshot dt =", dt_snap, "=", dt_snap/t_h, "T2(d_h)")
print("min child survival =", min_child_snapshots, "snapshots")
print("splits kept:", len(T_s), " of which sub-Hinze delta:", int((delta_s < d_h).sum()))
#
# %%
def D_at(pid, t, side="after"):
    h = diameter_history.get(pid)
    if h is None: return None
    tt = np.asarray(h["time"]); dd = np.asarray(h["diameter"])
    order = np.argsort(tt); tt, dd = tt[order], dd[order]
    m = tt >= t if side == "after" else tt <= t
    if not m.any(): return None
    return dd[m][0] if side == "after" else dd[m][-1]

rows = []
for record in physical_event_history:
    for ev in record["events"]:
        if ev.get("type") not in BREAKUP_TYPES: continue
        t, p, ch = ev["time"], ev["parent"], ev["children"]
        Dp = D_at(p, t, "before")
        Dc = [D_at(c, t, "after") for c in ch]
        if Dp is None or any(d is None for d in Dc): continue
        Dc = np.asarray(Dc)
        rows.append(dict(
            t=t, parent=p, n_child=len(ch),
            parent_in_children=p in ch,
            vol_ratio=(Dc**3).sum() / Dp**3,       # expect ~1
            max_ratio=Dc.max() / Dp,               # expect <= 1
            min_ratio=Dc.min() / Dp,               # expect < 0.8 for binary
            first_is_birth=[np.min(diameter_history[c]["time"]) >= t - 1e-12 for c in ch],
        ))

import pandas as pd
df = pd.DataFrame(rows)
print(df.describe())
print("children count distribution:\n", df.n_child.value_counts())
print("parent ID reused as child:", df.parent_in_children.mean())
print("events with a child history starting BEFORE the event:",
      (~df.first_is_birth.apply(all)).mean())
print(df.sort_values("vol_ratio").head(10))

# %%
#
# ============================================================
# PLOT
# ============================================================
#
set_fig_params()
fig,ax = plt.subplots(figsize=(9,6))
#
d_v = np.logspace(-1,1,256)
thd = d_v**(3/2)                                                     # T2(delta)/T2(d_h)
pre = (1/A)*cases[ens].mean_eps**(-1/3)*d_h**(-5/6)/np.sqrt(rho_liq/sigma)
tch = pre*d_v**(2./3)                                                # t_c(delta)/T2(d_h)
#
sc = ax.scatter(delta_s/d_h, T_s/t_h, c=Delta_s/d_h, cmap="viridis",
                vmin=0, vmax=5, s=20, alpha=0.6)
fig.colorbar(sc, ax=ax, label=r"$\Delta/d_h$")
#
# ensemble average over Delta at given delta (black squares in the paper)
bins = np.logspace(-1,1,21)
ib = np.digitize(delta_s/d_h, bins)
for k in np.unique(ib):
    m = ib == k
    if 0 < k < len(bins) and m.sum() >= 3:
        x = np.sqrt(bins[k-1]*bins[k])
        y = (T_s[m]/t_h).mean(); e = (T_s[m]/t_h).std()/np.sqrt(m.sum())
        ax.errorbar(x, y, yerr=e, fmt="ks", ms=6)
#
ax.plot(d_v,tch,"k:",linewidth=2.0,label=r"$t_c(\delta)/T_2(d_h)$")
ax.plot(d_v,thd,"k--",linewidth=2.0,label=r"$T_2(\delta)/T_2(d_h)$")
ax.set_xlim([0.1,10]); ax.set_ylim([1e-3,1e1])
ax.set_xlabel(r"$\delta/d_h$")
ax.set_ylabel(r"$T(\Delta,\delta)/T_2(d_h)$")
ax.legend(loc="lower right",fontsize=14)
ax.set_xscale("log"); ax.set_yscale("log")
#
plt.tight_layout()
plt.show()
#

# %%
#print(diameters)
