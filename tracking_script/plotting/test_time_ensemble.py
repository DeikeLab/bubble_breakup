# -*- coding: utf-8 -*-
"""
T(Delta, delta) as in Riviere et al., PRF 7, 083602 (2022), Fig. 3,
accumulated over the ensemble s00-s19 of the We = 15 cases.

Each realization is cleaned once and its splits are cached to
<work_dir>/splits_T.npz, so re-running (e.g. to restyle the plot) is fast.
Set force_recompute = True after changing the tracker or min_child_snapshots.
"""
# %%
import json
import time
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    import orjson                  # ~5x faster JSON parsing: pip install orjson
    _loads, _mode = orjson.loads, "rb"
except ImportError:
    _loads, _mode = json.loads, "r"
    print("orjson not found, using json (slower). Install with: pip install orjson")
import numpy as np
import matplotlib.pyplot as plt
from fun_track_fast import clean_genealogy_from_raw_history, count_raw_topological_events
from param_fig import set_fig_params
from read_case import (cases_ReL_038_We06p0_dL1p49,
                       cases_ReL_038_We15p0_dL1p49,
                       cases_ReL_038_We30p0_dL1p49,
                       cases_ReL_038_We45p0_dL1p49,)
# %%
# ============================================================
# PARAMETERS
# ============================================================
WeD0   = 6
WeD0_c = f"{WeD0:02d}"
cases  = globals()[f"cases_ReL_038_We{WeD0_c}p0_dL1p49"]
ens_list = range(20)                         # s00 ... s19
general_path = (
    Path("/home/nn8802/Documents/simulation/tracking_validation/3D_HIT")
    / f"We{WeD0_c}"
    / "statistics_output"
)
#
is_2d = 0
lbox_v = [120, 120, 120]
periodic_bc = [1, 1, 1]
separation_veto = True
do_debug = False
#
n_eddy = 4.0                 # splits with t < n_eddy * t_c(d0)
min_child_snapshots = 0      # 0 = paper definition (no satellite filter)
force_recompute = False      # ignore cached splits_T.npz
n_workers = 4                # realizations processed in parallel (1 = sequential).
                             # Each worker needs ~2-3 GB of RAM at peak.
#
A = 1/np.sqrt(12)
c0 = cases[0]                # physical parameters are the same for all realizations
rho_liq = c0.rho_r
mu_liq = c0.mu1_in
sigma = c0.sigma
d_h = c0.d_h
t_h = A*np.sqrt(rho_liq/sigma)*d_h**1.5      # T2(d_h)
eps = c0.mean_eps

# %%
# ============================================================
# SPLITS OF ONE CLEANED REALIZATION
# ============================================================
def compute_splits(cleaned, time_initial, tend, T_eddy):
    """Return Delta, delta, T for every split with t < n_eddy*T_eddy.

    Delta = parent diameter before the split, delta = smaller child,
    T = parent lifetime (birth -> split). A split counts only if its small child
    survives at least min_child_snapshots snapshots; otherwise it is a non-event
    (the parent lives on through the large child, and the satellite re-merging
    does not end its life either).
    """
    bubble_lifetime = cleaned["bubble_lifetime"]
    diameter_tree = cleaned["diameter_tree"]
    peh = cleaned["physical_event_history"]
    #
    t_window_start = time_initial
    t_window_end = time_initial + n_eddy*T_eddy
    tol = 1e-12*max(1.0, abs(t_window_end))
    dt_snap = np.median(np.diff([r["time_p1"] for r in peh]))
    min_child_life = min_child_snapshots*dt_snap
    #
    def child_survives(cid):
        v = bubble_lifetime[cid]
        tl = v["t_life"] if v["t_death"] is not None else tend - v["t_birth"]
        return tl >= min_child_life
    #
    split_of, satellites = {}, set()
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
        v = bubble_lifetime[pid]
        if v["birth_event"] == "breakup":
            ok, big, small, _ = split_of[v["parents"][0]]
            if not ok and pid == big:
                return v["parents"][0]
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
    Delta, delta, T = [], [], []
    for par, (ok, big, small, ev) in split_of.items():
        if not ok or not (t_window_start - tol <= ev["time_p1"] <= t_window_end + tol):
            continue
        Delta.append(ev["D_parent"])
        delta.append(min(ev["D_children"]))
        T.append(ev["time_p1"] - effective_birth(par))
    return np.asarray(Delta), np.asarray(delta), np.asarray(T), dt_snap

# %%
# ============================================================
# LOAD + CLEAN ONE REALIZATION
# ============================================================
def process_realization(ens):
    """Returns (ens, result or None, list of log lines)."""
    log = []
    t_start = time.time()
    work_dir = general_path/f"s{ens:02d}"
    raw_history_file = work_dir/"event_history_raw.out"
    initial_genealogy_file = work_dir/"initial_genealogy.json"
    cache_file = work_dir/"splits_T.npz"
    #
    if cache_file.exists() and not force_recompute:
        z = np.load(cache_file)
        if int(z["min_child_snapshots"]) == min_child_snapshots and float(z["n_eddy"]) == n_eddy:
            log.append(f"s{ens:02d}: loaded from cache ({len(z['T'])} splits)")
            return ens, (z["Delta"], z["delta"], z["T"]), log
    #
    if not raw_history_file.exists() or not initial_genealogy_file.exists():
        log.append(f"s{ens:02d}: missing files, skipped")
        return ens, None, log
    #
    with open(initial_genealogy_file, "r") as f:
        ini = json.load(f)
    idx_to_id_initial = {int(k): v for k, v in ini["idx_to_id_initial"].items()}
    id_to_birth_time_initial = {k: float(v) for k, v in ini["id_to_birth_time_initial"].items()}
    time_initial = float(ini["time_initial"])
    #
    event_history_raw = []
    with open(raw_history_file, _mode) as f:
        for line in f:
            if line.strip():
                rec = _loads(line)
                for k in ("ucmx0", "ucmy0", "ucmz0", "ucmx1", "ucmy1", "ucmz1", "summary"):
                    rec.pop(k, None)          # not used by the cleaning; saves memory
                event_history_raw.append(rec)
    t_load = time.time() - t_start
    if len(event_history_raw) == 0:
        log.append(f"s{ens:02d}: empty raw history, skipped")
        return ens, None, log
    if len(idx_to_id_initial) != len(event_history_raw[0]["voll0"]):
        raise RuntimeError(f"s{ens:02d}: initial genealogy does not match first raw frame.")
    tend = event_history_raw[-1]["time_p1"]
    #
    nb_raw, nc_raw, _ = count_raw_topological_events(event_history_raw)
    cleaned = clean_genealogy_from_raw_history(
        event_history_raw, idx_to_id_initial, id_to_birth_time_initial,
        rho_liq, mu_liq, sigma, int(ini["is_2d"]),
        debug=do_debug, lbox_v=lbox_v, periodic_bc=periodic_bc,
        separation_veto=separation_veto)
    del event_history_raw                     # free memory before the next realization
    #
    Delta, delta, T, dt_snap = compute_splits(cleaned, time_initial, tend, cases[ens].ted)
    log.append(f"s{ens:02d}: raw breakups {nb_raw}, clean breakups {cleaned['total_breakups']}, "
               f"splits kept {len(T)} (sub-Hinze {int((delta < d_h).sum())}), "
               f"dt_snap = {dt_snap/t_h:.4f} T2(d_h)  "
               f"[load {t_load:.0f} s, total {time.time()-t_start:.0f} s]")
    del cleaned
    #
    np.savez(cache_file, Delta=Delta, delta=delta, T=T,
             min_child_snapshots=min_child_snapshots, n_eddy=n_eddy)
    return ens, (Delta, delta, T), log

# %%
# ============================================================
# LOOP OVER THE ENSEMBLE
# ============================================================
results = {}
if n_workers <= 1:
    for ens in ens_list:
        print(f"I am doing s{ens:02d}", flush=True)
        _, out, log = process_realization(ens)
        print("\n".join(log), flush=True)
        results[ens] = out
else:
    ens_todo = list(ens_list)
    print(f"Running {len(ens_todo)} realizations on {n_workers} workers", flush=True)
    # "fork" lets the workers see the functions defined in this script, even when
    # it is run cell by cell in Spyder (Linux/macOS; on Windows use n_workers = 1).
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("fork")) as pool:
        futures = {pool.submit(process_realization, ens): ens for ens in ens_todo}
        for i, fut in enumerate(as_completed(futures), 1):
            ens, out, log = fut.result()
            print(f"[{i}/{len(ens_todo)}] done s{ens:02d}")
            print("\n".join(log), flush=True)
            results[ens] = out
#
Delta_all, delta_all, T_all, ens_all = [], [], [], []
for ens in sorted(results):
    if results[ens] is None:
        continue
    Delta, delta, T = results[ens]
    Delta_all.append(Delta); delta_all.append(delta); T_all.append(T)
    ens_all.append(np.full(len(T), ens))
#
Delta_all = np.concatenate(Delta_all)
delta_all = np.concatenate(delta_all)
T_all = np.concatenate(T_all)
ens_all = np.concatenate(ens_all)
n_real = len(np.unique(ens_all))
print("")
print(f"realizations used: {n_real}, total splits: {len(T_all)}, "
      f"sub-Hinze: {int((delta_all < d_h).sum())}")

# %%
# ============================================================
# ENSEMBLE BIN AVERAGE  <T(Delta,delta)>_Delta  in bins of delta
# ============================================================
bins = np.logspace(-1, 1, 21)
ib = np.digitize(delta_all/d_h, bins)
x_bin, y_bin, e_bin, n_bin = [], [], [], []
for k in range(1, len(bins)):
    m = ib == k
    if m.sum() < 3:
        continue
    y = T_all[m]/t_h
    x_bin.append(np.sqrt(bins[k-1]*bins[k]))
    y_bin.append(y.mean())
    e_bin.append(y.std()/np.sqrt(m.sum()))
    n_bin.append(m.sum())
x_bin, y_bin, e_bin, n_bin = map(np.asarray, (x_bin, y_bin, e_bin, n_bin))
for x, y, n in zip(x_bin, y_bin, n_bin):
    print(f"delta/d_h = {x:6.3f}  n = {n:5d}  <T>/T2(d_h) = {y:.4f}  T2(delta)/T2(d_h) = {x**1.5:.4f}")

# %%
# ============================================================
# PLOT (Fig. 3 of Riviere et al. 2022)
# ============================================================
set_fig_params()
fig, ax = plt.subplots(figsize=(9, 6))
#
d_v = np.logspace(-1, 1, 256)
thd = d_v**(3/2)                                          # T2(delta)/T2(d_h)
tch = (1/A)*eps**(-1/3)*d_h**(-5/6)/np.sqrt(rho_liq/sigma)*d_v**(2/3)   # t_c(delta)/T2(d_h)
#
sc = ax.scatter(delta_all/d_h, T_all/t_h, c=Delta_all/d_h, cmap="viridis",
                vmin=0, vmax=5, s=15, alpha=0.4)
fig.colorbar(sc, ax=ax, label=r"$\Delta/d_h$")
ax.errorbar(x_bin, y_bin, yerr=e_bin, fmt="ks", ms=6, zorder=5)
#
ax.plot(d_v, tch, "k:", linewidth=2.0, label=r"$t_c(\delta)/T_2(d_h)$")
ax.plot(d_v, thd, "k--", linewidth=2.0, label=r"$T_2(\delta)/T_2(d_h)$")
ax.set_xlim([0.1, 10]); ax.set_ylim([1e-3, 1e1])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\delta/d_h$")
ax.set_ylabel(r"$T(\Delta,\delta)/T_2(d_h)$")
ax.set_title(f"We = {WeD0}, {n_real} realizations, {len(T_all)} splits")
ax.legend(loc="lower right", fontsize=14)
#
plt.tight_layout()
plt.show()
