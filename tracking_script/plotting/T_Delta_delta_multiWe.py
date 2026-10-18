# -*- coding: utf-8 -*-
"""
T(Delta, delta) as in Riviere et al., PRF 7, 083602 (2022), Fig. 3,
pooled over the ensembles of the We = 6, 15, 30, 45 cases, plus diagnostics.

Each realization is cleaned ONCE and a light genealogy (bubble lifetimes,
parents, birth events, breakup events, dt_snap, window) is cached to
<work_dir>/genealogy_light.pkl.gz. Splits are recomputed from that cache for
any min_child_snapshots / n_eddy / filter choice in seconds, without re-running
the cleaning. Set force_recompute = True only after changing the tracker.
(The old splits_T.npz files are no longer used.)
"""
# %%
import gzip
import json
import pickle
import time
import traceback
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
WeD0_list = [6, 15, 30, 45]
ens_list  = range(20)                        # s00 ... s19 (capped by len(cases) per We)
base_path = Path("/home/nn8802/Documents/simulation/tracking_validation/3D_HIT")
#
is_2d = 0
lbox_v = [120, 120, 120]
periodic_bc = [1, 1, 1]
separation_veto = True
do_debug = False
level = 10                                   # grid level: dx = lbox/2**level
n_cut_cells = 6                              # bubbles below n_cut_cells*dx were excluded upstream
dx = lbox_v[0]/2**level
#
n_eddy = 4.0                  # splits with t < n_eddy * t_c(d0)
min_child_snapshots = 25      # used for the main plot (0 = paper definition)
min_child_scan = [0, 1, 2, 3, 5, 15, 20, 25]   # convergence diagnostic
min_parent_snapshots = 1      # a breakup-born big child that splits again within this many
                              # snapshots is part of the same burst: its lifetime runs back to
                              # its parent's birth (what a coarser output interval would see)
min_parent_scan = [0, 1, 2, 3, 5, 10, 15, 20, 25]   # burst diagnostic (at fixed min_child_snapshots)
satellite_merge_only = True  # True: a small child is a satellite only if it DISAPPEARS (coalescence)
                             # within min_child_snapshots; if it breaks up itself, the split is kept.
                             # False: any death (coalescence or breakup) counts (previous behaviour).
force_recompute = False      # re-run the cleaning even if genealogy_light.pkl.gz exists
n_workers = 4                # realizations cleaned in parallel (1 = sequential).
                             # Each worker needs ~2-3 GB of RAM at peak (more at high We).
#
A = 1/np.sqrt(12)            # paper convention; 1/(4*np.sqrt(6)) for delta as a true diameter
#
# ---- filters applied to the MAIN plot (diagnostics show their own subsets) ----
We_plot = WeD0_list 
delta_min_cells = 0          # e.g. 20: keep splits with delta >= 20 dx (above the grid-controlled pile-up)
delta_min_common = None      # e.g. 0.21 (= 6 dx/d_h at We = 45); None = no common cut
vol_tol = None               # e.g. 0.05 keeps |(delta^3 + delta_big^3)/Delta^3 - 1| < 5 %
drop_coal_lineage = False    # drop splits whose parent descends from a real coalescence
drop_coal_born = False       # drop only splits whose parent's lifetime STARTS at a real
                             # coalescence (root_birth == 2); descendants born by breakup are kept
drop_subhinze_parents = False  # drop splits with Delta < d_h
n_boot = 1000                # bootstrap over realizations for the error bars (0 = std/sqrt(n))
bins = np.logspace(-1, 1, 21)
rng = np.random.default_rng(0)


def case_list(WeD0):
    return globals()[f"cases_ReL_038_We{WeD0:02d}p0_dL1p49"]


def phys_params(WeD0):
    """Physical parameters of one We (identical for all its realizations)."""
    c0 = case_list(WeD0)[0]
    rho, sigma, d_h = c0.rho_r, c0.sigma, c0.d_h
    return dict(rho_liq=rho, mu_liq=c0.mu1_in, sigma=sigma, d_h=d_h,
                eps=c0.mean_eps,
                t_h=A*np.sqrt(rho/sigma)*d_h**1.5,           # T2(d_h)
                x_cut=n_cut_cells*dx/d_h,                   # smallest delta/d_h tracked
                stats_path=base_path/f"We{WeD0:02d}"/"statistics_output")


PHYS = {We: phys_params(We) for We in WeD0_list}
for We, P in PHYS.items():
    print(f"We = {We:2d}: d_h = {P['d_h']:.4g}, T2(d_h) = {P['t_h']:.4g}, eps = {P['eps']:.4g}, "
          f"sigma = {P['sigma']:.4g}, {n_cut_cells} dx/d_h = {P['x_cut']:.3f}")

# %%
# ============================================================
# LIGHT GENEALOGY (what is cached) AND SPLITS COMPUTED FROM IT
# ============================================================
BUBBLE_KEYS = ("t_birth", "t_death", "t_life", "birth_event", "parents")
BREAKUP_KEYS = ("parent", "children", "D_parent", "D_children", "time_p1")
BIRTH_CODE = {"breakup": 1, "coalescence": 2}     # anything else (initial bubbles) -> 0
BIRTH_NAME = {0: "initial", 1: "breakup", 2: "coalescence"}


def light_genealogy(cleaned, time_initial, tend, T_eddy, nb_raw, nc_raw):
    peh = cleaned["physical_event_history"]
    return dict(
        bubbles={k: {f: v[f] for f in BUBBLE_KEYS}
                 for k, v in cleaned["bubble_lifetime"].items()},
        breakups=[{f: ev[f] for f in BREAKUP_KEYS}
                  for ev in cleaned["diameter_tree"] if ev["type"] == "breakup"],
        dt_snap=float(np.median(np.diff([r["time_p1"] for r in peh]))),
        time_initial=float(time_initial), tend=float(tend), T_eddy=float(T_eddy),
        nb_raw=int(nb_raw), nc_raw=int(nc_raw),
        nb_clean=int(cleaned["total_breakups"]),
        nc_clean=int(cleaned.get("total_coalescences", -1)))


SPLIT_FIELDS = ("Delta", "delta", "delta_big", "T", "T_over_dt", "t_split", "vol_err",
                "parent_birth", "root_birth", "coal_lineage", "child_life", "child_censored",
                "chain")


def compute_splits(G, min_snap, n_eddy, min_parent=0):
    """All kept splits of one realization, with the fields in SPLIT_FIELDS.

    Delta = parent diameter before the split, delta = smaller child, T = parent
    lifetime (effective birth -> split). A split counts only if its small child
    survives at least min_snap snapshots; otherwise it is a non-event (the parent
    lives on through the large child, and the satellite re-merging does not end
    its life either).
    t_split      : split time since time_initial, in units of t_c(d0)
    parent_birth : how the parent itself was born (0 initial, 1 breakup, 2 coalescence)
    root_birth   : same, for the start of its effective lifetime
    coal_lineage : parent descends from a real (non-satellite) coalescence
    chain        : number of filtered satellite / burst events absorbed into the lifetime
    min_parent   : burst merging, see min_parent_snapshots
    """
    B = G["bubbles"]
    t0, tend, T_eddy, dt = G["time_initial"], G["tend"], G["T_eddy"], G["dt_snap"]
    t_win_end = t0 + n_eddy*T_eddy
    tol = 1e-12*max(1.0, abs(t_win_end))
    min_life = min_snap*dt
    burst_life = min_parent*dt*(1 + 1e-9)
    #
    def child_life(cid):
        v = B[cid]
        if v["t_death"] is not None:
            return v["t_life"], False
        return tend - v["t_birth"], True
    #
    breakup_parents = {ev["parent"] for ev in G["breakups"]}   # bubbles that died by breaking up
    #
    def is_satellite(cid):
        life, censored = child_life(cid)
        if life >= min_life:
            return False
        if satellite_merge_only and cid in breakup_parents:
            return False          # it broke up itself: a real bubble, not a vanished satellite
        return True
    #
    split_of, satellites = {}, set()
    for ev in G["breakups"]:
        (a, b), (Da, Db) = ev["children"], ev["D_children"]
        small, big = (a, b) if Da <= Db else (b, a)
        ok = not is_satellite(small)
        split_of[ev["parent"]] = (ok, big, small, ev)
        if not ok:
            satellites.add(small)
    #
    def predecessor(pid):
        v = B[pid]
        if v["birth_event"] == "breakup":
            ok, big, small, _ = split_of[v["parents"][0]]
            if pid == big and not ok:
                return v["parents"][0]
            if (pid == big and min_parent > 0 and pid in split_of
                    and v["t_life"] is not None and v["t_life"] <= burst_life):
                return v["parents"][0]
        elif v["birth_event"] == "coalescence":
            others = [q for q in v["parents"] if q not in satellites]
            if len(others) == 1 and len(v["parents"]) == 2:
                return others[0]
        return None
    #
    def root(pid):
        cur, n = pid, 0
        while (prev := predecessor(cur)) is not None:
            cur, n = prev, n + 1
        return cur, n
    #
    coal_anc = {}
    for k in sorted(B, key=lambda q: B[q]["t_birth"]):
        v = B[k]
        real_coal = v["birth_event"] == "coalescence" and predecessor(k) is None
        coal_anc[k] = real_coal or any(coal_anc.get(p, False) for p in (v["parents"] or []))
    #
    out = {f: [] for f in SPLIT_FIELDS}
    for par, (ok, big, small, ev) in split_of.items():
        ts = ev["time_p1"]
        if not ok or not (t0 - tol <= ts <= t_win_end + tol):
            continue
        r, n_chain = root(par)
        T = ts - B[r]["t_birth"]
        d_sm, d_bg = sorted(ev["D_children"])
        cl, cens = child_life(small)
        out["Delta"].append(ev["D_parent"])
        out["delta"].append(d_sm)
        out["delta_big"].append(d_bg)
        out["T"].append(T)
        out["T_over_dt"].append(T/dt)
        out["t_split"].append((ts - t0)/T_eddy)
        out["vol_err"].append((d_sm**3 + d_bg**3)/ev["D_parent"]**3 - 1.0)
        out["parent_birth"].append(BIRTH_CODE.get(B[par]["birth_event"], 0))
        out["root_birth"].append(BIRTH_CODE.get(B[r]["birth_event"], 0))
        out["coal_lineage"].append(coal_anc[par])
        out["child_life"].append(cl/dt)
        out["child_censored"].append(cens)
        out["chain"].append(n_chain)
    dtypes = dict(parent_birth=int, root_birth=int, chain=int,
                  coal_lineage=bool, child_censored=bool)
    return {f: np.asarray(v, dtype=dtypes.get(f, float)) for f, v in out.items()}

# %%
# ============================================================
# CLEAN ONE REALIZATION AND CACHE ITS LIGHT GENEALOGY
# ============================================================
def cache_path(WeD0, ens):
    return PHYS[WeD0]["stats_path"]/f"s{ens:02d}"/"genealogy_light.pkl.gz"


def process_realization(WeD0, ens):
    """Builds the cache if needed. Returns (WeD0, ens, ok, list of log lines)."""
    P = PHYS[WeD0]
    tag = f"We{WeD0:02d}/s{ens:02d}"
    t_start = time.time()
    work_dir = P["stats_path"]/f"s{ens:02d}"
    raw_history_file = work_dir/"event_history_raw.out"
    initial_genealogy_file = work_dir/"initial_genealogy.json"
    cache_file = cache_path(WeD0, ens)
    #
    if cache_file.exists() and not force_recompute:
        return WeD0, ens, True, [f"{tag}: cache found"]
    if not raw_history_file.exists() or not initial_genealogy_file.exists():
        return WeD0, ens, False, [f"{tag}: missing files, skipped"]
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
        return WeD0, ens, False, [f"{tag}: empty raw history, skipped"]
    if len(idx_to_id_initial) != len(event_history_raw[0]["voll0"]):
        raise RuntimeError(f"{tag}: initial genealogy does not match first raw frame.")
    tend = event_history_raw[-1]["time_p1"]
    #
    nb_raw, nc_raw, _ = count_raw_topological_events(event_history_raw)
    cleaned = clean_genealogy_from_raw_history(
        event_history_raw, idx_to_id_initial, id_to_birth_time_initial,
        P["rho_liq"], P["mu_liq"], P["sigma"], int(ini["is_2d"]),
        debug=do_debug, lbox_v=lbox_v, periodic_bc=periodic_bc,
        separation_veto=separation_veto)
    del event_history_raw
    G = light_genealogy(cleaned, time_initial, tend, case_list(WeD0)[ens].ted, nb_raw, nc_raw)
    del cleaned
    with gzip.open(cache_file, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    return WeD0, ens, True, [
        f"{tag}: raw b/c {nb_raw}/{nc_raw}, clean b/c {G['nb_clean']}/{G['nc_clean']}, "
        f"dt_snap = {G['dt_snap']/P['t_h']:.4f} T2(d_h)  "
        f"[load {t_load:.0f} s, total {time.time()-t_start:.0f} s]"]

# %%
# ============================================================
# BUILD / CHECK CACHES FOR ALL (We, realization) PAIRS
# ============================================================
tasks = [(We, ens) for We in WeD0_list for ens in ens_list if ens < len(case_list(We))]
t0_run = time.time()
if n_workers <= 1:
    for i, (We, ens) in enumerate(tasks, 1):
        try:
            _, _, ok, log = process_realization(We, ens)
        except Exception:
            log = [f"We{We:02d}/s{ens:02d}: FAILED\n{traceback.format_exc()}"]
        print(f"[{i}/{len(tasks)}] " + "\n".join(log), flush=True)
else:
    print(f"Checking/building {len(tasks)} caches on {n_workers} workers", flush=True)
    # "fork" lets the workers see the functions defined in this script, even when
    # it is run cell by cell in Spyder (Linux/macOS; on Windows use n_workers = 1).
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("fork")) as pool:
        futures = {pool.submit(process_realization, We, ens): (We, ens) for We, ens in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            We, ens = futures[fut]
            try:
                _, _, ok, log = fut.result()
            except Exception:
                log = [f"We{We:02d}/s{ens:02d}: FAILED\n{traceback.format_exc()}"]
            print(f"[{i}/{len(tasks)}] " + "\n".join(log), flush=True)
print(f"Caches ready in {time.time()-t0_run:.0f} s")
#
G_all = {}
for We, ens in tasks:
    cf = cache_path(We, ens)
    if cf.exists():
        with gzip.open(cf, "rb") as f:
            G_all[(We, ens)] = pickle.load(f)
print(f"Loaded {len(G_all)} light genealogies")

# %%
# ============================================================
# POOLED TABLE OF SPLITS (each We normalized by its own d_h, T2(d_h))
# ============================================================
def build_table(min_snap, We_sel=WeD0_list, min_parent=None):
    min_parent = min_parent_snapshots if min_parent is None else min_parent
    parts = []
    for (We, ens), G in sorted(G_all.items()):
        if We not in We_sel:
            continue
        s = compute_splits(G, min_snap, n_eddy, min_parent)
        P, n = PHYS[We], len(s["T"])
        s["x"] = s["delta"]/P["d_h"]            # delta/d_h
        s["y"] = s["T"]/P["t_h"]                # T/T2(d_h)
        s["c"] = s["Delta"]/P["d_h"]            # Delta/d_h
        s["We"] = np.full(n, We)
        s["ens"] = np.full(n, ens)
        s["rid"] = np.full(n, 100*We + ens)     # unique realization id
        parts.append(s)
    if not parts:
        raise RuntimeError("no data")
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main_mask(t):
    m = np.isin(t["We"], We_plot)
    if delta_min_cells is not None:
        m &= t["delta"]/dx >= delta_min_cells
    if delta_min_common is not None:
        m &= t["x"] >= delta_min_common
    if vol_tol is not None:
        m &= np.abs(t["vol_err"]) < vol_tol
    if drop_coal_lineage:
        m &= ~t["coal_lineage"]
    if drop_coal_born:
        m &= t["root_birth"] != 2
    if drop_subhinze_parents:
        m &= t["c"] >= 1.0
    return m


def bin_average(x, y, rid, n_boot=n_boot, min_count=3):
    """Bin means of y in bins of x. Errors: bootstrap over realizations (rid),
    since splits of one realization are correlated; std/sqrt(n) if n_boot = 0."""
    ib = np.digitize(x, bins)
    xb, yb, eb, nb = [], [], [], []
    for k in range(1, len(bins)):
        m = ib == k
        if m.sum() < min_count:
            continue
        ym = y[m]
        if n_boot:
            _, inv = np.unique(rid[m], return_inverse=True)
            sums, cnts = np.bincount(inv, weights=ym), np.bincount(inv)
            pick = rng.integers(0, len(sums), size=(n_boot, len(sums)))
            err = (sums[pick].sum(1)/cnts[pick].sum(1)).std()
        else:
            err = ym.std()/np.sqrt(m.sum())
        xb.append(np.sqrt(bins[k-1]*bins[k])); yb.append(ym.mean())
        eb.append(err); nb.append(m.sum())
    return tuple(map(np.asarray, (xb, yb, eb, nb)))


tab = build_table(min_child_snapshots)
M = main_mask(tab)
print(f"\nmin_child_snapshots = {min_child_snapshots}, min_parent_snapshots = "
      f"{min_parent_snapshots}: {len(tab['T'])} splits in total, "
      f"{int(M.sum())} after the main-plot filters")

# %%
# ============================================================
# D0. SUMMARY TABLE PER We
# ============================================================
def dup_fraction(delta, rid, rtol=1e-9):
    """Fraction of splits whose delta matches another split of the same realization."""
    n_dup = 0
    for r in np.unique(rid):
        d = np.sort(delta[rid == r])
        if len(d) > 1:
            n_dup += int((np.diff(d)/d[1:] < rtol).sum())
    return n_dup/max(len(delta), 1)


print(f"\n{'We':>3} {'real':>4} {'splits/real':>11} {'dup delta':>9} {'T<=2dt':>7} "
      f"{'|vol|>5%':>8} {'coal lin.':>9} {'coal born':>9} {'Delta<dh':>8} {'x<cut45':>8}")
for We in WeD0_list:
    m = tab["We"] == We
    if not m.any():
        continue
    n_real = len(np.unique(tab["rid"][m]))
    print(f"{We:3d} {n_real:4d} {m.sum()/n_real:11.1f} "
          f"{dup_fraction(tab['delta'][m], tab['rid'][m]):9.1%} "
          f"{(tab['T_over_dt'][m] <= 2).mean():7.1%} "
          f"{(np.abs(tab['vol_err'][m]) > 0.05).mean():8.1%} "
          f"{tab['coal_lineage'][m].mean():9.1%} "
          f"{(tab['root_birth'][m] == 2).mean():9.1%} "
          f"{(tab['c'][m] < 1).mean():8.1%} "
          f"{(tab['x'][m] < PHYS[max(WeD0_list)]['x_cut']).mean():8.1%}")
print("(paper: ~55 splits per realization, coalescence negligible)")

# %%
# ============================================================
# D1. CONVERGENCE WITH min_child_snapshots
# ============================================================
set_fig_params()
d_v = np.logspace(-1, 1, 256)
thd = d_v**(3/2)                                          # T2(delta)/T2(d_h)
fig, ax = plt.subplots(figsize=(9, 6))
cols = plt.cm.viridis(np.linspace(0, 0.9, len(min_child_scan)))
for ms, col in zip(min_child_scan, cols):
    t = tab if ms == min_child_snapshots else build_table(ms)
    m = main_mask(t)
    xb, yb, eb, _ = bin_average(t["x"][m], t["y"][m], t["rid"][m])
    ax.errorbar(xb, yb, yerr=eb, fmt="o-", color=col, ms=5, capsize=2,
                label=f"min child = {ms} ({int(m.sum())} splits)")
    print(f"min_child_snapshots = {ms}: {int(m.sum())} splits, "
          f"mean chain length {t['chain'][m].mean():.2f}")
ax.plot(d_v, thd, "k--", linewidth=2.0, label=r"$T_2(\delta)/T_2(d_h)$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim([0.1, 10]); ax.set_ylim([1e-3, 1e1])
ax.set_xlabel(r"$\delta/d_h$"); ax.set_ylabel(r"$\langle T\rangle_\Delta/T_2(d_h)$")
ax.set_title("D1: convergence with the satellite filter")
ax.legend(loc="lower right", fontsize=11)
plt.tight_layout(); plt.show()

# %%
# ============================================================
# D1b. BURST MERGING: min_parent_snapshots at fixed min_child_snapshots
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
cols_b = plt.cm.cividis(np.linspace(0, 0.9, len(min_parent_scan)))
for mp_, col in zip(min_parent_scan, cols_b):
    t = build_table(min_child_snapshots, min_parent=mp_)
    m = main_mask(t)
    xb, yb, eb, _ = bin_average(t["x"][m], t["y"][m], t["rid"][m])
    ax.errorbar(xb, yb, yerr=eb, fmt="o-", color=col, ms=5, capsize=2,
                label=f"min parent = {mp_}")
    print(f"min_parent_snapshots = {mp_:2d}: {int(m.sum())} splits, "
          f"T<=2dt {(t['T_over_dt'][m] <= 2).mean():.1%}, "
          f"mean chain length {t['chain'][m].mean():.2f}")
ax.plot(d_v, thd, "k--", linewidth=2.0, label=r"$T_2(\delta)/T_2(d_h)$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim([0.1, 10]); ax.set_ylim([1e-3, 1e1])
ax.set_xlabel(r"$\delta/d_h$"); ax.set_ylabel(r"$\langle T\rangle_\Delta/T_2(d_h)$")
ax.set_title(f"D1b: burst merging (min child = {min_child_snapshots})")
ax.legend(loc="lower right", fontsize=11)
plt.tight_layout(); plt.show()

# %%
# ============================================================
# D2. PER-We BIN AVERAGES (should collapse if T2(d_h) is the right scale)
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6))
cols = plt.cm.plasma(np.linspace(0, 0.8, len(WeD0_list)))
for We, col, mk in zip(WeD0_list, cols, "os^D"):
    m = (tab["We"] == We)
    if delta_min_cells is not None:
        m &= tab["delta"]/dx >= delta_min_cells
    if delta_min_common is not None:
        m &= tab["x"] >= delta_min_common
    if vol_tol is not None:
        m &= np.abs(tab["vol_err"]) < vol_tol
    if drop_coal_lineage:
        m &= ~tab["coal_lineage"]
    if drop_coal_born:
        m &= tab["root_birth"] != 2
    if drop_subhinze_parents:
        m &= tab["c"] >= 1.0
    if m.sum() < 3:
        continue
    xb, yb, eb, _ = bin_average(tab["x"][m], tab["y"][m], tab["rid"][m])
    ax.errorbar(xb, yb, yerr=eb, fmt=mk, color=col, ms=6, capsize=2,
                label=f"We = {We} ({int(m.sum())})")
    ax.axvline(PHYS[We]["x_cut"], color=col, ls=":", lw=1)
ax.plot(d_v, thd, "k--", linewidth=2.0, label=r"$T_2(\delta)/T_2(d_h)$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim([0.05, 10]); ax.set_ylim([1e-3, 1e1])
ax.set_xlabel(r"$\delta/d_h$"); ax.set_ylabel(r"$\langle T\rangle_\Delta/T_2(d_h)$")
ax.set_title(f"D2: per We (dotted: {n_cut_cells}$\\Delta x$ cut of each We)")
ax.legend(loc="lower right", fontsize=11)
plt.tight_layout(); plt.show()

# %%
# ============================================================
# D3. VOLUME CONSERVATION AND D4. SNAPSHOT PILE-UP
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
ve_bins = np.linspace(-0.3, 0.3, 61)
Tdt_bins = np.r_[np.arange(0.5, 10.5, 1.0),               # T/dt is an integer: unit bins
                 np.logspace(np.log10(10.5), 4, 40)[1:]]    # then log bins
for We, col in zip(WeD0_list, cols):
    m = tab["We"] == We
    if not m.any():
        continue
    axes[0].hist(np.clip(tab["vol_err"][m], -0.3, 0.3), bins=ve_bins, histtype="step",
                 color=col, lw=2, density=True, label=f"We = {We}")
    axes[1].hist(tab["T_over_dt"][m], bins=Tdt_bins, histtype="step",
                 color=col, lw=2, density=True, label=f"We = {We}")
axes[0].set_xlabel(r"$(\delta^3+\delta_{big}^3)/\Delta^3 - 1$  (clipped at $\pm 0.3$)")
axes[0].set_ylabel("pdf"); axes[0].set_yscale("log")
axes[0].set_title("D3: volume conservation at each split")
axes[1].set_xscale("log"); axes[1].set_yscale("log")
axes[1].set_xlabel(r"$T/\Delta t_{snap}$"); axes[1].set_ylabel("pdf")
axes[1].set_title("D4: parent lifetime in snapshots (pile-up at 1-2 = artefact)")
for a in axes:
    a.legend(fontsize=11)
plt.tight_layout(); plt.show()

# %%
# ============================================================
# D5. TIME WINDOW, FIRST BREAKUP, AND T vs SPLIT TIME
# ============================================================
first = []
for r in np.unique(tab["rid"]):
    m = (tab["rid"] == r) & (tab["parent_birth"] == 0)
    if m.any():
        first.append((r // 100, tab["t_split"][m].min()))
first = np.asarray(first)
print("\nFirst breakup of an initial bubble, in t_c(d0) (paper: ~1):")
for We in WeD0_list:
    f = first[first[:, 0] == We, 1] if len(first) else np.array([])
    if len(f):
        print(f"  We = {We:2d}: mean {f.mean():.2f}, min {f.min():.2f}, max {f.max():.2f}  "
              f"({len(f)} realizations)")
#
fig, ax = plt.subplots(figsize=(9, 6))
lin = tab["coal_lineage"]
ax.scatter(tab["t_split"][~lin], tab["y"][~lin], s=6, alpha=0.3, c="tab:blue",
           label="no coalescence in lineage")
ax.scatter(tab["t_split"][lin], tab["y"][lin], s=6, alpha=0.3, c="tab:red",
           label="coalescence in lineage")
ax.set_yscale("log"); ax.set_ylim([1e-3, 1e1]); ax.set_xlim([0, n_eddy])
ax.set_xlabel(r"$t_{split}/t_c(d_0)$"); ax.set_ylabel(r"$T/T_2(d_h)$")
ax.set_title("D5: lifetime vs split time")
ax.legend(fontsize=11, markerscale=3)
plt.tight_layout(); plt.show()

# %%
# ============================================================
# D6. SUBSETS: lineage type and sub-Hinze parents
# ============================================================
subsets = {
    "all":                           np.ones_like(M),
    "no coalescence in lineage":     ~tab["coal_lineage"],
    "coalescence in lineage":        tab["coal_lineage"],
    "lifetime starts at breakup":    tab["root_birth"] == 1,
    "lifetime starts at coalescence": tab["root_birth"] == 2,
    r"$\Delta \geq d_h$":            tab["c"] >= 1,
    r"$\Delta < d_h$":               tab["c"] < 1,
    r"$|$vol err$| < 5\%$":          np.abs(tab["vol_err"]) < 0.05,
}
fig, ax = plt.subplots(figsize=(9, 6))
for (name, sm), mk in zip(subsets.items(), "osD^v<>ph"):
    m = sm & np.isin(tab["We"], We_plot)
    if m.sum() < 3:
        continue
    xb, yb, eb, _ = bin_average(tab["x"][m], tab["y"][m], tab["rid"][m])
    ax.errorbar(xb, yb, yerr=eb, fmt=mk + "-", ms=5, capsize=2,
                label=f"{name} ({int(m.sum())})")
ax.plot(d_v, thd, "k--", linewidth=2.0, label=r"$T_2(\delta)/T_2(d_h)$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim([0.1, 10]); ax.set_ylim([1e-3, 1e1])
ax.set_xlabel(r"$\delta/d_h$"); ax.set_ylabel(r"$\langle T\rangle_\Delta/T_2(d_h)$")
ax.set_title(f"D6: subsets (min child = {min_child_snapshots})")
ax.legend(loc="lower right", fontsize=10)
plt.tight_layout(); plt.show()

# %%
# ============================================================
# MAIN PLOT (Fig. 3 of Riviere et al. 2022), all selected We pooled
# ============================================================
x_all, y_all, c_all, rid_all = tab["x"][M], tab["y"][M], tab["c"][M], tab["rid"][M]
x_bin, y_bin, e_bin, n_bin = bin_average(x_all, y_all, rid_all)
print("")
for x, y, e, n in zip(x_bin, y_bin, e_bin, n_bin):
    print(f"delta/d_h = {x:6.3f}  n = {n:5d}  <T>/T2(d_h) = {y:.4f} +- {e:.4f}  "
          f"T2(delta)/T2(d_h) = {x**1.5:.4f}")
#
We_used = [We for We in We_plot if (tab["We"][M] == We).any()]
pref = np.array([(1/A)*PHYS[We]["eps"]**(-1/3)*PHYS[We]["d_h"]**(-5/6)
                 / np.sqrt(PHYS[We]["rho_liq"]/PHYS[We]["sigma"]) for We in We_used])
if pref.max()/pref.min() > 1.05:
    print(f"WARNING: t_c(d_h)/T2(d_h) differs across We (min {pref.min():.3f}, "
          f"max {pref.max():.3f}); plotting the mean.")
tch = pref.mean()*d_v**(2/3)                              # t_c(delta)/T2(d_h)
#
fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(x_all, y_all, c=c_all, cmap="viridis", vmin=0, vmax=5, s=15, alpha=0.4)
fig.colorbar(sc, ax=ax, label=r"$\Delta/d_h$")
ax.errorbar(x_bin, y_bin, yerr=e_bin, fmt="ks", ms=6, zorder=5)
ax.plot(d_v, tch, "k:", linewidth=2.0, label=r"$t_c(\delta)/T_2(d_h)$")
ax.plot(d_v, thd, "k--", linewidth=2.0, label=r"$T_2(\delta)/T_2(d_h)$")
ax.set_xlim([0.1, 10]); ax.set_ylim([1e-3, 1e1])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$\delta/d_h$")
ax.set_ylabel(r"$T(\Delta,\delta)/T_2(d_h)$")
ax.set_title(f"We = {', '.join(map(str, We_used))}: "
             f"{len(np.unique(rid_all))} realizations, {len(y_all)} splits")
ax.legend(loc="lower right", fontsize=14)
plt.tight_layout(); plt.show()
