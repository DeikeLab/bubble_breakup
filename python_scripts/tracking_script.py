#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Tracking driver. run_case() validates the inputs, loads the tagging
# output, writes the initial genealogy and loops over time; each time step
# is processed by track_step(), which only calls the functions below:
#
#   extract_step         frame (DPS data) of one snapshot
#   check_counts         consistency of the frames with the tagging log
#   step_constants       C_V, C_X of the step (anchors or former calibration)
#   cover_local_gate     candidates + exact cover, gate g_i per DPS
#   cover_step_gate      candidates + exact cover, one gate g for the step
#   complete_summary     calibration and gate information of the step
#   save_step            one line of event_history_raw.out
#
# The settings of the run are held in a dict (cfg) and the loaded tagging
# output in another (data); both are only read by the step functions.
#
import argparse
import contextlib
import io
import json
import multiprocessing
import os
import warnings
from pathlib import Path
from time import perf_counter
import numpy as np
#
from tracking_functions import (
    EVENT_SHAPE,
    build_bubble_stat_index,
    remove_restart_duplicates,
    extract_frame,
    make_box,
    calibrate,
    calibrate_anchors,
    find_candidates,
    organize_events,
    failed_nodes,
    local_gate_candidates,
    build_record,
    get_log,
)
#
TIMING_KEYS = ("extract", "coefficients", "candidate_search", "organize", "history", "print")
EVENT_TYPES = ('continuation', 'breakup', 'coalescence', 'birth', 'death')
#
# ============================================================
# HELPER FUNCTIONS
# ============================================================
#
def ensure_slash(path):
    #
    path = os.path.abspath(path)
    if not path.endswith('/'):
        path += '/'
    #
    return path
#
def print_step_timing(step_timing):
    #
    total = sum(step_timing.values())
    print('----------------------------------------------------')
    print('Timing for this time step')
    if total <= 0.0:
        print('No timing recorded.')
    else:
        for key, val in step_timing.items():
            pct = 100.0 * val / total
            print(f'{key:15s}: {val:10.4f} s  ({pct:6.2f}%)')
        print(f'{"total":15s}: {total:10.4f} s  (100.00%)')
    #
    print('----------------------------------------------------')
#
def add_timing(timing, step_timing, key, t0):
    #
    dt = perf_counter() - t0
    timing[key] += dt
    step_timing[key] += dt
#
def tick(step_timing, key, t0):
    #
    step_timing[key] += perf_counter() - t0
#
def print_timing(timing, title, fmt="10.6f"):
    #
    print(title)
    total = sum(timing.values())
    for key, val in timing.items():
        pct = 100.0 * val / total if total > 0.0 else 0.0
        print(f"{key:15s}: {val:{fmt}} s  ({pct:6.2f}%)")
    print(f"{'total':15s}: {total:{fmt}} s  (100.00%)")
#
def to_json(x):
    #
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x
#
# ============================================================
# SETUP: inputs, tagging output, initial genealogy
# ============================================================
#
def make_config(input_dir, res, is_2d, axi, lbox_v, lmin_v, N_start, st, min_cells,
                periodic_bc, order, event_types, do_debug, outdir, contact, calibration,
                gate_max):
    #
    """
    Validate the inputs of run_case() and return them, normalized, as a
    dict (cfg) together with the output directory.
    """
    #
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f'Input directory does not exist: {input_dir}')
    #
    required_files = (
        input_dir / 'num_bubbles_droplets_f1.out',
        input_dir / 'bubbles_f1.out',
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing required input file(s):\n  ' + '\n  '.join(missing))
    #
    res, is_2d, axi = int(res), int(is_2d), int(axi)
    N_start, st, min_cells, do_debug = int(N_start), int(st), int(min_cells), int(do_debug)
    #
    if st < 1:
        raise ValueError('st must be >= 1.')
    if N_start < 0:
        raise ValueError('N_start must be >= 0.')
    if is_2d not in (0, 1):
        raise ValueError('is_2d must be 0 or 1.')
    if axi not in (0, 1):
        raise ValueError('axi must be 0 or 1.')
    #
    lbox_v = np.asarray(lbox_v, dtype=float)
    if lbox_v.shape != (3,) or np.any(lbox_v <= 0.0):
        raise ValueError('lbox_v must contain three positive values.')
    #
    if lmin_v is None:
        lmin_v = -0.5 * lbox_v
    else:
        lmin_v = np.asarray(lmin_v, dtype=float)
        if lmin_v.shape != (3,):
            raise ValueError('lmin_v must contain three values.')
    #
    periodic_bc = tuple(int(v) for v in periodic_bc)
    if len(periodic_bc) != 3 or any(v not in (0, 1) for v in periodic_bc):
        raise ValueError('periodic_bc must contain exactly three values, each 0 or 1.')
    #
    event_types = tuple(event_types)
    if len(event_types) == 0:
        raise ValueError('At least one candidate event type must be enabled.')
    if len(set(event_types)) != len(event_types):
        raise ValueError('event_types contains duplicates.')
    if not set(event_types).issubset(EVENT_TYPES):
        raise ValueError(f'event_types must be chosen from {sorted(EVENT_TYPES)}.')
    #
    order = int(order)
    if order not in (1, 2):
        raise ValueError('order must be 1 (first order) or 2 (second order).')
    #
    if outdir is None:
        outdir = input_dir.parent / f'event_output_v{order}'
    else:
        outdir = Path(outdir).expanduser().resolve()
    #
    cfg = {
        "input_dir": input_dir,
        "tagging_dir": ensure_slash(input_dir),
        "work_dir": str(input_dir.parent) + '/',
        "res": res, "is_2d": is_2d, "axi": axi,
        # axisymmetric DPSs are bodies of revolution: extract_frame converts the
        # per-radian data of 2D axi runs to 3D, and the rest of the tracking and
        # the cleaning treat them as 3D
        "is_2d_track": 0 if axi == 1 else is_2d,
        "lbox_v": lbox_v, "lmin_v": lmin_v, "periodic_bc": periodic_bc,
        "box": make_box(lbox_v, lmin_v, periodic_bc),
        "N_start": N_start, "st": st, "min_cells": min_cells,
        "order": order, "event_types": event_types, "do_debug": do_debug,
        "contact": contact, "calibration": calibration, "gate_max": gate_max,
        # anchors: gate g_i per DPS (cover_local_gate); all: one gate per step
        "local_gate": calibration == 'anchors',
    }
    #
    return cfg, outdir
#
def load_tagging(tagging_dir):
    #
    """
    Read num_bubbles_droplets_f1.out and bubbles_f1.out, remove the rows
    duplicated by simulation restarts and index the bubbles by time step.
    Returns data = {log_tag, bubble_stat_sorted, bubble_step_index}.
    """
    #
    log_tag = get_log(tagging_dir, string="num_bubbles_droplets_f1.out")
    bubble_stat = get_log(tagging_dir, string="bubbles_f1.out")
    #
    # Remove rows duplicated by simulation restarts (latest run wins).
    # Must run on the files in their raw order, before any sorting.
    log_tag, bubble_stat, restart_info = remove_restart_duplicates(log_tag, bubble_stat)
    #
    print("Restart cleaning:", restart_info)
    #
    if restart_info["n_restarts"] > 0:
        warnings.warn(
            f"{restart_info['n_restarts']} restart(s) detected: removed "
            f"{restart_info['n_log_rows_dropped']} log rows and "
            f"{restart_info['n_bubble_rows_dropped']} bubble rows "
            f"({restart_info['n_bubble_blocks_dropped']} steps, "
            f"{restart_info['n_back_to_back_splits']} back-to-back copies split).",
            RuntimeWarning,
        )
    #
    if restart_info["n_bubble_rows_unmatched"] > 0:
        warnings.warn(
            f"{restart_info['n_bubble_rows_unmatched']} rows of bubbles_f1.out, "
            f"from istep {restart_info['first_unmatched_step']} onwards, do not "
            "match the log and were discarded (e.g. output written after the "
            "last log line).",
            RuntimeWarning,
        )
    #
    # after cleaning, time must increase with the step in both files
    if np.any(np.diff(log_tag[:, 0].astype(float)) < 0):
        raise RuntimeError(
            "num_bubbles_droplets_f1.out: time decreases with istep "
            "after restart cleaning.")
    #
    b_order = np.argsort(bubble_stat[:, 1].astype(float), kind="stable")
    if np.any(np.diff(bubble_stat[b_order, 0].astype(float)) < 0):
        raise RuntimeError(
            "bubbles_f1.out: time decreases with istep after restart cleaning.")
    #
    bubble_stat_sorted, bubble_step_index = build_bubble_stat_index(bubble_stat)
    #
    return {"log_tag": log_tag, "bubble_stat_sorted": bubble_stat_sorted,
            "bubble_step_index": bubble_step_index}
#
def extract_step(data, cfg, istep):
    #
    """Frame (DPS data) of the snapshot at time step istep."""
    #
    return extract_frame(data["bubble_stat_sorted"], data["bubble_step_index"], cfg["lbox_v"],
                         cfg["min_cells"], cfg["res"], istep, cfg["is_2d"], cfg["order"],
                         cfg["axi"])
#
def log_step(data, row):
    #
    """Time step and time of the row of num_bubbles_droplets_f1.out."""
    #
    log_tag = data["log_tag"]
    return int(round(float(log_tag[row, 1]))), log_tag[row, 0]
#
def initial_genealogy(frame0, istep0, time0, cfg):
    #
    """
    Identifiers of the DPS of the first frame: b0 for the largest, b1, b2,
    ... for the others in index order, all born at time0.
    """
    #
    voll0 = frame0["vol"]
    if len(voll0) == 0:
        raise RuntimeError(
            "Cannot initialize genealogy: no bubbles found in the first frame.\n"
            f"N_start = {cfg['N_start']}\n"
            f"istep_p0 = {istep0}"
        )
    #
    idx_to_id = {}
    id_to_birth_time = {}
    imother0 = int(np.argmax(voll0))
    idx_to_id[imother0] = "b0"
    id_to_birth_time["b0"] = time0
    next_id = 1
    for i in range(len(voll0)):
        if i == imother0:
            continue
        bid = f"b{next_id}"
        idx_to_id[i] = bid
        id_to_birth_time[bid] = time0
        next_id += 1
    #
    # safety checks
    if len(idx_to_id) != len(voll0):
        raise RuntimeError("Initial genealogy ID count mismatch.")
    if len(id_to_birth_time) != len(voll0):
        raise RuntimeError("Initial genealogy birth-time count mismatch.")
    if len(set(idx_to_id.values())) != len(voll0):
        raise RuntimeError("Duplicate initial genealogy IDs.")
    if set(idx_to_id.values()) != set(id_to_birth_time.keys()):
        raise RuntimeError("Mismatch between initial IDs and birth-time IDs.")
    #
    return {
        "idx_to_id_initial": {str(k): v for k, v in idx_to_id.items()},
        "id_to_birth_time_initial": {k: float(v) for k, v in id_to_birth_time.items()},
        "Nb_initial": int(len(voll0)),
        "istep_initial": int(istep0),
        "time_initial": float(time0),
        "is_2d": int(cfg["is_2d_track"]),
        "is_2d_input": int(cfg["is_2d"]),
        "axi": int(cfg["axi"])}
#
# ============================================================
# ONE TIME STEP
# ============================================================
#
def check_counts(data, p, st, frame0, frame1):
    #
    """The numbers of DPS of the two frames must agree with the tagging log."""
    #
    log_tag = data["log_tag"]
    Nb_log0 = int(1 + log_tag[p + 0, 3]) - frame0["n_removed"]
    Nb_log1 = int(1 + log_tag[p + st, 3]) - frame1["n_removed"]
    if len(frame0["vol"]) != Nb_log0:
        raise ValueError(f"Inconsistent number of bubbles before event: "
                         f"tagging={len(frame0['vol'])}, log={Nb_log0} (event {p})")
    if len(frame1["vol"]) != Nb_log1:
        raise ValueError(f"Inconsistent number of bubbles after event: "
                         f"tagging={len(frame1['vol'])}, log={Nb_log1} (event {p})")
#
def step_constants(frame0, frame1, dt, cfg):
    #
    """
    C_V, C_X of each event type. anchors: the largest r_V, r_X of the anchor
    continuations (not less than 1), the same for every type.
    all: the former calibration, max over every combination of each type.
    Returns C_V, C_X (dicts by event type) and the anchor information.
    """
    #
    C_V, C_X = {}, {}
    anchor_info = None
    if cfg["calibration"] == 'anchors':
        cV, cX, anchor_info = calibrate_anchors(frame0, frame1, dt, cfg["box"], cfg["order"],
                                                contact=cfg["contact"])
        for event_type in cfg["event_types"]:
            C_V[event_type], C_X[event_type] = cV, cX
        print(f"   {anchor_info['n_anchors']} anchors (N per iteration "
              f"{anchor_info['N_iter']}): C_V = {cV:.6e}, C_X = {cX:.6e}")
    else:
        for event_type in cfg["event_types"]:
            C_V[event_type], C_X[event_type] = calibrate(
                frame0, frame1, dt, cfg["box"], event_type, cfg["order"])
    if cfg["do_debug"] == 1:
        for event_type in cfg["event_types"]:
            print(event_type, "C_V =", C_V[event_type], "C_X =", C_X[event_type])
    #
    return C_V, C_X, anchor_info
#
def search_candidates(frame0, frame1, dt, C_V, C_X, cfg, gate, step_timing):
    #
    """Candidate events with the bounds g C_V, g C_X (gate as in find_candidates)."""
    #
    t0 = perf_counter()
    print("2. Start candidate search")
    candidates = find_candidates(frame0, frame1, dt, C_V, C_X, cfg["box"], cfg["event_types"],
                                 gate=gate, order=cfg["order"], contact=cfg["contact"])
    if cfg["do_debug"] == 1:
        for event_type, c in candidates.items():
            print(event_type, "candidates =", len(c["cost"]))
    print("2. End candidate search")
    tick(step_timing, "candidate_search", t0)
    #
    return candidates
#
def organize(n0, n1, candidates, cfg, step_timing):
    #
    """Minimum-cost exact cover of the candidates, None if there is none."""
    #
    t0 = perf_counter()
    print("4. Start organization step")
    organized = organize_events(n0, n1, candidates, debug=cfg["do_debug"])
    tick(step_timing, "organize", t0)
    if organized is not None:
        print("4. End organization step")
    #
    return organized
#
def cover_local_gate(frame0, frame1, dt, C_V, C_X, cfg, label, step_timing):
    #
    """
    Local gate: each DPS has a gate g_i, 1 at first (i < n0 at t^n,
    n0 + j at t^(n+1)); a candidate on the DPS set S uses g_S C_V, g_S C_X,
    g_S = max over S of g_i. When no exact cover exists, g_i is doubled for
    the DPS of the components of the candidate graph without one.
    Returns (events, summary), max g_i, number of tries and the widened DPS
    as [0 (t^n) or 1 (t^(n+1)), index, g_i].
    """
    #
    n0, n1 = len(frame0["vol"]), len(frame1["vol"])
    g_node = np.ones(n0 + n1)
    levels = []                       # [(g, candidates with gate g)], g = 1, 2, 4, ...
    n_tries = 0
    while True:
        n_tries += 1
        gate = float(g_node.max())
        while not levels or levels[-1][0] < gate:
            g_new = 1.0 if not levels else 2.0 * levels[-1][0]
            levels.append((g_new, search_candidates(frame0, frame1, dt, C_V, C_X, cfg, g_new,
                                                    step_timing)))
        candidates = local_gate_candidates(n0, levels, g_node)
        if cfg["do_debug"] == 1 and len(levels) > 1:
            for event_type, c in candidates.items():
                print(event_type, "candidates (local gate) =", len(c["cost"]))
        organized = organize(n0, n1, candidates, cfg, step_timing)
        if organized is not None:
            break
        bad = failed_nodes(n0, n1, candidates)
        if bad.size == 0:
            raise RuntimeError(f"No exact cover at {label}, but every component has one.")
        if 2.0 * g_node[bad].max() > float(cfg["gate_max"]) * (1.0 + 1.0e-12):
            raise RuntimeError(
                f"No exact event cover at {label} with local gate g = {g_node[bad].max():g} "
                f"(gate_max = {float(cfg['gate_max']):g}) on {bad.size} DPS; run with "
                "--do_debug 1 for details.")
        g_node[bad] *= 2.0
        print(f"   local gate: no exact cover, g doubled on {bad.size} DPS "
              f"(t^n {np.sum(bad < n0)}, t^(n+1) {np.sum(bad >= n0)})")
    #
    gated = [[int(v >= n0), int(v - n0 if v >= n0 else v), float(g_node[v])]
             for v in np.flatnonzero(g_node > 1.0)]
    #
    return organized, gate, n_tries, gated
#
def cover_step_gate(frame0, frame1, dt, C_V, C_X, cfg, label, step_timing):
    #
    """
    One gate g for every DPS of the step, g = 1, 2, 4, ... until an exact
    cover exists. anchors: the bounds use g C, g <= gate_max. all: the
    former gate min(g, C), up to the full constants (gate None).
    Returns (events, summary), the final gate and the number of tries.
    """
    #
    n0, n1 = len(frame0["vol"]), len(frame1["vol"])
    anchors = cfg["calibration"] == 'anchors'
    if anchors:
        gate_max_step = float(cfg["gate_max"])
    else:
        gate_max_step = max(list(C_V.values()) + list(C_X.values()))
    gate = 1.0
    n_tries = 0
    while True:
        n_tries += 1
        gate_now = gate if anchors or gate < gate_max_step else None
        # 'all': former gate, the bounds use min(g, C)
        gate_arg = gate_now if anchors or gate_now is None else ("min", gate_now)
        candidates = search_candidates(frame0, frame1, dt, C_V, C_X, cfg, gate_arg, step_timing)
        organized = organize(n0, n1, candidates, cfg, step_timing)
        if organized is not None:
            break
        last = (2.0 * gate > gate_max_step * (1.0 + 1.0e-12) if anchors
                else gate_now is None)
        if last:
            raise RuntimeError(
                f"No exact event cover at {label} with g = {gate:g} "
                f"(gate_max = {gate_max_step:g}); run with --do_debug 1 for details.")
        print(f"   candidate gate g = {gate:g}: no exact cover, doubling g")
        gate *= 2.0
    #
    return organized, gate_now, n_tries
#
def complete_summary(summary, cfg, gate, n_tries, gated, anchor_info, C_V, C_X):
    #
    """Add the gate and calibration information of the step to its summary."""
    #
    print(
        "Candidate gate: g =",
        "all candidates (ungated)" if gate is None else f"{gate:g}",
        f"after {n_tries} tr{'y' if n_tries == 1 else 'ies'}",
    )
    summary["candidate_gate"] = gate
    summary["n_gate_tries"] = int(n_tries)
    summary["gate_scope"] = 'local' if cfg["local_gate"] else 'step'
    if cfg["local_gate"]:
        # DPS with a widened gate: [0 (t^n) or 1 (t^(n+1)), index, g_i]
        summary["gated_dps"] = gated
    summary["calibration"] = cfg["calibration"]
    if anchor_info is not None:
        summary["n_anchors"] = anchor_info["n_anchors"]
        summary["anchors"] = anchor_info["anchors"]
        summary["anchor_N_iter"] = anchor_info["N_iter"]
        for key in ("rV_max", "rX_max"):
            val = anchor_info[key]
            summary["anchor_" + key] = None if np.isnan(val) else float(val)
    summary["C_V"] = {key: float(val) for key, val in C_V.items()}
    summary["C_X"] = {key: float(val) for key, val in C_X.items()}
    #
    return summary
#
def track_step(p, frame0, data, cfg):
    #
    """
    One time step: t^n (row p of the log) -> t^(n+1) (row p + st). It depends
    only on its two snapshots, so that the steps can be tracked in any order.
    Returns the history record, the timing of the step and the frame at t^(n+1).
    """
    #
    step_timing = {key: 0.0 for key in TIMING_KEYS}
    st = cfg["st"]
    istep_p0, time_p0 = log_step(data, p)
    istep_p1, time_p1 = log_step(data, p + st)
    dt = float(time_p1 - time_p0)
    label = f"istep {istep_p0} -> {istep_p1}"
    if istep_p1 not in data["bubble_step_index"]:
        raise KeyError(f"istep {istep_p1} (log row {p + st}) is not present in bubbles_f1.out")
    #
    # 0. frame at t^(n+1)
    t0 = perf_counter()
    print("0. Start extraction of the bubble statistics")
    frame1 = extract_step(data, cfg, istep_p1)
    print("******************************************************")
    print("p =", p, "istep_p0 =", istep_p0, "istep_p1 =", istep_p1)
    print("Time_p0 =", time_p0, "Time_p1 =", time_p1)
    print("Nb_tag0 =", len(frame0["vol"]), "Nb_tag1 =", len(frame1["vol"]))
    print("Nb_rem0 =", frame0["n_removed"], "Nb_rem1 =", frame1["n_removed"])
    print("******************************************************")
    check_counts(data, p, st, frame0, frame1)
    print("0. End extraction of the bubble statistics")
    tick(step_timing, "extract", t0)
    #
    # 1. error constants C_V, C_X
    t0 = perf_counter()
    print("1. Start definition of event coefficients")
    C_V, C_X, anchor_info = step_constants(frame0, frame1, dt, cfg)
    print("1. End definition of event coefficients")
    tick(step_timing, "coefficients", t0)
    #
    # 2-4. candidate search and organization (gate, costs at the calibrated C)
    if cfg["local_gate"]:
        organized, gate, n_tries, gated = cover_local_gate(frame0, frame1, dt, C_V, C_X, cfg,
                                                           label, step_timing)
    else:
        organized, gate, n_tries = cover_step_gate(frame0, frame1, dt, C_V, C_X, cfg,
                                                   label, step_timing)
        gated = []
    events, summary = organized
    summary = complete_summary(summary, cfg, gate, n_tries, gated, anchor_info, C_V, C_X)
    #
    # 6. history record
    t0 = perf_counter()
    print("6. Start history record")
    record = build_record(istep_p0, istep_p1, time_p0, time_p1, events, summary,
                          frame0, frame1, cfg["order"])
    print("6. End history record")
    tick(step_timing, "history", t0)
    #
    return record, step_timing, frame1
#
def track_step_quiet(p, data, cfg):
    #
    """track_step() of a worker: extracts its own frame at t^n, returns the printout."""
    #
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        frame0 = extract_step(data, cfg, log_step(data, p)[0])
        record, step_timing, _ = track_step(p, frame0, data, cfg)
    #
    return record, step_timing, out.getvalue()
#
# worker state, set by run_case() before the pool forks (the workers inherit
# the loaded tagging output instead of receiving it with every step)
_WORKER = {}
#
def _run_track_step(p):
    """Pool entry point."""
    return track_step_quiet(p, _WORKER["data"], _WORKER["cfg"])
#
def save_step(raw_history_file, record, step_timing):
    #
    """Append the record to the raw history; returns the time spent."""
    #
    t0 = perf_counter()
    print("7. Start saving event_history_raw")
    with open(raw_history_file, "a") as f:
        f.write(json.dumps(record, default=to_json) + "\n")
    print("7. End saving event_history_raw")
    step_timing["print"] += perf_counter() - t0
    #
    print("")
    print_timing(step_timing, "Timing current time step:")
    print("")
#
# ============================================================
# TRACKING DRIVER
# ============================================================
#
def run_case(
    input_dir,
    res,
    is_2d,
    axi,
    lbox_v,
    lmin_v=None,
    N_start=0,
    N_end=None,
    st=1,
    min_cells=6,
    periodic_bc=(1, 1, 1),
    order=1,
    event_types=EVENT_TYPES,
    do_debug=0,
    outdir=None,
    overwrite=True,
    contact=None,
    workers=1,
    calibration='anchors',
    gate_max=np.inf,
):
    #
    """
    order = 1 : first-order method (unchanged). Input: _v1 tagging output.
    order = 2 : second-order method (Scapin & Deike). Input: _v2 tagging
                output (countDropsBubble_app_v2), which also provides per DPS
                  dV_hat (column 19): blended volume error, without C_V,
                  dX_hat (column 20): first-moment position error, without C_X.
                The event organization is the same for both orders.
    calibration = 'anchors': C_V, C_X of each pair of snapshots are the
                largest r_V, r_X of its anchor continuations (unique,
                isolated matches, calibrate_anchors), not less than
                1, and are applied to every event type. Births and deaths
                are tested against the volume V_c of the resolution
                threshold, V - V_c <= C_V dV_hat.
                'all': the former calibration, max over every combination
                of DPS of each event type (calibrate), which every
                combination passes by construction.
    gate_max  : largest gate g = 1, 2, 4, ... (default: no limit; 1: the
                calibrated bounds only). anchors: each DPS has its own gate
                g_i, 1 at first; when no exact cover exists, g_i is doubled
                only for the DPS of the components of the candidate graph
                without an exact cover, and a candidate on the DPS set S
                uses g_S C_V, g_S C_X, g_S = max over S of g_i. all: one
                gate for the step, min(g, C). The costs always use C_V, C_X.
    """
    #
    # 0. inputs and output files
    cfg, outdir = make_config(input_dir, res, is_2d, axi, lbox_v, lmin_v, N_start, st,
                              min_cells, periodic_bc, order, event_types, do_debug, outdir,
                              contact, calibration, gate_max)
    os.makedirs(outdir, exist_ok=True)
    initial_genealogy_file = outdir / 'initial_genealogy.json'
    raw_history_file = outdir / 'event_history_raw.out'
    if not overwrite:
        existing = [path for path in (initial_genealogy_file, raw_history_file) if path.exists()]
        if existing:
            raise FileExistsError(
                'Output file(s) already exist and overwrite=False:\n  '
                + '\n  '.join(str(path) for path in existing)
            )
    #
    print("====================================================")
    print("Running generic tracking case")
    print("order       =", cfg["order"], "(first order)" if cfg["order"] == 1 else "(second order)")
    print("tagging_dir =", cfg["tagging_dir"])
    print("work_dir    =", cfg["work_dir"])
    print("res         =", cfg["res"])
    print("N_start =", cfg["N_start"], "N_end =", N_end, "st =", cfg["st"])
    print("====================================================")
    #
    # 1. tagging output
    print("Start to load")
    data = load_tagging(cfg["tagging_dir"])
    print("Finish to load")
    #
    n_log, st, N_start = len(data["log_tag"]), cfg["st"], cfg["N_start"]
    N_end = n_log - st if N_end is None else min(int(N_end), n_log - st)
    if N_start >= N_end:
        raise ValueError(
            f'No snapshot pairs to process: N_start={N_start}, N_end={N_end}, st={st}, '
            f'len(log_tag)={n_log}')
    #
    # 2. initial genealogy from the first frame
    istep0, time0 = log_step(data, N_start)
    time0 = float(time0)
    if istep0 not in data["bubble_step_index"]:
        raise KeyError(f"istep {istep0} (log row {N_start}) is not present in bubbles_f1.out")
    frame0 = extract_step(data, cfg, istep0)
    with open(initial_genealogy_file, "w") as f:
        json.dump(initial_genealogy(frame0, istep0, time0, cfg), f, indent=2)
    #
    # ============================================================
    # LOOP OVER TIME
    # ============================================================
    #
    # ONE TIME STEP: t^n (row p) -> t^(n+1) (row p + st), tracked by
    # track_step() and appended to the raw history by save_step(). Each step
    # depends only on its two snapshots, so that the steps can be tracked in
    # parallel (--workers) with the same output.
    #
    with open(raw_history_file, "w"):
        pass
    timing = {key: 0.0 for key in TIMING_KEYS}
    steps = range(N_start, N_end, st)
    if workers <= 1:
        for p in steps:
            record, step_timing, frame0 = track_step(p, frame0, data, cfg)
            save_step(raw_history_file, record, step_timing)
            for key, val in step_timing.items():
                timing[key] += val
    else:
        # fork: the workers inherit the loaded tagging output through _WORKER
        _WORKER.update(data=data, cfg=cfg)
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            for record, step_timing, text in pool.imap(_run_track_step, steps):
                print(text, end="")
                save_step(raw_history_file, record, step_timing)
                for key, val in step_timing.items():
                    timing[key] += val
    #
    print('Done')
    print_timing(timing, 'Cumulative timing:')
    #
    return {
        'initial_genealogy_file': str(initial_genealogy_file),
        'raw_history_file': str(raw_history_file),
        'outdir': str(outdir),
        'timing': timing,
        'order': cfg["order"],
    }
#
# ============================================================
# COMMAND LINE INTERFACE
# ============================================================
#
def contact_arg(value):
    #
    return value if value == 'extent' else float(value)
#
def main():
    #
    parser = argparse.ArgumentParser(
        description=(
            'Generic DPS/bubble tracking post-processing script. '
            'input_dir must contain num_bubbles_droplets_f1.out and bubbles_f1.out.'
        )
    )
    #
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing num_bubbles_droplets_f1.out and bubbles_f1.out.',
    )
    parser.add_argument('--outdir', type=str, default=None, help='Output directory. Default: <input_dir>/../event_output_v1 (order 1) or event_output_v2 (order 2)')
    #
    parser.add_argument('--res', type=int, required=True, help='Grid refinement level used by the tagging output.')
    parser.add_argument('--is_2d', type=int, choices=(0, 1), required=True, help='1 for 2-D data, 0 for 3-D data.')
    parser.add_argument('--axi', type=int, choices=(0, 1), required=True,
                        help='0 for planar 2-D or 3-D data, 1 for axisymmetric data (axi.h). With '
                             '--is_2d 1 the volumes, areas and volume errors are per radian and are '
                             'multiplied by 2 pi; with --is_2d 0 they already include 2 pi. '
                             'Axisymmetric DPSs are tracked as 3D, with d = (6 V / pi)^(1/3).')
    #
    parser.add_argument('--N_start', type=int, default=0, help='First row/index in num_bubbles_droplets_f1.out.')
    parser.add_argument('--N_end', type=int, default=None, help='Exclusive upper index. Default: process all available pairs.')
    parser.add_argument('--st', type=int, default=1, help='Stride between snapshots.')
    parser.add_argument('--min_cells', type=int, default=6, help='Resolution threshold N_c: minimum equivalent diameter of a DPS, in cells of the finest level. Default: 6.')
    #
    parser.add_argument('--lbox', type=float, nargs=3, required=True, metavar=('LX', 'LY', 'LZ'), help='Domain lengths.')
    parser.add_argument(
        '--lmin',
        type=float,
        nargs=3,
        default=None,
        metavar=('XMIN', 'YMIN', 'ZMIN'),
        help='Domain minima. Default: -0.5*lbox in each direction.',
    )
    #
    parser.add_argument(
        '--periodic_bc',
        type=int,
        nargs=3,
        choices=(0, 1),
        default=(1, 1, 1),
        metavar=('PX', 'PY', 'PZ'),
        help='Periodic flags in x/y/z. Default: 1 1 1.',
    )
    #
    parser.add_argument('--order', type=int, choices=(1, 2),
                        help='1: 1st-order method (_v1 input).'
                             '2: 2nd-order method (_v2 input).')
    #
    parser.add_argument(
        '--event_types',
        nargs='+',
        choices=EVENT_TYPES,
        default=EVENT_TYPES,
        help='Candidate event classes to evaluate. Default: all five.',
    )
    parser.add_argument('--do_debug', type=int, choices=(0, 1), default=0)
    parser.add_argument(
        '--no_overwrite',
        action='store_false',
        dest='overwrite',
        help='Fail if the main output files already exist.',
    )
    parser.add_argument('--contact', type=contact_arg, default=None, metavar='extent|K',
                        help='Contact test of the two members of a breakup/coalescence. '
                             '"extent" (parameter-free): |x_b - x_a| <= l_a + l_b + dx_a + dx_b '
                             '+ |u_b - u_a| dt, with l the extent written by the tagging. '
                             'K (number): |x_b - x_a| <= K (R_a + R_b) + |u_b - u_a| dt. Default: off.')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of time steps tracked in parallel (same output as 1).')
    parser.add_argument('--calibration', choices=('anchors', 'all'), default='anchors',
                        help='anchors (default): C_V, C_X from the anchor continuations of each '
                             'pair of snapshots, applied to every event type. all: former '
                             'calibration, max over every combination of each event type.')
    parser.add_argument('--gate_max', type=float, default=np.inf,
                        help='Largest factor g of the bounds g C_V, g C_X (g = 1, 2, 4, ...) tried '
                             'when no exact cover exists; 1 = calibrated bounds only. Default: no limit.')
    parser.set_defaults(overwrite=True)
    #
    args = parser.parse_args()
    #
    run_case(
        input_dir=args.input_dir,
        res=args.res,
        is_2d=args.is_2d,
        axi=args.axi,
        lbox_v=args.lbox,
        lmin_v=args.lmin,
        N_start=args.N_start,
        N_end=args.N_end,
        st=args.st,
        min_cells=args.min_cells,
        periodic_bc=args.periodic_bc,
        event_types=args.event_types,
        do_debug=args.do_debug,
        outdir=args.outdir,
        overwrite=args.overwrite,
        order=args.order,
        contact=args.contact,
        workers=args.workers,
        calibration=args.calibration,
        gate_max=args.gate_max,
    )
#
if __name__ == '__main__':
    main()
#
