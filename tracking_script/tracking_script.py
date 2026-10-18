#
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
import argparse
import json
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
    find_candidates,
    organize_events,
    build_record,
)
from helper_functions import get_log
#
# ============================================================
# HELPER FUNCTIONS
# ============================================================
#
def ensure_slash(path):
    path = os.path.abspath(path)
    if not path.endswith('/'):
        path += '/'
    return path
#
def print_step_timing(step_timing):
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
    print('----------------------------------------------------')
#
def add_timing(timing, step_timing, key, t0):
    dt = perf_counter() - t0
    timing[key] += dt
    step_timing[key] += dt
#
# ============================================================
# TRACKING DRIVER
# ============================================================
#
def run_case(
    input_dir,
    res,
    is_2d,
    lbox_v,
    lmin_v=None,
    N_start=0,
    N_end=None,
    st=1,
    min_cells=10,
    periodic_bc=(1, 1, 1),
    order=1,
    event_types=('continuation', 'breakup', 'coalescence'),
    do_debug=0,
    outdir=None,
    overwrite=True,
):
    """
    order = 1 : first-order method. Input: _v1 tagging output.
    order = 2 : second-order method. Input: _v2 tagging output.
    """
    #
    # ========================================================
    # 0. Normalize and validate inputs
    # ========================================================
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
    res = int(res)
    is_2d = int(is_2d)
    N_start = int(N_start)
    st = int(st)
    min_cells = int(min_cells)
    do_debug = int(do_debug)
    #
    if st < 1:
        raise ValueError('st must be >= 1.')
    if N_start < 0:
        raise ValueError('N_start must be >= 0.')
    if is_2d not in (0, 1):
        raise ValueError('is_2d must be 0 or 1.')
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
    allowed_event_types = {'continuation', 'breakup', 'coalescence'}
    event_types = tuple(event_types)
    if len(event_types) == 0:
        raise ValueError('At least one candidate event type must be enabled.')
    if len(set(event_types)) != len(event_types):
        raise ValueError('event_types contains duplicates.')
    if not set(event_types).issubset(allowed_event_types):
        raise ValueError(f'event_types must be chosen from {sorted(allowed_event_types)}.')
    #
    order = int(order)
    if order not in (1, 2):
        raise ValueError('order must be 1 (first order) or 2 (second order).')
    #
    if outdir is None:
        outdir = input_dir.parent / f'event_output_v{order}'
    else:
        outdir = Path(outdir).expanduser().resolve()
    os.makedirs(outdir, exist_ok=True)
    #
    tagging_dir = ensure_slash(input_dir)
    work_dir = str(input_dir.parent) + '/'
    #
    initial_genealogy_file = outdir / 'initial_genealogy.json'
    raw_history_file = outdir / 'event_history_raw.out'
    #
    if not overwrite:
        existing = [path for path in (initial_genealogy_file, raw_history_file) if path.exists()]
        if existing:
            raise FileExistsError(
                'Output file(s) already exist and overwrite=False:\n  '
                + '\n  '.join(str(path) for path in existing)
            )
    #
    timing = {
        "extract": 0.0,
        "coefficients": 0.0,
        "candidate_search": 0.0,
        "organize": 0.0,
        "history": 0.0,
        "print": 0.0,
    }
    #
    print("====================================================")
    print("Running generic tracking case")
    print("order       =", order, "(first order)" if order == 1 else "(second order)")
    print("tagging_dir =", tagging_dir)
    print("work_dir    =", work_dir)
    print("res         =", res)
    print("N_start =", N_start, "N_end =", N_end, "st =", st)
    print("====================================================")
    #
    print("Start to load")
    #
    log_tag = get_log(tagging_dir, string="num_bubbles_droplets_f1.out")
    bubble_stat = get_log(tagging_dir, string="bubbles_f1.out")
    #
    # ------------------------------------------------------------
    # Remove rows duplicated by simulation restarts (latest run wins).
    # Must run on the files in their raw order, before any sorting.
    # ------------------------------------------------------------
    #
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
    print("Finish to load")
    #
    def extract(istep):
        return extract_frame(bubble_stat_sorted, bubble_step_index, lbox_v, min_cells,
                             res, istep, is_2d, order)
    #
    box = make_box(lbox_v, lmin_v, periodic_bc)
    #
    if N_end is None:
        N_end = len(log_tag) - st
    else:
        N_end = min(int(N_end), len(log_tag) - st)
    #
    if N_start >= N_end:
        raise ValueError(
            f'No snapshot pairs to process: N_start={N_start}, N_end={N_end}, st={st}, len(log_tag)={len(log_tag)}'
        )
    #
    # ============================================================
    # INITIALIZE GENEALOGY FROM FIRST FRAME
    # ============================================================
    #
    istep_p0 = int(round(float(log_tag[N_start,1])))
    time0 = float(log_tag[N_start,0])
    #
    if istep_p0 not in bubble_step_index:
        raise KeyError(
            f"istep {istep_p0} (log row {N_start}) is not present in "
            "bubbles_f1.out")
    #
    frame0 = extract(istep_p0)
    voll0 = frame0["vol"]
    #
    if len(voll0) == 0:
        raise RuntimeError(
        "Cannot initialize genealogy: no bubbles found in the first frame.\n"
        f"N_start = {N_start}\n"
        f"istep_p0 = {istep_p0}"
        )
    #
    idx_to_id_curr = {}
    id_to_birth_time_curr = {}
    #
    imother0 = int(np.argmax(voll0))
    idx_to_id_curr[imother0] = "b0"
    id_to_birth_time_curr["b0"] = time0
    #
    next_id = 1
    for i in range(len(voll0)):
        if i == imother0:
            continue
        bid = f"b{next_id}"
        idx_to_id_curr[i] = bid
        id_to_birth_time_curr[bid] = time0
        next_id += 1
    #
    Nb_initial = len(voll0)
    #
    # ============================================================
    # SAFETY CHECKS
    # ============================================================
    #
    if len(idx_to_id_curr) != len(voll0):
        raise RuntimeError("Initial genealogy ID count mismatch.")
    #
    if len(id_to_birth_time_curr) != len(voll0):
        raise RuntimeError("Initial genealogy birth-time count mismatch.")
    #
    if len(set(idx_to_id_curr.values())) != len(voll0):
        raise RuntimeError("Duplicate initial genealogy IDs.")
    #
    if set(idx_to_id_curr.values()) != set(id_to_birth_time_curr.keys()):
        raise RuntimeError("Mismatch between initial IDs and birth-time IDs.")
    #
    # ============================================================
    # SAVE INITIAL GENEALOGY
    # ============================================================
    #
    idx_to_id_initial = idx_to_id_curr.copy()
    id_to_birth_time_initial = id_to_birth_time_curr.copy()
    #
    initial_genealogy_file = outdir/"initial_genealogy.json"
    #
    initial_genealogy = {
        "idx_to_id_initial": {str(k):v for k,v in idx_to_id_initial.items()},
        "id_to_birth_time_initial": {k:float(v) for k,v in id_to_birth_time_initial.items()},
        "Nb_initial": int(Nb_initial),
        "istep_initial": int(istep_p0),
        "time_initial": float(time0),
        "is_2d": int(is_2d)}
    #
    with open(initial_genealogy_file,"w") as f:
        json.dump(initial_genealogy,f,indent=2)
    #
    # ============================================================
    # RAW HISTORY
    # ============================================================
    #
    event_history_raw = []
    raw_history_file = outdir/"event_history_raw.out"
    #
    with open(raw_history_file,"w") as f:
        pass
    #
    # ============================================================
    # LOOP OVER TIME
    # ============================================================
    #
    for p in range(N_start, N_end, st):
        #
        step_timing = {key: 0.0 for key in timing}
        #
        istep_p0 = int(round(float(log_tag[p + 0, 1])))
        istep_p1 = int(round(float(log_tag[p + st, 1])))
        #
        time_p0 = log_tag[p + 0, 0]
        time_p1 = log_tag[p + st, 0]
        #
        DeltaT = time_p1 - time_p0
        #
        if istep_p1 not in bubble_step_index:
            raise KeyError(
                f"istep {istep_p1} (log row {p + st}) is not present in "
                "bubbles_f1.out")
        #
        # ========================================================
        # 0. Extract p1 bubble statistics
        # ========================================================
        #
        t0 = perf_counter()
        #
        print("0. Start extraction of the bubble statistics")
        #
        frame1 = extract(istep_p1)
        #
        Nb_tag0, Nb_rem0 = len(frame0["vol"]), frame0["n_removed"]
        Nb_tag1, Nb_rem1 = len(frame1["vol"]), frame1["n_removed"]
        #
        print("******************************************************")
        print("p =", p, "istep_p0 =", istep_p0, "istep_p1 =", istep_p1)
        print("Time_p0 =", time_p0, "Time_p1 =", time_p1)
        print("Nb_tag0 =", Nb_tag0, "Nb_tag1 =", Nb_tag1)
        print("Nb_rem0 =", Nb_rem0, "Nb_rem1 =", Nb_rem1)
        print("******************************************************")
        #
        # the tagging log must agree with the bubble file
        Nb_log0 = int(1 + log_tag[p + 0, 3]) - Nb_rem0
        Nb_log1 = int(1 + log_tag[p + st, 3]) - Nb_rem1
        if Nb_tag0 != Nb_log0:
            raise ValueError(f"Inconsistent number of bubbles before event: "
                             f"tagging={Nb_tag0}, log={Nb_log0} (event {p})")
        if Nb_tag1 != Nb_log1:
            raise ValueError(f"Inconsistent number of bubbles after event: "
                             f"tagging={Nb_tag1}, log={Nb_log1} (event {p})")
        #
        print("0. End extraction of the bubble statistics")
        add_timing(timing, step_timing, "extract", t0)
        #
        # ========================================================
        # 1. Error constants C_V, C_X of each event type
        # ========================================================
        #
        t0 = perf_counter()
        print("1. Start definition of event coefficients")
        #
        C_V, C_X = {}, {}
        for event_type in event_types:
            C_V[event_type], C_X[event_type] = calibrate(
                frame0, frame1, float(DeltaT), box, event_type, order)
            if do_debug == 1:
                print(event_type, "C_V =", C_V[event_type], "C_X =", C_X[event_type])
        #
        print("1. End definition of event coefficients")
        add_timing(timing, step_timing, "coefficients", t0)
        #
        # ========================================================
        # 2-4. Candidate search and organization
        #
        # The error bounds use min(g, C) for g = 1, 2, 4, ... until an exact
        # cover exists; once g reaches the largest constant every candidate
        # within C is kept. Costs do not depend on g.
        # ========================================================
        #
        gate = 1.0
        gate_max = max(list(C_V.values()) + list(C_X.values()))
        n_gate_tries = 0
        while True:
            n_gate_tries += 1
            gate_now = None if gate >= gate_max else gate
            #
            t0 = perf_counter()
            print("2. Start candidate search")
            candidates = find_candidates(frame0, frame1, float(DeltaT), C_V, C_X, box,
                                         event_types, gate=gate_now, order=order)
            if do_debug == 1:
                for event_type in EVENT_SHAPE:
                    print(event_type, "candidates =", len(candidates[event_type]["cost"]))
            print("2. End candidate search")
            add_timing(timing, step_timing, "candidate_search", t0)
            #
            t0 = perf_counter()
            print("4. Start organization step")
            organized = organize_events(Nb_tag0, Nb_tag1, candidates, debug=do_debug)
            add_timing(timing, step_timing, "organize", t0)
            if organized is not None:
                print("4. End organization step")
                break
            if gate_now is None:
                raise RuntimeError(
                    f"No exact event cover at istep {istep_p0} -> {istep_p1} even with "
                    "every candidate kept (run with --do_debug 1 for details).")
            print(f"   candidate gate g = {gate:g}: no exact cover, doubling g")
            gate *= 2.0
        #
        events, summary = organized
        gate_final = gate if gate_now is not None else None
        print(
            "Candidate gate: g =",
            "all candidates (ungated)" if gate_final is None else f"{gate_final:g}",
            f"after {n_gate_tries} tr{'y' if n_gate_tries == 1 else 'ies'}",
        )
        summary["candidate_gate"] = gate_final
        summary["n_gate_tries"] = int(n_gate_tries)
        if order == 2:
            summary["C_V"] = {key: float(val) for key, val in C_V.items()}
            summary["C_X"] = {key: float(val) for key, val in C_X.items()}
        #
        # ========================================================
        # 6. History record
        # ========================================================
        #
        t0 = perf_counter()
        print("6. Start history record")
        record = build_record(istep_p0, istep_p1, time_p0, time_p1, events, summary,
                              frame0, frame1, order)
        event_history_raw.append(record)
        #
        print("6. End history record")
        #
        add_timing(timing, step_timing, "history", t0)
        #
        # ========================================================
        # 7. Save the event history
        # ========================================================
        #
        t0 = perf_counter()
        #
        print("7. Start saving event_history_raw")
        #
        with open(raw_history_file, "a") as f:
            f.write(
                json.dumps(
                    record,
                    default=lambda x:
                        x.tolist()
                        if isinstance(x, np.ndarray)
                        else x.item()
                        if isinstance(x, np.generic)
                        else x
                )
                + "\n"
            )
        #
        print("7. End saving event_history_raw")
        #
        add_timing(timing, step_timing, "print", t0)
        #
        # ========================================================
        # 8. Copy current event to old events
        # ========================================================
        #
        frame0 = frame1
        #
        # ========================================================
        # 9. Print of the timing of the current time step
        # ========================================================
        #
        print("")
        print("Timing current time step:")
        #
        total_timing_step = sum(step_timing.values())
        #
        for key,val in step_timing.items():
            #
            pct = (
                100.0*val/total_timing_step
                if total_timing_step > 0.0
                else 0.0
            )
            #
            print(
                f"{key:15s}: "
                f"{val:10.6f} s  "
                f"({pct:6.2f}%)"
            )
        #
        print(
            f"{'total':15s}: "
            f"{total_timing_step:10.6f} s  "
            f"(100.00%)"
        )
        #
        print("")

    #
    # ========================================================
    # FINAL TIMING
    # ========================================================
    #
    print('Done')
    print('Cumulative timing:')
    #
    total_timing = sum(timing.values())
    #
    for key, val in timing.items():
        pct = 100.0 * val / total_timing if total_timing > 0.0 else 0.0
        print(f'{key:15s}: {val:10.6f} s  ({pct:6.2f}%)')
    #
    print(f'{"total":15s}: {total_timing:10.6f} s  (100.00%)')
    #
    return {
        'initial_genealogy_file': str(initial_genealogy_file),
        'raw_history_file': str(raw_history_file),
        'outdir': str(outdir),
        'timing': timing,
        'order': order,
    }
#
# ============================================================
# COMMAND LINE INTERFACE
# ============================================================
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
    #
    parser.add_argument('--N_start', type=int, default=0, help='First row/index in num_bubbles_droplets_f1.out.')
    parser.add_argument('--N_end', type=int, default=None, help='Exclusive upper index. Default: process all available pairs.')
    parser.add_argument('--st', type=int, default=1, help='Stride between snapshots.')
    parser.add_argument('--min_cells', type=int, default=10, help='Minimum number of cells for a DPS.')
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
        choices=('continuation', 'breakup', 'coalescence'),
        default=('continuation', 'breakup', 'coalescence'),
        help='Candidate event classes to evaluate. Default: all three.',
    )
    parser.add_argument('--do_debug', type=int, choices=(0, 1), default=0)
    parser.add_argument(
        '--no_overwrite',
        action='store_false',
        dest='overwrite',
        help='Fail if the main output files already exist.',
    )
    parser.set_defaults(overwrite=True)
    #
    args = parser.parse_args()
    #
    run_case(
        input_dir=args.input_dir,
        res=args.res,
        is_2d=args.is_2d,
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
    )
#
if __name__ == '__main__':
    main()
#
