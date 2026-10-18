#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ============================================================
# DUMP TRACKED STATE
#
#     python3 dump_tracked.py 1900 1950
#     python3 dump_tracked.py 1900 1950 --case 5
#
# Prints, for each step, every tracked ID with its position and
# volume -- and then a side-by-side comparison showing which
# IDs appear, disappear, or move between the two steps.
#
# Colours in the movie follow the ID, so a contour that is
# geometrically unchanged but changes colour means its ID
# changed. This shows whether that happened in the genealogy.
# ============================================================
#
import sys
import json
#
from pathlib import Path
#
import numpy as np
#
from fun_track_fast import clean_genealogy_from_raw_history
from tracker_colors import (build_tracked_state, make_step_lookup,
                            validate_raw_history, ids_at_record)
#
# ============================================================
print("[dump_tracked] SCRIPT VERSION 2026-09-12b  (aligned idx_to_id_history)")
# ============================================================
#
#
case = 5
steps = [1850,1900,1950]
#
rho_liq = 1.0
mu_liq = 0.01
sigma = 6.944e+00
#
xlim = (-5.,5.)
ylim = (-5.,5.)
#
args = [a for a in sys.argv[1:]]
#
if "--case" in args:
    k = args.index("--case")
    case = int(args[k+1])
    del args[k:k+2]
#
if args:
    steps = [int(a) for a in args]
#
work_dir = Path(
    f"/home/nicolo-scapin/numerical_codes/cases_validation/case{case}"
)
#
with open(work_dir/"event_output/initial_genealogy.json","r") as f:
    initial_data = json.load(f)
#
idx_to_id_initial = {
    int(k):v for k,v in initial_data["idx_to_id_initial"].items()
}
#
id_to_birth_time_initial = {
    k:float(v) for k,v in initial_data["id_to_birth_time_initial"].items()
}
#
is_2d = int(initial_data["is_2d"])
#
event_history_raw = []
#
with open(work_dir/"event_output/event_history_raw.out","r") as f:
    for line in f:
        if line.strip():
            event_history_raw.append(json.loads(line))
#
validate_raw_history(event_history_raw)
#
genealogy = clean_genealogy_from_raw_history(
    event_history_raw,idx_to_id_initial,id_to_birth_time_initial,
    rho_liq,mu_liq,sigma,is_2d,debug=False,
)
#
tracked_state = build_tracked_state(
    event_history_raw,
    genealogy["diameter_history"],
    xlim=xlim,ylim=ylim,
    idx_to_id_history=genealogy["idx_to_id_history"],
    idx_to_id_initial=idx_to_id_initial,
    is_2d=is_2d,
)
#
lookup = make_step_lookup(tracked_state,max_step_gap=0)
#
snap = {}
#
for step in steps:
    #
    used,now = lookup(step)
    snap[step] = now
    #
    print("")
    print("==============================================")
    print("TRACKED STATE  step =",step)
    print("==============================================")
    print(f"{'id':>32}{'volume':>10}{'x':>10}{'y':>10}{'nraw':>6}")
    #
    for pid,info in sorted(now.items(),key=lambda kv:-kv[1]["volume"]):
        print(f"{str(pid):>32}{info['volume']:>10.5f}"
              f"{info['x']:>10.4f}{info['y']:>10.4f}{info['nraw']:>6}")
    #
    # --------------------------------------------------------
    # Raw components and the ID the genealogy gave each one.
    #
    # This is the mapping itself, before any grouping, so a
    # wrong label is visible directly instead of being inferred
    # from a summed volume.
    # --------------------------------------------------------
    #
    steps_p1 = [int(r["istep_p1"]) for r in event_history_raw]
    #
    if step in steps_p1:
        #
        ih = steps_p1.index(step)
        rec = event_history_raw[ih]
        ids = ids_at_record(genealogy["idx_to_id_history"],
                            event_history_raw,ih)
        #
        print("")
        print(f"{'idx':>5}{'volume':>10}{'x':>10}{'y':>10}  id from genealogy")
        #
        for i in sorted(ids,key=lambda k:int(k)):
            #
            i = int(i)
            #
            if i >= len(rec["xpos1"]):
                print(f"{i:>5}{'':>10}{'':>10}{'':>10}  {ids[i]}  <== NO POSITION")
                continue
            #
            bid = ids[i]
            #
            tag = ""
            #
            if isinstance(bid,tuple) and len(bid) == 3:
                tag = "  <== TRANSIENT PAIR"
                bid = f"pair({bid[1]},{bid[2]})"
            #
            print(f"{i:>5}{rec['voll1'][i]:>10.5f}{rec['xpos1'][i]:>10.4f}"
                  f"{rec['ypos1'][i]:>10.4f}  {bid}{tag}")
#
# ------------------------------------------------------------
# What changed
# ------------------------------------------------------------
#
if len(steps) == 2:
    #
    a,b = steps
    A,B = snap[a],snap[b]
    #
    print("")
    print("==============================================")
    print(f"CHANGES  {a} -> {b}")
    print("==============================================")
    #
    gone = [p for p in A if p not in B]
    new = [p for p in B if p not in A]
    kept = [p for p in A if p in B]
    #
    print("disappeared:",gone if gone else "none")
    print("appeared   :",new if new else "none")
    #
    print("")
    print("IDs present in both, and how far they moved:")
    print(f"{'id':>32}{'move':>10}{'dvol':>11}")
    #
    for p in sorted(kept,key=lambda p:-A[p]["volume"]):
        #
        dx = B[p]["x"]-A[p]["x"]
        dy = B[p]["y"]-A[p]["y"]
        #
        dist = float(np.hypot(dx,dy))
        dv = B[p]["volume"]-A[p]["volume"]
        #
        flag = "   <== jumped" if dist > 0.3 else ""
        #
        print(f"{str(p):>32}{dist:>10.4f}{dv:>+11.5f}{flag}")
    #
    print("")
    print("If a contour that barely moved changed colour, look for")
    print("its ID in 'disappeared' and a new ID in 'appeared' at")
    print("the same place -- that is the genealogy renaming it,")
    print("not the movie mismatching it.")
    print("==============================================")
