#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ============================================================
# SINGLE FRAME DIAGNOSTIC  (standalone)
#
# Loads everything itself. Just run it:
#
#     python3 diagnose_frame.py
#     python3 diagnose_frame.py 2700
#     python3 diagnose_frame.py 2700 3
#                                |    |
#                             step   case
#
# or in Spyder:   %runfile diagnose_frame.py
#                 (edit istep_check / case below)
#
# Answers two questions about one frame:
#
#   1. Is a grey contour grey because the MOVIE failed to claim
#      it, or because the TRACKER has no ID for it?
#
#   2. Does any ID own LESS enclosed area than its tracked
#      volume? That is the signature of VOF tagging merging two
#      touching blobs into one component: one ID, one centroid,
#      but two separate closed loops in the facets. The loop the
#      centroid misses has no owner and comes out grey.
# ============================================================
#
import os
import sys
import json
#
from pathlib import Path
#
import numpy as np
#
from fun_track_fast import clean_genealogy_from_raw_history
#
from tracker_colors import (
    validate_raw_history,
    ids_at_record,
    build_tracked_state,
    build_connected_regions,
    region_area_centroid,
    region_perimeter,
    assign_regions_by_containment,
    make_step_lookup,
)
#
# ============================================================
print("[diagnose] SCRIPT VERSION 2026-09-12b  (aligned idx_to_id_history)")
# ============================================================
#
#
# ============================================================
# SETTINGS  (command line overrides these)
# ============================================================
#
istep_check = 3650
case = 5
#
rho_liq = 1.0
mu_liq = 0.01
sigma = 6.944e+00
#
xlim = (-5.,5.)
ylim = (-5.,5.)
periodic = True
#
D_REAL = 0.30              # a CLOSED region this big is a real bubble
AREA_RATIO_FLAG = 1.10     # volume/claimed-area above this is a deficit
#
if len(sys.argv) > 1:
    istep_check = int(sys.argv[1])
#
if len(sys.argv) > 2:
    case = int(sys.argv[2])
#
work_dir = Path(
    f"/home/nicolo-scapin/numerical_codes/cases_validation/case{case}"
)
#
facets_path = work_dir/"interface/facets"
raw_history_file = work_dir/"event_output/event_history_raw.out"
initial_genealogy_file = work_dir/"event_output/initial_genealogy.json"
#
Lx = xlim[1]-xlim[0]
Ly = ylim[1]-ylim[0]
#
# ============================================================
# LOAD
# ============================================================
#
with open(initial_genealogy_file,"r") as f:
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
with open(raw_history_file,"r") as f:
    for line in f:
        if line.strip():
            event_history_raw.append(json.loads(line))
#
print("case",case,"  raw records:",len(event_history_raw))
#
validate_raw_history(event_history_raw)
#
genealogy = clean_genealogy_from_raw_history(
    event_history_raw,
    idx_to_id_initial,
    id_to_birth_time_initial,
    rho_liq,
    mu_liq,
    sigma,
    is_2d,
    debug=False,
)
#
diameter_history = genealogy["diameter_history"]
idx_to_id_history = genealogy["idx_to_id_history"]
#
tracked_state = build_tracked_state(
    event_history_raw,
    diameter_history,
    xlim=xlim,
    ylim=ylim,
    idx_to_id_history=idx_to_id_history,
    idx_to_id_initial=idx_to_id_initial,
    is_2d=is_2d,
)
#
# ============================================================
# THE FRAME
# ============================================================
#
filename = os.path.join(str(facets_path),f"facets_{istep_check:09d}.out")
#
if not os.path.exists(filename):
    raise RuntimeError(
        f"no facet file:\n  {filename}\n"
        "check the step number and the facets folder name"
    )
#
data = np.loadtxt(filename)
x,y = data[:,0],data[:,1]
#
regions = build_connected_regions(
    x,y,None,
    Lx=Lx if periodic else 0.,
    Ly=Ly if periodic else 0.,
)
#
lookup = make_step_lookup(tracked_state,max_step_gap=0)
istep_used,tracked_now = lookup(istep_check)
#
# ------------------------------------------------------------
# Raw record for this step: do the arrays agree in length?
#
# build_tracked_state drops a component whose index falls
# outside the position arrays. It has an ID, it exists, but it
# gets no tracked entry -- so its contour is drawn grey with no
# way to tell it apart from real debris.
# ------------------------------------------------------------
#
steps_p1 = [int(r["istep_p1"]) for r in event_history_raw]
#
print("")
print("==============================================")
print("RAW RECORD AT step =",istep_check)
print("==============================================")
#
if istep_check in steps_p1:
    #
    ih = steps_p1.index(istep_check)
    rec = event_history_raw[ih]
    #
    n_vol = len(rec["voll1"])
    n_x = len(rec["xpos1"])
    n_y = len(rec["ypos1"])
    ids_here = ids_at_record(idx_to_id_history,event_history_raw,ih)
    n_ids = len(ids_here)
    #
    print(f"voll1 = {n_vol}   xpos1 = {n_x}   ypos1 = {n_y}")
    print(f"ids assigned by genealogy = {n_ids}")
    print(f"entries in tracked_state  = {len(tracked_now)}")
    #
    if n_x != n_vol or n_y != n_vol:
        print("")
        print("*** LENGTH MISMATCH: Basilisk wrote fewer positions")
        print("*** than volumes. Components beyond the end of the")
        print("*** position arrays are dropped and show up grey.")
    #
    dropped = [
        i for i in ids_here
        if int(i) >= min(n_x,n_y)
    ]
    #
    if dropped:
        print("")
        print("components with an ID but no position:",dropped)
        for i in dropped:
            print(f"   idx {i}: volume={rec['voll1'][int(i)]:.5f}"
                  f"  id={ids_here[i]}")
    #
    if n_ids != len(tracked_now):
        print("")
        print(f"{n_ids-len(tracked_now)} component(s) have an ID but no")
        print("entry in tracked_state -> drawn grey.")
    #
    print("")
    print("raw component volumes:",
          [round(float(v),5) for v in rec["voll1"]])
#
else:
    print("step not present in the raw history")
#
print("==============================================")
#
assignment,stats = assign_regions_by_containment(
    regions,tracked_now,
    Lx=Lx if periodic else 0.,
    Ly=Ly if periodic else 0.,
)
#
print("")
print("==============================================")
print("FRAME DIAGNOSTIC  step =",istep_check)
print("==============================================")
print("regions total     =",len(regions))
print("tracked bubbles   =",len(tracked_now))
print("assigned          =",len(assignment))
print("stats             =",
      {k:stats[k] for k in ("contained","fallback","conflict",
                           "deficit_filled","unclaimed")})
#
# ------------------------------------------------------------
# Closed regions big enough to be real bubbles
# ------------------------------------------------------------
#
print("")
print("CLOSED REGIONS (D_eq >",D_REAL,")")
print(f"{'#':>4}{'D_eq':>9}{'area':>10}{'cx':>10}{'cy':>10}  owner")
#
n_real = 0
n_real_grey = 0
n_open = 0
#
for iregion,region in enumerate(regions):
    #
    A,cx,cy,ok = region_area_centroid(region,Lx,Ly)
    D_eq = region_perimeter(region,Lx,Ly)/np.pi
    #
    if D_eq < D_REAL:
        continue
    #
    if not ok:
        n_open += 1
        continue
    #
    n_real += 1
    owner = assignment.get(iregion,"--- GREY ---")
    #
    if iregion not in assignment:
        n_real_grey += 1
    #
    print(f"{iregion:>4}{D_eq:>9.4f}{A:>10.4f}{cx:>10.4f}{cy:>10.4f}  {owner}")
#
print("")
print("open/fragmented regions above D_eq (not real bubbles) =",n_open)
#
# ------------------------------------------------------------
# Volume owned vs volume tracked
# ------------------------------------------------------------
#
print("")
print("TRACKED BUBBLES vs THE AREA THEY CLAIMED")
print("")
print(f"{'id':>30}{'vol':>9}{'claimed A':>11}{'ratio':>8}{'nraw':>6}  regions")
#
owner_of = {}
#
for ireg,pid in assignment.items():
    owner_of.setdefault(pid,[]).append(ireg)
#
suspect = 0
#
for pid,info in tracked_now.items():
    #
    regs_of_pid = owner_of.get(pid,[])
    #
    A_tot = 0.
    #
    for ireg in regs_of_pid:
        A,_,_,ok = region_area_centroid(regions[ireg],Lx,Ly)
        if ok:
            A_tot += A
    #
    ratio = info["volume"]/A_tot if A_tot > 0. else float("inf")
    #
    flag = ""
    #
    if ratio > AREA_RATIO_FLAG:
        flag = "  <== claimed area too small"
        suspect += 1
    #
    print(f"{str(pid):>30}{info['volume']:>9.4f}{A_tot:>11.4f}"
          f"{ratio:>8.2f}{info['nraw']:>6}  {regs_of_pid}{flag}")
#
print("")
print("==============================================")
print("closed real bubbles      =",n_real)
print("of which GREY            =",n_real_grey)
print("tracked bubbles          =",len(tracked_now))
print("area-deficit bubbles     =",suspect)
#
if suspect > 0:
    print("")
    print("=> An ID owns LESS area than its tracked volume.")
    print("   One VOF component, two facet loops: that ID should")
    print("   also own the adjacent grey loop.")
elif n_real_grey > 0:
    print("")
    print("=> ",n_real_grey,"closed bubble(s) unclaimed, with every")
    print("   tracked ID's area accounted for. Those contours")
    print("   have no branch in the genealogy.")
    print("   (An ID may legitimately own several loops, so the")
    print("    raw counts above need not match.)")
else:
    print("")
    print("=> Every closed bubble is coloured in this frame.")
#
print("==============================================")
