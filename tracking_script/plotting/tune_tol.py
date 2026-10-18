#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ============================================================
# PAIRING TOLERANCE SWEEP
#
#     python3 tune_tol.py 1450
#     python3 tune_tol.py 1450 5
#
# The search radius for joining facet endpoints is
# tol_factor * h, where h is the median facet length (~ the
# grid spacing).
#
# Too SMALL and neighbouring facets never join, so the
# interface shatters -- that was the original 1e-6.
#
# Too LARGE and the two sides of a thin ligament, which are
# only about one cell apart, get joined to each other instead
# of along the contour. The walk then never closes, the region
# has no cycle, and the bubble cannot be claimed by
# containment.
#
# The real endpoint mismatch is 1e-6 to 1e-4, i.e. 0.0001 to
# 0.01 cells, so there is a lot of room below the current 2.0.
#
# What to look for: the largest tol_factor for which EVERY
# big region is closed, then back off from it.
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
from tracker_colors import (
    build_connected_regions,
    region_area_centroid,
    region_perimeter,
    report_region_cycles,
)
#
istep_check = 1450
case = 5
#
xlim = (-5.,5.)
ylim = (-5.,5.)
periodic = True
#
D_REAL = 0.30
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
#
Lx = xlim[1]-xlim[0]
Ly = ylim[1]-ylim[0]
#
filename = os.path.join(str(facets_path),f"facets_{istep_check:09d}.out")
#
data = np.loadtxt(filename)
x,y = data[:,0],data[:,1]
#
pts = np.stack((x,y),axis=1)
nseg = len(pts)//2
p0,p1 = pts[0:2*nseg:2],pts[1:2*nseg:2]
#
d = p1-p0
#
if periodic:
    d[:,0] -= Lx*np.round(d[:,0]/Lx)
    d[:,1] -= Ly*np.round(d[:,1]/Ly)
#
lengths = np.sqrt(np.sum(d*d,axis=1))
h = float(np.median(lengths[lengths > 0.]))
#
print("")
print("==============================================")
print("TOLERANCE SWEEP  step =",istep_check,"  case =",case)
print("==============================================")
print(f"segments = {nseg}   h_median = {h:.6f}")
print("")
print(f"{'tol_fac':>8}{'tol':>10}{'regions':>9}"
      f"{'big':>6}{'closed':>8}{'open':>6}{'closed area':>13}")
#
factors = [0.05,0.1,0.2,0.3,0.5,0.75,1.0,1.5,2.0,4.0,8.0]
#
rows = []
#
for f in factors:
    #
    regions = build_connected_regions(
        x,y,tol=f*h,
        Lx=Lx if periodic else 0.,
        Ly=Ly if periodic else 0.,
    )
    #
    n_big = 0
    n_closed = 0
    n_open = 0
    A_tot = 0.
    #
    for region in regions:
        #
        D_eq = region_perimeter(region,Lx,Ly)/np.pi
        #
        if D_eq < D_REAL:
            continue
        #
        n_big += 1
        #
        A,cx,cy,ok = region_area_centroid(region,Lx,Ly)
        #
        if ok:
            n_closed += 1
            A_tot += A
        else:
            n_open += 1
    #
    rows.append((f,len(regions),n_big,n_closed,n_open,A_tot))
    #
    print(f"{f:>8.2f}{f*h:>10.5f}{len(regions):>9}"
          f"{n_big:>6}{n_closed:>8}{n_open:>6}{A_tot:>13.4f}")
#
# ------------------------------------------------------------
# Best choice: every big region closed, most area recovered
# ------------------------------------------------------------
#
good = [r for r in rows if r[4] == 0 and r[2] > 0]
#
print("")
print("==============================================")
#
if good:
    #
    best = max(good,key=lambda r:r[5])
    #
    print("every big region closes at tol_factor =",
          [f"{r[0]:g}" for r in good])
    print("most closed area at tol_factor =",best[0],
          f" (area {best[5]:.4f})")
    print("")
    print("=> set tol_factor to that value in combined_movie_tree.py")
#
else:
    #
    print("NO tol_factor closes every big region.")
    print("The open one is not a tolerance problem -- it is")
    print("probably a genuinely open interface (a film that")
    print("leaves the domain, or facets missing from the dump).")
#
print("==============================================")
#
# ------------------------------------------------------------
# Detail at the current default, for comparison
# ------------------------------------------------------------
#
print("")
print("DETAIL AT tol_factor = 2.0 (current default)")
print(f"{'#':>4}{'D_eq':>9}{'area':>10}{'cx':>10}{'cy':>10}{'closed':>8}{'cycles':>8}")
#
regions = build_connected_regions(
    x,y,tol=2.0*h,
    Lx=Lx if periodic else 0.,
    Ly=Ly if periodic else 0.,
)
#
for iregion,region in enumerate(regions):
    #
    D_eq = region_perimeter(region,Lx,Ly)/np.pi
    #
    if D_eq < D_REAL:
        continue
    #
    A,cx,cy,ok = region_area_centroid(region,Lx,Ly)
    ncyc = len(region.get("loops") or [])
    #
    print(f"{iregion:>4}{D_eq:>9.4f}{A:>10.4f}{cx:>10.4f}{cy:>10.4f}"
          f"{str(ok):>8}{ncyc:>8}")
#
# ------------------------------------------------------------
# Raw cycle numbers for the regions that were REJECTED
#
# region_area_centroid returns zeros on rejection, which hides
# the measurement. These are the actual values.
# ------------------------------------------------------------
#
print("")
print("RAW CYCLE NUMBERS FOR REJECTED REGIONS")
#
for iregion,region in enumerate(regions):
    #
    D_eq = region_perimeter(region,Lx,Ly)/np.pi
    #
    if D_eq < D_REAL:
        continue
    #
    if region_area_centroid(region,Lx,Ly)[3]:
        continue
    #
    report_region_cycles(region,Lx,Ly,label=str(iregion))
#
print("==============================================")
