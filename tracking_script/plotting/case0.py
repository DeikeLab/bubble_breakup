#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#from IPython import get_ipython
#
#ip = get_ipython()
#ip.run_line_magic("clear", "")     # clear console
#
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from param_fig import set_fig_params, cm_u
from helper_functions import get_log,build_connected_curves
from matplotlib.collections import LineCollection
import subprocess
#
# ============================================================
# PARAMETERS
# ============================================================
#
fig_dir = "/home/nn8802/Documents/document_postdoc/articles/AMR_tracking/figures/"
set_fig_params()
#
general_path = Path("/home/nn8802/Documents/simulation/tracking_validation/");
#general_path = Path("/home/nicolo-scapin/numerical_codes/cases_validation/");
case = 0
#
rho_liq = 1.0
mu_liq = 0.01
sigma = 6.944e+00
do_debug = False
res_v = [4,5,6,7,8,9,10,11,12]
#
# ------------------------------------------------------------
# Calibration of C_V and C_X (static case)
# ------------------------------------------------------------
# In a static case there is no t^n -> t^{n+1}: the only "step" is the
# initialisation, i.e. exact shape -> discrete representation at t = 0.
# The ratio measured/estimated at t = 0 is therefore one calibration
# sample. To get statistics (the static analogue of sampling many time
# steps), the same circle is initialised at random sub-cell placements:
#   case0/LXX/shift000, shift001, ...
# each containing the usual tagging_v2/bubbles_f1.out plus a file
# center.out with one line "xc yc R" (exact centre and radius).
# If no shift directories exist, only the base placement is used.
#
Lmin_cal = 4          # first level of the asymptotic range (check p_meas / p_Eq9)
q_cal = 100.0         # percentile reported as a less pessimistic constant
#
# Generation of the list of shifted placements (to set up the runs)
write_shift_list = False
n_shift = 20          # placements per level
seed_shift = 1234
L_domain = 4.0        # CHECK: domain side length, so that dx = L_domain/2**Le
R_shift_rel = 0.0     # optional relative perturbation of R (e.g. 0.02)
#
def save_cropped(fig, path, margin=1):
    """Save fig, then crop the PDF to its real content plus margin (in pt)."""
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    subprocess.run(["pdfcrop", "--margins", str(margin), path, path],
                   check=True, stdout=subprocess.DEVNULL)
#
def read_cells(path):
    cells, current = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if current:
                    cells.append(current)
                    current = []
                continue
            x, y = map(float, line.split()[:2])
            current.append((x, y))
    if current:
        cells.append(current)
    return cells
#
def p_order(e):
    e = np.asarray(e, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(e[:-1]/e[1:])/np.log(2.0)
#
def read_tagging_v2(work_dir):
    """Read tagging_v2/bubbles_f1.out and check the column count."""
    log = get_log(str(work_dir), string="/tagging_v2/bubbles_f1.out")
    log = np.atleast_2d(log)
    if log.shape[1] != 21:
        raise ValueError(
            f"{work_dir}/tagging_v2/bubbles_f1.out has {log.shape[1]} columns; "
            "rerun with the updated my_funcs.h (21 columns expected)"
        )
    return log
#
def calibrate(meas, est, levels, Lmin):
    """C = max of measured/estimated over the asymptotic window Le >= Lmin.
    Returns the per-level ratios, C, the level where C is attained and
    whether C*est >= meas holds on the whole window."""
    meas = np.asarray(meas, dtype=float)
    est = np.asarray(est, dtype=float)
    levels = np.asarray(levels)
    r = meas/est
    win = levels >= Lmin
    C = r[win].max()
    Lstar = levels[win][np.argmax(r[win])]
    ok = bool(np.all(C*est[win] >= meas[win]*(1.0 - 1e-12)))
    return r, C, int(Lstar), ok
#
def collect_ensemble(level):
    """Ratios r_V, r_X over all shifted placements of one level."""
    base = general_path / f"case{case}" / f"L{level:02d}"
    rV, rX = [], []
    for sdir in sorted(base.glob("shift*")):
        f_tag = sdir / "tagging_v2" / "bubbles_f1.out"
        f_cen = sdir / "center.out"
        if not (f_tag.exists() and f_cen.exists()):
            continue
        xc, yc, Rc = np.loadtxt(f_cen).ravel()[:3]
        log = read_tagging_v2(sdir)
        dV_meas = abs(log[0,3] - np.pi*Rc**2)
        dX_meas = np.hypot(log[0,5] - xc, log[0,6] - yc)
        rV.append(dV_meas/log[0,19])
        rX.append(dX_meas/log[0,20])
    return np.array(rV), np.array(rX)
#
def write_shifts(path, levels, xc0, yc0, R0):
    """Random sub-cell offsets of the centre (uniform in [-dx/2, dx/2])."""
    rng = np.random.default_rng(seed_shift)
    with open(path, "w") as f:
        f.write("# level  shift  xc  yc  R\n")
        for lev in levels:
            dx = L_domain/2.0**lev
            for k in range(n_shift):
                ox, oy = rng.uniform(-0.5, 0.5, size=2)*dx
                Rk = R0*(1.0 + R_shift_rel*rng.uniform(-1.0, 1.0))
                f.write(f"{lev:3d}  {k:03d}  {xc0+ox:.15e}  {yc0+oy:.15e}  {Rk:.15e}\n")
#
# ============================================================
# Case B (base)
# ============================================================
#
R = 1.0
L = 2*R   # normalization length
work_dir = general_path/"caseB"
facet  = get_log(str(work_dir) + "/interface/facets/",
                 string="facets_000000000.out")
curves = build_connected_curves(facet[:,0], facet[:,1])
grid   = read_cells(str(work_dir) + "/interface/grid/grid_000000000.out")
#
fig0, ax = plt.subplots(figsize=(9*cm_u, 9*cm_u))
segs = [np.vstack([cell, cell[0]])/L for cell in map(np.asarray, grid)]
ax.add_collection(LineCollection(segs, colors="0.75", linewidths=0.5, zorder=1))
for curve in curves:
    ax.plot(curve[:,0]/L, curve[:,1]/L, color="C3", linewidth=2.0, zorder=10)
#
ax.set_aspect("equal")
#ax.autoscale_view()
ax.set_xlabel(r"$x/L_0$")
ax.set_ylabel(r"$y/L_0$")
#
ax.set_xlim([-1,1]);
ax.set_ylim([-1,1]);
#
save_cropped(fig0, fig_dir+"potato_grid.pdf")
plt.show()
#stop;
#
# ============================================================
# Case 0
# ============================================================
#
fig1, ax = plt.subplots(figsize=(9*cm_u, 9*cm_u))
#
res_vs = [4,5,6]
colors = plt.cm.viridis(np.linspace(0.20,0.88,len(res_vs)))
#
R = 5/5.
lw_grid = np.linspace(2.00, 0.25, len(res_vs))   # coarsest thickest
al_grid = np.linspace(0.10, 1.00, len(res_vs))   # coarsest most opaque
#
for p, res in enumerate(res_vs):
    #
    res_c = f"{res:02d}"
    work_dir = general_path / f"case{case}" / f"L{res_c}"
    #
    facet = get_log(
        str(work_dir) + "/interface/facets/",
        string="facets_000000000.out"
    )
    #
    curves = build_connected_curves(
        facet[:,0],
        facet[:,1]
    )
    #
    res_c1 = f"{res:01d}"
    for j, curve in enumerate(curves):
        #
        ax.plot(
            curve[:,0]/(2*R),
            curve[:,1]/(2*R),
            color=colors[p],
            linewidth=2.0,
            zorder=10,
            label=r"$\mathrm{Le}=$" + f" {res_c1}" if j == 0 else None
        )
    #
    grid = read_cells(str(work_dir)+"/interface/grid/grid_000000000.out")
    for cell in grid:
        #
        xs, ys = zip(*cell)
        ax.plot(np.array(xs)/(2*R), np.array(ys)/(2*R),
                color=colors[p], linewidth=lw_grid[p],
                zorder=len(res_vs) - p, alpha=al_grid[p])
#
# Exact shape
#
theta = np.linspace(0., 2.*np.pi, 500)
#
x_center = 0.2
y_center = 0.1
x_exact = R*np.cos(theta)+x_center
y_exact = R*np.sin(theta)+y_center
vol_true = np.pi*(5/5.)**2
#
ax.plot(
    x_exact/(2*R),
    y_exact/(2*R),
    linestyle=":",
    linewidth=5.5,
    color="k",
    label="Exact shape"
)
#
ax.scatter(
    x_center/(2*R),
    y_center/(2*R),
    color="red",
    zorder=10
)
#
# Dotted lines from center to axes
ax.plot(
    [x_center/(2*R),x_center/(2*R)],
    [-2.0/(2*R),y_center/(2*R)],
    linestyle=":",
    linewidth=1.5,
    color="red"
)
#
ax.plot(
    [-2.0/(2*R),x_center/(2*R)],
    [y_center/(2*R),y_center/(2*R)],
    linestyle=":",
    linewidth=1.5,
    color="red"
)
#
ax.set_xticks([-1.00,-0.50,0,0.50,1.00]);
ax.set_yticks([-1.00,-0.50,0,0.50,1.00]);
ax.set_xlim([-1.0,1.0])
ax.set_ylim([-1.0,1.0])
#
ax.set_xlabel(r"$x/d_0$");
ax.set_ylabel(r"$y/d_0$");
ax.set_aspect("equal")
ax.legend(loc="lower left",fontsize=10)
save_cropped(fig1, fig_dir+"circular_bubble.pdf")
#
# ============================================================
# Volume and position errors: collect the estimates
# ============================================================
#
vol1_res = [];
vol2_res = [];
area_res = [];
Xpos_res = [];
Ypos_res = [];
err1_bub = [];
err2_bub = [];
# Columns of tagging_v2 (0-based), see countDropsBubble_app_v2:
#   18 dV_err 1st order (Eq. 3), 19 dV_err blended (Eq. 5),
#   20 dX_err first-moment bound (Eq. 9).
# All estimates are written WITHOUT C_V / C_X.
dX_res = [];
for p, res in enumerate(res_v):
    #
    res_c = f"{res:02d}"
    work_dir = general_path / f"case{case}" / f"L{res_c}"
    log_vol = get_log(str(work_dir),string="/log_vol.out")
    log_err_v1 = np.atleast_2d(get_log(str(work_dir),string="/tagging_v1/bubbles_f1.out"))
    log_err_v2 = read_tagging_v2(work_dir)
    #
    vol1_res.append(log_err_v1[0,3])
    vol2_res.append(log_err_v2[0,3])
    area_res.append(log_err_v2[0,4])
    Xpos_res.append(log_err_v2[0,5])
    Ypos_res.append(log_err_v2[0,6])
    err1_bub.append(log_err_v2[0,18])
    err2_bub.append(log_err_v2[0,19])
    dX_res.append(log_err_v2[0,20])
    #
#
# ============================================================
# Calibration of C_V and C_X
# ============================================================
#
d0 = 2*R
dX_arr = np.array(dX_res)                       # Eq. (9), without C_X
diff2 = np.fabs(np.array(vol2_res) - vol_true)  # measured volume error
diffX = np.fabs(np.array(Xpos_res) - x_center)
diffY = np.fabs(np.array(Ypos_res) - y_center)
diffR = np.hypot(diffX, diffY)                  # |x_i - x_ex|
#
if write_shift_list:
    shift_file = str(general_path / f"case{case}" / "shifts.txt")
    write_shifts(shift_file, res_v, x_center, y_center, R)
    print(f"Shifted placements written to {shift_file}")
#
# (a) base placement only, asymptotic window Le >= Lmin_cal
r_V, C_V, LV, okV = calibrate(diff2, err2_bub, res_v, Lmin_cal)
r_X, C_X, LX, okX = calibrate(diffR, dX_arr, res_v, Lmin_cal)
#
# (b) ensemble over sub-cell placements (base placement included)
rV_lev, rX_lev = {}, {}
for i, lev in enumerate(res_v):
    rV_s, rX_s = collect_ensemble(lev)
    rV_lev[lev] = np.concatenate([[r_V[i]], rV_s])
    rX_lev[lev] = np.concatenate([[r_X[i]], rX_s])
n_ens = {lev: len(rV_lev[lev]) for lev in res_v}
has_ens = any(n > 1 for n in n_ens.values())
#
win_lev = [lev for lev in res_v if lev >= Lmin_cal]
rV_win = np.concatenate([rV_lev[lev] for lev in win_lev])
rX_win = np.concatenate([rX_lev[lev] for lev in win_lev])
C_V_ens, C_X_ens = rV_win.max(), rX_win.max()
C_V_q, C_X_q = np.percentile(rV_win, q_cal), np.percentile(rX_win, q_cal)
#
print("\nCalibration ratios per level (min / median / max over placements)")
print("Level  n    r_V min    r_V med    r_V max    r_X min    r_X med    r_X max")
for lev in res_v:
    a, b = rV_lev[lev], rX_lev[lev]
    flag = "" if lev >= Lmin_cal else "  (outside window)"
    print(f"L{lev:02d}  {n_ens[lev]:3d}  "
          f"{a.min():.3e}  {np.median(a):.3e}  {a.max():.3e}  "
          f"{b.min():.3e}  {np.median(b):.3e}  {b.max():.3e}{flag}")
#
print(f"\nWindow Le >= {Lmin_cal}")
print(f"Base placement : C_V = {C_V:.4g} (L{LV:02d}, bound holds: {okV}), "
      f"C_X = {C_X:.4g} (L{LX:02d}, bound holds: {okX})")
if has_ens:
    print(f"Ensemble (max) : C_V = {C_V_ens:.4g}, C_X = {C_X_ens:.4g}")
    print(f"Ensemble (p{q_cal:.0f}) : C_V = {C_V_q:.4g}, C_X = {C_X_q:.4g}")
    C_V, C_X = C_V_ens, C_X_ens       # use the ensemble bound from here on
else:
    print("No shift* directories found: using the base placement only.")
print(f"All levels (incl. coarse): max r_V = {np.max(r_V):.4g}, "
      f"max r_X = {np.max(r_X):.4g}")
if C_X < 1.0:
    print(f"Note: C_X = {C_X:.3g} < 1, i.e. Eq. (9) is conservative for the circle "
          "(error cancellation by symmetry); consider clamping C_X at 1.")
#
# ============================================================
# Volume errors: figures
# ============================================================
#
fig2, ax = plt.subplots(figsize=(12*cm_u, 8*cm_u))
#
#ax.plot(
#    res_v,
#    np.array(vol1_res)/vol_true,
#    linestyle="-",
#    marker="x",
#    label="1st order ID algorithm",
#    markersize=10
#)
#
ax.plot(
    res_v,
    np.array(vol2_res)/vol_true,
    linestyle="-",
    marker="x",
    #label="2nd order ID algorithm",
    markersize=10,
    color="tab:blue"
)
#
ax.set_xlabel(r"$\mathrm{Le}$")
ax.set_ylabel(r"$V/V_0$")
#
ax.axhline(
    1,
    linestyle=":",
    linewidth=3,
    label="Exact value",
    color="k"   
)
#
#ax.set_ylim([0.375,1.025])
ax.set_ylim([0.575,1.025])
ax.set_xticks(res_v)
ax.legend(loc="lower right")
plt.tight_layout()
save_cropped(fig2, fig_dir+"volume_variation.pdf")
plt.show()
#
plt.show()
#
# order_true1 = np.log(diff1[:-1]/diff1[1:])/np.log(2.0)
# order_true2 = np.log(diff2[:-1]/diff2[1:])/np.log(2.0)
# order_1st = np.log(np.array(err1_bub[:-1])/np.array(err1_bub[1:]))/np.log(2.0)
# order_2nd = np.log(np.array(err2_bub[:-1])/np.array(err2_bub[1:]))/np.log(2.0)
# #
# for i in range(len(res_v)-1):
#     #
#     print(
#         f"L{res_v[i]} -> L{res_v[i+1]}: "
#         f"p_true1 = {order_true1[i]:.4f}, "
#         f"p_true2 = {order_true2[i]:.4f}, "
#         f"p_1st = {order_1st[i]:.4f}, "
#         f"p_2nd = {order_2nd[i]:.4f}"
#     )
#
# ============================================================
# Position errors: figures
# ============================================================
#
fig4, ax = plt.subplots(figsize=(12*cm_u, 8*cm_u))
#
col_x = "tab:blue"
col_y = "tab:red"
col_r = "k"
#
Xpos_true = x_center;
Ypos_true = y_center;
ax.plot(
    res_v,
    np.array(Xpos_res)/Xpos_true,
    linestyle="-",
    marker="x",
    label=r"Position $x$",
    markersize=12,
    color=col_x,
)
ax.plot(
    res_v,
    np.array(Ypos_res)/Ypos_true,
    linestyle="-",
    marker="x",
    label=r"Position $y$",
    markersize=12,
    color=col_y,
)
#
ax.axhline(
    1,
    linestyle=":",
    linewidth=3,
    label="Exact value",
    color=col_r,
)
#
ax.set_xticks(res_v)
ax.set_yticks([1,1.1,1.2,1.3,1.4])
ax.set_ylim([0.95,1.45])
ax.set_xlabel(r"$\mathrm{Le}$")
ax.set_ylabel(r"$\mathbf{x}/d_0$")
ax.legend()
plt.tight_layout()
save_cropped(fig4, fig_dir+"pos_variation.pdf")
#
plt.show()
#
# Convergence orders of the POSITION errors
#
# order_R  = p_order(diffR)
# order_dX = p_order(dX_arr)
# #
# for i in range(len(res_v)-1):
#     #
#     print(
#         f"L{res_v[i]} -> L{res_v[i+1]}: "
#         f"p_meas = {order_R[i]:.4f}, "
#         f"p_Eq9 = {order_dX[i]:.4f}"
#     )
#
# ============================================================
# Calibration ratios: figure
# ============================================================
#
fig6, ax = plt.subplots(figsize=(12*cm_u, 8*cm_u))
for lev in res_v:
    ax.scatter(np.full(n_ens[lev], lev) - 0.08, rV_lev[lev], marker="o",
               s=18, color="tab:blue", alpha=0.6,
               label=r"$r_V$" if lev == res_v[0] else None)
    ax.scatter(np.full(n_ens[lev], lev) + 0.08, rX_lev[lev], marker="s",
               s=18, color="tab:red", alpha=0.6,
               label=r"$r_X$" if lev == res_v[0] else None)
ax.axhline(C_V, linestyle="--", color="tab:blue", linewidth=1.5,
           label=rf"$C_V={C_V:.2f}$")
ax.axhline(C_X, linestyle="--", color="tab:red", linewidth=1.5,
           label=rf"$C_X={C_X:.2f}$")
ax.axvspan(res_v[0] - 0.5, Lmin_cal - 0.5, color="0.9", zorder=0)
ax.set_yscale("log")
ax.set_xlim([res_v[0] - 0.5, res_v[-1] + 0.5])
ax.set_ylim([10**(-3),10**(0)])
ax.set_xticks(res_v)
ax.set_xlabel(r"$\mathrm{Le}$")
ax.set_ylabel(r"$C_i$")
ax.legend(loc="upper right")
plt.tight_layout()
save_cropped(fig6, fig_dir + "calibration_ratios.pdf")
plt.show()
#
fig3, ax = plt.subplots(figsize=(12*cm_u, 8*cm_u))
#
diff1 = np.fabs(np.array(vol1_res) - vol_true)
#
#ax.plot(
#    res_v,
#    diff1/vol_true,
#    linestyle="-",
#    marker="x",
#    label=r"Measured 1st order $\Delta\mathcal{V}_\mathrm{err}$",
#    markersize=20
#)
#
meas_V = diff2/vol_true                          # measured, normalised
est_V  = max(r_V)*np.array(err2_bub)/vol_true    # calibrated estimate
#
ax.plot(
    res_v,
    meas_V,
    linestyle="-",
    marker="x",
    label=r"Measured $\Delta\mathcal{V}_\mathrm{err}$",
    markersize=12,
    color="tab:blue",
)
#
#ax.plot(
#    res_v,
#    np.array(err1_bub)/vol_true,
#    linestyle="-",
#    marker="x",
#    label=r"Estimated $\Delta\mathcal{V}_\mathrm{err}$ (1st order)",
#   markersize=20
#)
#
ax.plot(
    res_v,
    est_V,
    linestyle="--",
    marker="s",
    label=r"Estimated $\Delta\mathcal{V}_\mathrm{err}$",
    markersize=12,
    color="k",
    markerfacecolor="none",
)
#
res_arr = np.asarray(res_v,dtype=float)
#
Lmin_guide = 8.0
Lmax_guide = 10.0
L0 = 8.0
#
mag_1st = 200.5e-2
mag_2nd = 200.5e-2
res_guide = np.linspace(Lmin_guide,Lmax_guide,100)
guide_1st = mag_1st*2.0**(-1.0*(res_guide-L0))
guide_2nd = 2e-3*mag_2nd*2.0**(-2.0*(res_guide-L0))
#
# ax.plot(
#     res_guide,
#     guide_1st,
#     linestyle=":",
#     linewidth=2,
#     color="k",
#     #label=r"$O(\Delta x)$"
# )
#
ax.plot(
    res_guide,
    guide_2nd,
    linestyle=":",
    linewidth=2,
    color="grey",
    #label=r"$O(\Delta x^2)$"
)
#
# First-order guide label
# ax.text(
#     res_guide[-1] + 0.05,
#     guide_1st[-1],
#     r"$\mathcal{O}(\Delta x)$",
#     fontsize=16,
#     va="center",
#     ha="left"
# )
#
# Second-order guide label
ax.text(
    1.02*res_guide[-1],
    1.00*guide_2nd[-1],
    r"$\mathcal{O}(\Delta x^2)$",
    #fontsize=16,
    va="center",
    ha="left",
)
#
ax.set_yscale("log")
#
ax.set_xlabel(r"$\mathrm{Le}$")
ax.set_ylabel(r"$\Delta\mathcal{V}_\mathrm{err}/V_0$")
#
ax.set_ylim([10**(-6),10**(0)])
ax.set_xlim([3.75,12.25])
ax.set_xticks(res_v)
#ax.set_yticks([10**(-6),10**(-5),10**(-4),10**(-3),10**(-2),10**(-1),10**(-0),10**(+1)])
#
ax.legend(
    loc="upper right",fontsize=10)
#
# Inset: zoom on a few levels, where the two curves are almost superposed
#
zoom_x = (9.8, 10.2)                   # Le range of the zoom (one level gives the strongest magnification)
inset_bounds = [0.06, 0.09, 0.36, 0.36] # [x0, y0, w, h] in axes fraction (empty lower-left corner)
zoom_pad = 1.25                        # vertical margin of the zoom (multiplicative)
#
axins = ax.inset_axes(inset_bounds)
axins.plot(res_v, meas_V, linestyle="-", marker="x", markersize=10,
           color="tab:blue")
axins.plot(res_v, est_V, linestyle="--", marker="s", markersize=10,
           color="k", markerfacecolor="none")
axins.set_yscale("log")
#
in_zoom = (np.asarray(res_v) >= zoom_x[0]) & (np.asarray(res_v) <= zoom_x[1])
y_lo = min(meas_V[in_zoom].min(), est_V[in_zoom].min())
y_hi = max(meas_V[in_zoom].max(), est_V[in_zoom].max())
axins.set_xlim(zoom_x)
axins.set_ylim(y_lo/zoom_pad, y_hi*zoom_pad)
axins.set_xticks([lev for lev in res_v if zoom_x[0] <= lev <= zoom_x[1]])
axins.tick_params(labelsize=8)
axins.yaxis.set_major_formatter(plt.NullFormatter())   # zoom spans < 1 decade:
axins.yaxis.set_minor_formatter(plt.NullFormatter())   # y labels would only clutter
axins.set_facecolor("white")
ax.indicate_inset_zoom(axins, edgecolor="0.4")
#
plt.tight_layout()
save_cropped(fig3, fig_dir+"volume_error.pdf")
#
fig5, ax = plt.subplots(figsize=(12*cm_u, 8*cm_u))
#
# Position error: measured centroid shift vs. the first-moment bound of
# Eq. (9). Both are normalised with the diameter d0 = 2R (normalising
# with the centroid coordinates would make the result depend on where
# the circle is placed). Eq. (9) bounds the NORM of the centroid shift,
# so the comparison is with |x_i - x_ex|; the components are shown for
# reference.
#
# d0, dX_arr, diffX, diffY, diffR and C_X come from the calibration block.
#
col_x = "tab:blue"
col_y = "tab:red"
col_r = "k"
#
# measured: solid + markers
ax.plot(res_v, diffX/d0, linestyle="-", marker="x", markersize=12,
        color=col_x, label=r"Measured $|\Delta x|$")
ax.plot(res_v, diffY/d0, linestyle="-", marker="x", markersize=12,
        color=col_y, label=r"Measured $|\Delta y|$")
#
# estimated, Eq. (9): raw and calibrated
ax.plot(res_v, max(r_X)*dX_arr/d0, linestyle="--", marker="s", markersize=10,
        markerfacecolor="none", color=col_r,
        label=r"Estimated $\Delta\mathcal{X}_\mathrm{err}$")
#
# O(dx^2) guide
res_guide = np.linspace(Lmin_guide,Lmax_guide,100)
guide_pos = max(r_X)*2.0*(dX_arr[res_v.index(int(L0))]/d0)*2.0**(-2.0*(res_guide-L0))*2.0
ax.plot(res_guide, guide_pos, linestyle=":", linewidth=2, color="gray")
ax.text(1.01*res_guide[-1], 1.1*guide_pos[-1], r"$\mathcal{O}(\Delta x^2)$", 
        va="center", ha="left")
#
ax.set_yscale("log")
ax.set_xlabel(r"$\mathrm{Le}$")
ax.set_ylabel(r"$\Delta\mathcal{X}_\mathrm{err}/d_0$")
ax.set_xlim([3.75,12.25])
ax.set_ylim([10**(-8),10**(0)])
ax.set_xticks(res_v)
ax.legend(loc="upper right",fontsize=10)
#
plt.tight_layout()
save_cropped(fig5, fig_dir + "pos_error.pdf")
plt.show()