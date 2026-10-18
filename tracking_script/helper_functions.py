#
import numpy as np
import pandas as pd
import os, glob
from collections import defaultdict, deque
from scipy.spatial import ConvexHull
from pathlib import Path
#
import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import matplotlib as mpl
plt.rcParams.update(plt.rcParamsDefault);
from matplotlib import rc;
import matplotlib.pylab as plt;
from matplotlib.ticker import FuncFormatter  
import os;
import array;
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
#
def get_log(work_dir,string):
    #
    log_file = pd.read_csv(work_dir+string, skiprows=0, sep=" "); 
    log_file = log_file.to_numpy();
    #
    return log_file;
#
def make_dist(D, nbins, Lx, MAXLEVEL, do_pdf, Nreal, factor):
    #
    D = np.asarray(D)
    D = D[np.isfinite(D) & (D > 0)]
    if D.size == 0:
        return np.array([]), np.array([])
    #
    dmin = D.min();
    dmax = D.max();
    #
    # log-spaced edges
    edges = np.logspace(np.log10(dmin), np.log10(dmax), nbins + 1)
    centers = np.sqrt(edges[:-1]*edges[1:])
    counts, _ = np.histogram(D, bins=edges, density=False)
    widths = np.diff(edges)
    if do_pdf:
        dist = counts / (counts.sum()*widths)  # pdf w.r.t. D; integrates to 1
    else:
        dist = counts / (Nreal*widths) # number density n(D); integrates to total number of bubbles
        #dist = counts / (widths) # number density n(D); integrates to total number of bubbles
    #
    return dist, centers
#
def make_dist_vol(D, nbins, Lx, MAXLEVEL, do_pdf, Nreal):
    #
    D = np.asarray(D)
    D = D[np.isfinite(D) & (D > 0)]
    if D.size == 0:
        return np.array([]), np.array([])
    #
    Vb = cmpt_vol(D);
    #
    #dmin = 4*Lx/(2**MAXLEVEL)
    #dmax = dmin*1000
    dmin = D.min();
    dmax = D.max();
    #
    # log-spaced edges
    edges   = np.logspace(np.log10(dmin), np.log10(dmax), nbins + 1)
    centers = np.sqrt(edges[:-1]*edges[1:])
    widths  = np.diff(edges)
    Vbin, _ = np.histogram(D, bins=edges, weights=Vb)
    if do_pdf:
        dist = Vbin / (Vbin.sum() * widths)
    else:
        dist = Vbin / (Nreal*widths) 
    #
    return dist, centers, edges
#
def process_exp(name_set1, name_set2, pwd, DeltaMin_exp):
    #
    # load distribution and remove unresolved
    #
    set_n = np.loadtxt(pwd + name_set1, skiprows=1)
    d = set_n[:, 0]
    p = set_n[:, 1]
    #
    mask = d >= DeltaMin_exp
    d_f = d[mask]
    p_f = p[mask]
    #
    # load filament data
    #
    raw = np.loadtxt(pwd + name_set2, skiprows=1)
    run = raw[:, 4].astype(int)
    #
    runs, Nb_per_filament = np.unique(run, return_counts=True)
    Nb_mean = Nb_per_filament.mean()
    #
    dd = np.empty_like(d)
    dd[1:-1] = 0.5 * (d[2:] - d[:-2])
    dd[0] = d[1] - d[0]
    dd[-1] = d[-1] - d[-2]
    #
    nd = Nb_mean * p / dd
    #
    return d, nd, Nb_mean
#
def normalize_dist_exp(dataset):
    #
    d = np.asarray(dataset[:,0])
    N = np.asarray(dataset[:,1])
    #
    # sort just in case
    ind = np.argsort(d)
    d = d[ind]
    N = N[ind]
    #
    # compute bin widths from centers
    dd = np.zeros_like(d)
    #
    dd[1:-1] = 0.5 * (d[2:] - d[:-2])
    dd[0] = d[1] - d[0]
    dd[-1] = d[-1] - d[-2]
    #
    A = np.sum(N * dd)
    n = N / A
    #
    return np.column_stack((d, n))
#
def get_log(work_dir,string):
    #
    log_file = pd.read_csv(work_dir+string, header=None, sep=" "); 
    log_file = log_file.to_numpy();
    #
    return log_file;
#
def cmpt_d(vol,if_2d):
    #
    if(if_2d == 1):
        d = (4.*vol/np.pi)**(1./2.);
    else:
        d = (6.*vol/np.pi)**(1./3.);
    #
    return d;
#
def get_diameters(cases,lx,do_remove_big,do_remove_small,MAXLEVEL,min_diam,istep_target=None):
    #
    D_list = []
    #
    Delta_min = lx / (2**MAXLEVEL)
    min_diam_abs = min_diam * Delta_min
    #
    for case in cases:

        bubbles_all = get_log(
            case.work_dir,
            string="tagging/bubbles_f1.out"
        )

        # use latest available timestep if not provided
        if istep_target is None:
            istep_case = int(np.max(bubbles_all[:, 1]))
        else:
            istep_case = int(istep_target)

        print(istep_case)
        mask = bubbles_all[:, 1].astype(int) == istep_case
        bubbles = bubbles_all[mask, :]

        if bubbles.shape[0] == 0:
            print(f"WARNING: no bubbles found for istep = {istep_case} in {case.work_dir}")
            D_list.append(np.array([]))
            continue

        D = cmpt_d(bubbles[:, 2])

        if do_remove_big == 1 and len(D) > 2:
            max_indices = np.argsort(D)[-2:]
            D = np.delete(D, max_indices)

        if do_remove_small == 1:
            D = D[D >= min_diam_abs]

        D_list.append(D)

    return D_list
#
def get_Nb_ss(cases, lx, MAXLEVEL):
    #
    Nb = []
    dmin = 4*lx/(2**MAXLEVEL)
    for case in cases:
        #
        log_bub = get_log(case.work_dir, string="tagging/num_bubbles_droplets_f1.out")
        istep = int(log_bub[-1,1])
        bubbles = get_log(case.work_dir, string=f"tagging/bubbles_f1_{istep:09d}.out")
        d = cmpt_d(bubbles[:,1])
        it = 0
        for p in range(len(d)):
            if(d[p] >= dmin):
              it += 1;
        Nb.append(it);
    #
    return Nb
#
def get_vol_ss(cases, lx, MAXLEVEL):
    #
    vol = []
    dmin = 4*lx/(2**MAXLEVEL)
    for case in cases:
        #
        log_bub = get_log(case.work_dir, string="tagging/num_bubbles_droplets_f1.out")
        istep = int(log_bub[-1,1])
        bubbles = get_log(case.work_dir, string=f"tagging/bubbles_f1_{istep:09d}.out")
        d = cmpt_d(bubbles[:,1])
        tot_vol = 0
        for p in range(len(d)):
            if(d[p] >= dmin and d[p] < max(d)):
              tot_vol += bubbles[p,1]
        vol.append(tot_vol);
    #
    return vol
#
def get_best_fit(centers,pdf):
    #
    mask = (pdf > 0) & (centers > 0)
    log_centers = np.log10(centers[mask])
    log_pdf = np.log10(pdf[mask])
    slope, intercept = np.polyfit(log_centers, log_pdf, 1)  # linear fit
    A = 10**intercept
    fit_pdf = A*centers**slope
    #
    return fit_pdf,A,slope;
#
def reorder(D_list):
    #
    D_all = np.concatenate(D_list);
    D_all = np.array(D_all);
    D_all = np.sort(D_all);
    #
    return D_all;
#
def build_connected_curves(x,y,tol=1e-6):
    #
    pts = np.stack((x, y), axis=1)
    pts = np.round(pts, 6)
    adj = defaultdict(list)
    #
    for i in range(0,len(pts)-1,2):
        p1, p2 = tuple(pts[i]), tuple(pts[i+1])
        adj[p1].append(p2)
        adj[p2].append(p1)
    visited = set()
    curves = []
    #
    for p in adj:
        if p in visited:
            continue
        comp = []
        stack = [p]
        while stack:
            curr = stack.pop()
            if curr in visited:
                continue
            visited.add(curr)
            comp.append(curr)
            for nb in adj[curr]:
                if nb not in visited:
                    stack.append(nb)
        curves.append(np.array(comp))
    #
    return curves
#
def build_connected_regions(x, y, tol):
    #
    pts = np.stack((x, y), axis=1)
    nseg = len(pts)//2
    #
    seg_points = []
    for k in range(nseg):
       p1 = pts[2*k]
       p2 = pts[2*k + 1]
       seg_points.append((p1, p2))
    nseg = len(seg_points)
    #
    # snap each endpoint to a tolerance-based key
    point_to_segs = defaultdict(list)
    seg_keys = []
    #
    for k, (p1, p2) in enumerate(seg_points):
        k1 = tuple(np.round(np.asarray(p1)/tol).astype(int))
        k2 = tuple(np.round(np.asarray(p2)/tol).astype(int))
        seg_keys.append((k1, k2))

        point_to_segs[k1].append(k)
        point_to_segs[k2].append(k)
    #
    # connected components of segments
    visited = set()
    regions = []
    #
    for k in range(nseg):
        if k in visited:
            continue
        #
        stack = [k]
        seg_ids = []
        verts = []
        #
        while stack:
            s = stack.pop()
            if s in visited:
                continue

            visited.add(s)
            seg_ids.append(s)

            p1, p2 = seg_points[s]
            verts.append(np.asarray(p1))
            verts.append(np.asarray(p2))

            k1, k2 = seg_keys[s]

            for nb in point_to_segs[k1]:
                if nb not in visited:
                    stack.append(nb)

            for nb in point_to_segs[k2]:
                if nb not in visited:
                    stack.append(nb)
        #
        verts = np.array(verts)
        #
        dx = verts[:, 0].max() - verts[:, 0].min()
        dy = verts[:, 1].max() - verts[:, 1].min()
        D_eq = max(dx, dy)
        #
        segs = np.array([[seg_points[s][0], seg_points[s][1]] for s in seg_ids])
        #
        regions.append({
            "seg_ids": seg_ids,
            "segments": segs,
            "vertices": verts,
            "dx": dx,
            "dy": dy,
            "D_eq": D_eq,
            "nseg": len(seg_ids)
        })
    #
    # sort from biggest to smallest
    regions = sorted(regions, key=lambda reg: reg["D_eq"], reverse=True)
    #
    return regions
#
def gen_label(gen):
    if gen == 0:
        return "0th generation"
    if gen == 1:
        return "1st generation"
    if gen == 2:
        return "2nd generation"
    if gen == 3:
        return "3rd generation"
    return f"{gen}th generation"
#
def get_generation(bubble_id):
    """
    Return the generation from a persistent bubble ID.

    Examples
    --------
    0       -> 0
    "0"     -> 0
    "0_1"   -> 1
    "0_1_2" -> 2
    """

    bubble_id = str(bubble_id)

    if bubble_id == "":
        return -1

    return bubble_id.count("_")
#
def draw_facets_from_istep_p0(
    test_case,
    istep_p0,
    time_p0,
    xpos0,
    ypos0,
    idx_to_id_curr,
    lx,
    frames_dir,
    facets_file,
    tol=1e-6,
):

    facets_file = Path(facets_file)

    #expected_name = f"facets_{istep_p0:09d}.out"
    #if facets_file.name != expected_name:
    #    raise RuntimeError(
    #        f"Wrong facets file for istep_p0={istep_p0}:\n"
    #        f"expected: {expected_name}\n"
    #        f"got:      {facets_file.name}"
    #    )

    data = np.loadtxt(facets_file)

    #if data.ndim == 1:
    #    data = data.reshape(1, -1)

    x = data[:, 0]
    y = data[:, 1]

    xpos0 = np.asarray(xpos0)

    regions = build_connected_regions(x, y, tol)

    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.get_cmap("tab10")

    gen_markers = {
        0: "x",
        1: "x",
        2: "x",
        3: "x",
        4: "x",
        5: "x",
        6: "x",
        7: "x",
    }

    all_segments = []

    for reg in regions:
        for seg in reg["segments"]:
            all_segments.append(
                np.column_stack((seg[:, 0] / lx, +seg[:, 1] / lx))
            )
            all_segments.append(
                np.column_stack((seg[:, 0] / lx, -seg[:, 1] / lx))
            )

    if all_segments:
        lc = LineCollection(
            all_segments,
            colors="0.5",
            linewidths=2.0,
            zorder=1,
        )
        ax.add_collection(lc)

    gens_present = set()

    for ib in range(len(xpos0)):
        pid = idx_to_id_curr.get(ib, None)

        if pid is None:
            marker = "x"
            color = "k"
            label_text = "None"
        else:
            gen = get_generation(pid)
            gens_present.add(gen)

            marker = gen_markers.get(gen, "*")
            color = cmap(gen % 10)
            label_text = str(pid)

        xplot = xpos0[ib] / lx
        yplot = 0.0

        ax.scatter(
            xplot,
            yplot,
            marker=marker,
            s=14,
            color=color,
            linewidths=1.0,
            zorder=10,
        )

        ax.text(
            xplot,
            yplot + 0.002,
            label_text,
            fontsize=5,
            ha="center",
            va="bottom",
            color=color,
            zorder=11,
        )

    legend_handles = []

    for gen in sorted(gens_present):
        marker = gen_markers.get(gen, "*")

        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                color=cmap(gen % 10),
                markersize=7,
                label=gen_label(gen),
            )
        )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=7,
            frameon=True,
        )

    ax.set_xlabel(r"$x/L_x$")
    ax.set_ylabel(r"$y/L_x$")
    ax.set_xlim([0,1]);
    ax.set_ylim([-0.2,0.2]);
    ax.set_aspect("equal")
    ax.set_title(fr"$t={time_p0:.4e}$, istep={istep_p0:09d}")

    fig.tight_layout()

    out_file = Path(frames_dir) / f"facets_generation_{istep_p0:09d}.png"
    fig.savefig(out_file, dpi=250)
    plt.close(fig)

    print(f"Wrote {out_file}")
#
def update_only_base(frame_idx,facets_files_base,time_base,Lx,
                     line_base_top,line_base_bot,time_text):
    #
    # add the base
    #
    facets_b = np.loadtxt(facets_files_base[frame_idx])
    x_b, y_b = facets_b[:, 0]/Lx, facets_b[:, 1]/Lx
    curves_b = build_connected_curves(x_b, y_b)
    xb_all, yb_all = [], []
    for c in curves_b:
        xb_all.extend(c[:, 0].tolist() + [np.nan])
        yb_all.extend(c[:, 1].tolist() + [np.nan])
    line_base_top.set_data(xb_all, yb_all)
    line_base_bot.set_data(xb_all, [-yy for yy in yb_all])
    #
    # add the time
    #
    t_nd = time_base[min(frame_idx, len(time_base)-1)]
    time_text.set_text(rf"$t/\sqrt{{\rho_w L_x^3/\sigma}} = {t_nd:.2f}$")
    #
    return [line_base_top, line_base_bot, time_text]
#
def update(frame_idx,facets_files_base,facets_files_perts,time_base,Lx,
           line_base_top,line_base_bot,lines_perts_top,lines_perts_bot,time_text):
    #
    # add the base
    #
    facets_b = np.loadtxt(facets_files_base[frame_idx])
    x_b, y_b = facets_b[:, 0]/Lx, facets_b[:, 1]/Lx
    curves_b = build_connected_curves(x_b, y_b)
    xb_all, yb_all = [], []
    for c in curves_b:
        xb_all.extend(c[:, 0].tolist() + [np.nan])
        yb_all.extend(c[:, 1].tolist() + [np.nan])
    line_base_top.set_data(xb_all, yb_all)
    line_base_bot.set_data(xb_all, [-yy for yy in yb_all])
    #
    # add the perturbations
    #
    for k, facets_list in enumerate(facets_files_perts):
        facets_p = np.loadtxt(facets_list[frame_idx])
        x_p, y_p = facets_p[:, 0]/Lx, facets_p[:, 1]/Lx
        curves_p = build_connected_curves(x_p, y_p)
        xp_all, yp_all = [], []
        for c in curves_p:
            xp_all.extend(c[:, 0].tolist() + [np.nan])
            yp_all.extend(c[:, 1].tolist() + [np.nan])
        lines_perts_top[k].set_data(xp_all, yp_all)
        lines_perts_bot[k].set_data(xp_all, [-yy for yy in yp_all])
    #
    # add the time
    #
    t_nd = time_base[min(frame_idx, len(time_base)-1)]
    time_text.set_text(rf"$t/\sqrt{{\rho_w L_x^3/\sigma}} = {t_nd:.2f}$")
    #
    return [line_base_top, line_base_bot, *lines_perts_top, *lines_perts_bot, time_text]
#
def load_times(case,tot):
    #
    log_file = np.loadtxt(os.path.join(case.work_dir, "log_vol.out"))
    #
    return log_file[::tot,0]/case.td
#
def get_bubble_diameters_vs_time(case, Lx, MAXLEVEL):
    #
    tagging_dir = os.path.join(case.work_dir, "tagging/")
    log_bub = get_log(case.work_dir, string="tagging/num_bubbles_droplets_f1.out")
    #
    bubbles_files = sorted(glob.glob(os.path.join(tagging_dir, "bubbles_f1_*.out")))
    dmin = 4*Lx/(2**MAXLEVEL);
    #
    time_all, D_all = [], []
    for p in range(len(bubbles_files)):
        time  = log_bub[p, 0]
        istep = int(log_bub[p, 1])
        #
        bubble_data = get_log(case.work_dir, string=f"tagging/bubbles_f1_{istep:09d}.out")
        vol = bubble_data[:, 1]
        D = cmpt_d(vol)
        #
        mask = D >= dmin
        D = D[mask]
        #
        time_all.extend([time] * len(D))
        D_all.extend(D)
    #
    return np.array(time_all), np.array(D_all)
#
def Nb_resolved_vs_time(case, Lx, MAXLEVEL):
    #
    log_bub = get_log(case.work_dir, string="tagging/num_bubbles_droplets_f1.out")
    time  = log_bub[:, 0]
    istep = log_bub[:, 1].astype(int)
    #
    dmin = 4.0*Lx/(2**MAXLEVEL)
    #
    Nb_res = np.zeros(len(time), dtype=int)
    for i, stp in enumerate(istep):
        bub = get_log(case.work_dir, string=f"tagging/bubbles_f1_{stp:09d}.out")
        D = cmpt_d(bub[:, 1])
        Nb_res[i] = np.sum(D >= dmin)
    #
    return time, Nb_res
#
def get_bubble_info_vs_time_table(case):
    #
    tagging_dir = os.path.join(case.work_dir, "tagging/")
    log_bub = get_log(case.work_dir, string="tagging/num_bubbles_droplets_f1.out")
    #
    bubbles_files = sorted(
        glob.glob(os.path.join(tagging_dir, "bubbles_f1_*.out"))
    )
    #
    rows = []  # each row: [istep, time, ib, vol, x, y, z]
    #
    for p in range(len(bubbles_files)):
        time  = float(log_bub[p, 0])
        istep = int(log_bub[p, 1])
        #
        bubble_data = get_log(
            case.work_dir,
            string=f"tagging/bubbles_f1_{istep:09d}.out"
        )
        #
        volb = bubble_data[:, 1]
        xpos = bubble_data[:, 2]
        ypos = bubble_data[:, 3]
        zpos = bubble_data[:, 4]
        #
        nb = volb.size
        ib = np.arange(nb, dtype=int)
        #
        block = np.column_stack([
            np.full(nb, istep),
            np.full(nb, time),
            ib,
            volb,
            xpos,
            ypos,
            zpos
        ])
        #
        rows.append(block)
    #
    if len(rows) == 0:
        return np.empty((0, 7))
    #
    return np.vstack(rows)
#
