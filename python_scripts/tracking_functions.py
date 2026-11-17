#
import sys
from collections import deque
import numpy as np
#
EPS = 1.0e-14
TOL = 1.0e-14
# The tracking needs numpy only.
AXI_FACTOR = 2.0*np.pi   # volume of an axisymmetric DPS / volume per radian
EVENT_SHAPE = {"continuation": (1, 1), "breakup": (1, 2), "coalescence": (2, 1),
               "birth": (0, 1), "death": (1, 0)}
#
# bubbles_f1.out, 0-based columns. Common to all layouts:
#   0 t, 1 istep, 2 tag, 3 V, 4 A, 5-7 centroid, 8-10 velocity.
# The other columns depend on the tagging version and on its layout, which
# is recognized from the number of columns:
#   _v1, legacy  (20): 12 dx max, 18 dV_err (first order)
#   _v2, legacy  (21): as _v1, plus 19 dV_hat (blended), 20 dX_hat (first moment)
#   _v1, current (14): 11 dx max, 12 dV_err, 13 extent
#   _v2, current (16): 11 dx max, 12 dV_err, 13 dV_hat, 14 dX_hat, 15 extent
# The current layouts only hold what the tracking uses; the extent, max
# |x_c - x_i| + sqrt(D)/2 Delta over the interfacial cells, is needed by the
# parameter-free contact test (contact = "extent").
#
LAYOUTS = {
    (1, 20): {"dxma": 12, "area_err": 18},
    (2, 21): {"dxma": 12, "area_err": 18, "dV_hat": 19, "dX_hat": 20},
    (1, 14): {"dxma": 11, "area_err": 12, "ext": 13},
    (2, 16): {"dxma": 11, "area_err": 12, "dV_hat": 13, "dX_hat": 14, "ext": 15},
}
BASE_COLS = {"vol": 3, "area": 4, "x": 5, "y": 6, "z": 7, "ux": 8, "uy": 9, "uz": 10}
#
# ------------------------------------------------------------------
# Input files
# ------------------------------------------------------------------
#
def get_log(work_dir, string):
    """Table of numbers separated by single spaces, e.g. bubbles_f1.out."""
    return np.loadtxt(work_dir + string, ndmin=2)
#
def cmpt_d(vol, if_2d, axi=0):
    #
    """
    Equivalent diameter of a DPS of volume vol:
      2D       (if_2d = 1, axi = 0): area,                 d = (4 V / pi)^(1/2)
      2D axi   (if_2d = 1, axi = 1): volume per radian,    d = (12 V)^(1/3)
      3D       (if_2d = 0, axi = 0): volume,               d = (6 V / pi)^(1/3)
      3D axi   (if_2d = 0, axi = 1): volume (2 pi incl.),  d = (6 V / pi)^(1/3)
    The 2D axi diameter is the 3D one of the volume AXI_FACTOR V.
    """
    #
    if if_2d == 1 and axi == 0:
        return (4.0 * vol / np.pi)**(1.0 / 2.0)
    if if_2d == 1:
        vol = AXI_FACTOR * vol
    #
    return (6.0 * vol / np.pi)**(1.0 / 3.0)
#
def build_bubble_stat_index(bubble_stat):
    #
    isteps = np.rint(bubble_stat[:, 1]).astype(np.int64)
    order = np.argsort(isteps, kind="stable")
    steps, start, count = np.unique(isteps[order], return_index=True, return_counts=True)
    index = {int(s): (int(a), int(a + c)) for s, a, c in zip(steps, start, count)}
    #
    return bubble_stat[order], index
#
def remove_restart_duplicates(log_tag, bubble_stat, step_col=1, count_col=3):
    #
    """
    Remove the rows written again after a simulation restart.

    A restart is a log row whose step is not larger than the previous one; a
    log row is kept only if its step is smaller than the start step of every
    later run. The bubble file is walked together with the raw log: each log
    row owns the next block of bubble rows with its step. A block longer than
    the logged count whose step appears again later in the log holds two
    copies written back to back, and is split by the count. Rows that cannot
    be matched to the log are dropped and reported.
    """
    #
    log_tag = np.asarray(log_tag)
    bubble_stat = np.asarray(bubble_stat)
    #
    log_steps = np.rint(log_tag[:, step_col]).astype(np.int64)
    log_counts = np.rint(1.0 + log_tag[:, count_col]).astype(np.int64)
    b_steps = np.rint(bubble_stat[:, step_col]).astype(np.int64)
    n_log, n_b = len(log_steps), len(b_steps)
    #
    last_row = {int(s): r for r, s in enumerate(log_steps)}
    if n_log:
        run_start = np.r_[True, log_steps[1:] <= log_steps[:-1]]
        run_id = np.cumsum(run_start) - 1
        start_steps = log_steps[run_start]
        later_min = np.full(len(start_steps), np.iinfo(np.int64).max, dtype=np.int64)
        for r in range(len(start_steps) - 2, -1, -1):
            later_min[r] = min(later_min[r + 1], start_steps[r + 1])
        valid = log_steps < later_min[run_id]
    else:
        start_steps = np.zeros(0, dtype=np.int64)
        valid = np.zeros(0, dtype=bool)
    #
    if n_b:
        starts = np.flatnonzero(np.r_[True, b_steps[1:] != b_steps[:-1]])
        ends = np.r_[starts[1:], n_b]
        run_end = np.repeat(ends, ends - starts)
    #
    blocks = []
    n_split = 0
    ptr = 0
    for r in range(n_log):
        if ptr >= n_b:
            break
        s = int(log_steps[r])
        if b_steps[ptr] != s:
            continue
        end = int(run_end[ptr])
        n = int(log_counts[r])
        if 0 < n < end - ptr and last_row[s] > r:
            end = ptr + n
            n_split += 1
        blocks.append((ptr, end, r))
        ptr = end
    #
    n_unmatched = n_b - ptr
    keep = [np.arange(a, b) for a, b, r in blocks if valid[r]]
    bubble_clean = bubble_stat[np.concatenate(keep)] if keep else bubble_stat[:0]
    #
    log_rows = np.flatnonzero(valid)
    log_rows = log_rows[np.argsort(log_steps[log_rows], kind="stable")]
    info = {
        "n_restarts": max(len(start_steps) - 1, 0),
        "n_log_rows_dropped": int(n_log - len(log_rows)),
        "n_bubble_blocks_dropped": sum(1 for a, b, r in blocks if not valid[r]),
        "n_bubble_rows_dropped": int(n_b - n_unmatched - len(bubble_clean)),
        "n_back_to_back_splits": int(n_split),
        "n_bubble_rows_unmatched": int(n_unmatched),
        "first_unmatched_step": int(b_steps[ptr]) if n_unmatched > 0 else None,
    }
    #
    return log_tag[log_rows], bubble_clean, info
#
def extract_frame(bubble_stat_sorted, step_index, lbox_v, min_cells, maxlevel,
                  istep, is_2d, order, axi=0):
    #
    """
    DPS of one snapshot, larger than min_cells cells, by decreasing volume.

    Axisymmetric data (axi = 1) are bodies of revolution. With is_2d = 1 the
    tagging gives the volume, area and volume errors per radian: they are
    multiplied by AXI_FACTOR here, so that the frame holds the 3D quantities
    and is processed as 3D (is_2d = 0) from here on. With is_2d = 0 the
    factor is already included.

    Besides the columns of LAYOUTS the frame holds pos, vel (n x 3) and the
    two error estimates without constants used by the tracking:
      order 1: err = area_err (first order),  xerr = area_err / area
      order 2: err = dV_hat (blended),        xerr = dX_hat (first moment)
    """
    #
    ncol = bubble_stat_sorted.shape[1]
    if (order, ncol) not in LAYOUTS:
        raise ValueError(
            f"bubbles_f1.out has {ncol} columns, which is not a known layout of the "
            f"_v{order} tagging output ({sorted(n for o, n in LAYOUTS if o == order)}). "
            f"Is --input_dir pointing to the _v{order} tagging output?")
    cols = {**BASE_COLS, **LAYOUTS[(order, ncol)]}
    #
    i0, i1 = step_index.get(int(istep), (0, 0))
    rows = bubble_stat_sorted[i0:i1]
    frame = {key: rows[:, c].astype(float) for key, c in cols.items()}
    has_ext = "ext" in frame
    #
    keep = cmpt_d(frame["vol"], is_2d, axi) >= min_cells * min(lbox_v) / 2**maxlevel
    n_removed = int(np.count_nonzero(~keep))
    by_vol = np.argsort(-frame["vol"][keep], kind="stable")
    frame = {key: v[keep][by_vol] for key, v in frame.items()}
    if axi == 1 and is_2d == 1:
        # vol, area, area_err (and dV_hat for order 2): per radian -> 3D;
        # dX_hat, dx and the extent are lengths and are not rescaled
        for key in ("vol", "area", "area_err", "dV_hat"):
            if key in frame:
                frame[key] = frame[key] * AXI_FACTOR
        is_2d = 0
    #
    frame["n_removed"] = n_removed
    # volume of a DPS at the resolution threshold (diameter N_c Delta_m):
    # the DPS below it are discarded, so an empty group of a birth or a
    # death stands for a volume between 0 and vcut
    d_cut = min_cells * min(lbox_v) / 2**maxlevel
    frame["vcut"] = np.pi * d_cut**2 / 4.0 if is_2d == 1 else np.pi * d_cut**3 / 6.0
    frame["pos"] = np.column_stack((frame["x"], frame["y"], frame["z"]))
    frame["vel"] = np.column_stack((frame["ux"], frame["uy"], frame["uz"]))
    frame["rad"] = 0.5 * cmpt_d(frame["vol"], is_2d)
    # extent: radius of the sphere centred at the centroid that contains the
    # DPS; a sphere of the same volume is a lower bound (NaN: legacy layout)
    frame["ext"] = (np.maximum(frame["ext"], frame["rad"]) if has_ext
                    else np.full(len(frame["vol"]), np.nan))
    if order == 1:
        frame["err"] = frame["area_err"]
        frame["xerr"] = frame["err"] / np.maximum(frame["area"], EPS)
    else:
        frame["err"] = frame["dV_hat"]
        frame["xerr"] = frame["dX_hat"]
        if np.any(frame["err"] < 0.0) or np.any(frame["xerr"] < 0.0):
            raise ValueError(f"Negative dV_hat / dX_hat at istep {istep}.")
    #
    return frame
#
# ------------------------------------------------------------------
# Geometry of groups of DPS
# ------------------------------------------------------------------
#
def make_box(lbox_v, lmin_v, periodic_bc):
    #
    mask = np.asarray(periodic_bc, dtype=bool)
    #
    return mask, np.asarray(lbox_v, dtype=float)[mask], np.asarray(lmin_v, dtype=float)[mask]
#
def min_image(d, box):
    #
    mask, length, _ = box
    if np.any(mask):
        d[..., mask] -= length * np.round(d[..., mask] / length)
    #
    return d
#
def wrap(x, box):
    #
    mask, length, origin = box
    if np.any(mask):
        x[..., mask] = (x[..., mask] - origin) % length + origin
    #
    return x
#
def vnorm(v):
    # a single vector keeps np.linalg.norm's own path (bit-identical to before)
    return np.linalg.norm(v) if v.ndim == 1 else np.linalg.norm(v, axis=-1)
#
def group(frame, members, box, terms=()):
    #
    """
    One DPS (members = (i,)) or the union of two (members = (a, b)), with
    i, a, b integers or index arrays: volume V, centroid x and velocity u of
    the group (volume-weighted), and the requested error terms without constants:
      E = sum of err,  S = sum of xerr,
      A, B, W: the position (A, B) and velocity (W) error terms of the
               group, multiplied by C_X (A) and C_V (B, W).
    """
    vol, pos, vel = frame["vol"], frame["pos"], frame["vel"]
    err, xerr = frame["err"], frame["xerr"]
    #
    if len(members) == 1:
        i = members[0]
        g = {"V": vol[i], "x": pos[i], "u": vel[i]}
        single = {"E": err[i], "S": xerr[i], "A": xerr[i], "B": 0.0, "W": 0.0}
        g.update((t, single[t]) for t in terms)
        return g
    #
    a, b = members
    Va, Vb = vol[a], vol[b]
    V = Va + Vb
    d = min_image(pos[b] - pos[a], box)
    g = {
        "V": V,
        "x": wrap(pos[a] + d * (Vb / V)[:, None], box),
        "u": (vel[a] * Va[:, None] + vel[b] * Vb[:, None]) / V[:, None],
    }
    if "E" in terms:
        g["E"] = err[a] + err[b]
    if "S" in terms:
        g["S"] = xerr[a] + xerr[b]
    if "A" in terms:
        g["A"] = (Va * xerr[a] + Vb * xerr[b]) / V
    if "B" in terms or "W" in terms:
        w = (Vb * err[a] + Va * err[b]) / V**2
        g["B"] = np.linalg.norm(d, axis=1) * w
        g["W"] = np.linalg.norm(vel[b] - vel[a], axis=1) * w
    #
    return g
#
def trapezoid_residual(g0, g1, dt, box):
    #
    """Trapezoidal residual |x1 - x0 - (u0 + u1) dt/2| and |u1 - u0| dt/2."""
    r = min_image(g1["x"] - g0["x"] - 0.5 * (g0["u"] + g1["u"]) * dt, box)
    # 
    return np.linalg.norm(r, axis=-1), 0.5 * np.linalg.norm(g1["u"] - g0["u"], axis=-1) * dt
#
# ------------------------------------------------------------------
# Calibration of C_V and C_X
# ------------------------------------------------------------------
#
def calibrate_reference(frame0, frame1, dt, box, event_type, order, block=64):
    #
    """
    C_V, C_X of one event type: the largest ratio, over every possible
    combination of DPS, between the mismatch tested in the candidate search
    and the error that the constant multiplies.

      order 1: r_V = |dV| / E,  r_X = |x1 - x0| / S
      order 2: r_V = |dV| / E,  r_X = max(0, residual - |u1 - u0| dt/2) / A

    In both cases every combination passes its own test at full C. In
    order 2 the C_V terms B, W are not subtracted from the residual, which
    can only make r_X larger. 1.0 is returned when no ratio exists.
    """
    #
    nb, na = EVENT_SHAPE[event_type]
    n0, n1 = len(frame0["vol"]), len(frame1["vol"])
    if n0 < nb or n1 < na:
        return 1.0, 1.0
    #
    terms = ("E", "S") if order == 1 else ("E", "A")
    def all_groups(frame, n, size):
        members = (np.arange(n),) if size == 1 else np.triu_indices(n, k=1)
        return group(frame, members, box, terms)
    #
    g0 = all_groups(frame0, n0, nb)
    g1 = all_groups(frame1, n1, na)
    #
    # loop over blocks of the side made of single DPS
    singles_first = nb == 1
    singles = g0 if singles_first else g1
    n_singles = n0 if singles_first else n1
    #
    C_V = C_X = -np.inf
    for s0 in range(0, n_singles, block):
        part = {key: val[s0:s0 + block, None] if np.ndim(val) else val
                for key, val in singles.items()}
        others = {key: val[None] if np.ndim(val) else val
                  for key, val in (g1 if singles_first else g0).items()}
        b0, b1 = (part, others) if singles_first else (others, part)
        #
        dV = np.abs(b1["V"] - b0["V"])
        E = b1["E"] + b0["E"]
        if order == 1:
            dR = np.linalg.norm(min_image(b1["x"] - b0["x"], box), axis=-1)
            D = b1["S"] + b0["S"]
        else:
            dR, du = trapezoid_residual(b0, b1, dt, box)
            dR = np.maximum(0.0, dR - du)
            D = b1["A"] + b0["A"]
        #
        mask_v = E > EPS
        mask_x = (D > EPS) & (b0["V"] > EPS) & (b1["V"] > EPS)
        if np.any(mask_v):
            C_V = max(C_V, float(np.max(dV[mask_v] / E[mask_v])))
        if np.any(mask_x):
            C_X = max(C_X, float(np.max(dR[mask_x] / D[mask_x])))
    #
    return (1.0 if C_V == -np.inf else C_V), (1.0 if C_X == -np.inf else C_X)
#
# ------------------------------------------------------------------
# Candidate search
# ------------------------------------------------------------------
#
def pairs_by_sum(values_sorted, target, tolerance):
    #
    """Index pairs a < b of a sorted array with |values[a] + values[b] - target| <= tolerance."""
    #
    n = len(values_sorted)
    if n < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    #
    ia = np.arange(n - 1)
    va = values_sorted[:-1]
    lo = np.maximum(np.searchsorted(values_sorted, target - tolerance - va, side="left"), ia + 1)
    hi = np.searchsorted(values_sorted, target + tolerance - va, side="right")
    count = np.maximum(hi - lo, 0)
    total = int(count.sum())
    if total == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    #
    a = np.repeat(ia, count)
    offset = np.repeat(np.cumsum(count) - count, count)
    #
    return a, lo[a] + np.arange(total) - offset
#
def volume_window(v, err, vol_sorted, err_sorted, cV, rv, n_other, order):
    #
    """
    Half-width of the volume window of the single DPS (volumes v, errors err)
    in which the sum of the volumes of its n_other partners must lie.
    The wide window uses the largest error of the other snapshot. Each
    partner has a positive volume, hence V <= v + bound, so its error is at
    most the largest error among the DPS of volume <= v + bound: the window
    is narrowed with that error, which bounds the exact volume test of
    evaluate() (EPS and TOL included), so no candidate is discarded.
    """
    #
    def width(e):
        b = cV * rv * (err + n_other * e)
        return b + np.maximum(EPS, TOL * b) if order == 2 else b
    def exact(e):
        g = np.maximum(EPS, cV * rv * (err + n_other * e))
        return g + np.maximum(EPS, TOL * g)
    prefix = np.maximum.accumulate(err_sorted)
    bound = width(prefix[-1])
    for _ in range(2):
        k = np.searchsorted(vol_sorted, v + bound, side="right") - 1
        e = np.where(k >= 0, prefix[np.maximum(k, 0)], 0.0)
        bound = np.minimum(bound, exact(e))
    #
    return bound
#
def gate_ratio(coef, gate):
    #
    """
    Factor of the constant coef in the error bounds: the bounds use
    g C (gate g), C when gate is None. The ratios dVrel, dRrel and the
    costs are always computed with C.
    """
    #
    if gate is None:
        return 1.0
    if isinstance(gate, tuple):
        # former gate ("min", g): the bounds use min(g, C)
        g = gate[1]
        return 1.0 if coef <= 0.0 or g >= coef else g / coef
    return float(gate)
#
def take(members, keep):
    #
    return tuple(m if np.ndim(m) == 0 else m[keep] for m in members)
#
def contact_reach(frame, a, b, dt, contact):
    #
    """
    Largest separation |x_b - x_a| of two DPS a, b of one snapshot that are
    in contact within the motion of one step:
      contact = K (float):  K (R_a + R_b) + |u_b - u_a| dt
      contact = "extent":   l_a + l_b + dx_a + dx_b + |u_b - u_a| dt
    with R the equivalent radius, l the extent (radius of the sphere centred
    at the centroid that contains the DPS) and dx the largest cell size of
    the DPS, which bounds an unresolved film and the motion of the interface
    over one step. The "extent" test contains no parameter.
    """
    #
    motion = np.linalg.norm(frame["vel"][b] - frame["vel"][a], axis=-1) * dt
    if isinstance(contact, str):
        if contact != "extent":
            raise ValueError(f"Unknown contact test {contact!r}.")
        if np.any(np.isnan(frame["ext"])):
            raise ValueError("contact = 'extent' needs the extent column of the tagging "
                             "(current layout of my_funcs.h, see LAYOUTS).")
        return (frame["ext"][a] + frame["ext"][b]
                + frame["dxma"][a] + frame["dxma"][b] + motion)
    #
    return contact * (frame["rad"][a] + frame["rad"][b]) + motion
#
def evaluate(frame0, frame1, be, ae, dt, cV, cX, rv, rx, box, order, contact=None):
    #
    """
    Test the candidates be -> ae (members as in group()) and return the
    survivors with dV / dV_err and dR / dR_max, both at full C.

    Volume:            |V1 - V0| <= C_V E
    Position, order 1: |x1 - x0| <= max(|u0| dt + |u1 - u0| dt/2, C_X S)
    Position, order 2: residual  <= |u1 - u0| dt/2 + C_V (B + W dt/2) + C_X A

    Contact (optional): the two members a, b of a breakup (after) or of a
    coalescence (before) touch within the motion of one step,
        |x_b - x_a| <= contact_reach(...)

    The test uses the constants times rv, rx (the gate), the ratios do not.
    Errors are summed member by member, in event order.
    """
    #
    def total(key):
        s = 0.0
        for m in be:
            s = s + frame0[key][m]
        for m in ae:
            s = s + frame1[key][m]
        return s
    #
    dV = np.abs(sum(frame1["vol"][m] for m in ae) - sum(frame0["vol"][m] for m in be))
    dV_err = np.maximum(EPS, cV * total("err"))
    gate_v = np.maximum(EPS, dV_err * rv)
    keep = dV <= gate_v + np.maximum(EPS, TOL * gate_v)
    if not np.any(keep):
        return None
    #
    be, ae, dV, dV_err = take(be, keep), take(ae, keep), dV[keep], dV_err[keep]
    if contact is not None and max(len(be), len(ae)) == 2:
        frame, (a, b) = (frame0, be) if len(be) == 2 else (frame1, ae)
        sep = np.linalg.norm(min_image(frame["pos"][b] - frame["pos"][a], box), axis=-1)
        keep = sep <= contact_reach(frame, a, b, dt, contact)
        if not np.any(keep):
            return None
        be, ae, dV, dV_err = take(be, keep), take(ae, keep), dV[keep], dV_err[keep]
    #
    terms = () if order == 1 else ("A", "B", "W")
    g0 = group(frame0, be, box, terms)
    g1 = group(frame1, ae, box, terms)
    if order == 1:
        dR = np.linalg.norm(min_image(g1["x"] - g0["x"], box), axis=1)
        motion = vnorm(g0["u"]) * dt + 0.5 * np.linalg.norm(g1["u"] - g0["u"], axis=1) * dt
        error = cX * total("xerr")
        dR_max = np.maximum(EPS, np.maximum(motion, error))
        gate_x = np.maximum(EPS, np.maximum(motion, error * rx))
    else:
        dR, du = trapezoid_residual(g0, g1, dt, box)
        A, B, W = g0["A"] + g1["A"], g0["B"] + g1["B"], g0["W"] + g1["W"]
        dR_max = np.maximum(EPS, du + 0.5 * cV * W * dt + cX * A + cV * B)
        gate_x = np.maximum(EPS, du + 0.5 * cV * rv * W * dt + cX * rx * A + cV * rv * B)
    #
    keep = dR <= gate_x + np.maximum(EPS, TOL * gate_x)
    if not np.any(keep):
        return None
    #
    return take(be, keep), take(ae, keep), dV[keep] / dV_err[keep], dR[keep] / dR_max[keep]
#
def find_candidates_reference(frame0, frame1, dt, C_V, C_X, box, event_types, gate=None, order=1,
                              contact=None, bd_threshold=True):
    #
    """
    Candidate events within the error bounds:

        continuation : [i]   -> [j]
        breakup      : [i]   -> [j, k]
        coalescence  : [i, k] -> [j]

    C_V, C_X are dicts by event type; gate=None uses them in full, a number g
    uses g C. Returns, for each type, p0 (N x n_before), p1
    (N x n_after), dVrel, dRrel and cost = dVrel + dRrel.
    """
    #
    out = {}
    for event_type, (nb, na) in EVENT_SHAPE.items():
        if 0 in (nb, na):
            continue                                   # birth/death: birth_death_candidates()
        p0, p1, dVrel, dRrel = [], [], [], []
        #
        # loop over the single DPS, search its partner(s) in the other snapshot
        single, other = (frame1, frame0) if event_type == "coalescence" else (frame0, frame1)
        n_other = max(nb, na)
        if event_type in event_types and len(single["vol"]) > 0 and len(other["vol"]) >= n_other:
            cV, cX = float(C_V[event_type]), float(C_X[event_type])
            rv, rx = gate_ratio(cV, gate), gate_ratio(cX, gate)
            #
            by_vol = np.argsort(other["vol"], kind="stable")
            vol_sorted = other["vol"][by_vol]
            bounds = volume_window(single["vol"], single["err"], vol_sorted,
                                   other["err"][by_vol], cV, rv, n_other, order)
            #
            for s in range(len(single["vol"])):
                bound = bounds[s]
                v = single["vol"][s]
                if n_other == 1:
                    found = by_vol[np.searchsorted(vol_sorted, v - bound, side="left"):
                                   np.searchsorted(vol_sorted, v + bound, side="right")]
                    partners = (found,)
                else:
                    a, b = pairs_by_sum(vol_sorted, v, bound)
                    a, b = by_vol[a], by_vol[b]
                    partners = (np.minimum(a, b), np.maximum(a, b))
                if partners[0].size == 0:
                    continue
                #
                be, ae = ((s,), partners) if single is frame0 else (partners, (s,))
                result = evaluate(frame0, frame1, be, ae, dt, cV, cX, rv, rx, box, order, contact)
                if result is None:
                    continue
                #
                be, ae, dv, dr = result
                n = len(dv)
                p0.append(np.column_stack([np.full(n, m) if np.ndim(m) == 0 else m for m in be]))
                p1.append(np.column_stack([np.full(n, m) if np.ndim(m) == 0 else m for m in ae]))
                dVrel.append(dv)
                dRrel.append(dr)
        #
        if p0:
            dVrel, dRrel = np.concatenate(dVrel), np.concatenate(dRrel)
            out[event_type] = {"p0": np.concatenate(p0).astype(int),
                               "p1": np.concatenate(p1).astype(int),
                               "dVrel": dVrel, "dRrel": dRrel, "cost": dVrel + dRrel}
        else:
            out[event_type] = {"p0": np.empty((0, nb), dtype=int), "p1": np.empty((0, na), dtype=int),
                               "dVrel": np.empty(0), "dRrel": np.empty(0), "cost": np.empty(0)}
    out.update(birth_death_candidates(frame0, frame1, C_V, event_types, gate, bd_threshold))
    #
    return out
#
# ------------------------------------------------------------------
# Birth and death
# ------------------------------------------------------------------
#
def calibrate_birth_death(frame):
    #
    """
    C_V of a birth (frame at t^(n+1)) or a death (frame at t^n): the
    largest V / dV_hat, as for the other types the largest ratio over every
    possible event. C_X is not used.
    """
    if len(frame["vol"]) == 0:
        return 1.0, 1.0
    #
    return float(np.max(frame["vol"] / np.maximum(EPS, frame["err"]))), 1.0
#
def birth_death_candidates(frame0, frame1, C_V, event_types, gate=None, bd_threshold=True):
    #
    """
    Births (empty -> [j]) and deaths ([i] -> empty). The empty group has no
    center of mass, so only the volume is tested. The empty group stands for
    the DPS below the resolution threshold, of volume between 0 and V_cut
    (frame["vcut"], diameter N_c Delta_m), so that the volume test with an
    empty group reads V - V_cut <= C_V dV_hat, with g C_V under the gate,
    and dVrel = max(0, V - V_cut) / (C_V dV_hat) (bd_threshold = True).
    With bd_threshold = False, V_cut = 0, i.e. V <= C_V dV_hat. The displacement ratio is set
    to its bound, dRrel = 1, so that a death and a birth always cost more
    than a continuation between the same DPS.
    """
    #
    out = {}
    for event_type, frame in (("birth", frame1), ("death", frame0)):
        nb, na = EVENT_SHAPE[event_type]
        idx, dv = np.empty(0, dtype=int), np.empty(0)
        if event_type in event_types and len(frame["vol"]) > 0:
            cV = float(C_V[event_type])
            dV_err = np.maximum(EPS, cV * frame["err"])
            gate_v = np.maximum(EPS, dV_err * gate_ratio(cV, gate))
            excess = np.maximum(0.0, frame["vol"] - (frame.get("vcut", 0.0) if bd_threshold else 0.0))
            idx = np.nonzero(excess <= gate_v + np.maximum(EPS, TOL * gate_v))[0]
            dv = excess[idx] / dV_err[idx]
        members, empty = idx[:, None], np.empty((len(idx), 0), dtype=int)
        out[event_type] = {"p0": members if nb else empty, "p1": members if na else empty,
                           "dVrel": dv, "dRrel": np.ones(len(idx)), "cost": dv + 1.0}
    #
    return out
#
# ------------------------------------------------------------------
# Fast calibration and candidate search: same results as
# calibrate_reference() and find_candidates_reference() (kept above as the
# definition and for checks), without enumerating every combination
# ------------------------------------------------------------------
#
def max_volume_ratio(V0, e0, V1, e1, n0, n1):
    #
    """
    max |sum V1 - sum V0| / (sum e0 + sum e1) over every group of n0 DPS of
    frame 0 and n1 DPS of frame 1 (n = 1 or 2), by Dinkelbach's iteration:
    for a fixed lam the objective |dV| - lam E is separable, its maximum is
    the largest single plus the two largest distinct members of the pair.
    Returns the members of the best combination.
    """
    #
    def best(values, n):
        if n == 1:
            k = int(np.argmax(values))
            return (k,)
        a, b = np.argpartition(-values, 1)[:2]
        return (int(min(a, b)), int(max(a, b)))
    #
    def ratio(m0, m1):
        Vg0, Eg0 = (V0[m0[0]], e0[m0[0]]) if n0 == 1 else (V0[m0[0]] + V0[m0[1]], e0[m0[0]] + e0[m0[1]])
        Vg1, Eg1 = (V1[m1[0]], e1[m1[0]]) if n1 == 1 else (V1[m1[0]] + V1[m1[1]], e1[m1[0]] + e1[m1[1]])
        return np.abs(Vg1 - Vg0) / (Eg1 + Eg0)
    #
    lam, arg = -np.inf, None
    lam_try = 0.0
    while True:
        new = []
        for s in (1.0, -1.0):
            m0 = best(-s * V0 - lam_try * e0, n0)
            m1 = best(s * V1 - lam_try * e1, n1)
            new.append((ratio(m0, m1), m0, m1))
        r, m0, m1 = max(new, key=lambda x: x[0])
        if r <= lam:
            return lam, arg
        lam, arg = r, (m0, m1)
        lam_try = lam
#
def calibrate(frame0, frame1, dt, box, event_type, order, chunk=2_000_000):
    #
    """
    Same C_V, C_X as calibrate() (max over every combination of DPS), without
    enumerating every combination:
      C_V: Dinkelbach's iteration on the separable |dV| - lam E;
      C_X: the mismatch is bounded by R_b (half the periodic box, the extent
           of the DPS otherwise), so a combination with error D >= R_b / C_X
           cannot raise C_X and is skipped. The combinations are visited in
           increasing D, which makes C_X large early on.
    """
    #
    if 0 in EVENT_SHAPE[event_type]:
        return calibrate_birth_death(frame1 if event_type == "birth" else frame0)
    nb, na = EVENT_SHAPE[event_type]
    n0, n1 = len(frame0["vol"]), len(frame1["vol"])
    if n0 < nb or n1 < na:
        return 1.0, 1.0
    if n_combinations(n0, n1, event_type) <= SMALL_PROBLEM:
        return calibrate_reference(frame0, frame1, dt, box, event_type, order)
    #
    # ---- C_V ----
    e0, e1 = frame0["err"], frame1["err"]
    def smallest(e, n):
        return np.sum(np.partition(e, n - 1)[:n])
    if smallest(e0, nb) + smallest(e1, na) > EPS:
        C_V, _ = max_volume_ratio(frame0["vol"], e0, frame1["vol"], e1, nb, na)
    else:
        C_V = None          # combinations with E <= EPS: see below
    #
    # ---- C_X ----
    terms = ("E", "S") if order == 1 else ("E", "A")
    dkey = "S" if order == 1 else "A"
    def members(n, size):
        return (np.arange(n),) if size == 1 else np.triu_indices(n, k=1)
    def error_of(frame, m):
        # D of each group without its geometry (same arithmetic as group())
        if len(m) == 1:
            return frame["xerr"][m[0]]
        a, b = m
        if order == 1:
            return frame["xerr"][a] + frame["xerr"][b]
        Va, Vb = frame["vol"][a], frame["vol"][b]
        return (Va * frame["xerr"][a] + Vb * frame["xerr"][b]) / (Va + Vb)
    m0, m1 = members(n0, nb), members(n1, na)
    D0, D1 = error_of(frame0, m0), error_of(frame1, m1)
    #
    # bound of the mismatch
    mask, length, _ = box
    pos0, pos1, vel0, vel1 = frame0["pos"], frame1["pos"], frame0["vel"], frame1["vel"]
    span = np.maximum(pos1.max(0) - pos0.min(0), pos0.max(0) - pos1.min(0))
    if order == 2:
        span = span + 0.5 * dt * (np.abs(vel0).max(0) + np.abs(vel1).max(0))
    full = np.zeros(3, dtype=bool)
    full[:len(mask)] = mask
    span[full] = 0.5 * length
    R_b = float(np.linalg.norm(span)) * (1.0 + 1.0e-9)
    #
    # singles side in blocks, other side filtered on its error
    singles_first = nb == 1
    if singles_first:
        Ds, ms, Do, mo = D0, m0, D1, m1
    else:
        Ds, ms, Do, mo = D1, m1, D0, m0
    order_s = np.argsort(Ds, kind="stable")
    order_o = np.argsort(Do, kind="stable")
    Ds_sorted, Do_sorted = Ds[order_s], Do[order_o]
    #
    C_X = -np.inf
    E_need = C_V is None
    C_V_slow = -np.inf
    i = 0
    n_s = len(order_s)
    while i < n_s:
        limit = np.inf if C_X <= 0.0 or E_need else R_b / C_X
        if Ds_sorted[i] + Do_sorted[0] >= limit:
            break
        n_o = int(np.searchsorted(Do_sorted, limit - Ds_sorted[i], side="left"))
        n_o = max(n_o, 1)
        block = max(1, min(n_s - i, chunk // n_o))
        si = order_s[i:i + block]
        oi = order_o[:n_o]
        ms_b = tuple(m[si] for m in ms)
        mo_b = tuple(m[oi] for m in mo)
        gs = group(frame0 if singles_first else frame1, ms_b, box, terms)
        go = group(frame1 if singles_first else frame0, mo_b, box, terms)
        part = {k: v[:, None] for k, v in gs.items()}
        others = {k: v[None] for k, v in go.items()}
        b0, b1 = (part, others) if singles_first else (others, part)
        if order == 1:
            dR = np.linalg.norm(min_image(b1["x"] - b0["x"], box), axis=-1)
        else:
            dR, du = trapezoid_residual(b0, b1, dt, box)
            dR = np.maximum(0.0, dR - du)
        D = b1[dkey] + b0[dkey]
        mask_x = (D > EPS) & (b0["V"] > EPS) & (b1["V"] > EPS)
        if np.any(mask_x):
            C_X = max(C_X, float(np.max(dR[mask_x] / D[mask_x])))
        if E_need:
            dV = np.abs(b1["V"] - b0["V"])
            E = b1["E"] + b0["E"]
            mask_v = E > EPS
            if np.any(mask_v):
                C_V_slow = max(C_V_slow, float(np.max(dV[mask_v] / E[mask_v])))
        i += block
    if E_need:
        C_V = C_V_slow
    return (1.0 if C_V == -np.inf else C_V), (1.0 if C_X == -np.inf else C_X)
#
SLACK = 1.0e-9   # relative margin of the pre-filters (exact tests in evaluate)
# below this number of combinations the reference functions are used (no
# vectorization overhead); both give the same result
SMALL_PROBLEM = 200_000
CHUNK = 1_000_000  # candidates tested at once
#
def n_combinations(n0, n1, event_type):
    #
    nb_, na = EVENT_SHAPE[event_type]
    count = lambda n, size: n if size == 1 else n * (n - 1) // 2
    #
    return count(n0, nb_) * count(n1, na)
#
def contact_pairs_reference(frame, dt, box, contact, chunk=CHUNK):
    """
    Pairs a < b of one snapshot that can pass the contact test of evaluate(),
    with a margin SLACK, by testing every pair (definition of contact_pairs()).
    """
    #
    n = len(frame["vol"])
    pos = frame["pos"]
    pa, pb = [np.empty(0, dtype=int)], [np.empty(0, dtype=int)]
    rows = max(1, chunk // max(n, 1))
    for a0 in range(0, n - 1, rows):
        a = np.arange(a0, min(a0 + rows, n - 1))
        b = np.arange(a0 + 1, n)
        sep = np.linalg.norm(min_image(pos[b][None] - pos[a][:, None], box), axis=-1)
        reach = contact_reach(frame, a[:, None], b[None], dt, contact)
        i, j = np.nonzero((sep <= reach * (1.0 + SLACK)) & (b[None] > a[:, None]))
        pa.append(a[i])
        pb.append(b[j])
    #
    return np.concatenate(pa), np.concatenate(pb)
#
def find_candidates(frame0, frame1, dt, C_V, C_X, box, event_types, gate=None, order=1,
                    contact=None, chunk=CHUNK, bd_threshold=True):
    #
    """
    Same output as find_candidates_reference(), vectorized. The groups of
    the other snapshot (single DPS, or pairs that can pass the contact test)
    are matched to the single DPS through the volume windows of
    find_candidates_reference(); evaluate() applies the exact tests, and the
    candidates are returned in the order of find_candidates_reference().
    Without the contact test every pair is a group, and the loop of
    find_candidates_reference() is faster.
    """
    #
    n0, n1 = len(frame0["vol"]), len(frame1["vol"])
    if contact is None or max(n_combinations(n0, n1, e) for e, shape in EVENT_SHAPE.items()
                                 if 0 not in shape) <= SMALL_PROBLEM:
        return find_candidates_reference(frame0, frame1, dt, C_V, C_X, box, event_types,
                                         gate=gate, order=order, contact=contact,
                                         bd_threshold=bd_threshold)
    out = {}
    for event_type, (nb_, na) in EVENT_SHAPE.items():
        if 0 in (nb_, na):
            continue                                   # birth/death: birth_death_candidates()
        single, other = (frame1, frame0) if event_type == "coalescence" else (frame0, frame1)
        n_other = max(nb_, na)
        found = []
        if event_type in event_types and len(single["vol"]) > 0 and len(other["vol"]) >= n_other:
            cV, cX = float(C_V[event_type]), float(C_X[event_type])
            rv, rx = gate_ratio(cV, gate), gate_ratio(cX, gate)
            by_vol = np.argsort(other["vol"], kind="stable")
            vol_sorted = other["vol"][by_vol]
            rank = np.empty_like(by_vol)
            rank[by_vol] = np.arange(len(by_vol))
            #
            # volume window of each single DPS, as in find_candidates_reference()
            bound = volume_window(single["vol"], single["err"], vol_sorted,
                                  other["err"][by_vol], cV, rv, n_other, order)
            by_s = np.argsort(single["vol"], kind="stable")
            lo = (single["vol"] - bound)[by_s]
            hi = (single["vol"] + bound)[by_s]
            #
            # groups of the other snapshot by rank in volume, va <= vb
            # (va = 0 for a single DPS: lo - va = lo exactly)
            if n_other == 1:
                rb = np.arange(len(vol_sorted))
                ra = rb
                va = np.zeros(len(rb))
            else:
                a, b = contact_pairs(other, dt, box, contact, chunk)
                ra, rb = np.minimum(rank[a], rank[b]), np.maximum(rank[a], rank[b])
                va = vol_sorted[ra]
            vb = vol_sorted[rb]
            #
            # single DPS whose window can contain va + vb: monotone envelopes
            # of the windows, then the exact test lo - va <= vb <= hi - va
            vg = va + vb
            margin = SLACK * (np.abs(vg) + np.max(np.abs(bound))) + EPS
            k0 = np.searchsorted(np.maximum.accumulate(hi), vg - margin, side="left")
            k1 = np.searchsorted(np.minimum.accumulate(lo[::-1])[::-1], vg + margin, side="right")
            count = np.maximum(k1 - k0, 0)
            cum = np.cumsum(count)
            g0 = 0
            while g0 < len(count):
                g1 = max(g0 + 1, int(np.searchsorted(cum, (cum[g0 - 1] if g0 else 0) + chunk,
                                                      side="right")))
                c = count[g0:g1]
                g = np.repeat(np.arange(g0, g1), c)
                k = k0[g] + np.arange(len(g)) - np.repeat(np.cumsum(c) - c, c)
                keep = (lo[k] - va[g] <= vb[g]) & (vb[g] <= hi[k] - va[g])
                g, k = g[keep], k[keep]
                g0 = g1
                if len(g) == 0:
                    continue
                s = by_s[k]
                if n_other == 1:
                    partners = (by_vol[rb[g]],)
                else:
                    pa, pb = by_vol[ra[g]], by_vol[rb[g]]
                    partners = (np.minimum(pa, pb), np.maximum(pa, pb))
                be, ae = ((s,), partners) if single is frame0 else (partners, (s,))
                result = evaluate(frame0, frame1, be, ae, dt, cV, cX, rv, rx, box, order, contact)
                if result is not None:
                    found.append(result)
        if found:
            be = tuple(np.concatenate([r[0][i] for r in found]) for i in range(nb_))
            ae = tuple(np.concatenate([r[1][i] for r in found]) for i in range(na))
            dv = np.concatenate([r[2] for r in found])
            dr = np.concatenate([r[3] for r in found])
            # order of find_candidates_reference(): single DPS, then ranks in volume
            s, partners = (be[0], ae) if single is frame0 else (ae[0], be)
            r = [rank[m] for m in partners]
            key = (r[0],) if n_other == 1 else (np.maximum(*r), np.minimum(*r))
            sort = np.lexsort(key + (s,))
            out[event_type] = {"p0": np.column_stack(be)[sort].astype(int),
                               "p1": np.column_stack(ae)[sort].astype(int),
                               "dVrel": dv[sort], "dRrel": dr[sort],
                               "cost": (dv + dr)[sort]}
        else:
            out[event_type] = {"p0": np.empty((0, nb_), dtype=int), "p1": np.empty((0, na), dtype=int),
                               "dVrel": np.empty(0), "dRrel": np.empty(0), "cost": np.empty(0)}
    out.update(birth_death_candidates(frame0, frame1, C_V, event_types, gate, bd_threshold))
    #
    return out
#
# ------------------------------------------------------------------
# Calibration of C_V and C_X on the anchor continuations
# ------------------------------------------------------------------
#
def continuation_ratios(frame0, frame1, dt, box, order, rows):
    #
    """
    r_V and r_X of the continuations i -> j, i in rows, j every DPS of
    frame1 (len(rows) x n1), with the uncalibrated errors (C_V = C_X = 1):
      r_V = |V_j - V_i| / (E_i + E_j)
      order 1: r_X = |x_j - x_i| / (S_i + S_j), 0 within the motion bound
      order 2: r_X = max(0, residual - |u_j - u_i| dt/2) / (A_i + A_j)
    i.e. the smallest C_V, C_X for which the continuation passes the
    volume and position tests of evaluate().
    """
    #
    V0, V1 = frame0["vol"][rows, None], frame1["vol"][None]
    E = frame0["err"][rows, None] + frame1["err"][None]
    r_V = np.abs(V1 - V0) / np.maximum(E, EPS)
    x0, x1 = frame0["pos"][rows, None], frame1["pos"][None]
    u0, u1 = frame0["vel"][rows, None], frame1["vel"][None]
    D = frame0["xerr"][rows, None] + frame1["xerr"][None]
    if order == 1:
        dR = np.linalg.norm(min_image(x1 - x0, box), axis=-1)
        motion = np.linalg.norm(u0, axis=-1) * dt + 0.5 * np.linalg.norm(u1 - u0, axis=-1) * dt
        dR = np.where(dR <= motion, 0.0, dR)
    else:
        dR = np.linalg.norm(min_image(x1 - x0 - 0.5 * (u0 + u1) * dt, box), axis=-1)
        dR = np.maximum(0.0, dR - 0.5 * np.linalg.norm(u1 - u0, axis=-1) * dt)
    r_X = dR / np.maximum(D, EPS)
    #
    return r_V, r_X
#
def in_contact(frame, dt, box, contact):
    #
    """DPS of one snapshot in contact (contact_reach) with another DPS of the snapshot."""
    #
    flag = np.zeros(len(frame["vol"]), dtype=bool)
    if contact is None or len(flag) < 2:
        return flag
    a, b = contact_pairs(frame, dt, box, contact)
    flag[a] = True
    flag[b] = True
    #
    return flag
#
def calibrate_anchors_reference(frame0, frame1, dt, box, order, contact=None, floor=1.0, chunk=CHUNK,
                      max_iter=20):
    #
    """
    C_V, C_X calibrated on the anchor continuations of the pair of snapshots,
    by testing every continuation (definition of calibrate_anchors(), which
    gives the same result with a neighbour search).

    For given constants, the continuation i -> j is admissible if
    r_V <= C_V and r_X <= C_X (continuation_ratios). N(C_V, C_X) is the
    number of admissible continuations i -> j such that j is the only
    admissible partner of i, i the only admissible partner of j, and
    neither i nor j is in contact with another DPS of its snapshot. N is
    maximized one constant at a time (C_X with C_V fixed, then C_V with
    C_X fixed, from C_V = inf), each time with the smallest value that
    gives the maximum; the iteration stops when N no longer increases.
    For a fixed C_V, i -> j counts for r_X,ij <= C_X < min(second best r_X
    of i, of j), and N(C_X) is the number of these intervals that contain
    C_X: its maximum is at the lower end of one of them.

    The anchors are the continuations counted in N at the solution, and
    C_V, C_X are the largest r_V, r_X among them, but not less than floor.
    The estimates dV_hat, dX_hat are upper bounds of the identification
    error, so floor = 1 (default) keeps at least the uncalibrated bounds,
    and the anchors raise C_V, C_X where the estimates are exceeded;
    floor = 0 uses the anchors alone. Without anchors (no unique, isolated
    continuation) C_V = C_X = max(floor, 1). Returns C_V, C_X and a dict
    with the anchors and their largest ratios.
    """
    #
    n0, n1 = len(frame0["vol"]), len(frame1["vol"])
    info = {"n_anchors": 0, "anchors": [], "N_iter": [], "calibrated": False}
    info.update(rV_max=np.nan, rX_max=np.nan)
    if n0 == 0 or n1 == 0:
        return max(floor, 1.0), max(floor, 1.0), info
    #
    # contact test of the anchors: the one of the candidate search, or the
    # parameter-free one when the extent is available
    if contact is None and not np.any(np.isnan(frame0["ext"])) and not np.any(np.isnan(frame1["ext"])):
        contact_a = "extent"
    else:
        contact_a = contact
    free0 = ~in_contact(frame0, dt, box, contact_a)
    free1 = ~in_contact(frame1, dt, box, contact_a)
    #
    rows_per_chunk = max(1, chunk // max(n1, 1))
    def best_two(var, C_other):
        """
        Best and second best ratio `var` (0: r_V, 1: r_X) of each DPS of
        both snapshots over its partners with the other ratio <= C_other.
        Returns (best0, arg0, second0, best1, arg1, second1, r_ij of the best).
        """
        b0 = np.full(n0, np.inf); a0 = np.full(n0, -1); s0 = np.full(n0, np.inf)
        b1 = np.full(n1, np.inf); a1 = np.full(n1, -1); s1 = np.full(n1, np.inf)
        for r0 in range(0, n0, rows_per_chunk):
            rows = np.arange(r0, min(r0 + rows_per_chunk, n0))
            ratios = continuation_ratios(frame0, frame1, dt, box, order, rows)
            r = np.where(ratios[1 - var] <= C_other, ratios[var], np.inf)
            # rows (DPS of frame0): all partners in this chunk
            if n1 >= 2:
                part = np.partition(r, 1, axis=1)[:, :2]
                b0[rows], s0[rows] = part[:, 0], part[:, 1]
            else:
                b0[rows] = r[:, 0]
            a0[rows] = np.argmin(r, axis=1)
            # columns (DPS of frame1): merge the chunk with the running best two
            k = np.argmin(r, axis=0)
            cb = r[k, np.arange(n1)]
            if len(rows) >= 2:
                cs = np.partition(r, 1, axis=0)[1]
            else:
                cs = np.full(n1, np.inf)
            new_best = cb < b1
            s1 = np.where(new_best, np.minimum(b1, cs), np.minimum(s1, cb))
            a1 = np.where(new_best, rows[k], a1)
            b1 = np.where(new_best, cb, b1)
        return b0, a0, s0, b1, a1, s1
    #
    def intervals(var, C_other):
        """Lower and upper ends of the uniqueness interval of each mutual best i -> j."""
        b0, a0, s0, b1, a1, s1 = best_two(var, C_other)
        i = np.nonzero(np.isfinite(b0))[0]
        j = a0[i]
        mutual = (a1[j] == i) & free0[i] & free1[j]
        i, j = i[mutual], j[mutual]
        lo, hi = b0[i], np.minimum(s0[i], s1[j])
        ok = lo < hi
        return i[ok], j[ok], lo[ok], hi[ok]
    #
    def maximize(var, C_other):
        """Smallest constant of ratio `var` that maximizes N, and that N."""
        i, j, lo, hi = intervals(var, C_other)
        if len(lo) == 0:
            return np.inf, 0
        los, his = np.sort(lo), np.sort(hi)
        # N at each lower end: intervals started minus intervals ended
        N = np.searchsorted(los, los, side="right") - np.searchsorted(his, los, side="right")
        k = int(np.argmax(N))                     # first (smallest) maximum
        return float(los[k]), int(N[k])
    #
    C = [np.inf, np.inf]          # C_V, C_X
    N_best, var = -1, 1           # start with C_X at C_V = inf
    for _ in range(max_iter):
        value, N = maximize(var, C[1 - var])
        if N <= N_best:
            break
        C[var], N_best = value, N
        info["N_iter"].append(int(N))
        var = 1 - var
    if N_best <= 0:
        return max(floor, 1.0), max(floor, 1.0), info
    #
    # anchors at the solution: continuations unique at (C_V, C_X)
    i, j, lo, hi = intervals(1, C[0])
    keep = (lo <= C[1]) & (C[1] < hi)
    i, j = i[keep], j[keep]
    ratios = [continuation_ratios(frame0, frame1, dt, box, order, np.array([a])) for a in i]
    r_V = np.array([rv[0, b] for (rv, _), b in zip(ratios, j)])
    r_X = np.array([rx[0, b] for (_, rx), b in zip(ratios, j)])
    rV_max = float(np.max(r_V)) if len(r_V) else np.nan
    rX_max = float(np.max(r_X)) if len(r_X) else np.nan
    C_V = max(floor, rV_max) if len(r_V) else max(floor, 1.0)
    C_X = max(floor, rX_max) if len(r_X) else max(floor, 1.0)
    # a constant equal to zero (floor = 0 and identical volumes) would
    # admit nothing but exact matches: keep the uncalibrated bound then
    C_V = C_V if C_V > EPS else 1.0
    C_X = C_X if C_X > EPS else 1.0
    info.update(n_anchors=int(len(i)), anchors=[[int(a), int(b)] for a, b in zip(i, j)],
                calibrated=True, rV_max=rV_max, rX_max=rX_max)
    #
    return C_V, C_X, info
#
# ------------------------------------------------------------------
# Neighbour search (cell list, numpy only)
# ------------------------------------------------------------------
#
# below this number of pairs the functions that test every pair are faster
# (one vectorized call) and are used instead; both give the same result
SMALL_GRID = 10_000
#
def make_grid(pts, box):
    #
    """
    Uniform grid of the points pts (n x 3) for radius queries (grid_query).
    The cell size, about one point per cell, only sets the cost: a query
    returns all the points within its radius whatever the cell size.
    Periodic directions of box wrap around; a direction in which all the
    points have the same coordinate (2D) has a single cell.
    """
    #
    mask, length, origin = box
    per = np.zeros(3, dtype=bool)
    per[:len(mask)] = mask
    L, O = np.zeros(3), np.zeros(3)
    L[per], O[per] = length, origin
    lo, hi = pts.min(0), pts.max(0)
    span = np.where(per, L, hi - lo)
    active = span > 0
    D = max(int(np.count_nonzero(active)), 1)
    h = (np.prod(span[active]) / len(pts))**(1.0 / D) if np.any(active) else 1.0
    nc = np.where(active, np.maximum(1, np.floor(span / h)), 1).astype(np.int64)
    grid = {"pts": pts, "box": box, "per": per, "L": L, "lo": np.where(per, O, lo),
            "pmin": lo, "pmax": hi, "nc": nc,
            "h": np.where(active, span / nc, np.inf)}
    key = grid_key(grid, grid_cell(grid, pts))
    grid["order"] = np.argsort(key, kind="stable")
    count = np.bincount(key, minlength=int(np.prod(nc)))
    grid["start"] = np.cumsum(count) - count
    grid["count"] = count
    #
    return grid
#
def grid_cell(grid, x):
    #
    """Cell of each point x (n x 3); out-of-grid points go to the nearest cell."""
    #
    h, per, nc = grid["h"], grid["per"], grid["nc"]
    fin = np.isfinite(h)
    c = np.where(fin, np.floor((x - grid["lo"]) / np.where(fin, h, 1.0)), 0).astype(np.int64)
    c[:, per] %= nc[per]
    # non-periodic: a point beyond the grid is in its nearest cell, from which
    # the reach of its query still covers every point within the radius
    c[:, ~per] = np.clip(c[:, ~per], 0, nc[~per] - 1)
    #
    return c
#
def grid_key(grid, c):
    #
    nc = grid["nc"]
    return c[:, 0] + nc[0] * (c[:, 1] + nc[1] * c[:, 2])
#
def grid_query(grid, q, radius):
    #
    """
    Pairs (k, t): point t of the grid within radius[k] of q[k] (minimum
    image). The queries are grouped by their reach in cells, rounded up to
    a power of two; for each offset in y and z the cells along x form one
    or two contiguous ranges of the cell table, read at once.
    """
    #
    pts, box, h, per, nc = grid["pts"], grid["box"], grid["h"], grid["per"], grid["nc"]
    if len(q) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    fin = np.isfinite(h)
    cq = grid_cell(grid, q)
    reach = np.ceil(radius[:, None] / np.where(fin, h, np.inf)[None]).max(1)
    reach = np.where(reach <= 1, 1, 2.0**np.ceil(np.log2(np.maximum(reach, 1)))).astype(np.int64)
    first = grid["start"]
    last = grid["start"] + grid["count"]          # end of each cell in the table
    out_k, out_t = [], []
    for m in np.unique(reach):
        ks = np.nonzero(reach == m)[0]
        absolute = [(not fin[d]) or 2 * m + 1 >= nc[d] for d in range(3)]
        offs = [np.arange(nc[d]) if absolute[d] else np.arange(-m, m + 1) for d in (1, 2)]
        # x ranges of the cells of each query: [a, b], one or two pieces
        if absolute[0]:
            pieces = [(np.zeros(len(ks), dtype=np.int64), np.full(len(ks), nc[0] - 1))]
        elif per[0]:
            a = (cq[ks, 0] - m) % nc[0]
            b = (cq[ks, 0] + m) % nc[0]
            wrap = a > b
            pieces = [(a, np.where(wrap, nc[0] - 1, b)),
                      (np.where(wrap, 0, a), np.where(wrap, b, -1))]   # empty when b < a
        else:
            pieces = [(np.maximum(cq[ks, 0] - m, 0), np.minimum(cq[ks, 0] + m, nc[0] - 1))]
        for o1 in offs[0]:
            for o2 in offs[1]:
                c = cq[ks].copy()
                ok = np.ones(len(ks), dtype=bool)
                for d, o in ((1, o1), (2, o2)):
                    if absolute[d]:
                        c[:, d] = o
                    elif per[d]:
                        c[:, d] = (c[:, d] + o) % nc[d]
                    else:
                        c[:, d] += o
                        ok &= (c[:, d] >= 0) & (c[:, d] < nc[d])
                for a, b in pieces:
                    use = ok & (b >= a)
                    if not np.any(use):
                        continue
                    ca, cb = c[use].copy(), c[use].copy()
                    ca[:, 0], cb[:, 0] = a[use], b[use]
                    s0, s1 = first[grid_key(grid, ca)], last[grid_key(grid, cb)]
                    cnt = s1 - s0
                    tot = int(cnt.sum())
                    if tot == 0:
                        continue
                    kk = ks[use]
                    rep = np.repeat(np.arange(len(kk)), cnt)
                    pos = s0[rep] + np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
                    out_k.append(kk[rep])
                    out_t.append(grid["order"][pos])
    if not out_k:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    k, t = np.concatenate(out_k), np.concatenate(out_t)
    keep = np.linalg.norm(min_image(pts[t] - q[k], box), axis=-1) <= radius[k]
    #
    return k[keep], t[keep]
#
def grid_farthest(grid, q):
    #
    """Upper bound of the distance from each q to any point of the grid."""
    #
    far = np.maximum(np.abs(grid["pmax"] - q), np.abs(q - grid["pmin"]))
    per = grid["per"]
    far[:, per] = np.minimum(far[:, per], 0.5 * grid["L"][per])
    #
    return np.linalg.norm(far, axis=-1)
#
def velocity_spread(u, U):
    #
    """max over the rows of U of |U - u|, for each row of u (bounding box of U)."""
    #
    lo, hi = U.min(0), U.max(0)
    return np.linalg.norm(np.maximum(np.abs(hi - u), np.abs(u - lo)), axis=-1)
#
def pair_ratios(frame0, frame1, dt, box, order, i, j):
    #
    """r_V and r_X of the continuations i -> j (index arrays), as continuation_ratios()."""
    #
    V0, V1 = frame0["vol"][i], frame1["vol"][j]
    E = frame0["err"][i] + frame1["err"][j]
    r_V = np.abs(V1 - V0) / np.maximum(E, EPS)
    x0, x1 = frame0["pos"][i], frame1["pos"][j]
    u0, u1 = frame0["vel"][i], frame1["vel"][j]
    D = frame0["xerr"][i] + frame1["xerr"][j]
    if order == 1:
        dR = np.linalg.norm(min_image(x1 - x0, box), axis=-1)
        motion = np.linalg.norm(u0, axis=-1) * dt + 0.5 * np.linalg.norm(u1 - u0, axis=-1) * dt
        dR = np.where(dR <= motion, 0.0, dR)
    else:
        dR = np.linalg.norm(min_image(x1 - x0 - 0.5 * (u0 + u1) * dt, box), axis=-1)
        dR = np.maximum(0.0, dR - 0.5 * np.linalg.norm(u1 - u0, axis=-1) * dt)
    #
    return r_V, dR / np.maximum(D, EPS)
#
def contact_pairs(frame, dt, box, contact, chunk=CHUNK):
    """
    Pairs a < b of one snapshot that can pass the contact test of evaluate(),
    with a margin SLACK, sorted by (a, b): same output as
    contact_pairs_reference(), with a cell list. The reach of a pair is at
    most reach_a + max_b(reach_b) + max |u_b - u_a| dt, which bounds the radius
    of the search around a.
    """
    #
    n = len(frame["vol"])
    if n < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if n * (n - 1) // 2 <= SMALL_GRID:
        return contact_pairs_reference(frame, dt, box, contact, chunk)
    if isinstance(contact, str):
        own = frame["ext"] + frame["dxma"]
    else:
        own = contact * frame["rad"]
    motion = velocity_spread(frame["vel"], frame["vel"]) * dt
    radius = (own + own.max() + motion) * (1.0 + SLACK) + EPS
    a, b = grid_query(make_grid(frame["pos"], box), frame["pos"], radius)
    keep = b > a
    a, b = a[keep], b[keep]
    sep = np.linalg.norm(min_image(frame["pos"][b] - frame["pos"][a], box), axis=-1)
    keep = sep <= contact_reach(frame, a, b, dt, contact) * (1.0 + SLACK)
    a, b = a[keep], b[keep]
    s = np.lexsort((b, a))
    #
    return a[s], b[s]
#
def calibrate_anchors(frame0, frame1, dt, box, order, contact=None, floor=1.0, chunk=CHUNK,
                      max_iter=20):
    #
    """
    Same C_V, C_X and anchors as calibrate_anchors_reference(), without
    testing every continuation: each step of the maximization needs only
    a subset of the pairs i -> j, found exactly.

      X step, C_V = inf: the two best r_X of each DPS of both snapshots.
          With p_i = x_i + u_i dt (order 2; x_i for order 1) and W_i the
          largest |u_j - u_i| dt (plus |u_i| dt for order 1), any j with
          |x_j - p_i| >= rho has r_X >= (rho - W_i) / (S_i + max_j S_j):
          rho is doubled from the cell size until the second best r_X found
          is below this bound (or rho covers the domain). Same for the DPS
          at t^(n+1), around x_j - u_j dt.
      V step, C_X fixed: the admissible partners have r_X <= C_X, hence
          |x_j - p_i| <= C_X (S_i + max S) + W_i: one radius query.
      X step, C_V finite: the admissible partners have r_V <= C_V, i.e.
          lie in a volume window (volume_window()): one sorted search.

    In each step the pairs found contain every admissible partner that
    can be the best or second best, so that N, the intervals and the
    anchors are those of the full enumeration.
    """
    #
    n0, n1 = len(frame0["vol"]), len(frame1["vol"])
    if n0 * n1 <= SMALL_GRID:
        return calibrate_anchors_reference(frame0, frame1, dt, box, order, contact=contact,
                                           floor=floor, chunk=chunk, max_iter=max_iter)
    info = {"n_anchors": 0, "anchors": [], "N_iter": [], "calibrated": False}
    info.update(rV_max=np.nan, rX_max=np.nan)
    #
    if contact is None and not np.any(np.isnan(frame0["ext"])) and not np.any(np.isnan(frame1["ext"])):
        contact_a = "extent"
    else:
        contact_a = contact
    free0 = ~in_contact(frame0, dt, box, contact_a)
    free1 = ~in_contact(frame1, dt, box, contact_a)
    #
    x0, x1, u0, u1 = frame0["pos"], frame1["pos"], frame0["vel"], frame1["vel"]
    S0, S1 = frame0["xerr"], frame1["xerr"]
    spread0 = velocity_spread(u0, u1) * dt     # max_j |u_j - u_i| dt, i at t^n
    spread1 = velocity_spread(u1, u0) * dt     # max_i |u_j - u_i| dt, j at t^(n+1)
    if order == 1:
        p0, W0 = x0, np.linalg.norm(u0, axis=-1) * dt + 0.5 * spread0
        p1, W1 = x1, np.linalg.norm(u0, axis=-1).max() * dt + 0.5 * spread1
    else:
        p0, W0 = x0 + u0 * dt, spread0
        p1, W1 = x1 - u1 * dt, spread1
    grid1, grid0 = make_grid(x1, box), make_grid(x0, box)
    #
    def unique_pairs(i, j):
        key = np.unique(i.astype(np.int64) * n1 + j)
        return key // n1, key % n1
    #
    def knn_pairs():
        """Pairs that contain the two best r_X of every DPS (C_V = inf)."""
        found_i, found_j = [], []
        for side in (0, 1):
            grid, p, W, Sself, Smax = ((grid1, p0, W0, S0, S1.max()) if side == 0
                                       else (grid0, p1, W1, S1, S0.max()))
            n = len(p)
            far = grid_farthest(grid, p) + W
            fin = np.isfinite(grid["h"])
            rho = W + (np.min(grid["h"][fin]) if np.any(fin) else 1.0)
            rho = np.minimum(rho, far)
            todo = np.arange(n)
            while len(todo):
                k, t = grid_query(grid, p[todo], rho[todo])
                q = todo[k]
                i, j = (q, t) if side == 0 else (t, q)
                _, rX = pair_ratios(frame0, frame1, dt, box, order, i, j)
                # second smallest r_X of each query
                second = np.full(n, np.inf)
                srt = np.lexsort((rX, q))
                qs, rs = q[srt], rX[srt]
                first = np.r_[True, qs[1:] != qs[:-1]]
                idx = np.nonzero(first)[0]
                has2 = (idx + 1 < len(qs)) & (np.r_[qs[1:], -1][idx] == qs[idx])
                second[qs[idx[has2]]] = rs[idx[has2] + 1]
                bound = np.maximum(0.0, rho - W) / np.maximum(Sself + Smax, EPS)
                done = (second <= bound) | (rho >= far)
                ok = done[q]
                found_i.append(i[ok]); found_j.append(j[ok])
                todo = todo[~done[todo]]
                rho[todo] = np.minimum(2.0 * rho[todo], far[todo])
        return unique_pairs(np.concatenate(found_i), np.concatenate(found_j))
    #
    def radius_pairs(C_X):
        """Every pair with r_X <= C_X (and possibly others)."""
        rho = C_X * (S0 + S1.max()) * (1.0 + SLACK) + W0 + EPS
        k, t = grid_query(grid1, p0, rho)
        return k, t
    #
    def volume_pairs(C_V):
        """Every pair with r_V <= C_V (and possibly others)."""
        by_vol = np.argsort(frame1["vol"], kind="stable")
        vs = frame1["vol"][by_vol]
        bound = volume_window(frame0["vol"], frame0["err"], vs, frame1["err"][by_vol],
                              C_V, 1.0, 1, 2) * (1.0 + SLACK) + EPS * max(1.0, C_V)
        a = np.searchsorted(vs, frame0["vol"] - bound, side="left")
        b = np.searchsorted(vs, frame0["vol"] + bound, side="right")
        cnt = np.maximum(b - a, 0)
        tot = int(cnt.sum())
        i = np.repeat(np.arange(n0), cnt)
        pos = np.repeat(a, cnt) + np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        return i, by_vol[pos]
    #
    cache = {}
    def pairs_for(var, C_other):
        """Pairs and ratios that contain every admissible best and second best."""
        if var == 1 and not np.isfinite(C_other):
            key = ("knn",)
            if key not in cache:
                cache[key] = knn_pairs()
            i, j = cache[key]
        elif var == 1:
            i, j = volume_pairs(C_other)
        else:
            i, j = radius_pairs(C_other)
        r_V, r_X = pair_ratios(frame0, frame1, dt, box, order, i, j)
        return i, j, r_V, r_X
    #
    def best_two_sparse(i, j, r):
        """Best ratio, its partner and second best of each DPS of both snapshots."""
        out = []
        for a, b, n in ((i, j, n0), (j, i, n1)):
            best, arg, second = np.full(n, np.inf), np.full(n, -1), np.full(n, np.inf)
            if len(a):
                srt = np.lexsort((b, r, a))
                as_, bs, rs = a[srt], b[srt], r[srt]
                first = np.r_[True, as_[1:] != as_[:-1]]
                idx = np.nonzero(first)[0]
                best[as_[idx]], arg[as_[idx]] = rs[idx], bs[idx]
                has2 = (idx + 1 < len(as_)) & (np.r_[as_[1:], -1][idx] == as_[idx])
                second[as_[idx[has2]]] = rs[idx[has2] + 1]
            out.append((best, arg, second))
        return out
    #
    def intervals(var, C_other):
        i, j, r_V, r_X = pairs_for(var, C_other)
        ratios = (r_V, r_X)
        adm = ratios[1 - var] <= C_other
        i, j, r = i[adm], j[adm], ratios[var][adm]
        (b0, a0, s0), (b1, a1, s1) = best_two_sparse(i, j, r)
        ii = np.nonzero(np.isfinite(b0))[0]
        jj = a0[ii]
        mutual = (a1[jj] == ii) & free0[ii] & free1[jj]
        ii, jj = ii[mutual], jj[mutual]
        lo, hi = b0[ii], np.minimum(s0[ii], s1[jj])
        ok = lo < hi
        return ii[ok], jj[ok], lo[ok], hi[ok]
    #
    def maximize(var, C_other):
        i, j, lo, hi = intervals(var, C_other)
        if len(lo) == 0:
            return np.inf, 0
        los, his = np.sort(lo), np.sort(hi)
        N = np.searchsorted(los, los, side="right") - np.searchsorted(his, los, side="right")
        k = int(np.argmax(N))
        return float(los[k]), int(N[k])
    #
    C = [np.inf, np.inf]
    N_best, var = -1, 1
    for _ in range(max_iter):
        value, N = maximize(var, C[1 - var])
        if N <= N_best:
            break
        C[var], N_best = value, N
        info["N_iter"].append(int(N))
        var = 1 - var
    if N_best <= 0:
        return max(floor, 1.0), max(floor, 1.0), info
    #
    i, j, lo, hi = intervals(1, C[0])
    keep = (lo <= C[1]) & (C[1] < hi)
    i, j = i[keep], j[keep]
    r_V, r_X = pair_ratios(frame0, frame1, dt, box, order, i, j)
    rV_max = float(np.max(r_V)) if len(r_V) else np.nan
    rX_max = float(np.max(r_X)) if len(r_X) else np.nan
    C_V = max(floor, rV_max) if len(r_V) else max(floor, 1.0)
    C_X = max(floor, rX_max) if len(r_X) else max(floor, 1.0)
    C_V = C_V if C_V > EPS else 1.0
    C_X = C_X if C_X > EPS else 1.0
    s = np.lexsort((j, i))
    info.update(n_anchors=int(len(i)), anchors=[[int(a), int(b)] for a, b in zip(i[s], j[s])],
                calibrated=True, rV_max=rV_max, rX_max=rX_max)
    #
    return C_V, C_X, info
#
# ------------------------------------------------------------------
# Event organization: minimum-cost exact cover
# ------------------------------------------------------------------
#
def reduce_problem(enodes, k):
    #
    """
    Exact cover preprocessing on DPS 0..k-1: a DPS with a single candidate
    forces it (repeated to a fixed point), then the rest is split into
    independent components. Returns (forced events, components as
    (DPS list, event list)), or None if a DPS is left without candidates.
    """
    #
    node_events = [[] for _ in range(k)]
    for e, nodes in enumerate(enodes):
        for v in nodes:
            node_events[v].append(e)
    if any(not events for events in node_events):
        return None
    #
    alive = [True] * len(enodes)
    covered = [False] * k
    n_live = [len(events) for events in node_events]
    forced = []
    queue = deque(v for v in range(k) if n_live[v] == 1)
    while queue:
        v = queue.popleft()
        if covered[v]:
            continue
        if n_live[v] == 0:
            return None
        e = next(e for e in node_events[v] if alive[e])
        forced.append(e)
        alive[e] = False
        for u in enodes[e]:
            covered[u] = True
        for u in enodes[e]:
            for e2 in node_events[u]:
                if not alive[e2]:
                    continue
                alive[e2] = False
                for w in enodes[e2]:
                    n_live[w] -= 1
                    if not covered[w]:
                        if n_live[w] == 0:
                            return None
                        if n_live[w] == 1:
                            queue.append(w)
    #
    comp_id = [-1] * k
    components = []
    for start in range(k):
        if covered[start] or comp_id[start] >= 0:
            continue
        comp_id[start] = len(components)
        bfs = deque([start])
        nodes, events, seen = [], [], set()
        while bfs:
            v = bfs.popleft()
            nodes.append(v)
            for e in node_events[v]:
                if alive[e] and e not in seen:
                    seen.add(e)
                    events.append(e)
                    for u in enodes[e]:
                        if comp_id[u] < 0:
                            comp_id[u] = len(components)
                            bfs.append(u)
        components.append((nodes, sorted(events)))
    #
    return forced, components
#
def lagrangian_multipliers(cand, enodes, ecost):
    #
    """
    Multipliers lam of the constraints 'each DPS is in exactly one event',
    by exact coordinate ascent of the Lagrangian dual
        L(lam) = sum_v lam_v + sum_e min(0, r_e),  r_e = c_e - sum_{v in e} lam_v,
    which is a lower bound of the minimum cost for any lam, and whose
    maximum is the bound of the linear relaxation. Along lam_v, L is concave
    with slope 1 - #{e containing v: r_e < 0}, so it is maximal when lam_v
    is raised by any amount between the two smallest r_e of v: the midpoint
    is taken (the smallest r_e if v has a single event). The sweeps over
    the DPS stop when L no longer increases, after at most k sweeps.
    Returns lam and the reduced costs r.
    """
    #
    k = len(cand)
    lam = [0.0] * k
    r = list(ecost)
    def dual():
        return sum(lam) + sum(x for x in r if x < 0.0)
    L = dual()
    for _ in range(k):
        for v in range(k):
            s1 = s2 = float("inf")
            for e in cand[v]:
                x = r[e]
                if x < s1:
                    s1, s2 = x, s1
                elif x < s2:
                    s2 = x
            d = s1 if s2 == float("inf") else 0.5 * (s1 + s2)
            if d != 0.0:
                lam[v] += d
                for e in cand[v]:
                    r[e] -= d
        L_new = dual()
        if L_new <= L + TOL * max(1.0, abs(L)):
            break
        L = L_new
    # reduced costs recomputed from lam, without the accumulated round-off
    r = [c - sum(lam[v] for v in nodes) for c, nodes in zip(ecost, enodes)]
    #
    return lam, r
#
def rcm_order(k, enodes):
    #
    """
    Reverse Cuthill-McKee ordering of the DPS 0..k-1, linked when they share
    an event: a breadth-first numbering from a pseudo-peripheral DPS, each
    DPS visiting its neighbours by increasing degree, then reversed. DPS
    linked by an event receive close positions, so that at any point of an
    ordered search only a narrow band of DPS ahead can be already covered.
    """
    #
    adj = [set() for _ in range(k)]
    for nodes in enodes:
        for a in nodes:
            adj[a].update(nodes)
    for a in range(k):
        adj[a].discard(a)
    deg = [len(a) for a in adj]
    seen, order = [False] * k, []
    for start in sorted(range(k), key=lambda v: deg[v]):
        if seen[start]:
            continue
        # pseudo-peripheral start: the lowest-degree DPS among the farthest
        # from the current start, twice
        for _ in range(2):
            dist = {start: 0}
            bfs = deque([start])
            while bfs:
                v = bfs.popleft()
                for u in adj[v]:
                    if u not in dist and not seen[u]:
                        dist[u] = dist[v] + 1
                        bfs.append(u)
            far = max(dist.values())
            start = min((v for v in dist if dist[v] == far), key=lambda v: deg[v])
        seen[start] = True
        bfs = deque([start])
        while bfs:
            v = bfs.popleft()
            order.append(v)
            for u in sorted(adj[v], key=lambda u: deg[u]):
                if not seen[u]:
                    seen[u] = True
                    bfs.append(u)
    #
    return order[::-1]
#
def cover_component(enodes, ecost, k):
    #
    """
    Exact minimum-cost exact cover of one component (DPS 0..k-1) by the
    given events. Returns the selected events, or None if there is no cover.

    The DPS are numbered by rcm_order, and the search always covers the
    first uncovered DPS p, by one of the events whose lowest DPS is p, by
    increasing reduced cost. All the DPS before p are then covered, so the
    state of a branch is the set of covered DPS in the narrow band after p,
    and a state already reached with a cost that is not higher is dropped
    (dynamic programming over the band). With the Lagrangian multipliers
    lam (lagrangian_multipliers) and the reduced costs r, a cover X of the
    uncovered DPS U costs sum_{v in U} lam_v + sum_{e in X} r_e, so a
    branch is dropped when its cost plus
        sum_{v in U} max(min_e c_e / |e|, lam_v + min_e r_e / |e|)
    (over each sum separately) is not below the best cover found. Any cover
    with the event e costs at least J_L + r_e, J_L = sum lam + sum min(0, r),
    so the events with r_e >= ub - J_L cannot be in a cover cheaper than ub.
    The search looks for a cover cheaper than ub = J_L + d, with d doubled
    from the round-off margin until one is found: the first cover found is
    optimal, and the events kept are few while d is small.
    """
    #
    order = rcm_order(k, enodes)
    pos = [0] * k
    for p, v in enumerate(order):
        pos[v] = p
    en = [tuple(sorted(pos[v] for v in nodes)) for nodes in enodes]
    emask = [sum(1 << v for v in nodes) for nodes in en]
    allc = [[] for _ in range(k)]
    first = [[] for _ in range(k)]
    for e, nodes in enumerate(en):
        first[nodes[0]].append(e)
        for v in nodes:
            allc[v].append(e)
    if any(not c for c in allc):
        #
        return None
    #
    lam, r = lagrangian_multipliers(allc, en, ecost)
    JL = sum(lam) + sum(x for x in r if x < 0.0)
    esize = [len(nodes) for nodes in en]
    share0 = [min(ecost[e] / esize[e] for e in allc[v]) for v in range(k)]
    share1 = [lam[v] + min(r[e] / esize[e] for e in allc[v]) for v in range(k)]
    eb0 = [sum(share0[v] for v in nodes) for nodes in en]
    eb1 = [sum(share1[v] for v in nodes) for nodes in en]
    for c in first:
        c.sort(key=lambda e: (r[e], e))
    tol = TOL * (k + 1)
    #
    def search_below(ub):
        # optimal cover cheaper than ub (None if there is none)
        best, best_cost = None, ub
        memo, chosen = {}, []
        def search(used, p, cost, rest0, rest1):
            nonlocal best, best_cost
            while used >> p & 1:
                p += 1
            if p == k:
                if cost < best_cost:
                    best_cost, best = cost, chosen.copy()
                return
            # margin against the round-off of sums of up to k + 1 terms: a
            # branch is dropped only if it cannot hold a strictly cheaper cover
            margin = tol * max(1.0, abs(best_cost)) if best_cost < float("inf") else 0.0
            if cost + max(rest0, rest1) >= best_cost + margin:
                return
            if memo.get(used, float("inf")) <= cost:
                return
            memo[used] = cost
            for e in first[p]:
                if r[e] >= best_cost - JL + margin:
                    break
                if emask[e] & used:
                    continue
                chosen.append(e)
                search(used | emask[e], p + 1, cost + ecost[e],
                       rest0 - eb0[e], rest1 - eb1[e])
                chosen.pop()
        search(0, 0, 0.0, sum(share0), sum(share1))
        return best
    #
    # largest possible cost of a cover: beyond it the search is unbounded
    cap = sum(max(0.0, max(ecost[e] for e in allc[v])) for v in range(k))
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, k + 200))
    try:
        d = tol * max(1.0, abs(JL))
        while True:
            ub = JL + d if JL + d <= cap else float("inf")
            best = search_below(ub)
            if best is not None or ub == float("inf"):
                break
            d *= 2.0
    finally:
        sys.setrecursionlimit(limit)
    #
    return None if best is None else sorted(best)
#
def organize_events(n0, n1, candidates, debug=False):
    #
    """
    Select the candidate events so that every DPS of both snapshots takes
    part in exactly one of them, at minimum total cost.

    Events forced by a DPS with a single candidate are taken first, the
    rest is split into independent components, each solved
    exactly by cover_component.

    Returns (events, summary), or None if no exact cover exists.
    """
    # event list: type, DPS at t^n, DPS at t^(n+1), cost
    events = []
    for event_type in EVENT_SHAPE:
        if event_type not in candidates:
            continue
        c = candidates[event_type]
        for p0, p1, cost in zip(c["p0"], c["p1"], c["cost"]):
            p0, p1 = tuple(int(i) for i in p0), tuple(int(j) for j in p1)
            if len(set(p0)) < len(p0) or len(set(p1)) < len(p1):
                raise ValueError(f"Invalid {event_type} candidate {p0} -> {p1}.")
            events.append((event_type, p0, p1, float(cost)))
    #
    # nodes: DPS i at t^n is i, DPS j at t^(n+1) is n0 + j
    n_nodes = n0 + n1
    ev_nodes = [p0 + tuple(n0 + j for j in p1) for _, p0, p1, _ in events]
    ecost = [ev[3] for ev in events]
    #
    reduced = reduce_problem(ev_nodes, n_nodes)
    chosen = None
    if reduced is not None:
        forced, components = reduced
        chosen = []
        for nodes, comp_events in components:
            bit = {v: b for b, v in enumerate(nodes)}
            sub = cover_component([tuple(bit[v] for v in ev_nodes[e]) for e in comp_events],
                                  [ecost[e] for e in comp_events], len(nodes))
            if sub is None:
                chosen = None
                break
            chosen.append([comp_events[i] for i in sub])
    if chosen is None:
        if debug:
            print("No exact cover of the candidate events.")
        return None
    #
    selected = forced + [e for sub in chosen for e in sub]
    total_cost = 0.0
    for e in forced:
        total_cost += ecost[e]
    for sub in chosen:
        total_cost += float(sum(ecost[e] for e in sub))
    n_forced = len(forced)
    #
    selected.sort()
    #
    count = np.zeros(n_nodes, dtype=int)
    for e in selected:
        count[list(ev_nodes[e])] += 1
    if np.any(count != 1):
        raise RuntimeError(f"Not an exact cover: participation {count.tolist()}")
    #
    result = [{"type": events[e][0], "p0": events[e][1], "p1": events[e][2], "cost": events[e][3]}
              for e in selected]
    n_type = {t: sum(ev["type"] == t for ev in result) for t in EVENT_SHAPE}
    n_cont, n_break, n_coal = n_type["continuation"], n_type["breakup"], n_type["coalescence"]
    n_birth, n_death = n_type["birth"], n_type["death"]
    if n_cont + n_break + 2 * n_coal + n_death != n0 or \
       n_cont + 2 * n_break + n_coal + n_birth != n1:
        raise RuntimeError("DPS-number balance violated.")
    n_cand = {t: len(candidates[t]["cost"]) if t in candidates else 0 for t in EVENT_SHAPE}
    #
    summary = {
        "n0": n0, "n1": n1,
        "n_continuation": n_cont, "n_breakup": n_break, "n_coalescence": n_coal,
        "n_birth": n_birth, "n_death": n_death,
        "n_lost": n_death, "n_new": n_birth,
        "total_cost": float(total_cost),
        "balance_lhs": n1 - n0, "balance_rhs": n_break - n_coal + n_birth - n_death,
        "balance_error": 0,
        "n_cont_candidates": n_cand["continuation"],
        "n_break_candidates": n_cand["breakup"],
        "n_coal_candidates": n_cand["coalescence"],
        "n_birth_candidates": n_cand["birth"],
        "n_death_candidates": n_cand["death"],
        "n_candidates_total": len(events),
        "n_components": len(components),
    }
    #
    if debug:
        print("organization:", f"{n_cont} continuation, {n_break} breakup, {n_coal} coalescence,",
              f"{n_birth} birth, {n_death} death;",
              f"{len(events)} candidates, {n_forced} forced, {len(components)} components",
              f"(sizes {[len(ce) for _, ce in components]}), cost {total_cost:.6e}")
    #
    return result, summary
#
# ------------------------------------------------------------------
# Local gate: the bounds are widened only for the DPS that cannot be covered
# ------------------------------------------------------------------
#
def candidate_nodes(n0, candidates):
    #
    """
    Candidates of every type as node tuples (DPS i at t^n is i, DPS j at
    t^(n+1) is n0 + j), with their type and index in candidates[type].
    """
    #
    out = []
    for event_type in EVENT_SHAPE:
        if event_type not in candidates:
            continue
        c = candidates[event_type]
        for e, (p0, p1) in enumerate(zip(c["p0"].tolist(), c["p1"].tolist())):
            out.append((event_type, e, tuple(p0) + tuple(n0 + j for j in p1)))
    return out
#
def failed_nodes(n0, n1, candidates):
    #
    """
    DPS of the connected components of the candidate graph (DPS linked by
    a common candidate) that have no exact cover: a DPS without any
    candidate is such a component by itself. The exact cover of the whole
    step exists if and only if this set is empty.
    """
    #
    n_nodes = n0 + n1
    ev_nodes = [nodes for _, _, nodes in candidate_nodes(n0, candidates)]
    parent = np.arange(n_nodes)
    def root(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    for nodes in ev_nodes:
        r0 = root(nodes[0])
        for v in nodes[1:]:
            r = root(v)
            if r != r0:
                parent[r] = r0
    comp = np.array([root(v) for v in range(n_nodes)], dtype=np.int64)
    comp_events = {}
    for e, nodes in enumerate(ev_nodes):
        comp_events.setdefault(int(comp[nodes[0]]), []).append(e)
    #
    bad = []
    for c in np.unique(comp):
        nodes = np.flatnonzero(comp == c)
        events = comp_events.get(int(c), [])
        ok = bool(events)
        if ok:
            bit = {int(v): b for b, v in enumerate(nodes)}
            sub = [tuple(bit[v] for v in ev_nodes[e]) for e in events]
            reduced = reduce_problem(sub, len(nodes))
            ok = reduced is not None
            if ok:
                for comp_nodes, comp_ev in reduced[1]:
                    b2 = {v: b for b, v in enumerate(comp_nodes)}
                    if cover_component([tuple(b2[v] for v in sub[e]) for e in comp_ev],
                                       [1.0] * len(comp_ev), len(comp_nodes)) is None:
                        ok = False
                        break
        if not ok:
            bad.append(nodes)
    #
    return np.concatenate(bad) if bad else np.empty(0, dtype=np.int64)
#
def candidate_levels(levels):
    #
    """
    levels: [(g, candidates found with gate g)] for g = 1, 2, 4, ... The
    bounds grow with g, so the candidate sets are nested. Returns, for
    each candidate of the last (largest) g, the smallest g that admits it.
    """
    #
    g_top, top = levels[-1]
    out = {}
    for event_type, c in top.items():
        rows = np.hstack([c["p0"], c["p1"]]).tolist()
        level = np.full(len(rows), float(g_top))
        for g, cand in reversed(levels[:-1]):
            low = cand[event_type]
            found = set(map(tuple, np.hstack([low["p0"], low["p1"]]).tolist()))
            hit = np.fromiter((tuple(r) in found for r in rows), dtype=bool, count=len(rows))
            level[hit] = g
        out[event_type] = level
    return out
#
def local_gate_candidates(n0, levels, g_node):
    #
    """
    Candidates admitted by the local gate: a candidate on the DPS set S is
    kept if the smallest gate that admits it is not larger than
    g_S = max over S of g_node (g_node = 1 for every DPS: the calibrated
    bounds). levels as in candidate_levels(), up to g = max(g_node).
    """
    #
    g_top, top = levels[-1]
    if len(levels) == 1:
        return top
    level = candidate_levels(levels)
    out = {}
    for event_type, c in top.items():
        g_S = np.ones(len(c["cost"]))
        for col in c["p0"].T:
            g_S = np.maximum(g_S, g_node[col])
        for col in c["p1"].T:
            g_S = np.maximum(g_S, g_node[n0 + col])
        keep = level[event_type] <= g_S * (1.0 + 1.0e-12)
        out[event_type] = {key: val[keep] for key, val in c.items()}
    return out
#
# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------
#
def build_record(istep_p0, istep_p1, time_p0, time_p1, events, summary, frame0, frame1, order):
    #
    record = {
        "istep_p0": int(istep_p0),
        "istep_p1": int(istep_p1),
        "time_p0": float(time_p0),
        "time_p1": float(time_p1),
        "DeltaT": float(time_p1 - time_p0),
        "events": events,
        "summary": dict(summary),
    }
    for tag, frame in (("0", frame0), ("1", frame1)):
        record["voll" + tag] = frame["vol"].copy()
        record["area" + tag] = frame["area"].copy()
        record[f"area{tag}_err"] = frame["area_err"].copy()
        for key in ("x", "y", "z"):
            record[f"{key}pos{tag}"] = frame[key].copy()
        for key in ("ux", "uy", "uz"):
            record[f"ucm{key[1]}{tag}"] = frame[key].copy()
    if order == 2:
        record["order"] = 2
        for tag, frame in (("0", frame0), ("1", frame1)):
            record["dV_hat" + tag] = frame["dV_hat"].copy()
            record["dX_hat" + tag] = frame["dX_hat"].copy()
    #
    return record
#
