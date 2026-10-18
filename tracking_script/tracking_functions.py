#
import sys
from collections import deque
import numpy as np
from helper_functions import cmpt_d
#
EPS = 1.0e-14
TOL = 1.0e-14
#
EVENT_SHAPE = {"continuation": (1, 1), "breakup": (1, 2), "coalescence": (2, 1)}
#
FIELDS = ("vol", "area", "x", "y", "z", "ux", "uy", "uz",
          "dxmi", "dxma", "dxav", "levmi", "levma", "levav", "ncells", "area_err")
N_COLS = {1: 19, 2: 21}
#
# ------------------------------------------------------------------
# Input files
# ------------------------------------------------------------------
#
def build_bubble_stat_index(bubble_stat):
    isteps = np.rint(bubble_stat[:, 1]).astype(np.int64)
    order = np.argsort(isteps, kind="stable")
    steps, start, count = np.unique(isteps[order], return_index=True, return_counts=True)
    index = {int(s): (int(a), int(a + c)) for s, a, c in zip(steps, start, count)}
    return bubble_stat[order], index
#
def remove_restart_duplicates(log_tag, bubble_stat, step_col=1, count_col=3):
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
    log_tag = np.asarray(log_tag)
    bubble_stat = np.asarray(bubble_stat)

    log_steps = np.rint(log_tag[:, step_col]).astype(np.int64)
    log_counts = np.rint(1.0 + log_tag[:, count_col]).astype(np.int64)
    b_steps = np.rint(bubble_stat[:, step_col]).astype(np.int64)
    n_log, n_b = len(log_steps), len(b_steps)

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

    if n_b:
        starts = np.flatnonzero(np.r_[True, b_steps[1:] != b_steps[:-1]])
        ends = np.r_[starts[1:], n_b]
        run_end = np.repeat(ends, ends - starts)

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

    n_unmatched = n_b - ptr
    keep = [np.arange(a, b) for a, b, r in blocks if valid[r]]
    bubble_clean = bubble_stat[np.concatenate(keep)] if keep else bubble_stat[:0]

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
    return log_tag[log_rows], bubble_clean, info
#
def extract_frame(bubble_stat_sorted, step_index, lbox_v, min_cells, maxlevel,
                  istep, is_2d, order):
    """
    DPS of one snapshot, larger than min_cells cells, by decreasing volume.
    """
    ncol = bubble_stat_sorted.shape[1]
    if ncol < N_COLS[order]:
        raise ValueError(
            f"bubbles_f1.out has {ncol} columns, order {order} needs {N_COLS[order]}. "
            f"Is --input_dir pointing to the _v{order} tagging output?")

    i0, i1 = step_index.get(int(istep), (0, 0))
    data = bubble_stat_sorted[i0:i1, 3:3 + len(FIELDS) + 2 * (order == 2)].astype(float)

    keep = cmpt_d(data[:, 0], is_2d) >= min_cells * min(lbox_v) / 2**maxlevel
    n_removed = int(np.count_nonzero(~keep))
    data = data[keep]
    data = data[np.argsort(-data[:, 0], kind="stable")]

    frame = {key: data[:, n] for n, key in enumerate(FIELDS)}
    frame["n_removed"] = n_removed
    frame["pos"] = np.column_stack((frame["x"], frame["y"], frame["z"]))
    frame["vel"] = np.column_stack((frame["ux"], frame["uy"], frame["uz"]))
    if order == 1:
        frame["err"] = frame["area_err"]
        frame["xerr"] = frame["err"] / np.maximum(frame["area"], EPS)
    else:
        frame["err"] = frame["dV_hat"] = data[:, -2]
        frame["xerr"] = frame["dX_hat"] = data[:, -1]
        if np.any(frame["err"] < 0.0) or np.any(frame["xerr"] < 0.0):
            raise ValueError(f"Negative dV_hat / dX_hat at istep {istep}.")
    return frame
#
# ------------------------------------------------------------------
# Geometry of groups of DPS
# ------------------------------------------------------------------
#
def make_box(lbox_v, lmin_v, periodic_bc):
    mask = np.asarray(periodic_bc, dtype=bool)
    return mask, np.asarray(lbox_v, dtype=float)[mask], np.asarray(lmin_v, dtype=float)[mask]
#
def min_image(d, box):
    mask, length, _ = box
    if np.any(mask):
        d[..., mask] -= length * np.round(d[..., mask] / length)
    return d
#
def wrap(x, box):
    mask, length, origin = box
    if np.any(mask):
        x[..., mask] = (x[..., mask] - origin) % length + origin
    return x
#
def vnorm(v):
    # a single vector keeps np.linalg.norm's own path (bit-identical to before)
    return np.linalg.norm(v) if v.ndim == 1 else np.linalg.norm(v, axis=-1)
#
def group(frame, members, box, terms=()):
    """
    One DPS (members = (i,)) or the union of two (members = (a, b)), with
    i, a, b integers or index arrays: volume V, centroid x and velocity u of
    the group.
    """
    vol, pos, vel = frame["vol"], frame["pos"], frame["vel"]
    err, xerr = frame["err"], frame["xerr"]

    if len(members) == 1:
        i = members[0]
        g = {"V": vol[i], "x": pos[i], "u": vel[i]}
        single = {"E": err[i], "S": xerr[i], "A": xerr[i], "B": 0.0, "W": 0.0}
        g.update((t, single[t]) for t in terms)
        return g

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
    return g
#
def trapezoid_residual(g0, g1, dt, box):
    """|x1 - x0 - (u0 + u1) dt/2| and |u1 - u0| dt/2."""
    r = min_image(g1["x"] - g0["x"] - 0.5 * (g0["u"] + g1["u"]) * dt, box)
    return np.linalg.norm(r, axis=-1), 0.5 * np.linalg.norm(g1["u"] - g0["u"], axis=-1) * dt
#
# ------------------------------------------------------------------
# Calibration of C_V and C_X
# ------------------------------------------------------------------
#
def calibrate(frame0, frame1, dt, box, event_type, order, block=64):
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
    nb, na = EVENT_SHAPE[event_type]
    n0, n1 = len(frame0["vol"]), len(frame1["vol"])
    if n0 < nb or n1 < na:
        return 1.0, 1.0

    terms = ("E", "S") if order == 1 else ("E", "A")

    def all_groups(frame, n, size):
        members = (np.arange(n),) if size == 1 else np.triu_indices(n, k=1)
        return group(frame, members, box, terms)

    g0 = all_groups(frame0, n0, nb)
    g1 = all_groups(frame1, n1, na)

    # loop over blocks of the side made of single DPS
    singles_first = nb == 1
    singles = g0 if singles_first else g1
    n_singles = n0 if singles_first else n1

    C_V = C_X = -np.inf
    for s0 in range(0, n_singles, block):
        part = {key: val[s0:s0 + block, None] if np.ndim(val) else val
                for key, val in singles.items()}
        others = {key: val[None] if np.ndim(val) else val
                  for key, val in (g1 if singles_first else g0).items()}
        b0, b1 = (part, others) if singles_first else (others, part)

        dV = np.abs(b1["V"] - b0["V"])
        E = b1["E"] + b0["E"]
        if order == 1:
            dR = np.linalg.norm(min_image(b1["x"] - b0["x"], box), axis=-1)
            D = b1["S"] + b0["S"]
        else:
            dR, du = trapezoid_residual(b0, b1, dt, box)
            dR = np.maximum(0.0, dR - du)
            D = b1["A"] + b0["A"]

        mask_v = E > EPS
        mask_x = (D > EPS) & (b0["V"] > EPS) & (b1["V"] > EPS)
        if np.any(mask_v):
            C_V = max(C_V, float(np.max(dV[mask_v] / E[mask_v])))
        if np.any(mask_x):
            C_X = max(C_X, float(np.max(dR[mask_x] / D[mask_x])))

    return (1.0 if C_V == -np.inf else C_V), (1.0 if C_X == -np.inf else C_X)
#
# ------------------------------------------------------------------
# Candidate search
# ------------------------------------------------------------------
#
def pairs_by_sum(values_sorted, target, tolerance):
    """Index pairs a < b of a sorted array with |values[a] + values[b] - target| <= tolerance."""
    n = len(values_sorted)
    if n < 2:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    ia = np.arange(n - 1)
    va = values_sorted[:-1]
    lo = np.maximum(np.searchsorted(values_sorted, target - tolerance - va, side="left"), ia + 1)
    hi = np.searchsorted(values_sorted, target + tolerance - va, side="right")
    count = np.maximum(hi - lo, 0)
    total = int(count.sum())
    if total == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    a = np.repeat(ia, count)
    offset = np.repeat(np.cumsum(count) - count, count)
    return a, lo[a] + np.arange(total) - offset
#
def gate_ratio(coef, gate):
    """min(gate, coef) / coef: 1 when the gate is open."""
    if gate is None or coef <= 0.0 or gate >= coef:
        return 1.0
    return gate / coef
#
def take(members, keep):
    return tuple(m if np.ndim(m) == 0 else m[keep] for m in members)
#
def evaluate(frame0, frame1, be, ae, dt, cV, cX, rv, rx, box, order):
    """
    Test the candidates be -> ae (members as in group()) and return the
    survivors with dV / dV_err and dR / dR_max, both at full C.

    The test uses the constants times rv, rx (the gate), the ratios do not.
    Errors are summed member by member, in event order.
    """
    def total(key):
        s = 0.0
        for m in be:
            s = s + frame0[key][m]
        for m in ae:
            s = s + frame1[key][m]
        return s

    dV = np.abs(sum(frame1["vol"][m] for m in ae) - sum(frame0["vol"][m] for m in be))
    dV_err = np.maximum(EPS, cV * total("err"))
    gate_v = np.maximum(EPS, dV_err * rv)
    keep = dV <= gate_v + np.maximum(EPS, TOL * gate_v)
    if not np.any(keep):
        return None
    be, ae, dV, dV_err = take(be, keep), take(ae, keep), dV[keep], dV_err[keep]

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

    keep = dR <= gate_x + np.maximum(EPS, TOL * gate_x)
    if not np.any(keep):
        return None
    return take(be, keep), take(ae, keep), dV[keep] / dV_err[keep], dR[keep] / dR_max[keep]
#
def find_candidates(frame0, frame1, dt, C_V, C_X, box, event_types, gate=None, order=1):
    """
    Candidate events within the error bounds:

        continuation : [i]   -> [j]
        breakup      : [i]   -> [j, k]
        coalescence  : [i, k] -> [j]

    C_V, C_X are dicts by event type; gate=None uses them in full, a number g
    uses min(g, C). Returns, for each type, p0 (N x n_before), p1
    (N x n_after), dVrel, dRrel and cost = dVrel + dRrel.
    """
    out = {}
    for event_type, (nb, na) in EVENT_SHAPE.items():
        p0, p1, dVrel, dRrel = [], [], [], []

        # loop over the single DPS, search its partner(s) in the other snapshot
        single, other = (frame1, frame0) if event_type == "coalescence" else (frame0, frame1)
        n_other = max(nb, na)
        if event_type in event_types and len(single["vol"]) > 0 and len(other["vol"]) >= n_other:
            cV, cX = float(C_V[event_type]), float(C_X[event_type])
            rv, rx = gate_ratio(cV, gate), gate_ratio(cX, gate)

            by_vol = np.argsort(other["vol"], kind="stable")
            vol_sorted = other["vol"][by_vol]
            max_err = float(np.max(other["err"]))

            for s in range(len(single["vol"])):
                bound = cV * rv * (single["err"][s] + n_other * max_err)
                if order == 2:
                    bound += max(EPS, TOL * bound)
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

                be, ae = ((s,), partners) if single is frame0 else (partners, (s,))
                result = evaluate(frame0, frame1, be, ae, dt, cV, cX, rv, rx, box, order)
                if result is None:
                    continue

                be, ae, dv, dr = result
                n = len(dv)
                p0.append(np.column_stack([np.full(n, m) if np.ndim(m) == 0 else m for m in be]))
                p1.append(np.column_stack([np.full(n, m) if np.ndim(m) == 0 else m for m in ae]))
                dVrel.append(dv)
                dRrel.append(dr)

        if p0:
            dVrel, dRrel = np.concatenate(dVrel), np.concatenate(dRrel)
            out[event_type] = {"p0": np.concatenate(p0).astype(int),
                               "p1": np.concatenate(p1).astype(int),
                               "dVrel": dVrel, "dRrel": dRrel, "cost": dVrel + dRrel}
        else:
            out[event_type] = {"p0": np.empty((0, nb), dtype=int), "p1": np.empty((0, na), dtype=int),
                               "dVrel": np.empty(0), "dRrel": np.empty(0), "cost": np.empty(0)}
    return out
#
# ------------------------------------------------------------------
# Event organization: minimum-cost exact cover (Sec. 5)
# ------------------------------------------------------------------
#
def reduce_problem(enodes, k):
    """
    Exact cover preprocessing on DPS 0..k-1: a DPS with a single candidate
    forces it (repeated to a fixed point), then the rest is split into
    independent components. Returns (forced events, components as
    (DPS list, event list)), or None if a DPS is left without candidates.
    """
    node_events = [[] for _ in range(k)]
    for e, nodes in enumerate(enodes):
        for v in nodes:
            node_events[v].append(e)
    if any(not events for events in node_events):
        return None

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
    return forced, components
#
def branch_and_bound(cand, emask, ecost, esize, full_mask, budget=None):
    """
    Exact minimum-cost exact cover by depth-first search: branch on the
    uncovered DPS with the fewest feasible events, cheapest first; drop a
    state already reached with a cost that is not lower; every uncovered DPS
    b still costs at least min over its events e of cost(e) / |e|.
    Stops after `budget` search nodes if given. Returns the best cover
    (None if none) and whether the search was complete.
    """
    k = len(cand)
    share = [min(ecost[e] / esize[e] for e in cand[b]) for b in range(k)]
    ebound = [0.0] * len(emask)
    for b in range(k):
        for e in cand[b]:
            ebound[e] += share[b]

    best, best_cost = None, float("inf")
    memo = {}
    chosen = []
    nodes = 0
    complete = True

    def search(used, cost, rest):
        nonlocal best_cost, best, nodes, complete
        nodes += 1
        if budget is not None and nodes > budget:
            complete = False
        if not complete:
            return
        if used == full_mask:
            if cost < best_cost:
                best_cost, best = cost, chosen.copy()
            return
        if cost + rest >= best_cost or memo.get(used, float("inf")) <= cost:
            return
        memo[used] = cost

        branch = None
        todo = full_mask & ~used
        while todo:
            low = todo & -todo
            todo ^= low
            feasible = [e for e in cand[low.bit_length() - 1] if not emask[e] & used]
            if not feasible:
                return
            if branch is None or len(feasible) < len(branch):
                branch = feasible
                if len(branch) == 1:
                    break

        for e in branch:
            chosen.append(e)
            search(used | emask[e], cost + ecost[e], rest - ebound[e])
            chosen.pop()

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, k + 200))
    try:
        search(0, 0.0, sum(share))
    finally:
        sys.setrecursionlimit(limit)
    return best, complete
#
def highs_cover(enodes, ecost, k):
    """
    Minimum-cost exact cover with HiGHS (scipy): the LP relaxation first,
    which is enough when its optimum is 0/1, the 0/1 program (zero optimality
    gap) otherwise. Returns the selected events, or None if there is no cover.
    """
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    from scipy.sparse import csc_matrix

    rows = [v for nodes in enodes for v in nodes]
    cols = [e for e, nodes in enumerate(enodes) for _ in nodes]
    A = csc_matrix((np.ones(len(rows)), (rows, cols)), shape=(k, len(enodes)))
    c = np.asarray(ecost)

    lp = linprog(c, A_eq=A, b_eq=np.ones(k), bounds=(0.0, 1.0), method="highs-ds")
    if lp.status == 2:
        return None
    if lp.status == 0 and np.all(np.abs(lp.x - np.rint(lp.x)) <= 1.0e-7):
        return np.flatnonzero(lp.x > 0.5).tolist()

    res = milp(c, constraints=LinearConstraint(A, 1.0, 1.0), integrality=np.ones(len(c)),
               bounds=Bounds(0.0, 1.0), options={"mip_rel_gap": 0.0})
    if res.status == 2:
        return None
    if res.status != 0 or res.x is None:
        raise RuntimeError(f"HiGHS failed on a component of {k} DPS: {res.message}")
    return np.flatnonzero(res.x > 0.5).tolist()
#
def cover_component(enodes, ecost, k):
    """
    Minimum-cost exact cover of one component (DPS 0..k-1) by the given
    events: the branch-and-bound, with as many search nodes as there are
    events, then HiGHS if it did not finish (or the branch-and-bound to the
    end when scipy is not installed). Returns the selected events, or None
    if there is no cover.
    """
    full_mask = (1 << k) - 1
    emask = [sum(1 << v for v in nodes) for nodes in enodes]
    cand = [[] for _ in range(k)]
    for e, nodes in enumerate(enodes):
        for v in nodes:
            cand[v].append(e)
    for c in cand:
        c.sort(key=lambda e: (ecost[e], e))
    esize = [len(nodes) for nodes in enodes]

    best, complete = branch_and_bound(cand, emask, ecost, esize, full_mask,
                                      budget=len(enodes))
    if complete:
        return best
    try:
        return highs_cover(enodes, ecost, k)
    except ImportError:
        return branch_and_bound(cand, emask, ecost, esize, full_mask)[0]
#
def organize_events(n0, n1, candidates, debug=False):
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
        c = candidates[event_type]
        for p0, p1, cost in zip(c["p0"], c["p1"], c["cost"]):
            p0, p1 = tuple(int(i) for i in p0), tuple(int(j) for j in p1)
            if len(set(p0)) < len(p0) or len(set(p1)) < len(p1):
                raise ValueError(f"Invalid {event_type} candidate {p0} -> {p1}.")
            events.append((event_type, p0, p1, float(cost)))

    # nodes: DPS i at t^n is i, DPS j at t^(n+1) is n0 + j
    n_nodes = n0 + n1
    ev_nodes = [p0 + tuple(n0 + j for j in p1) for _, p0, p1, _ in events]
    ecost = [ev[3] for ev in events]

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

    selected = forced + [e for sub in chosen for e in sub]
    total_cost = 0.0
    for e in forced:
        total_cost += ecost[e]
    for sub in chosen:
        total_cost += float(sum(ecost[e] for e in sub))
    n_forced = len(forced)

    selected.sort()

    count = np.zeros(n_nodes, dtype=int)
    for e in selected:
        count[list(ev_nodes[e])] += 1
    if np.any(count != 1):
        raise RuntimeError(f"Not an exact cover: participation {count.tolist()}")

    result = [{"type": events[e][0], "p0": events[e][1], "p1": events[e][2], "cost": events[e][3]}
              for e in selected]
    n_type = {t: sum(ev["type"] == t for ev in result) for t in EVENT_SHAPE}
    n_cont, n_break, n_coal = n_type["continuation"], n_type["breakup"], n_type["coalescence"]
    if n_cont + n_break + 2 * n_coal != n0 or n_cont + 2 * n_break + n_coal != n1:
        raise RuntimeError("DPS-number balance violated, Eqs. (47)-(49).")

    summary = {
        "n0": n0, "n1": n1,
        "n_continuation": n_cont, "n_breakup": n_break, "n_coalescence": n_coal,
        "n_lost": 0, "n_new": 0,
        "total_cost": float(total_cost),
        "balance_lhs": n1 - n0, "balance_rhs": n_break - n_coal, "balance_error": 0,
        "n_cont_candidates": len(candidates["continuation"]["cost"]),
        "n_break_candidates": len(candidates["breakup"]["cost"]),
        "n_coal_candidates": len(candidates["coalescence"]["cost"]),
        "n_candidates_total": len(events),
        "n_components": len(components),
    }

    if debug:
        print("organization:", f"{n_cont} continuation, {n_break} breakup, {n_coal} coalescence;",
              f"{len(events)} candidates, {n_forced} forced, {len(components)} components",
              f"(sizes {[len(ce) for _, ce in components]}), cost {total_cost:.6e}")

    return result, summary
#
# ------------------------------------------------------------------
# Output
# ------------------------------------------------------------------
#
def build_record(istep_p0, istep_p1, time_p0, time_p1, events, summary, frame0, frame1, order):
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
    return record
#
