"""
Post-processing of the tracking output (event_history_raw.out): genealogy
cleaning, topology statistics, region assignment and plotting.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from helper_functions import cmpt_d
#
eps_gen = 1.0e-14
DEBRIS = ("__DEBRIS__",)
_TP = "__TRANSIENT_PAIR__"
#
def count_raw_topological_events(
    event_history_raw,
):
    #
    total_breakups_raw = 0
    total_coalescences_raw = 0
    #
    raw_event_series = []
    #
    for ih,record in enumerate(event_history_raw):
        #
        n_breakup = 0
        n_coalescence = 0
        #
        for ev in record["events"]:
            #
            if ev["type"] == "breakup":
                #
                n_breakup += 1
                total_breakups_raw += 1
            #
            elif ev["type"] == "coalescence":
                #
                n_coalescence += 1
                total_coalescences_raw += 1
        #
        raw_event_series.append([
            float(record["time_p1"]),
            int(record["istep_p1"]),
            int(n_breakup),
            int(n_coalescence),
            int(total_breakups_raw),
            int(total_coalescences_raw),
        ])
    #
    return (
        total_breakups_raw,
        total_coalescences_raw,
        raw_event_series,
    )
#
def get_events_by_type(events,event_type):
    #
    return [
        ev for ev in events
        if ev["type"] == event_type
    ]
#
def print_topological_event_history(event_history):
    #
    for ih,record in enumerate(event_history):
        #
        events_topology = [
            ev for ev in record["events"]
            if ev["type"] != "continuation"
        ]
        #
        if len(events_topology) == 0:
            continue
        #
        print(
            "----------------------------------------------------"
        )
        #
        print(
            f"history index = {ih}, "
            f"istep = {record['istep_p0']} -> {record['istep_p1']}, "
            f"time = {record['time_p0']:.8e} -> "
            f"{record['time_p1']:.8e}"
        )
        #
        for ev in events_topology:
            #
            print(
                f"{ev['type']:12s} "
                f"{ev['p0']} -> {ev['p1']} "
                f"cost = {ev['cost']:.6e}"
            )
#
def rebuild_indexes(events):
    #
    ind_np0 = []
    ind_np1 = []
    #
    ind_par = []
    ind_ch1 = []
    ind_ch2 = []
    #
    ind_par1 = []
    ind_par2 = []
    ind_ch = []
    #
    for ev in events:
        #
        ev_type = ev["type"]
        #
        if ev_type == "continuation":
            ind_np0.append(ev["p0"][0])
            ind_np1.append(ev["p1"][0])
        #
        elif ev_type == "breakup":
            ind_par.append(ev["p0"][0])
            ind_ch1.append(ev["p1"][0])
            ind_ch2.append(ev["p1"][1])
        #
        elif ev_type == "coalescence":
            ind_par1.append(ev["p0"][0])
            ind_par2.append(ev["p0"][1])
            ind_ch.append(ev["p1"][0])
        #
        else:
            raise ValueError(f"Unknown event type: {ev_type}")
    #
    return (
        np.asarray(ind_np0, dtype=int),
        np.asarray(ind_np1, dtype=int),

        np.asarray(ind_par, dtype=int),
        np.asarray(ind_ch1, dtype=int),
        np.asarray(ind_ch2, dtype=int),

        np.asarray(ind_par1, dtype=int),
        np.asarray(ind_par2, dtype=int),
        np.asarray(ind_ch, dtype=int),
    )
#
def print_clean_genealogy_events(clean_result):
    #
    history = clean_result["physical_event_history"]
    #
    for record in history:
        #
        events = record["events"]
        #
        events_topology = [
            ev for ev in events
            if ev["type"] != "continuation"
        ]
        #
        if len(events_topology) == 0:
            continue
        #
        print("----------------------------------------------------")
        print(
            f"istep = {record['istep_p0']} "
            f"-> {record['istep_p1']}"
        )
        #
        for ev in events_topology:
            #
            ev_type = ev["type"]
            #
            if ev_type == "coalescence":
                #
                print(
                    "TRUE COALESCENCE:",
                    ev["parents"],
                    "->",
                    ev["child"],
                )
            #
            elif ev_type == "fake_coalescence":
                #
                print(
                    "FAKE COALESCENCE:",
                    ev["raw_p0"],
                    "->",
                    ev["raw_p1"],
                    "ID =",
                    ev["physical_id"],
                )
            #
            elif ev_type == "breakup":
                #
                print(
                    "TRUE BREAKUP:",
                    ev["parent"],
                    "->",
                    ev["children"],
                )
            #
            elif ev_type == "fake_breakup":
                #
                print(
                    "FAKE BREAKUP:",
                    ev["raw_p0"],
                    "->",
                    ev["raw_p1"],
                    "ID =",
                    ev["physical_id"],
                )
#
def _is_inside_drainage_window(
    physical_id,
    time_now,
    drainage_state,
    max_drainage_time,
):
    #
    if physical_id not in drainage_state:
        return False
    #
    time_start = drainage_state[physical_id]["time_start"]
    #
    return (time_now-time_start) <= max_drainage_time
#
def _active_indices_for_physical_id(
    idx_to_id,
    physical_id,
):
    #
    return [
        int(i)
        for i,bid in idx_to_id.items()
        if bid == physical_id
    ]
#
def store_physical_diameter_snapshot(
    diameter_history,
    idx_to_id,
    voll,
    istep,
    time,
    is_2d,
):
    #
    # ========================================================
    # Group raw tagged components by physical ID
    # ========================================================
    #
    id_to_indices = {}
    #
    for index,physical_id in idx_to_id.items():
        #
        if physical_id not in id_to_indices:
            #
            id_to_indices[physical_id] = []
        #
        id_to_indices[physical_id].append(int(index))
    #
    # ========================================================
    # Compute physical volume and diameter
    # ========================================================
    #
    for physical_id,indices in id_to_indices.items():
        #
        # During drainage several raw components can have the
        # same physical ID. Their volumes must therefore be
        # summed before computing the physical diameter.
        #
        V = float(
            np.sum([
                voll[i]
                for i in indices
            ])
        )
        #
        D = float(
            cmpt_d(
                V,
                is_2d,
            )
        )
        #
        # ====================================================
        # Initialize physical trajectory
        # ====================================================
        #
        if physical_id not in diameter_history:
            #
            diameter_history[physical_id] = {
                "time": [],
                "istep": [],
                "volume": [],
                "diameter": [],
                "raw_indices": [],
            }
        #
        # ====================================================
        # Store snapshot
        # ====================================================
        #
        diameter_history[physical_id]["time"].append(
            float(time)
        )
        #
        diameter_history[physical_id]["istep"].append(
            int(istep)
        )
        #
        diameter_history[physical_id]["volume"].append(
            V
        )
        #
        diameter_history[physical_id]["diameter"].append(
            D
        )
        #
        diameter_history[physical_id]["raw_indices"].append(
            tuple(sorted(indices))
        )
#
def _make_new_breakup_id(parent_id, used_ids):
    counter = 1
    while True:
        new_id = f"{parent_id}_B{counter}"
        if new_id not in used_ids:
            used_ids.add(new_id)
            return new_id
        counter += 1
#
def _make_two_new_breakup_ids(parent_id, used_ids):
    child_id_a = _make_new_breakup_id(parent_id, used_ids)
    child_id_b = _make_new_breakup_id(parent_id, used_ids)
    return child_id_a, child_id_b
#
def _make_new_coalescence_id(parent_id, used_ids):
    counter = 1
    while True:
        new_id = f"{parent_id}_C{counter}"
        if new_id not in used_ids:
            used_ids.add(new_id)
            return new_id
        counter += 1
#
def get_breakup_generation(bid):
    return sum(part.startswith("B") for part in str(bid).split("_"))
#
def _register_birth(
    bubble_lifetime,
    bid,
    generation,
    time,
    istep,
    event_type,
    parents=(),
):
    if bid in bubble_lifetime:
        raise RuntimeError(
            f"Trying to birth an already-existing physical ID: {bid}"
        )
    bubble_lifetime[bid] = {
        "generation": int(generation),
        "t_birth": float(time),
        "istep_birth": None if istep is None else int(istep),
        "birth_event": str(event_type),
        "t_death": None,
        "istep_death": None,
        "death_event": None,
        "t_life": None,
        "parents": tuple(parents),
        "children": (),
    }
#
def _register_death(
    bubble_lifetime,
    bid,
    time,
    istep,
    event_type,
    children=(),
):
    if bid not in bubble_lifetime:
        raise RuntimeError(f"Trying to kill unknown physical ID: {bid}")
    rec = bubble_lifetime[bid]
    if rec["t_death"] is not None:
        raise RuntimeError(
            f"Physical ID {bid} already died at "
            f"t={rec['t_death']}, istep={rec['istep_death']}"
        )
    rec["t_death"] = float(time)
    rec["istep_death"] = int(istep)
    rec["death_event"] = str(event_type)
    rec["t_life"] = float(time) - rec["t_birth"]
    rec["children"] = tuple(children)
#
def compute_topology_relaxation_time_phy(
    V,rho,mu,sigma,is_2d,eps=eps_gen,
):
    #
    # ========================================================
    # 1. Shape-agnostic event length from measure only
    #
    # In 2D, V is an area  -> L = V^(1/2).
    # In 3D, V is a volume -> L = V^(1/3).
    #
    # No circle/sphere radius and no interface-area-based
    # length are used, so strong deformation does not shorten
    # the physical topology-relaxation time.
    # ========================================================
    #
    V = float(V)
    rho = float(rho)
    mu = float(mu)
    sigma = float(sigma)
    #
    if V <= eps:
        #
        return 0.0
    #
    if sigma <= eps:
        #
        raise ValueError("Non-positive surface tension in topology-relaxation time.")
    #
    if is_2d:
        #
        L = np.sqrt(V)
    else:
        #
        L = V**(1.0/3.0)
    #
    # ========================================================
    # 2. Capillary-inertial and viscous-capillary limits
    # ========================================================
    #
    t_capillary = np.sqrt(max(rho,0.0)*L**3/sigma)
    t_viscous = max(mu,0.0)*L/sigma
    #
    # ========================================================
    # 3. Unified physical relaxation time
    #
    # From
    #
    #     sigma/L = rho*(L/t)^2 + mu/t
    #
    # taking the positive root. No fitted coefficient.
    # ========================================================
    #
    t_relax = 0.5*(
        t_viscous
        +
        np.sqrt(t_viscous**2+4.0*t_capillary**2)
    )
    #
    return float(t_relax)
#
def compute_event_topology_relaxation_time(
    V_before,V_after,delta_t,rho,mu,sigma,is_2d,eps=eps_gen,
):
    #
    # ========================================================
    # Event-level topology-relaxation time.
    #
    # V_before and V_after are TOTAL dispersed-phase measures
    # participating in the event. Their maximum is used so the
    # scale is not reduced by the numerical volume mismatch.
    #
    # The tracking/sampling interval delta_t is also a hard
    # temporal-resolution limit: a reversal occurring within one
    # sampled interval cannot be resolved more accurately than
    # that interval.
    # ========================================================
    #
    V_event = max(
        abs(float(V_before)),
        abs(float(V_after)),
    )
    #
    t_physical = compute_topology_relaxation_time_phy(
        V_event,rho,mu,sigma,is_2d,eps=eps,
    )
    #
    t_sampling = abs(float(delta_t))
    #
    return float(max(t_physical,t_sampling))
#
def _has_quick_recoalescence(
    event_history_raw,history_index,child_a,child_b,
    event_time,resolution_time):
    #
    # ========================================================
    # Follow both daughter branches through continuations and
    # return the ACTUAL recoalescence event if they merge again
    # inside the numerical topology-resolution window.
    # ========================================================
    #
    branch_a = int(child_a)
    branch_b = int(child_b)
    #
    for j in range(history_index+1,len(event_history_raw)):
        #
        rec = event_history_raw[j]
        #
        if float(rec["time_p0"])-event_time > resolution_time:
            #
            return None
        #
        next_a = None
        next_b = None
        #
        for ev in rec["events"]:
            #
            etype = ev["type"]
            #
            if etype == "continuation":
                #
                i0 = int(ev["p0"][0])
                i1 = int(ev["p1"][0])
                #
                if i0 == branch_a:
                    #
                    next_a = i1
                #
                if i0 == branch_b:
                    #
                    next_b = i1
            #
            elif etype == "coalescence":
                #
                i0a = int(ev["p0"][0])
                i0b = int(ev["p0"][1])
                parents = {i0a,i0b}
                #
                if parents == {branch_a,branch_b}:
                    #
                    return {
                        "history_index": j,
                        "record": rec,
                        "event": ev,
                    }
                #
                if branch_a in parents or branch_b in parents:
                    #
                    return None
            #
            elif etype == "breakup":
                #
                i0 = int(ev["p0"][0])
                #
                if i0 == branch_a or i0 == branch_b:
                    #
                    return None
        #
        if next_a is None or next_b is None:
            #
            return None
        #
        branch_a = next_a
        branch_b = next_b
    #
    return None
#
def _find_quick_rebreakup(
    event_history_raw,history_index,merged_child,
    event_time,resolution_time):
    #
    # ========================================================
    # Follow the merged raw branch through continuations and
    # return the ACTUAL breakup event occurring inside the
    # numerical topology-resolution window.
    # ========================================================
    #
    branch = int(merged_child)
    #
    for j in range(history_index+1,len(event_history_raw)):
        #
        rec = event_history_raw[j]
        #
        if float(rec["time_p0"])-event_time > resolution_time:
            #
            return None
        #
        next_branch = None
        #
        for ev in rec["events"]:
            #
            etype = ev["type"]
            #
            if etype == "continuation":
                #
                i0 = int(ev["p0"][0])
                i1 = int(ev["p1"][0])
                #
                if i0 == branch:
                    #
                    next_branch = i1
            #
            elif etype == "breakup":
                #
                i0 = int(ev["p0"][0])
                #
                if i0 == branch:
                    #
                    return {
                        "history_index": j,
                        "record": rec,
                        "event": ev,
                    }
            #
            elif etype == "coalescence":
                #
                i0a = int(ev["p0"][0])
                i0b = int(ev["p0"][1])
                #
                if branch == i0a or branch == i0b:
                    #
                    return None
        #
        if next_branch is None:
            #
            return None
        #
        branch = next_branch
    #
    return None
#
def _volume_uncertainty_ratio(
    Va,dVa_err,Vb,dVb_err,eps=eps_gen,
):
    #
    # ========================================================
    # Difference between two measured DPS volumes normalized
    # by their combined interface-discretization uncertainty:
    #
    #       |Va - Vb|
    #     ----------------
    #     dVa_err+dVb_err
    #
    # A ratio <= 1 means that the two uncertainty intervals
    # overlap. The threshold 1 is therefore not a fitted
    # tolerance; it follows directly from the error bounds.
    # ========================================================
    #
    dV = abs(float(Va)-float(Vb))
    dV_err = abs(float(dVa_err))+abs(float(dVb_err))
    #
    if dV_err <= eps:
        #
        return 0.0 if dV <= eps else np.inf
    #
    return float(dV/dV_err)
#
def _volumes_numerically_indistinguishable(
    Va,dVa_err,Vb,dVb_err,eps=eps_gen,
):
    #
    ratio = _volume_uncertainty_ratio(
        Va,dVa_err,Vb,dVb_err,eps=eps,
    )
    #
    return bool(ratio <= 1.0),float(ratio)
#
def _rebreakup_reconstructs_original_pair(
    V_parent_a,dV_parent_a,
    V_parent_b,dV_parent_b,
    V_child_a,dV_child_a,
    V_child_b,dV_child_b,
):
    #
    # ========================================================
    # Determine whether
    #
    #     A + B -> C -> D + E
    #
    # reconstructs the original pair within the numerical
    # volume uncertainties dV_err = sum_c(A_c*dx_c).
    #
    # Both daughter permutations are tested:
    #
    #     A -> D, B -> E
    #     A -> E, B -> D
    #
    # No relative percentage tolerance is used.
    # ========================================================
    #
    _,error_direct_a = _volumes_numerically_indistinguishable(
        V_parent_a,dV_parent_a,V_child_a,dV_child_a,
    )
    #
    _,error_direct_b = _volumes_numerically_indistinguishable(
        V_parent_b,dV_parent_b,V_child_b,dV_child_b,
    )
    #
    error_direct = max(error_direct_a,error_direct_b)
    #
    _,error_cross_a = _volumes_numerically_indistinguishable(
        V_parent_a,dV_parent_a,V_child_b,dV_child_b,
    )
    #
    _,error_cross_b = _volumes_numerically_indistinguishable(
        V_parent_b,dV_parent_b,V_child_a,dV_child_a,
    )
    #
    error_cross = max(error_cross_a,error_cross_b)
    #
    if error_direct <= error_cross:
        #
        reconstruction_error = error_direct
        swap_children = False
    else:
        #
        reconstruction_error = error_cross
        swap_children = True
    #
    reconstructs_pair = reconstruction_error <= 1.0
    #
    return reconstructs_pair,float(reconstruction_error),swap_children
#
def _recoalescence_reconstructs_original_parent(
    V_parent,dV_parent,V_child,dV_child,
):
    #
    # ========================================================
    # Parameter-free reconstruction test for
    #
    #     A -> B + C -> D.
    # ========================================================
    #
    reconstructs,error = _volumes_numerically_indistinguishable(
        V_parent,dV_parent,V_child,dV_child,
    )
    #
    return reconstructs,float(error)
#
def _make_transient_pair(id_a,id_b,swap_children=False):
    #
    # swap_children stores which daughter permutation best
    # reconstructs the original pre-coalescence pair.
    #
    return ("__TRANSIENT_PAIR__",id_a,id_b,bool(swap_children))
#
def _is_transient_pair(value):
    return (
        isinstance(value,tuple)
        and len(value) >= 3
        and value[0] == "__TRANSIENT_PAIR__"
    )
#
def count_clean_topological_events(physical_event_history):
    #
    # ========================================================
    # Authoritative count of PHYSICAL topology events.
    #
    # IMPORTANT:
    #
    # A coalescence A+B->C creates two parent death records in
    # bubble_lifetime, but it is only ONE topology event.
    #
    # Therefore topology counts must be obtained from the
    # physical event history, never from birth/death records.
    #
    # Fake/transient topology events are intentionally ignored.
    # ========================================================
    #
    total_breakups = 0
    total_coalescences = 0
    #
    for record in physical_event_history:
        #
        for ev in record["events"]:
            #
            if ev["type"] == "breakup":
                #
                total_breakups += 1
            #
            elif ev["type"] == "coalescence":
                #
                total_coalescences += 1
    #
    return total_breakups,total_coalescences
#
def clean_genealogy_from_raw_history_legacy(
    event_history_raw,idx_to_id_initial,id_to_birth_time_initial,
    rho,mu,sigma,is_2d,debug=False,
):
    #
    # ========================================================
    # Local genealogy builder
    #
    # This is the ONLY place where the genealogy metadata
    # exposed to post-processing is assembled.
    #
    # The physical ID itself is used as the genealogy label:
    #
    # b0
    # b0_B1
    # b0_B1_C_2
    # ...
    # ========================================================
    #
    def build_genealogy(bubble_lifetime):
        #
        genealogy = {}
        #
        for physical_id,data in bubble_lifetime.items():
            #
            genealogy[physical_id] = {
                "label": str(physical_id),
                "generation": int(data["generation"]),
                "parents": tuple(data["parents"]),
                "children": tuple(data["children"]),
                "birth_event": data["birth_event"],
                "death_event": data["death_event"],
                "t_birth": data["t_birth"],
                "t_death": data["t_death"],
            }
        #
        return genealogy
    #
    # ========================================================
    # Empty history
    # ========================================================
    #
    if len(event_history_raw) == 0:
        #
        bubble_lifetime = {}
        #
        for bid,t_birth in id_to_birth_time_initial.items():
            #
            bubble_lifetime[bid] = {
                "generation": 0,
                "t_birth": float(t_birth),
                "istep_birth": None,
                "birth_event": "initial",
                "t_death": None,
                "istep_death": None,
                "death_event": None,
                "t_life": None,
                "parents": (),
                "children": (),
            }
        #
        genealogy = build_genealogy(
            bubble_lifetime
        )
        #
        return {
            "physical_event_history": [],
            "idx_to_id_history": [],
            "idx_to_id_final": dict(idx_to_id_initial),
            "id_to_birth_time": dict(id_to_birth_time_initial),
            "bubble_lifetime": bubble_lifetime,
            "genealogy": genealogy,
            "total_breakups": 0,
            "total_coalescences": 0,
            "breakup_event": [],
            "diameter_tree": [],
            "diameter_history": {},
        }
    #
    # ========================================================
    # Initial state
    # ========================================================
    #
    idx_to_id_curr = {
        int(i):bid
        for i,bid in idx_to_id_initial.items()
    }
    #
    id_to_birth_time = dict(
        id_to_birth_time_initial
    )
    #
    used_ids = set(
        id_to_birth_time.keys()
    )
    #
    bubble_lifetime = {}
    #
    for bid,t_birth in id_to_birth_time_initial.items():
        #
        bubble_lifetime[bid] = {
            "generation": 0,
            "t_birth": float(t_birth),
            "istep_birth": None,
            "birth_event": "initial",
            "t_death": None,
            "istep_death": None,
            "death_event": None,
            "t_life": None,
            "parents": (),
            "children": (),
        }
    #
    physical_event_history = []
    idx_to_id_history = [
        dict(idx_to_id_curr)
    ]
    #
    breakup_event = []
    diameter_tree = []
    diameter_history = {}
    #
    total_breakups = 0
    total_coalescences = 0
    #
    # ========================================================
    # Initial diameter
    # ========================================================
    #
    first_record = event_history_raw[0]
    #
    store_physical_diameter_snapshot(
        diameter_history,
        idx_to_id_curr,
        first_record["voll0"],
        first_record["istep_p0"],
        first_record["time_p0"],
        is_2d,
    )
    #
    # ========================================================
    # Raw history
    # ========================================================
    #
    for ih,record in enumerate(event_history_raw):
        #
        events = record["events"]
        #
        time_p0 = float(
            record["time_p0"]
        )
        #
        time_p1 = float(
            record["time_p1"]
        )
        #
        istep_p0 = int(
            record["istep_p0"]
        )
        #
        istep_p1 = int(
            record["istep_p1"]
        )
        #
        voll0 = record["voll0"]
        voll1 = record["voll1"]
        area0 = record["area0"]
        area1 = record["area1"]
        area0_err = record["area0_err"]
        area1_err = record["area1_err"]
        #
        idx_to_id_next = {}
        physical_events_step = []
        #
        if debug:
            #
            print("====================================================")
            print(
                f"Clean genealogy: history={ih}, "
                f"istep={istep_p0}->{istep_p1}"
            )
        #
        # ====================================================
        # Events
        # ====================================================
        #
        for ev in events:
            #
            ev_type = ev["type"]
            #
            # =================================================
            # Continuation
            # =================================================
            #
            if ev_type == "continuation":
                #
                i0 = int(
                    ev["p0"][0]
                )
                #
                i1 = int(
                    ev["p1"][0]
                )
                #
                if i0 not in idx_to_id_curr:
                    #
                    raise RuntimeError(
                        f"Missing physical ID for continuation "
                        f"p0={i0}, istep={istep_p0}"
                    )
                #
                idx_to_id_next[i1] = idx_to_id_curr[i0]
            #
            # =================================================
            # Coalescence
            # =================================================
            #
            elif ev_type == "coalescence":
                #
                i0a = int(
                    ev["p0"][0]
                )
                #
                i0b = int(
                    ev["p0"][1]
                )
                #
                i1 = int(
                    ev["p1"][0]
                )
                #
                if (
                    i0a not in idx_to_id_curr
                    or
                    i0b not in idx_to_id_curr
                ):
                    #
                    raise RuntimeError(
                        f"Missing physical coalescence parent "
                        f"at istep={istep_p0}"
                    )
                #
                id_a = idx_to_id_curr[i0a]
                id_b = idx_to_id_curr[i0b]
                #
                # -------------------------------------------------
                # Breakup -> quick recoalescence
                #
                # Both branches already carry the same physical ID.
                # -------------------------------------------------
                #
                if id_a == id_b:
                    #
                    idx_to_id_next[i1] = id_a
                    #
                    physical_events_step.append({
                        "type": "fake_coalescence",
                        "istep": istep_p1,
                        "time": time_p1,
                        "raw_p0": (i0a,i0b),
                        "raw_p1": (i1,),
                        "physical_id": id_a,
                        "raw_cost": float(ev["cost"]),
                    })
                    #
                    if debug:
                        #
                        print(
                            "FAKE COALESCENCE:",
                            id_a,
                        )
                    #
                    continue
                #
                # -------------------------------------------------
                # Finish coalescence -> quick rebreakup transient
                # -------------------------------------------------
                #
                if _is_transient_pair(id_a) or _is_transient_pair(id_b):
                    #
                    raise RuntimeError(
                        "Transient pair reached unexpected coalescence. "
                        "Check topology sequence."
                    )
                #
                resolution_time = compute_event_topology_relaxation_time(
                    float(voll0[i0a])+float(voll0[i0b]),
                    float(voll1[i1]),
                    time_p1-time_p0,
                    rho,
                    mu,
                    sigma,
                    is_2d,
                )
                #
                quick_rebreakup = _find_quick_rebreakup(
                    event_history_raw,
                    ih,
                    i1,
                    time_p1,
                    resolution_time,
                )
                #
                collapse_as_transient = False
                reconstruction_error = None
                swap_children = False
                rebreakup_dt = None
                #
                if quick_rebreakup is not None:
                    #
                    rebreakup_record = quick_rebreakup["record"]
                    rebreakup_event = quick_rebreakup["event"]
                    #
                    i1a_rebreak = int(rebreakup_event["p1"][0])
                    i1b_rebreak = int(rebreakup_event["p1"][1])
                    #
                    V_parent_a = float(voll0[i0a])
                    V_parent_b = float(voll0[i0b])
                    dV_parent_a = float(area0_err[i0a])
                    dV_parent_b = float(area0_err[i0b])
                    #
                    V_child_a = float(rebreakup_record["voll1"][i1a_rebreak])
                    V_child_b = float(rebreakup_record["voll1"][i1b_rebreak])
                    dV_child_a = float(rebreakup_record["area1_err"][i1a_rebreak])
                    dV_child_b = float(rebreakup_record["area1_err"][i1b_rebreak])
                    #
                    (
                        collapse_as_transient,
                        reconstruction_error,
                        swap_children,
                    ) = _rebreakup_reconstructs_original_pair(
                            V_parent_a,dV_parent_a,
                            V_parent_b,dV_parent_b,
                            V_child_a,dV_child_a,
                            V_child_b,dV_child_b,
                        )
                    #
                    rebreakup_dt = float(rebreakup_record["time_p1"])-time_p1
                    #
                    if debug:
                        #
                        print(
                            "QUICK COALESCENCE -> BREAKUP CHECK:",
                            id_a,
                            "+",
                            id_b,
                            "dt =",
                            rebreakup_dt,
                            "resolution_time =",
                            resolution_time,
                            "reconstruction_error =",
                            reconstruction_error,
                            "collapse =",
                            collapse_as_transient,
                        )
                #
                if collapse_as_transient:
                    #
                    pair = _make_transient_pair(
                        id_a,
                        id_b,
                        swap_children=swap_children,
                    )
                    #
                    idx_to_id_next[i1] = pair
                    #
                    physical_events_step.append({
                        "type": "transient_coalescence",
                        "istep": istep_p1,
                        "time": time_p1,
                        "raw_p0": (i0a,i0b),
                        "raw_p1": (i1,),
                        "parents": (id_a,id_b),
                        "resolution_time": resolution_time,
                        "rebreakup_dt": rebreakup_dt,
                        "reconstruction_error": reconstruction_error,
                        "swap_children": swap_children,
                        "raw_cost": float(ev["cost"]),
                    })
                    #
                    if debug:
                        #
                        print(
                            "TRANSIENT COALESCENCE:",
                            id_a,
                            "+",
                            id_b,
                            "resolution_time =",
                            resolution_time,
                            "reconstruction_error =",
                            reconstruction_error,
                        )
                    #
                    continue
                #
                if debug and quick_rebreakup is not None:
                    #
                    print(
                        "KEEP PHYSICAL COALESCENCE:",
                        id_a,
                        "+",
                        id_b,
                        "quick breakup does not reconstruct parents;",
                        "reconstruction_error =",
                        reconstruction_error,
                    )
                #
                # -------------------------------------------------
                # True coalescence
                # -------------------------------------------------
                #
                dominant_id = (
                    id_a
                    if voll0[i0a] >= voll0[i0b]
                    else id_b
                )
                #
                new_id = _make_new_coalescence_id(
                    dominant_id,
                    used_ids,
                )
                #
                idx_to_id_next[i1] = new_id
                id_to_birth_time[new_id] = time_p1
                #
                generation_a = bubble_lifetime[id_a]["generation"]
                generation_b = bubble_lifetime[id_b]["generation"]
                #
                # -------------------------------------------------
                # IMPORTANT
                #
                # Genealogy generation is defined HERE.
                #
                # The plot does not reconstruct this.
                # -------------------------------------------------
                #
                new_generation = (
                    max(
                        generation_a,
                        generation_b,
                    )
                    +1
                )
                #
                _register_death(
                    bubble_lifetime,
                    id_a,
                    time_p1,
                    istep_p1,
                    "coalescence",
                    children=(new_id,),
                )
                #
                _register_death(
                    bubble_lifetime,
                    id_b,
                    time_p1,
                    istep_p1,
                    "coalescence",
                    children=(new_id,),
                )
                #
                _register_birth(
                    bubble_lifetime,
                    new_id,
                    new_generation,
                    time_p1,
                    istep_p1,
                    "coalescence",
                    parents=(id_a,id_b),
                )
                #
                D_parent_a = float(
                    cmpt_d(
                        voll0[i0a],
                        is_2d,
                    )
                )
                #
                D_parent_b = float(
                    cmpt_d(
                        voll0[i0b],
                        is_2d,
                    )
                )
                #
                D_child = float(
                    cmpt_d(
                        voll1[i1],
                        is_2d,
                    )
                )
                #
                diameter_tree.append({
                    "type": "coalescence",
                    "time_p0": time_p0,
                    "time_p1": time_p1,
                    "istep_p0": istep_p0,
                    "istep_p1": istep_p1,
                    "parents": (id_a,id_b),
                    "parent_generations": (
                        generation_a,
                        generation_b,
                    ),
                    "child": new_id,
                    "child_generation": new_generation,
                    "D_parents": (
                        D_parent_a,
                        D_parent_b,
                    ),
                    "D_child": D_child,
                })
                #
                total_coalescences += 1
                #
                physical_events_step.append({
                    "type": "coalescence",
                    "istep": istep_p1,
                    "time": time_p1,
                    "raw_p0": (i0a,i0b),
                    "raw_p1": (i1,),
                    "parents": (id_a,id_b),
                    "child": new_id,
                    "generation": new_generation,
                    "resolution_time": resolution_time,
                    "raw_cost": float(ev["cost"]),
                })
                #
                if debug:
                    #
                    print(
                        "TRUE COALESCENCE:",
                        id_a,
                        "+",
                        id_b,
                        "->",
                        new_id,
                        "generation =",
                        new_generation,
                    )
            #
            # =================================================
            # Breakup
            # =================================================
            #
            elif ev_type == "breakup":
                #
                i0 = int(
                    ev["p0"][0]
                )
                #
                i1a = int(
                    ev["p1"][0]
                )
                #
                i1b = int(
                    ev["p1"][1]
                )
                #
                if i0 not in idx_to_id_curr:
                    #
                    raise RuntimeError(
                        f"Missing physical breakup parent "
                        f"p0={i0}, istep={istep_p0}"
                    )
                #
                parent_id = idx_to_id_curr[i0]
                #
                # -------------------------------------------------
                # Finish transient coalescence -> rebreakup
                # -------------------------------------------------
                #
                if _is_transient_pair(parent_id):
                    #
                    id_a = parent_id[1]
                    id_b = parent_id[2]
                    swap_children = (
                        bool(parent_id[3])
                        if len(parent_id) >= 4
                        else False
                    )
                    #
                    # -------------------------------------------------
                    # Restore the original physical parents using the
                    # daughter permutation selected by the volume
                    # reconstruction test.
                    # -------------------------------------------------
                    #
                    if swap_children:
                        #
                        idx_to_id_next[i1a] = id_b
                        idx_to_id_next[i1b] = id_a
                    else:
                        #
                        idx_to_id_next[i1a] = id_a
                        idx_to_id_next[i1b] = id_b
                    #
                    physical_events_step.append({
                        "type": "transient_breakup",
                        "istep": istep_p1,
                        "time": time_p1,
                        "raw_p0": (i0,),
                        "raw_p1": (i1a,i1b),
                        "physical_ids": (id_a,id_b),
                        "raw_cost": float(ev["cost"]),
                    })
                    #
                    if debug:
                        #
                        print(
                            "TRANSIENT REBREAKUP:",
                            parent_id,
                            "->",
                            id_a,
                            "+",
                            id_b,
                        )
                    #
                    continue
                #
                # -------------------------------------------------
                # Check breakup -> quick recoalescence
                # -------------------------------------------------
                #
                resolution_time = compute_event_topology_relaxation_time(
                    float(voll0[i0]),
                    float(voll1[i1a])+float(voll1[i1b]),
                    time_p1-time_p0,
                    rho,
                    mu,
                    sigma,
                    is_2d,
                )
                #
                quick_recoalescence = _has_quick_recoalescence(
                    event_history_raw,
                    ih,
                    i1a,
                    i1b,
                    time_p1,
                    resolution_time,
                )
                #
                collapse_as_transient = False
                reconstruction_error = None
                recoalescence_dt = None
                #
                if quick_recoalescence is not None:
                    #
                    recoalescence_record = quick_recoalescence["record"]
                    recoalescence_event = quick_recoalescence["event"]
                    i1_recoal = int(recoalescence_event["p1"][0])
                    #
                    V_parent = float(voll0[i0])
                    dV_parent = float(area0_err[i0])
                    V_recoal = float(recoalescence_record["voll1"][i1_recoal])
                    dV_recoal = float(recoalescence_record["area1_err"][i1_recoal])
                    #
                    (
                        collapse_as_transient,
                        reconstruction_error,
                    ) = _recoalescence_reconstructs_original_parent(
                        V_parent,dV_parent,
                        V_recoal,dV_recoal,
                    )
                    #
                    recoalescence_dt = float(recoalescence_record["time_p1"])-time_p1
                    #
                    if debug:
                        #
                        print(
                            "QUICK BREAKUP -> COALESCENCE CHECK:",
                            parent_id,
                            "dt =",
                            recoalescence_dt,
                            "resolution_time =",
                            resolution_time,
                            "reconstruction_error =",
                            reconstruction_error,
                            "collapse =",
                            collapse_as_transient,
                        )
                #
                if collapse_as_transient:
                    #
                    idx_to_id_next[i1a] = parent_id
                    idx_to_id_next[i1b] = parent_id
                    #
                    physical_events_step.append({
                        "type": "transient_breakup",
                        "istep": istep_p1,
                        "time": time_p1,
                        "raw_p0": (i0,),
                        "raw_p1": (i1a,i1b),
                        "physical_id": parent_id,
                        "resolution_time": resolution_time,
                        "recoalescence_dt": recoalescence_dt,
                        "reconstruction_error": reconstruction_error,
                        "raw_cost": float(ev["cost"]),
                    })
                    #
                    if debug:
                        #
                        print(
                            "TRANSIENT BREAKUP:",
                            parent_id,
                            "resolution_time =",
                            resolution_time,
                            "reconstruction_error =",
                            reconstruction_error,
                        )
                    #
                    continue
                #
                # -------------------------------------------------
                # True breakup
                # -------------------------------------------------
                #
                t_birth = bubble_lifetime[parent_id]["t_birth"]
                t_death = time_p1
                t_life = t_death-t_birth
                #
                V1 = float(
                    voll1[i1a]
                )
                #
                V2 = float(
                    voll1[i1b]
                )
                #
                child_id_a,child_id_b = _make_two_new_breakup_ids(
                    parent_id,
                    used_ids,
                )
                #
                idx_to_id_next[i1a] = child_id_a
                idx_to_id_next[i1b] = child_id_b
                #
                id_to_birth_time[child_id_a] = time_p1
                id_to_birth_time[child_id_b] = time_p1
                #
                parent_generation = bubble_lifetime[parent_id]["generation"]
                child_generation = parent_generation+1
                #
                _register_death(
                    bubble_lifetime,
                    parent_id,
                    time_p1,
                    istep_p1,
                    "breakup",
                    children=(
                        child_id_a,
                        child_id_b,
                    ),
                )
                #
                _register_birth(
                    bubble_lifetime,
                    child_id_a,
                    child_generation,
                    time_p1,
                    istep_p1,
                    "breakup",
                    parents=(parent_id,),
                )
                #
                _register_birth(
                    bubble_lifetime,
                    child_id_b,
                    child_generation,
                    time_p1,
                    istep_p1,
                    "breakup",
                    parents=(parent_id,),
                )
                #
                D_parent = float(
                    cmpt_d(
                        voll0[i0],
                        is_2d,
                    )
                )
                #
                D_child_a = float(
                    cmpt_d(
                        voll1[i1a],
                        is_2d,
                    )
                )
                #
                D_child_b = float(
                    cmpt_d(
                        voll1[i1b],
                        is_2d,
                    )
                )
                #
                diameter_tree.append({
                    "type": "breakup",
                    "time_p0": time_p0,
                    "time_p1": time_p1,
                    "istep_p0": istep_p0,
                    "istep_p1": istep_p1,
                    "parent": parent_id,
                    "parent_generation": parent_generation,
                    "children": (
                        child_id_a,
                        child_id_b,
                    ),
                    "children_generation": (
                        child_generation,
                        child_generation,
                    ),
                    "D_parent": D_parent,
                    "D_children": (
                        D_child_a,
                        D_child_b,
                    ),
                })
                #
                if V1 <= V2:
                    #
                    child_id_min = child_id_a
                    V_child_min = V1
                #
                else:
                    #
                    child_id_min = child_id_b
                    V_child_min = V2
                #
                D_child_min = cmpt_d(
                    V_child_min,
                    is_2d,
                )
                #
                breakup_event.append([
                    istep_p1,
                    time_p1,
                    parent_id,
                    t_birth,
                    t_death,
                    t_life,
                    child_id_min,
                    D_child_min,
                ])
                #
                total_breakups += 1
                #
                physical_events_step.append({
                    "type": "breakup",
                    "istep": istep_p1,
                    "time": time_p1,
                    "raw_p0": (i0,),
                    "raw_p1": (i1a,i1b),
                    "parent": parent_id,
                    "parent_generation": parent_generation,
                    "children": (
                        child_id_a,
                        child_id_b,
                    ),
                    "children_generation": (
                        child_generation,
                        child_generation,
                    ),
                    "resolution_time": resolution_time,
                    "raw_cost": float(ev["cost"]),
                })
                #
                if debug:
                    #
                    print(
                        "TRUE BREAKUP:",
                        parent_id,
                        "->",
                        child_id_a,
                        "+",
                        child_id_b,
                        "generation =",
                        child_generation,
                    )
            #
            else:
                #
                raise ValueError(
                    f"Unknown raw event type: {ev_type}"
                )
        #
        # ====================================================
        # Strict raw-index coverage
        # ====================================================
        #
        n1 = len(
            voll1
        )
        #
        assigned = np.asarray(
            sorted(
                idx_to_id_next.keys()
            ),
            dtype=int,
        )
        #
        expected = np.arange(
            n1,
            dtype=int,
        )
        #
        if not np.array_equal(
            assigned,
            expected,
        ):
            #
            missing = np.setdiff1d(
                expected,
                assigned,
            )
            #
            raise RuntimeError(
                "Clean genealogy did not assign every raw p1 component at "
                f"istep {istep_p1}.\n"
                f"missing = {missing.tolist()}"
            )
        #
        # ====================================================
        # Diameter
        # ====================================================
        #
        diameter_map = {
            i:bid
            for i,bid in idx_to_id_next.items()
            if not _is_transient_pair(bid)
        }
        #
        if len(diameter_map) > 0:
            #
            store_physical_diameter_snapshot(
                diameter_history,
                diameter_map,
                voll1,
                istep_p1,
                time_p1,
                is_2d,
            )
        #
        # ====================================================
        # History
        # ====================================================
        #
        physical_event_history.append({
            "history_index": ih,
            "istep_p0": istep_p0,
            "istep_p1": istep_p1,
            "time_p0": time_p0,
            "time_p1": time_p1,
            "events": physical_events_step,
        })
        #
        idx_to_id_curr = idx_to_id_next
        #
        idx_to_id_history.append(
            dict(idx_to_id_curr)
        )
    #
    # ========================================================
    # Final check
    # ========================================================
    #
    unresolved = {
        i:bid
        for i,bid in idx_to_id_curr.items()
        if _is_transient_pair(bid)
    }
    #
    if unresolved:
        #
        raise RuntimeError(
            "Unresolved transient coalescence at end of history: "
            f"{unresolved}"
        )
    #
    # ========================================================
    # Genealogy
    #
    # This is the authoritative genealogy information.
    #
    # Plotting / movies / analysis should read this structure
    # directly and should NEVER reconstruct generations.
    # ========================================================
    #
    genealogy = build_genealogy(
        bubble_lifetime
    )
    #
    # ========================================================
    # Authoritative topology counts
    #
    # Never infer the number of coalescences from bubble_lifetime:
    # one A+B->C event correctly creates TWO parent deaths.
    #
    # Count only explicit physical topology events.
    # ========================================================
    #
    (
        total_breakups,
        total_coalescences,
    ) = count_clean_topological_events(
        physical_event_history
    )
    #
    # ========================================================
    # Independent consistency check against diameter_tree
    # ========================================================
    #
    n_breakups_tree = sum(
        event["type"] == "breakup"
        for event in diameter_tree
    )
    #
    n_coalescences_tree = sum(
        event["type"] == "coalescence"
        for event in diameter_tree
    )
    #
    if total_breakups != n_breakups_tree:
        #
        raise RuntimeError(
            "Inconsistent breakup bookkeeping:\n"
            f"physical_event_history = {total_breakups}\n"
            f"diameter_tree = {n_breakups_tree}"
        )
    #
    if total_coalescences != n_coalescences_tree:
        #
        raise RuntimeError(
            "Inconsistent coalescence bookkeeping:\n"
            f"physical_event_history = {total_coalescences}\n"
            f"diameter_tree = {n_coalescences_tree}"
        )
    #
    # ========================================================
    # Return
    # ========================================================
    #
    return {
        "physical_event_history": physical_event_history,
        "idx_to_id_history": idx_to_id_history,
        "idx_to_id_final": idx_to_id_curr,
        "id_to_birth_time": id_to_birth_time,
        "bubble_lifetime": bubble_lifetime,
        "genealogy": genealogy,
        "total_breakups": total_breakups,
        "total_coalescences": total_coalescences,
        "breakup_event": breakup_event,
        "diameter_tree": diameter_tree,
        "diameter_history": diameter_history,
    }
#
def plot_diameter_tree_v1(
    diameter_history,
    diameter_tree,
    genealogy=None,
    ax=None,
    show_ids=False,
    tmax=None,
    ylim=None,
    quantity="diameter",
):
    #
    import numpy as np
    import matplotlib.pyplot as plt
    #
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
    #
    if genealogy is None:
        #
        raise ValueError(
            "genealogy must be provided from "
            "clean_genealogy_from_raw_history()['genealogy']"
        )
    #
    if ax is None:
        #
        fig,ax = plt.subplots(
            figsize=(10,6)
        )
    #
    # ========================================================
    # 1. Quantity
    # ========================================================
    #
    if quantity == "diameter":
        #
        quantity_key = "diameter"
        ylabel = "Equivalent diameter"
    #
    elif quantity == "volume":
        #
        quantity_key = "volume"
        ylabel = "Volume"
    #
    else:
        #
        raise ValueError(
            "quantity must be 'diameter' or 'volume'"
        )
    #
    # ========================================================
    # 2. Representative value for every physical branch
    # ========================================================
    #
    branch_value = {}
    #
    for physical_id,data in diameter_history.items():
        #
        if quantity_key not in data:
            #
            raise KeyError(
                f"'{quantity_key}' not available for "
                f"physical bubble {physical_id}"
            )
        #
        values = np.asarray(
            data[quantity_key],
            dtype=float,
        )
        #
        if len(values) == 0:
            #
            continue
        #
        branch_value[physical_id] = float(
            np.median(values)
        )
    #
    # ========================================================
    # 3. Physical IDs
    # ========================================================
    #
    physical_ids = [
        physical_id
        for physical_id in diameter_history
        if physical_id in branch_value
    ]
    #
    # ========================================================
    # 4. Validate genealogy
    #
    # Genealogy is NOT reconstructed here.
    # ========================================================
    #
    for physical_id in physical_ids:
        #
        if physical_id not in genealogy:
            #
            raise KeyError(
                f"Missing genealogy information for "
                f"physical bubble {physical_id}"
            )
        #
        if "generation" not in genealogy[physical_id]:
            #
            raise KeyError(
                f"Missing genealogy generation for "
                f"physical bubble {physical_id}"
            )
        #
        if "label" not in genealogy[physical_id]:
            #
            raise KeyError(
                f"Missing genealogy label for "
                f"physical bubble {physical_id}"
            )
    #
    # ========================================================
    # 5. Unique colors
    #
    # THIS FUNCTION IS THE ONLY OWNER OF THE COLORS.
    # Black and grey tones are intentionally excluded.
    # ========================================================
    #
    color_pool = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#17becf",
        "#8c564b",
        "#e377c2",
        "#bcbd22",
        "#393b79",
        "#e6550d",
        "#31a354",
        "#a50f15",
        "#756bb1",
        "#3182bd",
        "#843c39",
        "#ce6dbd",
        "#637939",
        "#7b4173",
        "#6baed6",
        "#fd8d3c",
        "#74c476",
        "#fb6a4a",
        "#9e9ac8",
        "#41b6c4",
        "#ad494a",
        "#de9ed6",
        "#8ca252",
        "#08519c",
        "#a63603",
        "#006d2c",
        "#67000d",
        "#54278f",
        "#016c59",
        "#7f2704",
        "#980043",
        "#3f007d",
        "#5254a3",
        "#00a6d6",
        "#f28e2b",
        "#59a14f",
        "#e15759",
        "#b07aa1",
        "#edc948",
        "#76b7b2",
        "#ff9da7",
        "#af7aa1",
        "#4e79a7",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#4daf4a",
        "#377eb8",
        "#e41a1c",
        "#ff7f00",
        "#ffff33",
        "#a6cee3",
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#8dd3c7",
        "#ffffb3",
        "#bebada",
        "#fb8072",
        "#80b1d3",
        "#fdb462",
        "#b3de69",
        "#fccde5",
        "#ccebc5",
        "#ffed6f",
        "#00bfc4",
        "#f8766d",
        "#7cae00",
        "#c77cff",
        "#cd9600",
        "#00be67",
        "#00a9ff",
        "#ff61cc",
    ]
    #
    if len(physical_ids) > len(color_pool):
        #
        raise RuntimeError(
            "\nNot enough unique colors.\n"
            f"Physical bubbles = {len(physical_ids)}\n"
            f"Available colors = {len(color_pool)}\n"
        )
    #
    branch_color = {
        physical_id:color_pool[i]
        for i,physical_id in enumerate(physical_ids)
    }
    #
    # ========================================================
    # 6. Horizontal physical branches
    # ========================================================
    #
    for physical_id,data in diameter_history.items():
        #
        if physical_id not in branch_value:
            #
            continue
        #
        time = np.asarray(
            data["time"],
            dtype=float,
        )
        #
        if len(time) == 0:
            #
            continue
        #
        t_start = float(
            time[0]
        )
        #
        t_end = float(
            time[-1]
        )
        #
        if tmax is not None:
            #
            if t_start > tmax:
                #
                continue
            #
            t_end = min(
                t_end,
                tmax,
            )
        #
        ax.plot(
            [t_start,t_end],
            [
                branch_value[physical_id],
                branch_value[physical_id],
            ],
            "-",
            color=branch_color[physical_id],
            linewidth=1.5,
            zorder=4,
        )
    #
    # ========================================================
    # 7. Global scales
    # ========================================================
    #
    all_times = []
    all_values = []
    #
    for physical_id,data in diameter_history.items():
        #
        time = np.asarray(
            data["time"],
            dtype=float,
        )
        #
        if len(time) > 0:
            #
            all_times.extend(
                time.tolist()
            )
        #
        if physical_id in branch_value:
            #
            all_values.append(
                branch_value[physical_id]
            )
    #
    if len(all_times) > 1:
        #
        global_time_range = (
            max(all_times)
            -min(all_times)
        )
    #
    else:
        #
        global_time_range = 1.
    #
    if global_time_range <= 0.:
        #
        global_time_range = 1.
    #
    if len(all_values) > 1:
        #
        global_value_range = (
            max(all_values)
            -min(all_values)
        )
    #
    else:
        #
        global_value_range = 1.
    #
    if global_value_range <= 0.:
        #
        global_value_range = 1.
    #
    # ========================================================
    # 8. Event-tip geometry
    # ========================================================
    #
    event_tip_size = 9.
    event_tip_dx = 0.010*global_time_range
    event_tip_dy = 0.020*global_value_range
    #
    # ========================================================
    # 9. Split horizontal arrow helper
    # ========================================================
    #
    def draw_split_horizontal_arrow(
        x0,
        x1,
        y,
        color_a,
        color_b,
        shaft_height,
        head_height,
        head_length,
        zorder=8,
    ):
        #
        top = [
            (x1,y),
            (x1-head_length,y+head_height/2.),
            (x1-head_length,y+shaft_height/2.),
            (x0,y+shaft_height/2.),
            (x0,y),
        ]
        #
        bottom = [
            (x1,y),
            (x1-head_length,y-head_height/2.),
            (x1-head_length,y-shaft_height/2.),
            (x0,y-shaft_height/2.),
            (x0,y),
        ]
        #
        ax.add_patch(
            Polygon(
                top,
                closed=True,
                facecolor=color_a,
                edgecolor="none",
                zorder=zorder,
            )
        )
        #
        ax.add_patch(
            Polygon(
                bottom,
                closed=True,
                facecolor=color_b,
                edgecolor="none",
                zorder=zorder,
            )
        )
    #
    # ========================================================
    # 10. Topological events
    # ========================================================
    #
    for event in diameter_tree:
        #
        event_type = event["type"]
        #
        # ====================================================
        # BREAKUP
        # ====================================================
        #
        if event_type == "breakup":
            #
            t_event = float(
                event["time_p1"]
            )
            #
            if tmax is not None and t_event > tmax:
                #
                continue
            #
            parent_id = event["parent"]
            child_a_id = event["children"][0]
            child_b_id = event["children"][1]
            #
            value_parent = branch_value[parent_id]
            value_child_a = branch_value[child_a_id]
            value_child_b = branch_value[child_b_id]
            #
            parent_color = branch_color[parent_id]
            child_a_color = branch_color[child_a_id]
            child_b_color = branch_color[child_b_id]
            #
            value_min = min(
                value_child_a,
                value_child_b,
            )
            #
            # ------------------------------------------------
            # Vertical connector
            # ------------------------------------------------
            #
            ax.plot(
                [t_event,t_event],
                [value_min,value_parent],
                "-.",
                color="black",
                linewidth=2.,
                zorder=6,
            )
            #
            # ------------------------------------------------
            # Parent vertical tip
            # ------------------------------------------------
            #
            if value_min < value_parent:
                #
                marker = "v"
                marker_y = value_parent+event_tip_dy
            #
            else:
                #
                marker = "^"
                marker_y = value_parent-event_tip_dy
            #
            ax.plot(
                t_event,
                marker_y,
                marker=marker,
                markersize=event_tip_size,
                markerfacecolor=parent_color,
                markeredgecolor=parent_color,
                markeredgewidth=0.,
                linestyle="None",
                zorder=8,
            )
            #
            # ------------------------------------------------
            # Child A tip
            # ------------------------------------------------
            #
            ax.plot(
                t_event-0.5*event_tip_dx,
                value_child_a,
                marker=">",
                markersize=event_tip_size,
                markerfacecolor=child_a_color,
                markeredgecolor=child_a_color,
                markeredgewidth=0.,
                linestyle="None",
                zorder=8,
            )
            #
            # ------------------------------------------------
            # Child B tip
            # ------------------------------------------------
            #
            ax.plot(
                t_event-0.5*event_tip_dx,
                value_child_b,
                marker=">",
                markersize=event_tip_size,
                markerfacecolor=child_b_color,
                markeredgecolor=child_b_color,
                markeredgewidth=0.,
                linestyle="None",
                zorder=8,
            )
        #
        # ====================================================
        # COALESCENCE
        # ====================================================
        #
        elif event_type == "coalescence":
            #
            t_event = float(
                event["time_p1"]
            )
            #
            if tmax is not None and t_event > tmax:
                #
                continue
            #
            parent_a_id = event["parents"][0]
            parent_b_id = event["parents"][1]
            child_id = event["child"]
            #
            value_parent_a = branch_value[parent_a_id]
            value_parent_b = branch_value[parent_b_id]
            value_child = branch_value[child_id]
            #
            color_a = branch_color[parent_a_id]
            color_b = branch_color[parent_b_id]
            #
            value_min = min(
                value_parent_a,
                value_parent_b,
            )
            #
            # ------------------------------------------------
            # Vertical connector
            # ------------------------------------------------
            #
            ax.plot(
                [t_event,t_event],
                [value_min,value_child],
                "-.",
                color="grey",
                linewidth=2.,
                zorder=6,
            )
            #
            # ------------------------------------------------
            # Parent A tip
            # ------------------------------------------------
            #
            ax.plot(
                t_event-0.5*event_tip_dx,
                value_parent_a,
                marker=">",
                markersize=event_tip_size,
                markerfacecolor=color_a,
                markeredgecolor=color_a,
                markeredgewidth=0.,
                linestyle="None",
                zorder=8,
            )
            #
            # ------------------------------------------------
            # Parent B tip
            # ------------------------------------------------
            #
            ax.plot(
                t_event-0.5*event_tip_dx,
                value_parent_b,
                marker=">",
                markersize=event_tip_size,
                markerfacecolor=color_b,
                markeredgecolor=color_b,
                markeredgewidth=0.,
                linestyle="None",
                zorder=8,
            )
            #
            # ------------------------------------------------
            # Child split-color tip
            # ------------------------------------------------
            #
            draw_split_horizontal_arrow(
                t_event-event_tip_dx,
                t_event,
                value_child,
                color_a,
                color_b,
                0.020*global_value_range,
                0.050*global_value_range,
                event_tip_dx,
                zorder=8,
            )
        #
        else:
            #
            raise ValueError(
                f"Unknown event type: {event_type}"
            )
    #
    # ========================================================
    # 11. Genealogy information
    #
    # IMPORTANT:
    #
    # There is deliberately NO reconstruction here.
    #
    # generation
    # parents
    # children
    # genealogy label
    #
    # all come directly from clean_genealogy_from_raw_history.
    # ========================================================
    #
    branch_generation = {
        physical_id:int(
            genealogy[physical_id]["generation"]
        )
        for physical_id in physical_ids
    }
    #
    branch_label = {
        physical_id:str(
            genealogy[physical_id]["label"]
        )
        for physical_id in physical_ids
    }
    #
    # ========================================================
    # 12. Physical-ID / genealogy legend
    # ========================================================
    #
    if show_ids:
        #
        handles_by_generation = {}
        #
        for physical_id in physical_ids:
            #
            generation = branch_generation[physical_id]
            #
            if generation not in handles_by_generation:
                #
                handles_by_generation[generation] = []
            #
            handles_by_generation[generation].append(
                Line2D(
                    [0],[0],
                    color=branch_color[physical_id],
                    linewidth=3.,
                    label=branch_label[physical_id],
                )
            )
        #
        generations = sorted(
            handles_by_generation
        )
        #
        max_gen_per_row = 4
        #
        generation_groups = [
            generations[i:i+max_gen_per_row]
            for i in range(
                0,
                len(generations),
                max_gen_per_row,
            )
        ]
        #
        for igroup,generation_group in enumerate(
            generation_groups
        ):
            #
            handles_group = []
            #
            max_bubbles = max(
                len(
                    handles_by_generation[generation]
                )
                for generation in generation_group
            )
            #
            for generation in generation_group:
                #
                generation_handles = [
                    Line2D(
                        [0],[0],
                        color="none",
                        linewidth=0.,
                        label=f"Gen {generation}",
                    )
                ]
                #
                generation_handles.extend(
                    handles_by_generation[generation]
                )
                #
                while len(generation_handles) < max_bubbles+1:
                    #
                    generation_handles.append(
                        Line2D(
                            [0],[0],
                            color="none",
                            linewidth=0.,
                            label="",
                        )
                    )
                #
                handles_group.extend(
                    generation_handles
                )
            #
            legend_ids = ax.legend(
                handles=handles_group,
                loc="upper left",
                bbox_to_anchor=(
                    0.01,
                    0.98-0.22*igroup,
                ),
                fontsize=8,
                frameon=True,
                ncol=len(generation_group),
                columnspacing=1.5,
                handlelength=2.0,
            )
            #
            ax.add_artist(
                legend_ids
            )
    #
    # ========================================================
    # 13. Event legend
    # ========================================================
    #
    event_types_present = {
        event["type"]
        for event in diameter_tree
    }
    #
    topology_handles = []
    #
    if "breakup" in event_types_present:
        #
        topology_handles.append(
            Line2D(
                [0],[0],
                color="black",
                linestyle="-.",
                linewidth=2.,
                label="Breakup event",
            )
        )
    #
    if "coalescence" in event_types_present:
        #
        topology_handles.append(
            Line2D(
                [0],[0],
                color="grey",
                linestyle="-.",
                linewidth=2.,
                label="Coalescence event",
            )
        )
    #
    if len(topology_handles) > 0:
        #
        legend_topology = ax.legend(
            handles=topology_handles,
            loc="upper right",
            fontsize=12,
            frameon=True,
        )
        #
        ax.add_artist(
            legend_topology
        )
    #
    # ========================================================
    # 14. Formatting
    # ========================================================
    #
    if tmax is not None:
        #
        ax.set_xlim(
            [0,tmax]
        )
    #
    if ylim is not None:
        #
        ax.set_ylim(
            [0,ylim]
        )
    #
    ax.set_xlabel(
        "Time"
    )
    #
    ax.set_ylabel(
        ylabel
    )
    #
    ax.grid(
        True,
        alpha=0.25,
    )
    #
    # ========================================================
    # 15. Return tree colors to movie
    # ========================================================
    #
    return ax,branch_color,branch_value
#
def periodic_weighted_mean(values,weights,vmin,length):
    #
    values = np.asarray(values,dtype=float)
    weights = np.asarray(weights,dtype=float)
    #
    if len(values) == 0:
        return 0.
    #
    if len(values) == 1:
        return float(values[0])
    #
    if length is None or length <= 0.:
        return float(np.average(values,weights=weights))
    #
    theta = 2.*np.pi*(values-vmin)/length
    #
    c = float(np.sum(weights*np.cos(theta)))
    s = float(np.sum(weights*np.sin(theta)))
    #
    theta_mean = np.arctan2(s,c)
    #
    if theta_mean < 0.:
        theta_mean += 2.*np.pi
    #
    return float(vmin+length*theta_mean/(2.*np.pi))
#
def periodic_distance_matrix(points_a,points_b,lbox_v,lmin_v,periodic_bc):
    """
    Minimum-image 2D distance matrix between two point sets.

    lbox_v      : [Lx, Ly]  box size (need not be cubic: Lx != Ly is fine)
    lmin_v      : [xmin, ymin]  box origin
    periodic_bc : [px, py]  1/True = periodic on that axis, 0/False = not

    Periodicity is taken strictly from periodic_bc, not inferred from
    the sign or value of lbox_v: a finite, non-periodic domain still
    has Lx,Ly > 0, so that would silently wrap a wall-bounded axis.
    """
    #
    points_a = np.asarray(points_a,dtype=float)
    points_b = np.asarray(points_b,dtype=float)
    #
    if len(points_a) == 0 or len(points_b) == 0:
        return np.empty((len(points_a),len(points_b)),dtype=float)
    #
    lbox_xy = np.array([lbox_v[0],lbox_v[1]],dtype=float)
    lmin_xy = np.array([lmin_v[0],lmin_v[1]],dtype=float)
    periodic_mask = np.array([periodic_bc[0],periodic_bc[1]],dtype=bool)
    #
    pa = np.array(points_a[:,:2],dtype=float,copy=True)
    pb = np.array(points_b[:,:2],dtype=float,copy=True)
    #
    # Defensive: fold both point sets back into the canonical box on
    # the periodic axes first, so any upstream drift (e.g. from a
    # periodic weighted mean) can never leave a point just outside
    # [lmin_v, lmin_v + lbox_v) and throw off the minimum image below.
    #
    if np.any(periodic_mask):
        pa[:,periodic_mask] = (
            (pa[:,periodic_mask]-lmin_xy[periodic_mask])
            % lbox_xy[periodic_mask]
            + lmin_xy[periodic_mask]
        )
        pb[:,periodic_mask] = (
            (pb[:,periodic_mask]-lmin_xy[periodic_mask])
            % lbox_xy[periodic_mask]
            + lmin_xy[periodic_mask]
        )
    #
    dx = pa[:,None,0]-pb[None,:,0]
    dy = pa[:,None,1]-pb[None,:,1]
    #
    if periodic_mask[0]:
        dx -= lbox_xy[0]*np.round(dx/lbox_xy[0])
    #
    if periodic_mask[1]:
        dy -= lbox_xy[1]*np.round(dy/lbox_xy[1])
    #
    return np.sqrt(dx*dx+dy*dy)
#
def build_step_positions(event_history_raw):
    #
    step_positions = {}
    #
    first = event_history_raw[0]
    #
    step_positions[int(first["istep_p0"])] = {
        "x":np.asarray(first["xpos0"],dtype=float),
        "y":np.asarray(first["ypos0"],dtype=float),
        "v":np.asarray(first["voll0"],dtype=float),
    }
    #
    for record in event_history_raw:
        #
        step_positions[int(record["istep_p1"])] = {
            "x":np.asarray(record["xpos1"],dtype=float),
            "y":np.asarray(record["ypos1"],dtype=float),
            "v":np.asarray(record["voll1"],dtype=float),
        }
    #
    return step_positions
#
def build_tracked_state(
    event_history_raw,
    diameter_history,
    lbox_v,
    lmin_v,
    periodic_bc,
    is_2d,
):
    """
    lbox_v, lmin_v, periodic_bc : [x, y] box size, box origin, and
    periodic flags. lbox_v need not be cubic (Lx != Ly), and each
    axis is independently periodic or not, per periodic_bc.
    """
    #
    if int(is_2d) != 1:
        raise ValueError(
            "build_tracked_state only supports 2D tracking "
            "(is_2d must be 1)."
        )
    #
    step_positions = build_step_positions(event_history_raw)
    #
    xmin,ymin = float(lmin_v[0]),float(lmin_v[1])
    Lx,Ly = float(lbox_v[0]),float(lbox_v[1])
    #
    periodic_x = bool(periodic_bc[0])
    periodic_y = bool(periodic_bc[1])
    #
    tracked_state = {}
    n_missing = 0
    #
    for physical_id,data in diameter_history.items():
        #
        isteps = data["istep"]
        raw_indices = data["raw_indices"]
        volumes = data["volume"]
        diameters = data["diameter"]
        #
        for k in range(len(isteps)):
            #
            istep = int(isteps[k])
            #
            if istep not in step_positions:
                n_missing += 1
                continue
            #
            pos = step_positions[istep]
            #
            indices = [
                int(i)
                for i in raw_indices[k]
                if 0 <= int(i) < len(pos["x"])
            ]
            #
            if len(indices) == 0:
                n_missing += 1
                continue
            #
            weights = np.maximum(pos["v"][indices],1.e-30)
            #
            xc = periodic_weighted_mean(
                pos["x"][indices],weights,xmin,
                Lx if periodic_x else 0.,
            )
            #
            yc = periodic_weighted_mean(
                pos["y"][indices],weights,ymin,
                Ly if periodic_y else 0.,
            )
            #
            tracked_state.setdefault(istep,{})[physical_id] = {
                "x":xc,
                "y":yc,
                "volume":float(volumes[k]),
                "diameter":float(diameters[k]),
                "nraw":len(indices),
            }
    #
    if n_missing > 0:
        print(
            "build_tracked_state: skipped",n_missing,
            "snapshots without usable raw positions",
        )
    #
    return tracked_state
#
def make_step_lookup(tracked_state,max_step_gap=0):
    #
    steps = np.asarray(sorted(tracked_state.keys()),dtype=int)
    #
    def lookup(istep):
        #
        istep = int(istep)
        #
        if istep in tracked_state:
            return istep,tracked_state[istep]
        #
        if len(steps) == 0 or max_step_gap <= 0:
            return None,{}
        #
        j = int(np.argmin(np.abs(steps-istep)))
        #
        if abs(int(steps[j])-istep) > max_step_gap:
            return None,{}
        #
        return int(steps[j]),tracked_state[int(steps[j])]
    #
    return lookup
#
def region_perimeter(region):
    #
    segments = np.asarray(region["segments"],dtype=float)
    #
    if len(segments) == 0:
        return 0.
    #
    dxy = segments[:,1,:]-segments[:,0,:]
    #
    return float(np.sum(np.sqrt(np.sum(dxy*dxy,axis=1))))
#
def region_length_centroid(region):
    #
    # Arc-length weighted centroid: robust against uneven
    # facet density, unlike a plain mean of the endpoints.
    #
    segments = np.asarray(region["segments"],dtype=float)
    #
    if len(segments) == 0:
        return np.zeros(2,dtype=float)
    #
    mid = 0.5*(segments[:,0,:]+segments[:,1,:])
    #
    dxy = segments[:,1,:]-segments[:,0,:]
    w = np.sqrt(np.sum(dxy*dxy,axis=1))
    #
    wsum = float(np.sum(w))
    #
    if wsum <= 0.:
        return np.mean(mid,axis=0)
    #
    return np.asarray(
        [
            float(np.sum(w*mid[:,0])/wsum),
            float(np.sum(w*mid[:,1])/wsum),
        ],
        dtype=float,
    )
#
def region_geometry(regions):
    #
    centroids = np.empty((len(regions),2),dtype=float)
    perimeters = np.empty(len(regions),dtype=float)
    #
    for iregion,region in enumerate(regions):
        #
        centroids[iregion] = region_length_centroid(region)
        perimeters[iregion] = region_perimeter(region)
    #
    return centroids,perimeters
#
def assign_regions_from_tracker(
    regions,
    centroids,
    perimeters,
    tracked_now,
    lbox_v,
    lmin_v,
    periodic_bc,
    is_2d,
    size_weight=0.25,
    fragment_ambiguity=0.70,
):
    #
    if int(is_2d) != 1:
        raise ValueError(
            "assign_regions_from_tracker only supports 2D tracking "
            "(is_2d must be 1)."
        )
    #
    assignment = {}
    #
    if len(regions) == 0 or len(tracked_now) == 0:
        return assignment
    #
    ids = list(tracked_now.keys())
    #
    tracked_xy = np.asarray(
        [[tracked_now[pid]["x"],tracked_now[pid]["y"]] for pid in ids],
        dtype=float,
    )
    #
    tracked_d = np.asarray(
        [tracked_now[pid]["diameter"] for pid in ids],
        dtype=float,
    )
    #
    d_scale = float(np.mean(tracked_d))
    #
    if not np.isfinite(d_scale) or d_scale <= 0.:
        d_scale = 1.
    #
    # --------------------------------------------------------
    # Equivalent diameter of each facet region
    #
    # 2D closed interface: perimeter = pi*D
    # --------------------------------------------------------
    #
    region_d = perimeters/np.pi
    #
    # --------------------------------------------------------
    # Anchors = the n largest regions, n = number of bubbles
    #
    # Small fragments are never allowed to define identity.
    # --------------------------------------------------------
    #
    nanchor = min(len(ids),len(regions))
    #
    order = np.argsort(-np.asarray(perimeters, dtype=float), kind="stable")
    anchors = [int(i) for i in order[:nanchor]]
    #
    D = periodic_distance_matrix(
        centroids[anchors],tracked_xy,lbox_v,lmin_v,periodic_bc,
    )
    #
    dD = region_d[anchors][:,None]-tracked_d[None,:]
    #
    cost = (D/d_scale)**2+size_weight*(dD/d_scale)**2
    #
    ianchor_match,iid_match = linear_sum_assignment(cost)
    #
    for ia,ii in zip(ianchor_match,iid_match):
        assignment[int(anchors[int(ia)])] = ids[int(ii)]
    #
    # --------------------------------------------------------
    # Remaining fragments
    #
    # A fragment inherits an ID only when it is clearly closer
    # to one tracked bubble than to any other. Otherwise it
    # stays grey.
    # --------------------------------------------------------
    #
    fragments = [
        iregion
        for iregion in range(len(regions))
        if iregion not in assignment
    ]
    #
    if len(fragments) == 0:
        return assignment
    #
    Df = periodic_distance_matrix(
        centroids[fragments],tracked_xy,lbox_v,lmin_v,periodic_bc,
    )
    #
    for ilocal,iregion in enumerate(fragments):
        #
        distances = Df[ilocal]
        #
        iorder = np.argsort(distances, kind="stable")
        #
        if len(iorder) == 1:
            assignment[int(iregion)] = ids[int(iorder[0])]
            continue
        #
        d1 = float(distances[iorder[0]])
        d2 = float(distances[iorder[1]])
        #
        if d2 <= 1.e-14:
            continue
        #
        if d1 >= fragment_ambiguity*d2:
            continue
        #
        assignment[int(iregion)] = ids[int(iorder[0])]
    #
    return assignment
#
def check_tracker_alignment(
    regions,
    tracked_now,
    istep,
    lbox_v,
    lmin_v,
    periodic_bc,
    is_2d,
):
    #
    if int(is_2d) != 1:
        raise ValueError(
            "check_tracker_alignment only supports 2D tracking "
            "(is_2d must be 1)."
        )
    #
    print("")
    print("==============================================")
    print("TRACKER / FACET ALIGNMENT CHECK")
    print("==============================================")
    print("istep            =",istep)
    print("facet regions    =",len(regions))
    print("tracked bubbles  =",len(tracked_now))
    #
    if len(regions) == 0 or len(tracked_now) == 0:
        print("==============================================")
        return
    #
    centroids,perimeters = region_geometry(regions)
    #
    ids = list(tracked_now.keys())
    #
    tracked_xy = np.asarray(
        [[tracked_now[pid]["x"],tracked_now[pid]["y"]] for pid in ids],
        dtype=float,
    )
    #
    D = periodic_distance_matrix(centroids,tracked_xy,lbox_v,lmin_v,periodic_bc)
    #
    print("")
    print(
        f"{'region':>7}{'xc':>12}{'yc':>12}"
        f"{'D_facet':>12}{'nearest id':>16}"
        f"{'dist':>12}{'D_track':>12}"
    )
    #
    for iregion in range(len(regions)):
        #
        j = int(np.argmin(D[iregion]))
        #
        print(
            f"{iregion:>7}"
            f"{centroids[iregion,0]:>12.4f}"
            f"{centroids[iregion,1]:>12.4f}"
            f"{perimeters[iregion]/np.pi:>12.4f}"
            f"{str(ids[j]):>16}"
            f"{D[iregion,j]:>12.4f}"
            f"{tracked_now[ids[j]]['diameter']:>12.4f}"
        )
    #
    print("")
    print("Distances should be a small fraction of the bubble")
    print("diameter. Large values mean the facet frame and the")
    print("tracker frame do not coincide.")
    print("==============================================")
#
def _is_debris(label):
    return isinstance(label,tuple) and len(label) == 1 and label[0] == DEBRIS[0]
#
def _is_pair(label):
    return isinstance(label,tuple) and len(label) >= 3 and label[0] == _TP
#
def _is_physical(label):
    return isinstance(label,str)
#
class _Frames:
    #
    def __init__(self,H):
        #
        self.nrec = len(H)
        self.V,self.A,self.E,self.X,self.T,self.S = [],[],[],[],[],[]
        #
        for j in range(self.nrec+1):
            #
            rec,sfx = (H[j],"0") if j < self.nrec else (H[-1],"1")
            V = np.asarray(rec["voll"+sfx],dtype=float)
            #
            self.V.append(V)
            self.A.append(np.asarray(rec["area"+sfx],dtype=float))
            n = V.size
            self.X.append(np.column_stack([
                np.zeros(n) if rec.get(ax+"pos"+sfx) is None
                else np.asarray(rec[ax+"pos"+sfx],dtype=float)
                for ax in ("x","y","z")
            ]))
            self.E.append(np.asarray(rec["area"+sfx+"_err"],dtype=float))
            self.T.append(float(rec["time_p"+sfx]))
            self.S.append(int(rec["istep_p"+sfx]))
        #
        for j in range(self.nrec-1):
            #
            if len(H[j]["voll1"]) != len(H[j+1]["voll0"]):
                raise RuntimeError(
                    f"Record {j} p1 and record {j+1} p0 have different "
                    "numbers of regions; the raw history is not contiguous."
                )
    #
    def ell(self,j,i):
        # Interface-weighted resolution length area_err/area (= dx on
        # a uniform grid, shape-agnostic, valid in 2D and 3D).
        A = self.A[j][i]
        return 0.0 if A <= eps_gen else float(self.E[j][i]/A)
    #
    def s(self,j,i):
        # Volume signal-to-uncertainty ratio of region i in frame j.
        E = self.E[j][i]
        return np.inf if E <= eps_gen else float(self.V[j][i]/E)
#
def _min_image_3d(d,lbox_v,periodic_bc):
    #
    # Minimum-image displacement on the periodic axes, same
    # convention as locate_tag_events_fast: lbox_v = [Lx,Ly,Lz],
    # periodic_bc = [px,py,pz]. Periodicity is taken strictly from
    # periodic_bc; a wall-bounded axis is never wrapped.
    #
    if lbox_v is None or periodic_bc is None:
        return d
    d = np.array(d,dtype=float,copy=True)
    for ax in range(min(len(lbox_v),len(periodic_bc),3)):
        if periodic_bc[ax] and float(lbox_v[ax]) > 0.0:
            L = float(lbox_v[ax])
            d[ax] -= L*np.round(d[ax]/L)
    return d
#
def analyze_raw_topology(
    event_history_raw,rho,mu,sigma,is_2d,
    lbox_v=None,periodic_bc=None,separation_veto=True,
):
    #
    H = event_history_raw
    F = _Frames(H)
    nrec = F.nrec
    events = [rec["events"] for rec in H]
    #
    # --------------------------------------------------------
    # 1. Continuation segments and their resolvability (test R)
    # --------------------------------------------------------
    #
    seg = [np.full(F.V[j].size,-1,dtype=int) for j in range(nrec+1)]
    seg_s = []
    seg_last = []
    #
    def _new_seg(j,i):
        seg[j][i] = len(seg_s)
        seg_s.append(F.s(j,i))
        seg_last.append((j,i))
    #
    for i in range(F.V[0].size):
        _new_seg(0,i)
    #
    inp = [dict() for _ in range(nrec)]
    #
    for j in range(nrec):
        #
        for e,ev in enumerate(events[j]):
            #
            for i0 in ev["p0"]:
                inp[j][int(i0)] = e
            #
            if ev["type"] == "continuation":
                #
                i0,i1 = int(ev["p0"][0]),int(ev["p1"][0])
                sid = seg[j][i0]
                seg[j+1][i1] = sid
                seg_s[sid] = max(seg_s[sid],F.s(j+1,i1))
                seg_last[sid] = (j+1,i1)
            else:
                #
                for i1 in ev["p1"]:
                    _new_seg(j+1,int(i1))
        #
        if np.any(seg[j+1] < 0):
            raise RuntimeError(
                f"Unassigned p1 regions in record {j}: "
                f"{np.where(seg[j+1] < 0)[0].tolist()}"
            )
        if len(inp[j]) != F.V[j].size:
            raise RuntimeError(f"Unassigned p0 regions in record {j}.")
    #
    def res(j,i):
        return seg_s[seg[j][int(i)]] > 1.0
    #
    # --------------------------------------------------------
    # 2. Static event classes
    # --------------------------------------------------------
    #
    cls = []
    #
    for j in range(nrec):
        #
        cj = []
        #
        for ev in events[j]:
            #
            t = ev["type"]
            #
            if t == "continuation":
                cj.append({"kind": "cont"})
            #
            elif t == "breakup":
                #
                i0 = int(ev["p0"][0])
                a,b = int(ev["p1"][0]),int(ev["p1"][1])
                #
                if res(j,i0) and res(j+1,a) and res(j+1,b):
                    cj.append({"kind": "topo"})
                else:
                    keep,drop = (a,b) if F.V[j+1][a] >= F.V[j+1][b] else (b,a)
                    cj.append({"kind": "shed","keep": keep,"drop": drop})
            #
            elif t == "coalescence":
                #
                a,b = int(ev["p0"][0]),int(ev["p0"][1])
                ra,rb = res(j,a),res(j,b)
                #
                if ra and rb:
                    cj.append({"kind": "topo"})
                else:
                    if ra != rb:
                        keep_in = a if ra else b
                    else:
                        keep_in = a if F.V[j][a] >= F.V[j][b] else b
                    cj.append({
                        "kind": "absorb",
                        "keep_in": keep_in,
                        "other_in": b if keep_in == a else a,
                    })
            else:
                raise ValueError(f"Unknown raw event type: {t}")
        #
        cls.append(cj)
    #
    # --------------------------------------------------------
    # 3. Branch following through non-topological and collapsed
    #    events
    # --------------------------------------------------------
    #
    open_info = {}      # (j,e) of the first event of a collapsed reversal
    close_to_open = {}  # (k,e2) of its closing event -> (j,e)
    #
    def _step(j,i):
        #
        if j >= nrec:
            return ("end",)
        e = inp[j][i]
        key = (j,e)
        if key in open_info:
            info = open_info[key]
            return ("next",info["k"]+1,info["out_map"][i])
        if key in close_to_open:
            return ("lost",)
        c = cls[j][e]
        ev = events[j][e]
        if c["kind"] == "cont":
            return ("next",j+1,int(ev["p1"][0]))
        if c["kind"] == "shed":
            return ("next",j+1,c["keep"])
        if c["kind"] == "absorb":
            if i != c["keep_in"]:
                return ("lost",)
            return ("next",j+1,int(ev["p1"][0]))
        return ("stop",j,e,i)
    #
    def _follow(j,i):
        # Next active topological event reached by node (j,i).
        while True:
            if j < nrec and cls[j][inp[j][i]]["kind"] == "cont" \
               and (j,inp[j][i]) not in open_info:
                j,i = seg_last[seg[j][i]]
                continue
            r = _step(j,i)
            if r[0] == "next":
                j,i = r[1],r[2]
                continue
            return r if r[0] == "stop" else None
    #
    def _walk(j,i,j_end):
        # Frame -> region index along a branch, up to frame j_end.
        path = {}
        while j <= j_end:
            path[j] = i
            if j == j_end:
                break
            r = _step(j,i)
            if r[0] != "next":
                break
            j,i = r[1],r[2]
        return path
    #
    def _separation_ratio(j,a,b,k):
        #
        # Range of the daughters' centroid distance over the window,
        # divided by their combined resolution length. <= 1 means the
        # two daughters never moved apart by more than the grid can
        # distinguish. Minimum image on periodic axes.
        #
        pa,pb = _walk(j+1,a,k),_walk(j+1,b,k)
        common = sorted(set(pa) & set(pb))
        if len(common) < 2:
            return 0.0
        d,l = [],[]
        for f in common:
            dx = _min_image_3d(F.X[f][pb[f]]-F.X[f][pa[f]],lbox_v,periodic_bc)
            d.append(float(np.linalg.norm(dx)))
            l.append(F.ell(f,pa[f])+F.ell(f,pb[f]))
        l_max = max(l)
        return float((max(d)-min(d))/l_max) if l_max > eps_gen else np.inf
    #
    # --------------------------------------------------------
    # 4. Tests for a candidate reversal
    # --------------------------------------------------------
    #
    def _window_test(V_event,j,k):
        #
        # Shortest duration of the intermediate topology compatible
        # with the sampling: the opening event happened no later than
        # time_p1 of record j, the closing one no earlier than time_p0
        # of record k. A reversal seen in a single frame therefore has
        # dt_min = 0 whatever the output interval is.
        #
        t_window = compute_topology_relaxation_time_phy(V_event,rho,mu,sigma,is_2d)
        dt_min = max(F.T[k]-F.T[j+1],0.0)
        if t_window > eps_gen:
            ratio = dt_min/t_window
        else:
            ratio = 0.0 if dt_min <= eps_gen else np.inf
        return float(t_window),float(dt_min),float(ratio)
    #
    def _test_breakup_reversal(j,e,k,e2):
        #
        ev,ev2 = events[j][e],events[k][e2]
        i0 = int(ev["p0"][0])
        c = int(ev2["p1"][0])
        #
        vol_ratio = _volume_uncertainty_ratio(
            F.V[j][i0],F.E[j][i0],F.V[k+1][c],F.E[k+1][c],
        )
        t_window,dt_min,time_ratio = _window_test(
            max(F.V[j][i0],F.V[k+1][c]),j,k,
        )
        #
        a,b = int(ev["p1"][0]),int(ev["p1"][1])
        sep_ratio = _separation_ratio(j,a,b,k) if separation_veto else 0.0
        #
        return {
            "kind": "breakup_reversal",
            "open": (j,e),"close": (k,e2),
            "time_open": F.T[j+1],"time_close": F.T[k+1],
            "dt_min": dt_min,"t_window": t_window,
            "volume_ratio": float(vol_ratio),
            "time_ratio": time_ratio,
            "separation_ratio": float(sep_ratio),
            "numerical": bool(
                vol_ratio <= 1.0 and time_ratio <= 1.0 and sep_ratio <= 1.0
            ),
            "out_map": {i0: c},
        }
    #
    def _test_coalescence_reversal(j,e,k,e2):
        #
        ev,ev2 = events[j][e],events[k][e2]
        a,b = int(ev["p0"][0]),int(ev["p0"][1])
        c = int(ev["p1"][0])
        p,q = int(ev2["p1"][0]),int(ev2["p1"][1])
        #
        ok_vol,vol_ratio,swap = _rebreakup_reconstructs_original_pair(
            F.V[j][a],F.E[j][a],F.V[j][b],F.E[j][b],
            F.V[k+1][p],F.E[k+1][p],F.V[k+1][q],F.E[k+1][q],
        )
        t_window,dt_min,time_ratio = _window_test(
            max(F.V[j][a]+F.V[j][b],F.V[j+1][c]),j,k,
        )
        out_a,out_b = (q,p) if swap else (p,q)
        #
        return {
            "kind": "coalescence_reversal",
            "open": (j,e),"close": (k,e2),
            "time_open": F.T[j+1],"time_close": F.T[k+1],
            "dt_min": dt_min,"t_window": t_window,
            "volume_ratio": float(vol_ratio),
            "time_ratio": time_ratio,
            "numerical": bool(ok_vol and time_ratio <= 1.0),
            "swap": bool(swap),
            "out_map": {a: out_a,b: out_b},
        }
    #
    # --------------------------------------------------------
    # 5. Innermost-first reduction
    # --------------------------------------------------------
    #
    diagnostics = {}
    changed = True
    n_pass = 0
    #
    while changed:
        #
        changed = False
        n_pass += 1
        # Diagnostics of rejected candidates are recomputed each pass,
        # because collapsing inner reversals changes what is adjacent.
        diagnostics = {k: v for k,v in diagnostics.items() if k in open_info}
        #
        for j in range(nrec):
            #
            for e,ev in enumerate(events[j]):
                #
                key = (j,e)
                if cls[j][e]["kind"] != "topo" or key in open_info or key in close_to_open:
                    continue
                #
                if ev["type"] == "breakup":
                    #
                    a,b = int(ev["p1"][0]),int(ev["p1"][1])
                    ra,rb = _follow(j+1,a),_follow(j+1,b)
                    if ra is None or rb is None or ra[1:3] != rb[1:3]:
                        continue
                    k,e2 = ra[1],ra[2]
                    ev2 = events[k][e2]
                    if ev2["type"] != "coalescence" or (k,e2) in close_to_open:
                        continue
                    if ra[3] == rb[3] or {ra[3],rb[3]} != {int(x) for x in ev2["p0"]}:
                        continue
                    diag = _test_breakup_reversal(j,e,k,e2)
                #
                elif ev["type"] == "coalescence":
                    #
                    r = _follow(j+1,int(ev["p1"][0]))
                    if r is None:
                        continue
                    k,e2 = r[1],r[2]
                    ev2 = events[k][e2]
                    if ev2["type"] != "breakup" or (k,e2) in close_to_open:
                        continue
                    diag = _test_coalescence_reversal(j,e,k,e2)
                else:
                    continue
                #
                diagnostics[key] = diag
                #
                if diag["numerical"]:
                    open_info[key] = {"k": k,"e2": e2,"out_map": diag["out_map"],"diag": diag}
                    close_to_open[(k,e2)] = key
                    changed = True
    #
    return {
        "frames": F,
        "classes": cls,
        "resolved": res,
        "segment_resolvability": seg_s,
        "open_info": open_info,
        "close_to_open": close_to_open,
        "diagnostics": list(diagnostics.values()),
        "diagnostics_by_open": diagnostics,
        "n_passes": n_pass,
    }
#
def clean_genealogy_from_raw_history(
    event_history_raw,idx_to_id_initial,id_to_birth_time_initial,
    rho,mu,sigma,is_2d,debug=False,
    lbox_v=None,periodic_bc=None,separation_veto=True,
):
    #
    H = event_history_raw
    #
    bubble_lifetime = {}
    for bid,t_birth in id_to_birth_time_initial.items():
        bubble_lifetime[bid] = {
            "generation": 0,"t_birth": float(t_birth),"istep_birth": None,
            "birth_event": "initial","t_death": None,"istep_death": None,
            "death_event": None,"t_life": None,"parents": (),"children": (),
        }
    #
    def build_genealogy():
        return {
            bid: {
                "label": str(bid),
                "generation": int(d["generation"]),
                "parents": tuple(d["parents"]),
                "children": tuple(d["children"]),
                "birth_event": d["birth_event"],
                "death_event": d["death_event"],
                "t_birth": d["t_birth"],
                "t_death": d["t_death"],
            }
            for bid,d in bubble_lifetime.items()
        }
    #
    empty_counts = {
        "total_sheddings": 0,"total_absorptions": 0,"total_emergences": 0,
        "total_transient_breakups": 0,"total_transient_coalescences": 0,
    }
    #
    if len(H) == 0:
        return {
            "physical_event_history": [],"idx_to_id_history": [],
            "idx_to_id_final": dict(idx_to_id_initial),
            "id_to_birth_time": dict(id_to_birth_time_initial),
            "bubble_lifetime": bubble_lifetime,"genealogy": build_genealogy(),
            "total_breakups": 0,"total_coalescences": 0,
            "breakup_event": [],"diameter_tree": [],"diameter_history": {},
            "reversal_diagnostics": [],**empty_counts,
        }
    #
    topo = analyze_raw_topology(
        H,rho,mu,sigma,is_2d,
        lbox_v=lbox_v,periodic_bc=periodic_bc,separation_veto=separation_veto,
    )
    F = topo["frames"]
    cls = topo["classes"]
    res = topo["resolved"]
    open_info = topo["open_info"]
    close_to_open = topo["close_to_open"]
    #
    idx_to_id_curr = {int(i): bid for i,bid in idx_to_id_initial.items()}
    id_to_birth_time = dict(id_to_birth_time_initial)
    used_ids = set(id_to_birth_time.keys())
    #
    physical_event_history = []
    idx_to_id_history = [dict(idx_to_id_curr)]
    breakup_event = []
    diameter_tree = []
    diameter_history = {}
    counts = dict(empty_counts)
    #
    def _physical_map(m):
        return {i: b for i,b in m.items() if _is_physical(b)}
    #
    store_physical_diameter_snapshot(
        diameter_history,_physical_map(idx_to_id_curr),
        F.V[0],F.S[0],F.T[0],is_2d,
    )
    #
    for ih,record in enumerate(H):
        #
        j = ih
        time_p0,time_p1 = F.T[j],F.T[j+1]
        istep_p0,istep_p1 = F.S[j],F.S[j+1]
        voll0,voll1 = F.V[j],F.V[j+1]
        #
        idx_to_id_next = {}
        step_events = []
        #
        # ------------------------------------------------
        # Label helpers
        # ------------------------------------------------
        #
        def _emerge(i1):
            n = len(used_ids)
            new_id = f"e{n}"
            while new_id in used_ids:
                n += 1
                new_id = f"e{n}"
            used_ids.add(new_id)
            id_to_birth_time[new_id] = time_p1
            _register_birth(bubble_lifetime,new_id,0,time_p1,istep_p1,"emergence")
            counts["total_emergences"] += 1
            step_events.append({"type": "emergence","istep": istep_p1,"time": time_p1,
                                "raw_p1": (i1,),"physical_id": new_id})
            return new_id
        #
        def _inherit(label,i1):
            if _is_debris(label) and res(j+1,i1):
                return _emerge(i1)
            return label
        #
        def _orphan(i1):
            return _emerge(i1) if res(j+1,i1) else DEBRIS
        #
        for e,ev in enumerate(record["events"]):
            #
            key = (j,e)
            etype = ev["type"]
            c = cls[j][e]
            cost = float(ev.get("cost",0.0))
            #
            # ================================================
            # Continuation
            # ================================================
            #
            if etype == "continuation":
                idx_to_id_next[int(ev["p1"][0])] = idx_to_id_curr[int(ev["p0"][0])]
                continue
            #
            # ================================================
            # Breakup
            # ================================================
            #
            if etype == "breakup":
                #
                i0 = int(ev["p0"][0])
                i1a,i1b = int(ev["p1"][0]),int(ev["p1"][1])
                parent = idx_to_id_curr[i0]
                #
                if key in open_info:
                    # A -> B+C that will be undone: both keep A.
                    idx_to_id_next[i1a] = parent
                    idx_to_id_next[i1b] = parent
                    counts["total_transient_breakups"] += 1
                    step_events.append({"type": "transient_breakup","istep": istep_p1,
                        "time": time_p1,"raw_p0": (i0,),"raw_p1": (i1a,i1b),
                        "physical_id": parent,"diagnostic": open_info[key]["diag"],
                        "raw_cost": cost})
                    if debug:
                        print("TRANSIENT BREAKUP:",parent,open_info[key]["diag"])
                    continue
                #
                if key in close_to_open:
                    # Closing A+B -> C -> A'+B': restore A and B.
                    if not _is_pair(parent):
                        raise RuntimeError(f"Expected transient pair at {key}, got {parent}.")
                    okey = close_to_open[key]
                    ev_open = H[okey[0]]["events"][okey[1]]
                    pa,pb = int(ev_open["p0"][0]),int(ev_open["p0"][1])
                    out_map = open_info[okey]["out_map"]
                    idx_to_id_next[out_map[pa]] = parent[1]
                    idx_to_id_next[out_map[pb]] = parent[2]
                    step_events.append({"type": "transient_rebreakup","istep": istep_p1,
                        "time": time_p1,"raw_p0": (i0,),"raw_p1": (i1a,i1b),
                        "physical_ids": (parent[1],parent[2]),"raw_cost": cost})
                    continue
                #
                if c["kind"] == "shed":
                    keep,drop = c["keep"],c["drop"]
                    idx_to_id_next[keep] = _inherit(parent,keep)
                    idx_to_id_next[drop] = _orphan(drop)
                    counts["total_sheddings"] += 1
                    step_events.append({"type": "shedding","istep": istep_p1,
                        "time": time_p1,"raw_p0": (i0,),"raw_p1": (keep,drop),
                        "physical_id": idx_to_id_next[keep],
                        "s_drop": F.s(j+1,drop),"raw_cost": cost})
                    continue
                #
                # ---------------- true breakup ----------------
                #
                if not _is_physical(parent):
                    raise RuntimeError(
                        f"Physical breakup of non-physical label {parent} at record {j}."
                    )
                t_birth = bubble_lifetime[parent]["t_birth"]
                child_a,child_b = _make_two_new_breakup_ids(parent,used_ids)
                idx_to_id_next[i1a] = child_a
                idx_to_id_next[i1b] = child_b
                id_to_birth_time[child_a] = time_p1
                id_to_birth_time[child_b] = time_p1
                g_parent = bubble_lifetime[parent]["generation"]
                g_child = g_parent+1
                _register_death(bubble_lifetime,parent,time_p1,istep_p1,"breakup",
                                children=(child_a,child_b))
                _register_birth(bubble_lifetime,child_a,g_child,time_p1,istep_p1,
                                "breakup",parents=(parent,))
                _register_birth(bubble_lifetime,child_b,g_child,time_p1,istep_p1,
                                "breakup",parents=(parent,))
                D_p = float(cmpt_d(voll0[i0],is_2d))
                D_a = float(cmpt_d(voll1[i1a],is_2d))
                D_b = float(cmpt_d(voll1[i1b],is_2d))
                diameter_tree.append({"type": "breakup","time_p0": time_p0,"time_p1": time_p1,
                    "istep_p0": istep_p0,"istep_p1": istep_p1,"parent": parent,
                    "parent_generation": g_parent,"children": (child_a,child_b),
                    "children_generation": (g_child,g_child),"D_parent": D_p,
                    "D_children": (D_a,D_b)})
                if voll1[i1a] <= voll1[i1b]:
                    cid_min,D_min = child_a,D_a
                else:
                    cid_min,D_min = child_b,D_b
                breakup_event.append([istep_p1,time_p1,parent,t_birth,time_p1,
                                      time_p1-t_birth,cid_min,D_min])
                step_events.append({"type": "breakup","istep": istep_p1,"time": time_p1,
                    "raw_p0": (i0,),"raw_p1": (i1a,i1b),"parent": parent,
                    "parent_generation": g_parent,"children": (child_a,child_b),
                    "children_generation": (g_child,g_child),
                    "diagnostic": topo_diag_lookup(topo,key),"raw_cost": cost})
                if debug:
                    print("TRUE BREAKUP:",parent,"->",child_a,"+",child_b)
                continue
            #
            # ================================================
            # Coalescence
            # ================================================
            #
            if etype == "coalescence":
                #
                i0a,i0b = int(ev["p0"][0]),int(ev["p0"][1])
                i1 = int(ev["p1"][0])
                la,lb = idx_to_id_curr[i0a],idx_to_id_curr[i0b]
                #
                if key in close_to_open:
                    # Closing A -> B+C -> A': both halves carry A.
                    if la != lb:
                        raise RuntimeError(
                            f"Breakup reversal closed with different labels {la},{lb} at {key}."
                        )
                    idx_to_id_next[i1] = la
                    step_events.append({"type": "fake_coalescence","istep": istep_p1,
                        "time": time_p1,"raw_p0": (i0a,i0b),"raw_p1": (i1,),
                        "physical_id": la,"raw_cost": cost})
                    continue
                #
                if key in open_info:
                    # A+B -> C that will be undone: carry the pair.
                    idx_to_id_next[i1] = (_TP,la,lb,open_info[key]["diag"]["swap"])
                    counts["total_transient_coalescences"] += 1
                    step_events.append({"type": "transient_coalescence","istep": istep_p1,
                        "time": time_p1,"raw_p0": (i0a,i0b),"raw_p1": (i1,),
                        "parents": (la,lb),"diagnostic": open_info[key]["diag"],
                        "raw_cost": cost})
                    if debug:
                        print("TRANSIENT COALESCENCE:",la,"+",lb,open_info[key]["diag"])
                    continue
                #
                if c["kind"] == "absorb":
                    kin,oin = c["keep_in"],c["other_in"]
                    lk,lo = idx_to_id_curr[kin],idx_to_id_curr[oin]
                    if _is_pair(lo):
                        raise RuntimeError(f"Transient pair absorbed as debris at {key}.")
                    new_label = _inherit(lk,i1)
                    idx_to_id_next[i1] = new_label
                    if _is_physical(lo) and lo != new_label:
                        _register_death(bubble_lifetime,lo,time_p1,istep_p1,"absorption",
                            children=(new_label,) if _is_physical(new_label) else ())
                    counts["total_absorptions"] += 1
                    step_events.append({"type": "absorption","istep": istep_p1,
                        "time": time_p1,"raw_p0": (kin,oin),"raw_p1": (i1,),
                        "physical_id": new_label,"absorbed": lo,
                        "s_absorbed": F.s(j,oin),"raw_cost": cost})
                    continue
                #
                # ---------------- true coalescence ----------------
                #
                if not (_is_physical(la) and _is_physical(lb)):
                    raise RuntimeError(
                        f"Physical coalescence of non-physical labels {la},{lb} at record {j}."
                    )
                if la == lb:
                    raise RuntimeError(f"Unmatched self-coalescence of {la} at record {j}.")
                dominant = la if voll0[i0a] >= voll0[i0b] else lb
                new_id = _make_new_coalescence_id(dominant,used_ids)
                idx_to_id_next[i1] = new_id
                id_to_birth_time[new_id] = time_p1
                ga = bubble_lifetime[la]["generation"]
                gb = bubble_lifetime[lb]["generation"]
                g_new = max(ga,gb)+1
                _register_death(bubble_lifetime,la,time_p1,istep_p1,"coalescence",children=(new_id,))
                _register_death(bubble_lifetime,lb,time_p1,istep_p1,"coalescence",children=(new_id,))
                _register_birth(bubble_lifetime,new_id,g_new,time_p1,istep_p1,
                                "coalescence",parents=(la,lb))
                diameter_tree.append({"type": "coalescence","time_p0": time_p0,
                    "time_p1": time_p1,"istep_p0": istep_p0,"istep_p1": istep_p1,
                    "parents": (la,lb),"parent_generations": (ga,gb),"child": new_id,
                    "child_generation": g_new,
                    "D_parents": (float(cmpt_d(voll0[i0a],is_2d)),float(cmpt_d(voll0[i0b],is_2d))),
                    "D_child": float(cmpt_d(voll1[i1],is_2d))})
                step_events.append({"type": "coalescence","istep": istep_p1,"time": time_p1,
                    "raw_p0": (i0a,i0b),"raw_p1": (i1,),"parents": (la,lb),"child": new_id,
                    "generation": g_new,"diagnostic": topo_diag_lookup(topo,key),
                    "raw_cost": cost})
                if debug:
                    print("TRUE COALESCENCE:",la,"+",lb,"->",new_id)
                continue
            #
            raise ValueError(f"Unknown raw event type: {etype}")
        #
        # ------------------------------------------------
        # Coverage, diameters, history
        # ------------------------------------------------
        #
        n1 = F.V[j+1].size
        if sorted(idx_to_id_next.keys()) != list(range(n1)):
            missing = sorted(set(range(n1))-set(idx_to_id_next.keys()))
            raise RuntimeError(f"Unassigned p1 regions at istep {istep_p1}: {missing}")
        #
        pmap = _physical_map(idx_to_id_next)
        if pmap:
            store_physical_diameter_snapshot(diameter_history,pmap,voll1,istep_p1,time_p1,is_2d)
        #
        physical_event_history.append({"history_index": ih,"istep_p0": istep_p0,
            "istep_p1": istep_p1,"time_p0": time_p0,"time_p1": time_p1,
            "events": step_events})
        idx_to_id_curr = idx_to_id_next
        idx_to_id_history.append(dict(idx_to_id_curr))
    #
    unresolved = {i: b for i,b in idx_to_id_curr.items() if _is_pair(b)}
    if unresolved:
        raise RuntimeError(f"Unresolved transient pairs at end of history: {unresolved}")
    #
    total_breakups,total_coalescences = count_clean_topological_events(physical_event_history)
    #
    if total_breakups != sum(t["type"] == "breakup" for t in diameter_tree) or \
       total_coalescences != sum(t["type"] == "coalescence" for t in diameter_tree):
        raise RuntimeError("Inconsistent bookkeeping between event history and diameter tree.")
    #
    return {
        "physical_event_history": physical_event_history,
        "idx_to_id_history": idx_to_id_history,
        "idx_to_id_final": idx_to_id_curr,
        "id_to_birth_time": id_to_birth_time,
        "bubble_lifetime": bubble_lifetime,
        "genealogy": build_genealogy(),
        "total_breakups": total_breakups,
        "total_coalescences": total_coalescences,
        "breakup_event": breakup_event,
        "diameter_tree": diameter_tree,
        "diameter_history": diameter_history,
        "reversal_diagnostics": topo["diagnostics"],
        **counts,
    }
#
def topo_diag_lookup(topo,key):
    # Diagnostic of a KEPT candidate reversal opened by this event, if any.
    return topo["diagnostics_by_open"].get(key)
#
def print_cleaning_summary(cleaned,total_breakups_raw,total_coalescences_raw):
    #
    diags = cleaned["reversal_diagnostics"]
    kept = [d for d in diags if not d["numerical"]]
    print("")
    print("==============================================")
    print("RESOLUTION-BASED CLEANING")
    print("==============================================")
    print("Raw breakups / coalescences      =",total_breakups_raw,"/",total_coalescences_raw)
    print("Physical breakups / coalescences =",cleaned["total_breakups"],"/",cleaned["total_coalescences"])
    print("Transient breakups (A->B+C->A)   =",cleaned["total_transient_breakups"])
    print("Transient coalescences           =",cleaned["total_transient_coalescences"])
    print("Sheddings of unresolved debris   =",cleaned["total_sheddings"])
    print("Absorptions of unresolved debris =",cleaned["total_absorptions"])
    print("Emergences (debris -> resolved)  =",cleaned["total_emergences"])
    print("Candidate reversals kept as real =",len(kept))
    for d in kept:
        name = "breakup " if d["kind"] == "breakup_reversal" else "coalesc."
        print(f"   {name} t={d['time_open']:.4g}->{d['time_close']:.4g}  "
              f"vol={d['volume_ratio']:.2f}  dt_min/t_window={d['time_ratio']:.2f}"
              + (f"  sep={d['separation_ratio']:.2f}" if "separation_ratio" in d else ""))
    print("==============================================")
#
