#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic("clear","")
#
# ============================================================
# IMPORTS
# ============================================================
#
import json
from pathlib import Path
import matplotlib.pyplot as plt
from fun_track_fast import count_raw_topological_events,plot_diameter_tree_v1, \
                           clean_genealogy_from_raw_history,print_cleaning_summary
#
# ============================================================
# PARAMETERS
# ============================================================
#
general_path = Path("/home/nn8802/Documents/simulation/tracking_validation/");
case_i = 5
res_v = [9,10,11,12]
order = 2
#
if order == 1:
    event_dir = "event_output_v1"
else:
    event_dir = "event_output_v2"
#
rho_liq = 1.0
mu_liq = 0.01
sigma = 6.944e+00
do_debug = False
#
# Box used only by the separation veto (breakup reversals).
# Same convention as locate_tag_events_fast: [Lx,Ly,Lz], [px,py,pz].
# Set periodic_bc = None for a wall-bounded box.
lbox_v = [10,10,10]
periodic_bc = [1,1,1]
separation_veto = True
#
fig_names = {
    1: "continuation.mp4",
    2: "breakup.mp4",
    3: "coalescence.mp4",
    4: "comb_breakup_coalescence.mp4",
    5: "three_bubbles.mp4",
}
#
# ============================================================
# LOOP OVER RESOLUTION
# ============================================================
#
for res in res_v:
    #
    print("")
    print("============================================================")
    print(f"RESOLUTION L{res:02d}")
    print("============================================================")
    #
    # ========================================================
    # WORKING DIRS
    # ========================================================
    #
    res_c = f"{res:02d}"
    work_dir = general_path/f"case{case_i}"/f"L{res_c}"
    raw_history_file = work_dir/event_dir/"event_history_raw.out"
    initial_genealogy_file = work_dir/event_dir/"initial_genealogy.json"
    #
    # ========================================================
    # LOAD INITIAL GENEALOGY
    # ========================================================
    #
    with open(initial_genealogy_file,"r") as f:
        initial_data = json.load(f)
    #
    idx_to_id_initial = {
        int(k):v
        for k,v in initial_data["idx_to_id_initial"].items()
    }
    #
    id_to_birth_time_initial = {
        k:float(v)
        for k,v in initial_data["id_to_birth_time_initial"].items()
    }
    #
    Nb_initial = int(initial_data["Nb_initial"])
    istep_initial = int(initial_data["istep_initial"])
    time_initial = float(initial_data["time_initial"])
    is_2d = int(initial_data["is_2d"])
    #
    print("Initial genealogy loaded")
    print("Nb_initial =",Nb_initial)
    print("istep_initial =",istep_initial)
    print("time_initial =",time_initial)
    #
    # ========================================================
    # LOAD RAW EVENT HISTORY
    # ========================================================
    #
    event_history_raw = []
    #
    with open(raw_history_file,"r") as f:
        for line in f:
            if line.strip():
                event_history_raw.append(json.loads(line))
    #
    if len(event_history_raw) == 0:
        raise RuntimeError(
            f"No records found in {raw_history_file}"
        )
    #
    print("event_history_raw loaded:",len(event_history_raw),"records")
    tend = event_history_raw[-1]["time_p1"]
    print("end time =",tend)
    #
    # ========================================================
    # SAFETY CHECK INITIAL STATE
    # ========================================================
    #
    n0_raw = len(event_history_raw[0]["voll0"])
    #
    if len(idx_to_id_initial) != n0_raw:
        raise RuntimeError(
            "Initial genealogy does not match first raw-history frame.\n"
            f"Initial IDs = {len(idx_to_id_initial)}\n"
            f"First raw frame bubbles = {n0_raw}"
        )
    #
    if set(idx_to_id_initial.values()) != set(id_to_birth_time_initial.keys()):
        raise RuntimeError(
            "Mismatch between initial physical IDs and birth-time IDs."
        )
    #
    # ========================================================
    # RAW TOPOLOGICAL EVENT COUNTS
    # ========================================================
    #
    (
        total_breakups_raw,
        total_coalescences_raw,
        raw_event_series,
    ) = count_raw_topological_events(event_history_raw)
    #
    print("")
    print("==============================================")
    print("RAW TOPOLOGICAL EVENTS")
    print("==============================================")
    print("Total raw breakups     =",total_breakups_raw)
    print("Total raw coalescences =",total_coalescences_raw)
    print("==============================================")
    #
    # ========================================================
    # CLEAN PHYSICAL GENEALOGY
    # ========================================================
    #
    cleaned = clean_genealogy_from_raw_history(
        event_history_raw,
        idx_to_id_initial,
        id_to_birth_time_initial,
        rho_liq,
        mu_liq,
        sigma,
        is_2d,
        debug=do_debug,
        lbox_v=lbox_v,
        periodic_bc=periodic_bc,
        separation_veto=separation_veto,
    )
    #
    print_cleaning_summary(cleaned,total_breakups_raw,total_coalescences_raw)
    #
    total_breakups = cleaned["total_breakups"]
    total_coalescences = cleaned["total_coalescences"]
    breakup_event = cleaned["breakup_event"]
    diameter_tree = cleaned["diameter_tree"]
    diameter_history = cleaned["diameter_history"]
    physical_event_history = cleaned["physical_event_history"]
    bubble_lifetime = cleaned["bubble_lifetime"]
    idx_to_id_final = cleaned["idx_to_id_final"]
    id_to_birth_time = cleaned["id_to_birth_time"]
    genealogy = cleaned["genealogy"]
    #
    # ========================================================
    # CLEAN TOPOLOGICAL EVENT COUNTS
    # ========================================================
    #
    print("")
    print("==============================================")
    print("CLEAN TOPOLOGICAL EVENTS")
    print("==============================================")
    print("Total clean breakups     =",total_breakups)
    print("Total clean coalescences =",total_coalescences)
    print("Removed breakups         =",total_breakups_raw-total_breakups)
    print("Removed coalescences     =",total_coalescences_raw-total_coalescences)
    print("==============================================")
    #
    # ========================================================
    # DIAMETER TREE
    # ========================================================
    #
    fig,ax = plt.subplots(figsize=(10,6))
    #
    plot_diameter_tree_v1(
        diameter_history,
        diameter_tree,
        genealogy,
        ax=ax,
        show_ids=True,
        tmax=tend,
        ylim=6.,
        quantity="diameter"
    )
    #
    plt.tight_layout()
    #
    fig_name = Path(fig_names[case_i]).stem
    fig.savefig(work_dir/event_dir/f"{fig_name}.pdf")
    #
    plt.show()
    plt.close(fig)
#