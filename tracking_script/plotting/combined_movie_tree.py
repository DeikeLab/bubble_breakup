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
import os
import re
import numpy as np
from pathlib import Path
from fun_postproc_short import plot_diameter_tree_v1, \
                               count_raw_topological_events, \
                               clean_genealogy_from_raw_history, \
                               print_cleaning_summary
#
from tracker_colors import (
    validate_raw_history,
    build_tracked_state,
    make_step_lookup,
    region_geometry,
    assign_regions_by_containment,
    report_fragmented_frame,
    report_containment_failure,
    check_tracker_alignment,
    build_connected_regions
)
#
# ============================================================
# PARAMETERS
# ============================================================
#
general_path = Path("/home/nn8802/Documents/simulation/tracking_validation/");
case_i = 4
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
# ------------------------------------------------------------
# Domain: ONE definition used by both the genealogy cleaning
# (separation veto) and the movie (facet regions, periodic
# distances). Must match the domain the facets are written in.
# ------------------------------------------------------------
#
xlim = (-5,5)
ylim = (-5,5)
zlim = (-5,5)                 # unused in 2D
periodic_bc = [1,1,1]         # [px,py,pz]; wall-bounded axis -> 0
#
lbox_v = [
    xlim[1]-xlim[0],
    ylim[1]-ylim[0],
    zlim[1]-zlim[0],
]
separation_veto = True
#
movie_names = {
    1: "continuation.mp4",
    2: "breakup.mp4",
    3: "coalescence.mp4",
    4: "comb_breakup_coalescence.mp4",
    5: "three_bubbles.mp4",
}
#
tol = None
fps = 20
dpi = 150
tree_ylim = (0.,5.)
write_png = True                # per-frame PNGs in event_output/frames
png_every = 1                   # 2 = every other frame
#
# ============================================================
# READ ONE FACETS FILE
# ============================================================
#
def read_facets_file(filename):
    #
    data = np.loadtxt(filename)
    #
    if data.ndim == 1:
        data = data.reshape(1,-1)
    #
    return data[:,0],data[:,1]
#
# ============================================================
# EXTRACT FACET STEP
#
# facets_000000000.out -> 0
# facets_000000050.out -> 50
# etc.
# ============================================================
#
def get_facet_timestep(filename):
    #
    basename = os.path.basename(filename)
    match = re.search(r"facets_(\d+)\.out",basename)
    #
    if match is None:
        raise ValueError(f"Cannot extract timestep from {filename}")
    #
    return int(match.group(1))
#
# ============================================================
# BUILD GLOBAL PHYSICAL TIME ARRAY
#
# Tracking is available every simulation timestep.
#
# Rather than using event_history_raw, collect all physical
# times appearing in diameter_history.
# ============================================================
#
def build_global_time_array(diameter_history,time_initial):
    #
    times = []
    #
    for data in diameter_history.values():
        #
        tt = np.asarray(data["time"],dtype=float)
        #
        if len(tt) > 0:
            times.extend(tt.tolist())
    #
    if len(times) == 0:
        raise RuntimeError("diameter_history does not contain any time information")
    #
    # Round only to suppress possible tiny floating-point
    # differences between identical simulation times.
    #
    times = np.asarray(times,dtype=float)
    times = np.unique(np.round(times,14))
    times = np.sort(times)
    #
    # Make sure the known initial physical time is included.
    #
    if not np.any(np.isclose(times,time_initial,rtol=0.0,atol=1.e-12)):
        times = np.sort(np.append(times,time_initial))
    #
    return times
#
# ============================================================
# COMBINED FACETS + DIAMETER TREE MOVIE
# ============================================================
#
def make_facets_tree_movie(
    facets_path,
    diameter_history,
    diameter_tree,
    genealogy,
    istep_initial,
    time_initial,
    tracked_state,
    output_file="facets_tree.mp4",
    tol=1.e-6,
    fps=20,
    dpi=150,
    xlim=(-5.,5.),
    ylim=(-5.,5.),
    tree_ylim=(0.,3.),
    show_ids=False,
    quantity="diameter",
    max_step_gap=0,
    size_weight=0.25,
    check_alignment=True,
    periodic=True,
    n_rays=5,
    tol_factor=2.0,
    png_dir=None,
    png_every=1,
    debug_fragmentation=True,
    max_frag_reports=8,
    untracked_color="lightgrey",
):
    #
    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    #
    from matplotlib.collections import LineCollection
    from scipy.optimize import linear_sum_assignment
    #
    # ========================================================
    # 1. Load facet files
    # ========================================================
    #
    facets_files = glob.glob(
        os.path.join(
            facets_path,
            "facets_*.out",
        )
    )
    #
    facets_files = sorted(
        facets_files,
        key=get_facet_timestep,
    )
    #
    if len(facets_files) == 0:
        #
        raise RuntimeError(
            f"No facets_*.out files found in {facets_path}"
        )
    #
    facet_steps = np.asarray(
        [
            get_facet_timestep(filename)
            for filename in facets_files
        ],
        dtype=int,
    )
    #
    # ========================================================
    # 2. Global tracking time
    # ========================================================
    #
    global_time = build_global_time_array(
        diameter_history,
        time_initial,
    )
    #
    # ========================================================
    # 3. Synchronize facet files and tracking
    # ========================================================
    #
    valid_files = []
    valid_steps = []
    frame_times = []
    #
    for filename,istep in zip(
        facets_files,
        facet_steps,
    ):
        #
        itrack = int(
            istep-istep_initial
        )
        #
        if itrack < 0:
            continue
        #
        if itrack >= len(global_time):
            continue
        #
        valid_files.append(
            filename
        )
        #
        valid_steps.append(
            istep
        )
        #
        frame_times.append(
            global_time[itrack]
        )
    #
    facets_files = valid_files
    #
    facet_steps = np.asarray(
        valid_steps,
        dtype=int,
    )
    #
    frame_times = np.asarray(
        frame_times,
        dtype=float,
    )
    #
    if len(facets_files) == 0:
        #
        raise RuntimeError(
            "No facet frames overlap with the tracking history."
        )
    #
    # ========================================================
    # 4. Figure
    # ========================================================
    #
    fig,(ax_facets,ax_tree) = plt.subplots(
        2,
        1,
        figsize=(10,10),
        gridspec_kw={
            "height_ratios":[1.15,1.0]
        },
        constrained_layout=True,
    )
    #
    # ========================================================
    # 5. Plot diameter / volume tree
    #
    # This function is the ONLY owner of bubble colors.
    # ========================================================
    #
    (
        ax_tree,
        branch_color,
        branch_value,
    ) = plot_diameter_tree_v1(
        diameter_history,
        diameter_tree,
        genealogy,
        ax=ax_tree,
        show_ids=show_ids,
        tmax=frame_times[-1],
        quantity=quantity,
    )
    #
    ax_tree.set_xlim(
        frame_times[0],
        frame_times[-1],
    )
    #
    ax_tree.set_ylim(
        tree_ylim
    )
    #
    # ========================================================
    # 6. Domain size
    # ========================================================
    #
    Lx = xlim[1]-xlim[0]
    Ly = ylim[1]-ylim[0]
    #
    # periodic: bool (both axes) or [px,py(,pz)] per axis, same
    # convention as periodic_bc in the tracker. A wall-bounded
    # axis gets length 0, i.e. it is never wrapped.
    #
    if np.ndim(periodic) == 0:
        periodic_x = periodic_y = bool(periodic)
    else:
        periodic_x,periodic_y = bool(periodic[0]),bool(periodic[1])
    #
    Lx_p = Lx if periodic_x else 0.
    Ly_p = Ly if periodic_y else 0.
    #
    # ========================================================
    # 7. Active physical IDs by timestep
    # ========================================================
    #
    active_ids_by_step = {}
    #
    for physical_id,data in diameter_history.items():
        #
        steps = np.asarray(
            data["istep"],
            dtype=int,
        )
        #
        for istep in steps:
            #
            istep = int(
                istep
            )
            #
            active_ids_by_step.setdefault(
                istep,
                [],
            ).append(
                physical_id
            )
    #
    # ========================================================
    # 8. Active IDs helper
    # ========================================================
    #
    def get_active_ids(
        istep,
    ):
        #
        return active_ids_by_step.get(
            int(istep),
            [],
        )
    #
    # ========================================================
    # 9. Region centroids
    # ========================================================
    #
    def get_region_centroids(
        regions,
    ):
        #
        centroids = np.empty(
            (
                len(regions),
                2,
            ),
            dtype=float,
        )
        #
        for iregion,region in enumerate(
            regions
        ):
            #
            segments = np.asarray(
                region["segments"],
                dtype=float,
            )
            #
            points = segments.reshape(
                -1,
                2,
            )
            #
            centroids[
                iregion
            ] = np.mean(
                points,
                axis=0,
            )
        #
        return centroids
    #
    # ========================================================
    # 10. Interface measure
    # ========================================================
    #
    def region_measure(
        region,
    ):
        #
        segments = np.asarray(
            region["segments"],
            dtype=float,
        )
        #
        if len(segments) == 0:
            #
            return 0.
        #
        dxy = (
            segments[:,1,:]
            -
            segments[:,0,:]
        )
        #
        return float(
            np.sum(
                np.sqrt(
                    np.sum(
                        dxy*dxy,
                        axis=1,
                    )
                )
            )
        )
    #
    # ========================================================
    # 11. Periodic distance matrix
    # ========================================================
    #
    def distance_matrix(
        points_a,
        points_b,
    ):
        #
        if len(points_a) == 0 or len(points_b) == 0:
            #
            return np.empty(
                (
                    len(points_a),
                    len(points_b),
                ),
                dtype=float,
            )
        #
        dx = (
            points_a[:,None,0]
            -
            points_b[None,:,0]
        )
        #
        dy = (
            points_a[:,None,1]
            -
            points_b[None,:,1]
        )
        #
        if Lx_p > 0.:
            #
            dx -= Lx_p*np.round(
                dx/Lx_p
            )
        #
        if Ly_p > 0.:
            #
            dy -= Ly_p*np.round(
                dy/Ly_p
            )
        #
        return np.sqrt(
            dx*dx+dy*dy
        )
    #
    # ========================================================
    # 12. Periodic mean
    # ========================================================
    #
    def periodic_mean(
        values,
        weights,
        xmin,
        length,
    ):
        #
        values = np.asarray(
            values,
            dtype=float,
        )
        #
        weights = np.asarray(
            weights,
            dtype=float,
        )
        #
        if len(values) == 0:
            #
            return 0.
        #
        if length <= 0.:
            #
            return float(
                np.average(
                    values,
                    weights=weights,
                )
            )
        #
        theta = (
            2.*np.pi
            *(values-xmin)
            /length
        )
        #
        c = np.sum(
            weights*np.cos(theta)
        )
        #
        s = np.sum(
            weights*np.sin(theta)
        )
        #
        theta_mean = np.arctan2(
            s,
            c,
        )
        #
        if theta_mean < 0.:
            #
            theta_mean += 2.*np.pi
        #
        return float(
            xmin
            +
            length*theta_mean/(2.*np.pi)
        )
    #
    # ========================================================
    # 13. Physical bubble centroids
    #
    # Combine all facet regions carrying the same physical ID.
    # ========================================================
    #
    def get_physical_centroids(
        regions,
        region_centroids,
        assignment,
    ):
        #
        physical_centroids = {}
        #
        physical_ids = set(
            assignment.values()
        )
        #
        for physical_id in physical_ids:
            #
            region_indices = [
                iregion
                for iregion,pid in assignment.items()
                if pid == physical_id
            ]
            #
            if len(region_indices) == 0:
                continue
            #
            weights = np.asarray(
                [
                    max(
                        region_measure(
                            regions[iregion]
                        ),
                        1.e-14,
                    )
                    for iregion in region_indices
                ],
                dtype=float,
            )
            #
            points = region_centroids[
                region_indices
            ]
            #
            xc = periodic_mean(
                points[:,0],
                weights,
                xlim[0],
                Lx_p,
            )
            #
            yc = periodic_mean(
                points[:,1],
                weights,
                ylim[0],
                Ly_p,
            )
            #
            physical_centroids[
                physical_id
            ] = np.asarray(
                [xc,yc],
                dtype=float,
            )
        #
        return physical_centroids
    #
    # ========================================================
    # 14. Select dominant anchor regions
    #
    # For N physical bubbles, select the N largest connected
    # facet regions.
    #
    # Small fragments are NOT allowed to define identity.
    # ========================================================
    #
    def get_anchor_regions(
        regions,
        nphysical,
    ):
        #
        if nphysical <= 0:
            #
            return []
        #
        measures = np.asarray(
            [
                region_measure(region)
                for region in regions
            ],
            dtype=float,
        )
        #
        order = np.argsort(
            measures
        )[::-1]
        #
        nanchor = min(
            nphysical,
            len(regions),
        )
        #
        return [
            int(iregion)
            for iregion in order[:nanchor]
        ]
    #
    # ========================================================
    # 15. Assign secondary regions
    #
    # Small fragments inherit a bubble ID only when clearly
    # closer to one anchor than to the others.
    #
    # Otherwise they remain grey.
    # ========================================================
    #
    def assign_secondary_regions(
        regions,
        centroids,
        assignment,
        anchor_regions,
        ambiguity_factor=0.70,
    ):
        #
        if len(anchor_regions) == 0:
            #
            return assignment
        #
        anchor_regions = [
            iregion
            for iregion in anchor_regions
            if iregion in assignment
        ]
        #
        if len(anchor_regions) == 0:
            #
            return assignment
        #
        anchor_centroids = centroids[
            anchor_regions
        ]
        #
        secondary_regions = [
            iregion
            for iregion in range(
                len(regions)
            )
            if iregion not in anchor_regions
        ]
        #
        if len(secondary_regions) == 0:
            #
            return assignment
        #
        secondary_centroids = centroids[
            secondary_regions
        ]
        #
        D = distance_matrix(
            secondary_centroids,
            anchor_centroids,
        )
        #
        for ilocal,iregion in enumerate(
            secondary_regions
        ):
            #
            distances = D[
                ilocal
            ]
            #
            order = np.argsort(
                distances
            )
            #
            # ------------------------------------------------
            # Only one possible anchor
            # ------------------------------------------------
            #
            if len(order) == 1:
                #
                ianchor = anchor_regions[
                    int(order[0])
                ]
                #
                assignment[
                    int(iregion)
                ] = assignment[
                    ianchor
                ]
                #
                continue
            #
            d1 = float(
                distances[
                    order[0]
                ]
            )
            #
            d2 = float(
                distances[
                    order[1]
                ]
            )
            #
            # ------------------------------------------------
            # Ambiguous fragment
            # ------------------------------------------------
            #
            if d2 <= 1.e-14:
                #
                continue
            #
            if d1 >= ambiguity_factor*d2:
                #
                continue
            #
            # ------------------------------------------------
            # Clearly belongs to nearest anchor
            # ------------------------------------------------
            #
            ianchor = anchor_regions[
                int(order[0])
            ]
            #
            assignment[
                int(iregion)
            ] = assignment[
                ianchor
            ]
        #
        return assignment
    #
    # ========================================================
    # 16. Initial region assignment
    # ========================================================
    #
    def assign_initial_regions(
        regions,
        centroids,
        active_ids,
    ):
        #
        assignment = {}
        #
        nphysical = len(
            active_ids
        )
        #
        if nphysical == 0:
            #
            return assignment
        #
        # ----------------------------------------------------
        # One physical bubble
        # ----------------------------------------------------
        #
        if nphysical == 1:
            #
            physical_id = active_ids[0]
            #
            return {
                iregion:physical_id
                for iregion in range(
                    len(regions)
                )
            }
        #
        # ----------------------------------------------------
        # Select dominant anchors
        # ----------------------------------------------------
        #
        anchor_regions = get_anchor_regions(
            regions,
            nphysical,
        )
        #
        if len(anchor_regions) == 0:
            #
            return assignment
        #
        # ----------------------------------------------------
        # Initial identification by relative size
        # ----------------------------------------------------
        #
        anchors_sorted = sorted(
            anchor_regions,
            key=lambda iregion:region_measure(
                regions[
                    iregion
                ]
            ),
        )
        #
        ids_sorted = sorted(
            active_ids,
            key=lambda physical_id:branch_value[
                physical_id
            ],
        )
        #
        for ianchor,physical_id in zip(
            anchors_sorted,
            ids_sorted,
        ):
            #
            assignment[
                int(ianchor)
            ] = physical_id
        #
        # ----------------------------------------------------
        # Secondary fragments
        # ----------------------------------------------------
        #
        assignment = assign_secondary_regions(
            regions,
            centroids,
            assignment,
            anchor_regions,
        )
        #
        return assignment
    #
    # ========================================================
    # 17. Continuation
    #
    # No topology change:
    #
    # - physical identity is preserved;
    # - only dominant regions are used for matching;
    # - secondary fragments cannot force an identity change.
    # ========================================================
    #
    def propagate_continuation(
        previous_regions,
        previous_centroids,
        previous_assignment,
        current_regions,
        current_centroids,
        active_ids,
    ):
        #
        assignment = {}
        #
        nphysical = len(
            active_ids
        )
        #
        if len(current_regions) == 0:
            #
            return assignment
        #
        if nphysical == 0:
            #
            return assignment
        #
        # ----------------------------------------------------
        # One physical bubble
        # ----------------------------------------------------
        #
        if nphysical == 1:
            #
            physical_id = active_ids[0]
            #
            return {
                iregion:physical_id
                for iregion in range(
                    len(current_regions)
                )
            }
        #
        # ----------------------------------------------------
        # Previous physical bubble centroids
        # ----------------------------------------------------
        #
        previous_physical_centroids = get_physical_centroids(
            previous_regions,
            previous_centroids,
            previous_assignment,
        )
        #
        valid_ids = [
            physical_id
            for physical_id in active_ids
            if physical_id in previous_physical_centroids
        ]
        #
        if len(valid_ids) == 0:
            #
            return assign_initial_regions(
                current_regions,
                current_centroids,
                active_ids,
            )
        #
        # ----------------------------------------------------
        # Current dominant anchors
        # ----------------------------------------------------
        #
        anchor_regions = get_anchor_regions(
            current_regions,
            nphysical,
        )
        #
        if len(anchor_regions) == 0:
            #
            return assignment
        #
        anchor_centroids = current_centroids[
            anchor_regions
        ]
        #
        previous_bubble_centroids = np.asarray(
            [
                previous_physical_centroids[
                    physical_id
                ]
                for physical_id in valid_ids
            ],
            dtype=float,
        )
        #
        # ----------------------------------------------------
        # Global physical bubble -> anchor matching
        # ----------------------------------------------------
        #
        D = distance_matrix(
            previous_bubble_centroids,
            anchor_centroids,
        )
        #
        previous_match,current_match = linear_sum_assignment(
            D
        )
        #
        for iprevious,icurrent in zip(
            previous_match,
            current_match,
        ):
            #
            physical_id = valid_ids[
                int(iprevious)
            ]
            #
            ianchor = anchor_regions[
                int(icurrent)
            ]
            #
            assignment[
                ianchor
            ] = physical_id
        #
        # ----------------------------------------------------
        # Missing IDs / unused anchors
        # ----------------------------------------------------
        #
        assigned_ids = set(
            assignment.values()
        )
        #
        missing_ids = [
            physical_id
            for physical_id in active_ids
            if physical_id not in assigned_ids
        ]
        #
        unused_anchors = [
            ianchor
            for ianchor in anchor_regions
            if ianchor not in assignment
        ]
        #
        if (
            len(missing_ids) > 0
            and
            len(unused_anchors) > 0
        ):
            #
            missing_ids = sorted(
                missing_ids,
                key=lambda physical_id:branch_value[
                    physical_id
                ],
            )
            #
            unused_anchors = sorted(
                unused_anchors,
                key=lambda iregion:region_measure(
                    current_regions[
                        iregion
                    ]
                ),
            )
            #
            for ianchor,physical_id in zip(
                unused_anchors,
                missing_ids,
            ):
                #
                assignment[
                    int(ianchor)
                ] = physical_id
        #
        # ----------------------------------------------------
        # Secondary fragments
        # ----------------------------------------------------
        #
        assignment = assign_secondary_regions(
            current_regions,
            current_centroids,
            assignment,
            anchor_regions,
        )
        #
        return assignment
    #
    # ========================================================
    # 18. Topology change
    # ========================================================
    #
    def propagate_topology_change(
        previous_regions,
        previous_centroids,
        previous_assignment,
        previous_active_ids,
        current_regions,
        current_centroids,
        current_active_ids,
    ):
        #
        assignment = {}
        #
        previous_set = set(
            previous_active_ids
        )
        #
        current_set = set(
            current_active_ids
        )
        #
        continued_ids = (
            previous_set
            &
            current_set
        )
        #
        born_ids = list(
            current_set
            -
            previous_set
        )
        #
        disappeared_ids = (
            previous_set
            -
            current_set
        )
        #
        if len(current_regions) == 0:
            #
            return assignment
        #
        if len(current_active_ids) == 0:
            #
            return assignment
        #
        # ----------------------------------------------------
        # One resulting bubble
        # ----------------------------------------------------
        #
        if len(current_active_ids) == 1:
            #
            physical_id = current_active_ids[0]
            #
            return {
                iregion:physical_id
                for iregion in range(
                    len(current_regions)
                )
            }
        #
        # ----------------------------------------------------
        # Previous physical centroids
        # ----------------------------------------------------
        #
        previous_physical_centroids = get_physical_centroids(
            previous_regions,
            previous_centroids,
            previous_assignment,
        )
        #
        # ----------------------------------------------------
        # Current anchors
        # ----------------------------------------------------
        #
        anchor_regions = get_anchor_regions(
            current_regions,
            len(current_active_ids),
        )
        #
        if len(anchor_regions) == 0:
            #
            return assignment
        #
        anchor_centroids = current_centroids[
            anchor_regions
        ]
        #
        used_anchors = set()
        #
        # ====================================================
        # 18a. Preserve continued physical IDs first
        # ====================================================
        #
        continued_valid_ids = [
            physical_id
            for physical_id in continued_ids
            if physical_id in previous_physical_centroids
        ]
        #
        if len(continued_valid_ids) > 0:
            #
            continued_centroids = np.asarray(
                [
                    previous_physical_centroids[
                        physical_id
                    ]
                    for physical_id in continued_valid_ids
                ],
                dtype=float,
            )
            #
            D = distance_matrix(
                continued_centroids,
                anchor_centroids,
            )
            #
            id_match,anchor_match = linear_sum_assignment(
                D
            )
            #
            for iid,ianchor_local in zip(
                id_match,
                anchor_match,
            ):
                #
                physical_id = continued_valid_ids[
                    int(iid)
                ]
                #
                ianchor = anchor_regions[
                    int(ianchor_local)
                ]
                #
                assignment[
                    ianchor
                ] = physical_id
                #
                used_anchors.add(
                    ianchor
                )
        #
        # ====================================================
        # 18b. New physical IDs
        # ====================================================
        #
        unused_anchors = [
            ianchor
            for ianchor in anchor_regions
            if ianchor not in used_anchors
        ]
        #
        if len(born_ids) > 0 and len(unused_anchors) > 0:
            #
            born_sorted = sorted(
                born_ids,
                key=lambda physical_id:branch_value[
                    physical_id
                ],
            )
            #
            unused_anchors_sorted = sorted(
                unused_anchors,
                key=lambda iregion:region_measure(
                    current_regions[
                        iregion
                    ]
                ),
            )
            #
            for ianchor,physical_id in zip(
                unused_anchors_sorted,
                born_sorted,
            ):
                #
                assignment[
                    int(ianchor)
                ] = physical_id
        #
        # ====================================================
        # 18c. Any still-unassigned active ID
        # ====================================================
        #
        assigned_ids = set(
            assignment.values()
        )
        #
        missing_ids = [
            physical_id
            for physical_id in current_active_ids
            if physical_id not in assigned_ids
        ]
        #
        remaining_anchors = [
            ianchor
            for ianchor in anchor_regions
            if ianchor not in assignment
        ]
        #
        if (
            len(missing_ids) > 0
            and
            len(remaining_anchors) > 0
        ):
            #
            missing_ids = sorted(
                missing_ids,
                key=lambda physical_id:branch_value[
                    physical_id
                ],
            )
            #
            remaining_anchors = sorted(
                remaining_anchors,
                key=lambda iregion:region_measure(
                    current_regions[
                        iregion
                    ]
                ),
            )
            #
            for ianchor,physical_id in zip(
                remaining_anchors,
                missing_ids,
            ):
                #
                assignment[
                    int(ianchor)
                ] = physical_id
        #
        # ====================================================
        # 18d. Secondary fragments
        # ====================================================
        #
        assignment = assign_secondary_regions(
            current_regions,
            current_centroids,
            assignment,
            anchor_regions,
        )
        #
        return assignment
    #
    # ========================================================
    # 19. Build colored segments
    # ========================================================
    #
    def build_colored_segments(
        regions,
        assignment,
    ):
        #
        segments = []
        colors = []
        #
        for iregion,region in enumerate(
            regions
        ):
            #
            physical_id = assignment.get(
                iregion,
                None,
            )
            #
            if physical_id is None:
                #
                # No branch in the diameter tree: a droplet the
                # genealogy discarded. Drawn dark grey.
                #
                color = untracked_color
            #
            else:
                #
                # Same dictionary the tree was drawn with, so
                # facet colour == branch colour by construction.
                #
                color = branch_color.get(
                    physical_id,
                    untracked_color,
                )
            #
            region_segments = region[
                "segments"
            ]
            #
            segments.extend(
                region_segments
            )
            #
            colors.extend(
                [
                    color
                ]*len(region_segments)
            )
        #
        return segments,colors
    #
    # ========================================================
    # 20. Precompute temporal assignments
    # ========================================================
    #
    all_regions = []
    all_assignments = []
    all_active_ids = []
    #
    previous_regions = None
    previous_centroids = None
    previous_assignment = None
    previous_active_ids = None
    #
    lookup_tracked = make_step_lookup(
        tracked_state,
        max_step_gap=max_step_gap,
    )
    #
    n_tracked_frames = 0
    n_fallback_frames = 0
    #
    n_frag_reported = 0
    #
    containment_stats = {
        "contained":0,
        "fallback":0,
        "conflict":0,
        "unclaimed":0,
    }
    #
    print(
        "Precomputing facet/branch correspondence..."
    )
    #
    for iframe,filename in enumerate(
        facets_files
    ):
        #
        istep = int(
            facet_steps[iframe]
        )
        #
        x,y = read_facets_file(
            filename
        )
        #
        regions = build_connected_regions(
            x,
            y,
            tol,
            tol_factor=tol_factor,
            Lx=Lx_p,
            Ly=Ly_p,
        )
        #
        # ----------------------------------------------------
        # Tracker state at this timestep
        #
        # This is the authoritative identity source.
        # ----------------------------------------------------
        #
        istep_tracked,tracked_now = lookup_tracked(
            istep
        )
        #
        if check_alignment and iframe == 0:
            #
            check_tracker_alignment(
                regions,
                tracked_now,
                istep,
                Lx=Lx_p,
                Ly=Ly_p,
                xmin=xlim[0],
                ymin=ylim[0],
            )
        #
        # ----------------------------------------------------
        # Tracker-driven assignment
        # ----------------------------------------------------
        #
        if len(tracked_now) > 0:
            #
            centroids,perimeters = region_geometry(
                regions,
                Lx=Lx_p,
                Ly=Ly_p,
                xmin=xlim[0],
                ymin=ylim[0],
            )
            #
            active_ids = list(
                tracked_now.keys()
            )
            #
            # Walk out from each resolved bubble's centre of
            # mass and claim the closed region enclosing it.
            # Whatever is left over is under-resolved and has
            # no branch in the tree, so it stays grey.
            #
            assignment,stats = assign_regions_by_containment(
                regions,
                tracked_now,
                Lx=Lx_p,
                Ly=Ly_p,
                n_rays=n_rays,
            )
            #
            for key in ("contained","fallback","conflict","unclaimed"):
                containment_stats[key] += stats[key]
            #
            # Containment failed: no closed region encloses the
            # bubble centre, so the interface itself is broken
            # in this frame. Report why.
            #
            if debug_fragmentation and stats["fallback"] > 0:
                #
                if n_frag_reported == 0:
                    print("")
                    print("FRAGMENTED FRAMES")
                #
                if n_frag_reported < max_frag_reports:
                    #
                    report_fragmented_frame(
                        regions,
                        istep,
                        Lx=Lx_p,
                        Ly=Ly_p,
                    )
                    #
                    for physical_id,point in stats["missed"]:
                        #
                        report_containment_failure(
                            regions,
                            point,
                            physical_id,
                            istep,
                            Lx=Lx_p,
                            Ly=Ly_p,
                            n_rays=n_rays,
                        )
                #
                n_frag_reported += 1
            #
            n_tracked_frames += 1
        #
        # ----------------------------------------------------
        # Fallback: no tracker snapshot for this facet dump
        #
        # Purely geometric continuation from the previous
        # frame. Should not normally be needed.
        # ----------------------------------------------------
        #
        else:
            #
            centroids = get_region_centroids(
                regions
            )
            #
            active_ids = (
                list(previous_active_ids)
                if previous_active_ids is not None
                else []
            )
            #
            if previous_regions is None:
                #
                assignment = assign_initial_regions(
                    regions,
                    centroids,
                    active_ids,
                )
            #
            else:
                #
                assignment = propagate_continuation(
                    previous_regions,
                    previous_centroids,
                    previous_assignment,
                    regions,
                    centroids,
                    active_ids,
                )
            #
            n_fallback_frames += 1
        #
        all_regions.append(
            regions
        )
        #
        all_assignments.append(
            assignment
        )
        #
        all_active_ids.append(
            active_ids
        )
        #
        previous_regions = regions
        previous_centroids = centroids
        previous_assignment = assignment
        previous_active_ids = active_ids
    #
    print(
        "Facet/branch correspondence completed."
    )
    #
    print(
        "  tracker-driven frames =",n_tracked_frames,
    )
    #
    print(
        "  fallback frames       =",n_fallback_frames,
    )
    #
    n_regions_tot = sum(len(r) for r in all_regions)
    n_assigned_tot = sum(len(a) for a in all_assignments)
    #
    print(
        "  regions coloured      =",n_assigned_tot,
        "of",n_regions_tot,
    )
    #
    print(
        "  untracked (grey)      =",n_regions_tot-n_assigned_tot,
    )
    #
    print(
        "  claimed by containment=",containment_stats["contained"],
    )
    #
    print(
        "  claimed by fallback   =",containment_stats["fallback"],
    )
    #
    if containment_stats["conflict"] > 0:
        #
        print(
            "  WARNING conflicts     =",containment_stats["conflict"],
            "(two IDs inside one region)",
        )
    #
    if n_fallback_frames > 0:
        #
        print(
            "  WARNING: some facet dumps have no tracking\n"
            "           snapshot. Increase max_step_gap or\n"
            "           check the facet output frequency."
        )
    #
    # ========================================================
    # 21. Initial facet collection
    # ========================================================
    #
    segments0,colors0 = build_colored_segments(
        all_regions[0],
        all_assignments[0],
    )
    #
    facet_collection = LineCollection(
        segments0,
        colors=colors0,
        linewidths=1.5,
    )
    #
    ax_facets.add_collection(
        facet_collection
    )
    #
    ax_facets.set_xlim(
        xlim
    )
    #
    ax_facets.set_ylim(
        ylim
    )
    #
    ax_facets.set_aspect(
        "equal",
        adjustable="box",
    )
    #
    ax_facets.set_xlabel("x")
    ax_facets.set_ylabel("y")
    #
    time_text = ax_facets.text(
        0.02,
        0.96,
        "",
        transform=ax_facets.transAxes,
        fontsize=13,
        verticalalignment="top",
    )
    #
    # ========================================================
    # 22. Tree synchronization markers
    # ========================================================
    #
    red_dots = ax_tree.scatter(
        [],
        [],
        s=70,
        color="red",
        edgecolor="black",
        linewidth=0.6,
        zorder=100,
    )
    #
    time_cursor = ax_tree.axvline(
        frame_times[0],
        color="red",
        linestyle="--",
        linewidth=1.,
        alpha=0.4,
        zorder=90,
    )
    #
    # ========================================================
    # 23. Active tree points
    # ========================================================
    #
    def get_active_tree_points(
        iframe,
    ):
        #
        t = frame_times[
            iframe
        ]
        #
        points = [
            [
                t,
                branch_value[
                    physical_id
                ],
            ]
            for physical_id in all_active_ids[
                iframe
            ]
            if physical_id in branch_value
        ]
        #
        if len(points) == 0:
            #
            return np.empty(
                (0,2),
                dtype=float,
            )
        #
        return np.asarray(
            points,
            dtype=float,
        )
    #
    # ========================================================
    # 24. Initial state
    # ========================================================
    #
    t0 = float(
        frame_times[0]
    )
    #
    istep0 = int(
        facet_steps[0]
    )
    #
    red_dots.set_offsets(
        get_active_tree_points(
            0
        )
    )
    #
    time_cursor.set_xdata(
        [t0,t0]
    )
    #
    time_text.set_text(
        f"t = {t0:.6f}    step = {istep0}"
    )
    #
    # ========================================================
    # 25. Animation update
    # ========================================================
    #
    def update(
        iframe,
    ):
        #
        t = float(
            frame_times[iframe]
        )
        #
        istep = int(
            facet_steps[iframe]
        )
        #
        segments,colors = build_colored_segments(
            all_regions[iframe],
            all_assignments[iframe],
        )
        #
        facet_collection.set_segments(
            segments
        )
        #
        facet_collection.set_color(
            colors
        )
        #
        red_dots.set_offsets(
            get_active_tree_points(
                iframe
            )
        )
        #
        time_cursor.set_xdata(
            [t,t]
        )
        #
        time_text.set_text(
            f"t = {t:.6f}    step = {istep}"
        )
        #
        return (
            facet_collection,
            red_dots,
            time_cursor,
            time_text,
        )
    #
    # ========================================================
    # 26. Create and save animation
    # ========================================================
    #
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(facets_files),
        interval=1000./fps,
        blit=False,
    )
    #
    writer = animation.FFMpegWriter(
        fps=fps,
        bitrate=4000,
    )
    #
    ani.save(
        output_file,
        writer=writer,
        dpi=dpi,
    )
    #
    # ========================================================
    # 27. Optional per-frame PNGs
    #
    # Named by the SIMULATION STEP, not by frame index, so a
    # filename can be matched directly against the diagnostics
    # and against diagnose_frame.py.
    # ========================================================
    #
    if png_dir is not None:
        #
        os.makedirs(png_dir,exist_ok=True)
        #
        for iframe in range(len(facets_files)):
            #
            if iframe % png_every != 0:
                continue
            #
            update(iframe)
            #
            fig.savefig(
                os.path.join(
                    png_dir,
                    f"frame_{int(facet_steps[iframe]):09d}.png",
                ),
                dpi=dpi,
            )
        #
        n_png = len(range(0,len(facets_files),png_every))
        #
        print("wrote",n_png,"PNG frames to",png_dir)
    #
    plt.close(
        fig
    )
    #
    return ani
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
    work_dir = general_path / f"case{case_i}" / f"L{res_c}"
    #
    facets_path = work_dir/"interface/facets"
    raw_history_file = work_dir/event_dir/"event_history_raw.out"
    initial_genealogy_file = work_dir/event_dir/"initial_genealogy.json"
    output_movie = work_dir/event_dir/movie_names[case_i]
    png_dir = str(work_dir/event_dir/"frames") if write_png else None
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
    print("end time =",event_history_raw[-1]["time_p1"])
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
    validate_raw_history(event_history_raw)
    #
    # ========================================================
    # RAW + CLEAN TOPOLOGICAL EVENTS
    # ========================================================
    #
    (
        total_breakups_raw,
        total_coalescences_raw,
        raw_event_series,
    ) = count_raw_topological_events(event_history_raw)
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
    diameter_tree = cleaned["diameter_tree"]
    diameter_history = cleaned["diameter_history"]
    genealogy = cleaned["genealogy"]
    #
    # --------------------------------------------------------
    # The cleaned index map now also carries unresolved-debris
    # labels (non-string tuples with no branch in the tree).
    # They are dropped here so the movie draws debris in grey,
    # like any other untracked region. Physical IDs and
    # transient pairs are passed through unchanged.
    # --------------------------------------------------------
    #
    idx_to_id_history = [
        {
            i:pid
            for i,pid in frame_map.items()
            if not (isinstance(pid,tuple) and pid[:1] == ("__DEBRIS__",))
        }
        for frame_map in cleaned["idx_to_id_history"]
    ]
    #
    # ========================================================
    # TRACKED STATE
    #
    # istep -> {physical_id: centroid, volume, diameter}
    #
    # Identity comes from the tracker. The movie never re-derives
    # it from the facet geometry.
    # ========================================================
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
    print("tracked_state built for",len(tracked_state),"timesteps")
    #
    # ========================================================
    # MAKE MOVIE
    # ========================================================
    #
    ani = make_facets_tree_movie(
        facets_path,
        diameter_history,
        diameter_tree,
        genealogy,
        istep_initial,
        time_initial,
        tracked_state,
        output_file=output_movie,
        tol=tol,
        fps=fps,
        dpi=dpi,
        xlim=xlim,
        ylim=ylim,
        tree_ylim=tree_ylim,
        show_ids=False,
        max_step_gap=0,
        size_weight=0.25,
        check_alignment=True,
        periodic=periodic_bc,
        n_rays=5,
        tol_factor=4.0,                   # see note in tracker_colors
        png_dir=png_dir,
        png_every=png_every,
        debug_fragmentation=True,
        max_frag_reports=8,
    )
    #
    print("movie written:",output_movie)
#
