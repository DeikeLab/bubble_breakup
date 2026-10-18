#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# ============================================================
# TRACKER-DRIVEN FACET COLORING
#
# The physical identity of every facet region is taken
# DIRECTLY from the tracker, not re-derived from geometry.
#
# The tracker already knows, at every stored timestep:
#
#   - which raw components belong to which physical ID
#     (diameter_history[pid]["raw_indices"])
#   - where every raw component is
#     (event_history_raw[...]["xpos1"/"ypos1"])
#   - how big it is
#     (voll1 / diameter)
#
# Colours therefore match the diameter tree BY CONSTRUCTION.
# ============================================================
#
import numpy as np
#
from scipy.optimize import linear_sum_assignment
#
# ============================================================
# PERIODIC WEIGHTED MEAN
# ============================================================
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
# ============================================================
# PERIODIC DISTANCE MATRIX
# ============================================================
#
def periodic_distance_matrix(points_a,points_b,Lx,Ly):
    #
    points_a = np.asarray(points_a,dtype=float)
    points_b = np.asarray(points_b,dtype=float)
    #
    if len(points_a) == 0 or len(points_b) == 0:
        return np.empty((len(points_a),len(points_b)),dtype=float)
    #
    dx = points_a[:,None,0]-points_b[None,:,0]
    dy = points_a[:,None,1]-points_b[None,:,1]
    #
    if Lx is not None and Lx > 0.:
        dx -= Lx*np.round(dx/Lx)
    #
    if Ly is not None and Ly > 0.:
        dy -= Ly*np.round(dy/Ly)
    #
    return np.sqrt(dx*dx+dy*dy)
#
# ============================================================
# RAW COMPONENT POSITIONS PER TIMESTEP
#
# Snapshots are stored by clean_genealogy_from_raw_history as:
#
#   - record[0] at istep_p0 using voll0 / xpos0 / ypos0
#   - every record at istep_p1 using voll1 / xpos1 / ypos1
#
# so the raw indices stored in diameter_history refer to the
# arrays selected here.
# ============================================================
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
# ============================================================
# TRACKED STATE
#
# istep -> {physical_id: {"x","y","volume","diameter"}}
#
# This is the single source of truth for the movie.
# ============================================================
#
def _diameter_from_volume(v,is_2d):
    #
    v = max(float(v),0.)
    #
    if int(is_2d) == 1:
        return float(np.sqrt(4.*v/np.pi))
    #
    return float((6.*v/np.pi)**(1./3.))
#
def _last_diameter_before(diameter_history,pid,istep):
    #
    data = diameter_history.get(pid)
    #
    if data is None or len(data["istep"]) == 0:
        return 0.
    #
    steps = data["istep"]
    best = 0.
    #
    for k in range(len(steps)-1,-1,-1):
        #
        if int(steps[k]) <= istep:
            best = float(data["diameter"][k])
            break
    #
    return best
#
def build_tracked_state(
    event_history_raw,
    diameter_history,
    xlim=None,
    ylim=None,
    idx_to_id_history=None,
    idx_to_id_initial=None,
    is_2d=1,
):
    #
    step_positions = build_step_positions(event_history_raw)
    #
    if xlim is None:
        xmin,Lx = 0.,0.
    else:
        xmin,Lx = float(xlim[0]),float(xlim[1]-xlim[0])
    #
    if ylim is None:
        ymin,Ly = 0.,0.
    else:
        ymin,Ly = float(ylim[0]),float(ylim[1]-ylim[0])
    #
    tracked_state = {}
    n_missing = 0
    #
    # --------------------------------------------------------
    # Preferred path: build from idx_to_id_history
    #
    # diameter_history has already had __TRANSIENT_PAIR__
    # components filtered out by clean_genealogy_from_raw_history,
    # so a blob waiting out a transient coalescence has no entry
    # there at all -- it exists, it has an ID, and it would be
    # drawn grey. idx_to_id_history covers every component, so
    # the transient ones can be shown under one of their two
    # parent IDs instead of vanishing.
    # --------------------------------------------------------
    #
    if idx_to_id_history is not None:
        #
        frames = []
        #
        # --------------------------------------------------
        # Align the ID history with the records.
        #
        # clean_genealogy_from_raw_history seeds the list with
        # the INITIAL state before the loop and then appends
        # once per record:
        #
        #     idx_to_id_history[0]      state at istep_p0 of record 0
        #     idx_to_id_history[ih+1]   state at istep_p1 of record ih
        #
        # so it holds len(records)+1 entries. Pairing entry ih
        # with record ih labels every step with the PREVIOUS
        # step's IDs -- which is invisible while the component
        # count is unchanged, and produces swapped identities
        # and phantom indices at every step where it changes.
        # --------------------------------------------------
        #
        n_rec = len(event_history_raw)
        #
        offset = 1 if len(idx_to_id_history) == n_rec+1 else 0
        #
        if offset == 0 and len(idx_to_id_history) != n_rec:
            print(
                "build_tracked_state: WARNING idx_to_id_history has",
                len(idx_to_id_history),"entries for",n_rec,
                "records -- cannot align them reliably",
            )
        #
        first_ids = (
            idx_to_id_history[0] if offset == 1
            else idx_to_id_initial
        )
        #
        if first_ids is not None:
            frames.append(
                (int(event_history_raw[0]["istep_p0"]),first_ids)
            )
        #
        for ih in range(n_rec):
            #
            k = ih+offset
            #
            if k < len(idx_to_id_history):
                frames.append(
                    (int(event_history_raw[ih]["istep_p1"]),
                     idx_to_id_history[k])
                )
        #
        n_transient = 0
        n_transient_reassigned = 0
        n_transient_orphan = 0
        n_split = 0
        #
        for istep,ids in frames:
            #
            if istep not in step_positions:
                n_missing += 1
                continue
            #
            pos = step_positions[istep]
            #
            grouped = {}
            #
            # --------------------------------------------------
            # Pass 1: components with a plain physical ID.
            #
            # These own their ID outright. A transient pair must
            # not be allowed to borrow one of them.
            # --------------------------------------------------
            #
            pending = []
            #
            for i,bid in ids.items():
                #
                i = int(i)
                #
                if i < 0 or i >= len(pos["x"]):
                    n_missing += 1
                    continue
                #
                if (isinstance(bid,tuple) and len(bid) == 3
                        and bid[0] == "__TRANSIENT_PAIR__"):
                    pending.append((i,bid))
                    continue
                #
                grouped.setdefault(bid,[]).append(i)
            #
            # --------------------------------------------------
            # Pass 2: transient pairs.
            #
            # A pair is shown under whichever parent was larger
            # going into the window -- but ONLY if that parent is
            # not already on screen as a component of its own.
            #
            # Without that check, the pair and the real bubble
            # collapse into one entry: their volumes add and
            # their positions get averaged across the domain.
            # That is what produced an ID holding two components
            # 5 units apart, with the volume of one bubble and
            # the colour of another.
            #
            # If both parents are taken, the pair keeps its own
            # key and is simply not coloured, which is honest.
            # --------------------------------------------------
            #
            for i,bid in pending:
                #
                da = _last_diameter_before(diameter_history,bid[1],istep)
                db = _last_diameter_before(diameter_history,bid[2],istep)
                #
                first,second = (bid[1],bid[2]) if da >= db else (bid[2],bid[1])
                #
                if first not in grouped:
                    display = first
                elif second not in grouped:
                    display = second
                    n_transient_reassigned += 1
                else:
                    display = bid
                    n_transient_orphan += 1
                #
                n_transient += 1
                #
                grouped.setdefault(display,[]).append(i)
            #
            for display,indices in grouped.items():
                #
                # Sanity check: components of one physical ID
                # must be near each other. Anything else means
                # the grouping is wrong, and a silently averaged
                # position across the domain is worse than a
                # loud complaint.
                #
                if len(indices) > 1:
                    #
                    px = pos["x"][indices]
                    py = pos["y"][indices]
                    #
                    dx = px.max()-px.min()
                    dy = py.max()-py.min()
                    #
                    if Lx > 0.:
                        dx = min(dx,Lx-dx)
                    #
                    if Ly > 0.:
                        dy = min(dy,Ly-dy)
                    #
                    spread = float(np.sqrt(dx*dx+dy*dy))
                    #
                    D_ref = _diameter_from_volume(
                        float(np.sum(pos["v"][indices])),is_2d,
                    )
                    #
                    if spread > 4.*max(D_ref,1.e-12):
                        n_split += 1
                #
                weights = np.maximum(pos["v"][indices],1.e-30)
                #
                xc = periodic_weighted_mean(
                    pos["x"][indices],weights,xmin,Lx,
                )
                #
                yc = periodic_weighted_mean(
                    pos["y"][indices],weights,ymin,Ly,
                )
                #
                vol = float(np.sum(pos["v"][indices]))
                #
                tracked_state.setdefault(istep,{})[display] = {
                    "x":xc,
                    "y":yc,
                    "volume":vol,
                    "diameter":_diameter_from_volume(vol,is_2d),
                    "nraw":len(indices),
                    "points":[
                        (float(pos["x"][i]),float(pos["y"][i]))
                        for i in indices
                    ],
                    "point_volumes":[float(pos["v"][i]) for i in indices],
                }
        #
        if n_transient > 0:
            print(
                "build_tracked_state:",n_transient,
                "transient-pair components shown under a parent ID",
            )
        #
        if n_transient_reassigned > 0:
            print(
                "build_tracked_state:",n_transient_reassigned,
                "transient pairs moved to their other parent"
                " (first parent already on screen)",
            )
        #
        if n_transient_orphan > 0:
            print(
                "build_tracked_state:",n_transient_orphan,
                "transient pairs left uncoloured (both parents"
                " already on screen)",
            )
        #
        if n_split > 0:
            print(
                "build_tracked_state: WARNING",n_split,
                "physical IDs hold components more than 4 diameters"
                " apart -- check the genealogy at those steps",
            )
        #
        if n_missing > 0:
            print(
                "build_tracked_state: skipped",n_missing,
                "components without usable raw positions",
            )
        #
        return tracked_state
    #
    # --------------------------------------------------------
    # Fallback: diameter_history only (transient pairs absent)
    # --------------------------------------------------------
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
                pos["x"][indices],weights,xmin,Lx,
            )
            #
            yc = periodic_weighted_mean(
                pos["y"][indices],weights,ymin,Ly,
            )
            #
            # The centroid of EACH raw component is kept as
            # well. During drainage one physical ID owns
            # several blobs, and their merged centroid falls in
            # the gap between them -- inside no interface at
            # all. The individual centroids are each safely
            # inside their own blob.
            #
            points = [
                (float(pos["x"][i]),float(pos["y"][i]))
                for i in indices
            ]
            #
            point_volumes = [float(pos["v"][i]) for i in indices]
            #
            tracked_state.setdefault(istep,{})[physical_id] = {
                "x":xc,
                "y":yc,
                "volume":float(volumes[k]),
                "diameter":float(diameters[k]),
                "nraw":len(indices),
                "points":points,
                "point_volumes":point_volumes,
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
# ============================================================
# NEAREST TRACKED TIMESTEP
#
# Facets are usually dumped less often than the tracking is
# performed, and the two step counters need not coincide
# exactly.
# ============================================================
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
# ============================================================
# REGION GEOMETRY
# ============================================================
#
def minimum_image(dxy,Lx=0.,Ly=0.):
    #
    # A facet segment never spans the domain. If it appears to,
    # its endpoints were written on opposite sides of a
    # periodic boundary, and the raw difference would give a
    # segment of length ~L with a midpoint in the middle of
    # the domain -- which would dominate both the perimeter
    # and the centroid.
    #
    dxy = np.array(dxy,dtype=float,copy=True)
    #
    if Lx is not None and Lx > 0.:
        dxy[:,0] -= Lx*np.round(dxy[:,0]/Lx)
    #
    if Ly is not None and Ly > 0.:
        dxy[:,1] -= Ly*np.round(dxy[:,1]/Ly)
    #
    return dxy
#
def region_perimeter(region,Lx=0.,Ly=0.):
    #
    segments = np.asarray(region["segments"],dtype=float)
    #
    if len(segments) == 0:
        return 0.
    #
    dxy = minimum_image(segments[:,1,:]-segments[:,0,:],Lx,Ly)
    #
    return float(np.sum(np.sqrt(np.sum(dxy*dxy,axis=1))))
#
def region_length_centroid(region,Lx=0.,Ly=0.,xmin=0.,ymin=0.):
    #
    # Arc-length weighted centroid: robust against uneven
    # facet density, unlike a plain mean of the endpoints.
    #
    # When Lx/Ly are given the average is taken on the circle,
    # so a region that wraps around a periodic boundary gets
    # its true centroid instead of one in the middle of the
    # domain.
    #
    segments = np.asarray(region["segments"],dtype=float)
    #
    if len(segments) == 0:
        return np.zeros(2,dtype=float)
    #
    dxy = minimum_image(segments[:,1,:]-segments[:,0,:],Lx,Ly)
    #
    mid = segments[:,0,:]+0.5*dxy
    #
    w = np.sqrt(np.sum(dxy*dxy,axis=1))
    #
    wsum = float(np.sum(w))
    #
    if wsum <= 0.:
        w = np.ones(len(mid),dtype=float)
    #
    xc = periodic_weighted_mean(mid[:,0],w,xmin,Lx)
    yc = periodic_weighted_mean(mid[:,1],w,ymin,Ly)
    #
    return np.asarray([xc,yc],dtype=float)
#
def polygon_area_centroid(loop):
    #
    x = loop[:,0]
    y = loop[:,1]
    #
    x0,y0 = float(np.mean(x)),float(np.mean(y))
    #
    xs,ys = x-x0,y-y0
    #
    cross = xs[:-1]*ys[1:]-xs[1:]*ys[:-1]
    #
    A = 0.5*float(np.sum(cross))
    #
    if abs(A) < 1.e-12:
        return 0.,0.,0.
    #
    cx = float(np.sum((xs[:-1]+xs[1:])*cross))/(6.*A)+x0
    cy = float(np.sum((ys[:-1]+ys[1:])*cross))/(6.*A)+y0
    #
    return abs(A),cx,cy
#
def loop_perimeter(loop):
    #
    d = loop[1:]-loop[:-1]
    #
    return float(np.sum(np.sqrt(np.sum(d*d,axis=1))))
#
def region_segments_single_image(region,Lx=0.,Ly=0.):
    #
    # All segments of a region placed in one consistent image.
    #
    loops = region.get("loops")
    #
    pieces = [
        np.stack([np.asarray(l)[:-1],np.asarray(l)[1:]],axis=1)
        for l in (loops or []) if l is not None and len(l) >= 4
    ]
    #
    if pieces:
        return np.concatenate(pieces,axis=0)
    #
    segs = np.asarray(region["segments"],dtype=float).copy()
    #
    ref = segs[0,0,:]
    #
    for k in (0,1):
        #
        dxy = segs[:,k,:]-ref
        #
        if Lx > 0.:
            dxy[:,0] -= Lx*np.round(dxy[:,0]/Lx)
        #
        if Ly > 0.:
            dxy[:,1] -= Ly*np.round(dxy[:,1]/Ly)
        #
        segs[:,k,:] = ref+dxy
    #
    return segs
#
def region_area_centroid(region,Lx=0.,Ly=0.,n_target=220,return_grid=False):
    #
    # Cached: sampling is the expensive part of a frame, and
    # the same region is measured many times during matching.
    #
    if not return_grid:
        #
        cached = region.get("_area_cache")
        #
        if cached is not None:
            return cached
    #
    # Enclosed area and AREA centroid, measured by sampling the
    # interior with the even-odd rule.
    #
    # The obvious way -- the shoelace formula over the ordered
    # loop -- needs the traversal to wind consistently, and it
    # does not. Where an interface pinches, the two branches
    # meet TANGENTIALLY, so the walk slips from one to the
    # other without any long step or sharp turn to betray it.
    # The two lobes then wind opposite ways and the signed area
    # cancels: a real bubble of area 0.33 measured 0.0019 and
    # its centroid landed 100 units outside the domain.
    #
    # Sampling does not care about ordering or self-crossing.
    # It costs more, but it is measuring the same thing the
    # tracker measures: the area the fluid occupies.
    #
    segs = region_segments_single_image(region,Lx,Ly)
    #
    if len(segs) < 3:
        region["_area_cache"] = (0.,0.,0.,False)
        return 0.,0.,0.,False
    #
    v = segs.reshape(-1,2)
    #
    xmin,ymin = v[:,0].min(),v[:,1].min()
    xmax,ymax = v[:,0].max(),v[:,1].max()
    #
    w,hgt = xmax-xmin,ymax-ymin
    #
    if w <= 0. or hgt <= 0.:
        return 0.,0.,0.,False
    #
    # Grid spacing: n_target cells across the longer side.
    #
    dl = max(w,hgt)/float(n_target)
    #
    nx = max(int(np.ceil(w/dl))+1,2)
    ny = max(int(np.ceil(hgt/dl))+1,2)
    #
    gx = xmin+(np.arange(nx)+0.5)*dl
    gy = ymin+(np.arange(ny)+0.5)*dl
    #
    # Even-odd test: for each sample row, count segment
    # crossings of the ray going in +x.
    #
    ya = segs[:,0,1]
    yb = segs[:,1,1]
    xa = segs[:,0,0]
    xb = segs[:,1,0]
    #
    inside = np.zeros((ny,nx),dtype=bool)
    #
    dy = yb-ya
    dy_safe = np.where(np.abs(dy) < 1.e-30,1.e-30,dy)
    #
    for j,yy in enumerate(gy):
        #
        straddle = (ya > yy) != (yb > yy)
        #
        if not np.any(straddle):
            continue
        #
        t = (yy-ya[straddle])/dy_safe[straddle]
        #
        xint = xa[straddle]+t*(xb[straddle]-xa[straddle])
        #
        # count crossings strictly to the right of each sample
        #
        cnt = np.sum(gx[:,None] < xint[None,:],axis=1)
        #
        inside[j] = (cnt%2) == 1
    #
    n_in = int(np.count_nonzero(inside))
    #
    if n_in == 0:
        region["_area_cache"] = (0.,0.,0.,False)
        return 0.,0.,0.,False
    #
    A = n_in*dl*dl
    #
    jj,ii = np.nonzero(inside)
    #
    cx = float(np.mean(gx[ii]))
    cy = float(np.mean(gy[jj]))
    #
    # A point that is definitely INSIDE.
    #
    # The area centroid is not: on a crescent it falls in the
    # hollow, outside the fluid. Probing nesting with it misses
    # a crescent sitting inside another contour, and the host
    # then keeps area that is not its own.
    #
    px = gx[ii]-cx
    py = gy[jj]-cy
    #
    k_in = int(np.argmin(px*px+py*py))
    #
    region["_inside_point"] = (float(gx[ii[k_in]]),float(gy[jj[k_in]]))
    #
    if return_grid:
        return A,cx,cy,True,(gx,gy,inside)
    #
    region["_area_cache"] = (A,cx,cy,True)
    #
    return A,cx,cy,True
#
def resolve_nested_regions(regions,Lx=0.,Ly=0.,n_rays=5):
    #
    # Subtract nested loops from their host.
    #
    # Sampling a region's interior by parity counts everything
    # the contour encloses -- including a separate bubble that
    # happens to sit inside a concavity of it. That area then
    # belongs to two regions at once, and the host reads too
    # large: at one frame the tracked volumes summed to 5.85
    # while the measured areas summed to 6.79.
    #
    # A region is nested in another when its centroid tests
    # inside that other region. The host's area and first
    # moment are corrected by subtraction, which is exact.
    #
    raw = []
    #
    for region in regions:
        #
        A,cx,cy,ok = region_area_centroid(region,Lx,Ly)
        #
        probe = region.get("_inside_point") or (cx,cy)
        #
        raw.append((A,cx,cy,ok,probe))
    #
    order = sorted(
        [i for i in range(len(regions)) if raw[i][3]],
        key=lambda i:raw[i][0],
    )
    #
    host_of = {}
    #
    for a_i,i in enumerate(order):
        #
        # Smallest possible host that contains it.
        #
        for j in order[a_i+1:]:
            #
            if point_in_region(raw[i][4],regions[j],Lx,Ly,n_rays):
                host_of[i] = j
                break
    #
    if not host_of:
        return
    #
    A_corr = {i:raw[i][0] for i in range(len(regions)) if raw[i][3]}
    Mx = {i:raw[i][0]*raw[i][1] for i in A_corr}
    My = {i:raw[i][0]*raw[i][2] for i in A_corr}
    #
    # Innermost first, so a chain of nesting resolves correctly.
    #
    for i in order:
        #
        j = host_of.get(i)
        #
        if j is None:
            continue
        #
        A_corr[j] -= A_corr[i]
        Mx[j] -= Mx[i]
        My[j] -= My[i]
    #
    for i,A in A_corr.items():
        #
        if A <= 1.e-12:
            regions[i]["_area_cache"] = (0.,0.,0.,False)
            continue
        #
        regions[i]["_area_cache"] = (A,Mx[i]/A,My[i]/A,True)
#
def region_geometry(regions,Lx=0.,Ly=0.,xmin=0.,ymin=0.):
    #
    centroids = np.empty((len(regions),2),dtype=float)
    perimeters = np.empty(len(regions),dtype=float)
    #
    for iregion,region in enumerate(regions):
        #
        centroids[iregion] = region_length_centroid(
            region,Lx,Ly,xmin,ymin,
        )
        #
        perimeters[iregion] = region_perimeter(region,Lx,Ly)
    #
    return centroids,perimeters
#
# ============================================================
# ASSIGN FACET REGIONS TO PHYSICAL IDS
#
# Cost = periodic centroid distance (dominant)
#      + size mismatch (tie breaker only)
#
# Both terms are normalised by the local bubble scale, so the
# cost is dimensionless and case independent.
# ============================================================
#
def assign_regions_from_tracker(
    regions,
    centroids,
    perimeters,
    tracked_now,
    Lx=0.,
    Ly=0.,
    size_weight=0.25,
    fragment_ambiguity=0.70,
    max_offset=0.75,
    size_ratio_max=4.0,
    size_ratio_min=0.25,
):
    #
    # Returns {region index: physical_id}. A region that does
    # not correspond to any TRACKED bubble is simply absent
    # from the mapping: small droplets the genealogy discarded
    # have no branch in the tree, so they must not be given a
    # branch colour.
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
    # Acceptance test
    #
    # The arc-length centroid of a CLOSED interface sits at the
    # bubble centre, so a genuine match is offset from the
    # tracked centroid by only a few percent of the diameter.
    # A discarded droplet fails on distance, on size, or both.
    # --------------------------------------------------------
    #
    def accepts(iregion,iid,dist):
        #
        dref = max(tracked_d[iid],region_d[iregion],1.e-30)
        #
        if dist > max_offset*dref:
            return False
        #
        ratio = region_d[iregion]/max(tracked_d[iid],1.e-30)
        #
        if ratio > size_ratio_max or ratio < size_ratio_min:
            return False
        #
        return True
    #
    # --------------------------------------------------------
    # Anchors = the n largest regions, n = number of bubbles
    # --------------------------------------------------------
    #
    nanchor = min(len(ids),len(regions))
    #
    order = np.argsort(perimeters)[::-1]
    anchors = [int(i) for i in order[:nanchor]]
    #
    D = periodic_distance_matrix(centroids[anchors],tracked_xy,Lx,Ly)
    #
    dD = region_d[anchors][:,None]-tracked_d[None,:]
    #
    cost = (D/d_scale)**2+size_weight*(dD/d_scale)**2
    #
    ianchor_match,iid_match = linear_sum_assignment(cost)
    #
    matched_ids = set()
    #
    for ia,ii in zip(ianchor_match,iid_match):
        #
        ia,ii = int(ia),int(ii)
        iregion = int(anchors[ia])
        #
        if not accepts(iregion,ii,float(D[ia,ii])):
            continue
        #
        assignment[iregion] = ids[ii]
        matched_ids.add(ii)
    #
    # --------------------------------------------------------
    # Second pass
    #
    # A tracked bubble must never be left grey just because an
    # untracked droplet occupied an anchor slot. Any ID still
    # unmatched looks over every free region.
    # --------------------------------------------------------
    #
    free = [
        iregion
        for iregion in range(len(regions))
        if iregion not in assignment
    ]
    #
    for ii in range(len(ids)):
        #
        if ii in matched_ids or len(free) == 0:
            continue
        #
        Dfree = periodic_distance_matrix(
            centroids[free],
            tracked_xy[ii:ii+1],
            Lx,Ly,
        )[:,0]
        #
        best = None
        best_cost = None
        #
        for ilocal,iregion in enumerate(free):
            #
            dist = float(Dfree[ilocal])
            #
            if not accepts(iregion,ii,dist):
                continue
            #
            c = (dist/d_scale)**2
            #
            if best_cost is None or c < best_cost:
                best_cost = c
                best = iregion
        #
        if best is not None:
            #
            assignment[best] = ids[ii]
            matched_ids.add(ii)
            free.remove(best)
    #
    # --------------------------------------------------------
    # Remaining fragments
    #
    # A fragment inherits an ID only when it is unambiguously
    # closest to one tracked bubble AND close enough to belong
    # to it. Everything else stays unassigned.
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
    Df = periodic_distance_matrix(centroids[fragments],tracked_xy,Lx,Ly)
    #
    for ilocal,iregion in enumerate(fragments):
        #
        distances = Df[ilocal]
        #
        iorder = np.argsort(distances)
        #
        i1 = int(iorder[0])
        d1 = float(distances[i1])
        #
        if not accepts(int(iregion),i1,d1):
            continue
        #
        if len(iorder) > 1:
            #
            d2 = float(distances[int(iorder[1])])
            #
            if d2 <= 1.e-14 or d1 >= fragment_ambiguity*d2:
                continue
        #
        assignment[int(iregion)] = ids[i1]
    #
    return assignment
#
# ============================================================
# POINT IN REGION
#
# Ray casting (crossing number) against the region's facet
# segments. The segments are unordered, which does not matter:
# a ray crosses a closed curve an odd number of times if and
# only if its origin is inside.
#
# PLIC interfaces are not watertight -- neighbouring facets
# very nearly meet but not exactly -- so a single ray can slip
# through a gap and give the wrong parity. Several rays are
# cast in different directions and the majority wins.
# ============================================================
#
def count_crossings(segments,point,theta,Lx=0.,Ly=0.):
    #
    # Ray crossings for a set of segments already placed in a
    # single consistent image (see point_in_region).
    #
    a = np.asarray(segments[:,0,:],dtype=float)-np.asarray(point,dtype=float)
    b = np.asarray(segments[:,1,:],dtype=float)-np.asarray(point,dtype=float)
    #
    c,s = np.cos(theta),np.sin(theta)
    #
    ax = c*a[:,0]+s*a[:,1]
    ay = -s*a[:,0]+c*a[:,1]
    #
    bx = c*b[:,0]+s*b[:,1]
    by = -s*b[:,0]+c*b[:,1]
    #
    straddles = (ay > 0.) != (by > 0.)
    #
    if not np.any(straddles):
        return 0
    #
    ay_s,by_s = ay[straddles],by[straddles]
    ax_s,bx_s = ax[straddles],bx[straddles]
    #
    t = ay_s/(ay_s-by_s)
    #
    xint = ax_s+t*(bx_s-ax_s)
    #
    return int(np.count_nonzero(xint > 0.))
#
def region_in_image_of(region,point,Lx=0.,Ly=0.):
    #
    # Place the WHOLE region in the periodic image nearest the
    # test point, as one rigid piece.
    #
    # Imaging each endpoint separately -- which is what the
    # earlier version did -- tears a region apart whenever part
    # of it lies beyond L/2 from the point and part does not.
    # A torn contour has meaningless crossing parity, and a
    # bubble on the far side of the domain then tests as
    # "inside" a region it is nowhere near.
    #
    loops = region.get("loops")
    #
    if not loops:
        loop = region.get("loop")
        loops = [loop] if loop is not None else []
    #
    pieces = [
        np.stack([np.asarray(l)[:-1],np.asarray(l)[1:]],axis=1)
        for l in loops if l is not None and len(l) >= 4
    ]
    #
    if pieces:
        segs = np.concatenate(pieces,axis=0)
    #
    else:
        segs = np.asarray(region["segments"],dtype=float).copy()
    #
    ref = segs.reshape(-1,2).mean(axis=0)
    #
    shift = np.zeros(2,dtype=float)
    #
    if Lx is not None and Lx > 0.:
        shift[0] = -Lx*np.round((ref[0]-point[0])/Lx)
    #
    if Ly is not None and Ly > 0.:
        shift[1] = -Ly*np.round((ref[1]-point[1])/Ly)
    #
    return segs+shift
#
def point_in_region(point,region,Lx=0.,Ly=0.,n_rays=5):
    #
    # A region needs a measurable interior for containment to
    # mean anything.
    #
    # This used to test region["closed"], the flag from the
    # loop traversal. That was wrong: the traversal fails to
    # close on complex contours -- at one frame the largest
    # blob in the domain had zero cycles -- while the ray test
    # answered [1,1,1,1,1] for points plainly inside it. Every
    # such region was refused containment and pushed into the
    # fallback, and nesting went undetected.
    #
    # The sampled area is cached, so this is cheap.
    #
    if not region_area_centroid(region,Lx,Ly)[3]:
        return False
    #
    segments = region_in_image_of(region,point,Lx,Ly)
    #
    if len(segments) < 3:
        return False
    #
    votes = 0
    #
    for iray in range(n_rays):
        #
        theta = 2.*np.pi*iray/n_rays+0.37
        #
        if count_crossings(segments,point,theta)%2 == 1:
            votes += 1
    #
    return 2*votes > n_rays
#
# ============================================================
# ASSIGN BY CONTAINMENT
#
# For every tracked bubble, walk out from the centroid of each
# of its raw components and claim the closed region that
# encloses it. Regions nothing claims are under-resolved blobs
# with no branch in the tree: they stay grey.
# ============================================================
#
def assign_regions_by_containment(
    regions,
    tracked_now,
    Lx=0.,
    Ly=0.,
    n_rays=5,
    fallback_nearest=True,
    fill_area_deficit=True,
    area_tol=0.08,
    max_extra_regions=4,
    extra_reach=1.5,
):
    #
    assignment = {}
    #
    stats = {
        "contained":0,
        "fallback":0,
        "conflict":0,
        "unclaimed":0,
        "missed":[],
    }
    #
    if len(regions) == 0 or len(tracked_now) == 0:
        stats["unclaimed"] = len(regions)
        return assignment,stats
    #
    # Correct any nested loops before the areas are used.
    #
    resolve_nested_regions(regions,Lx,Ly,n_rays)
    #
    perims = np.asarray(
        [region_perimeter(r,Lx,Ly) for r in regions],
        dtype=float,
    )
    #
    # --------------------------------------------------------
    # Candidate regions for every tracked point
    #
    # A point can test as inside more than one region: the ray
    # parity is not perfectly reliable on a large or partly
    # broken contour. The SMALLEST containing region is the
    # right answer -- a point inside a small bubble is inside
    # the small bubble, whatever a leaky test says about the
    # big one nearby. Ordering candidates by size also gives a
    # second choice to fall back on when the first is taken.
    # --------------------------------------------------------
    #
    def candidates(point):
        #
        inside = [
            iregion
            for iregion in range(len(regions))
            if point_in_region(point,regions[iregion],Lx,Ly,n_rays)
        ]
        #
        return sorted(inside,key=lambda i:perims[i])
    #
    # Area centroid of every region, precomputed once.
    #
    area_c = [region_area_centroid(r,Lx,Ly) for r in regions]
    #
    def nearest_free(point):
        #
        # Compare the tracked VOLUME centroid against each
        # region's AREA centroid: like with like. A crescent
        # whose centroid falls outside its own interface --
        # so containment cannot succeed -- is still matched
        # correctly here, whereas "nearest facet" hands it to
        # whichever bubble happens to sit in its concavity.
        #
        best,best_d = None,np.inf
        #
        for iregion,region in enumerate(regions):
            #
            if iregion in assignment:
                continue
            #
            A,cx,cy,ok = area_c[iregion]
            #
            if ok:
                #
                dx,dy = cx-point[0],cy-point[1]
                #
                if Lx is not None and Lx > 0.:
                    dx -= Lx*np.round(dx/Lx)
                #
                if Ly is not None and Ly > 0.:
                    dy -= Ly*np.round(dy/Ly)
                #
                d = float(np.sqrt(dx*dx+dy*dy))
            #
            else:
                #
                v = np.asarray(region["vertices"],dtype=float)
                #
                dx = v[:,0]-point[0]
                dy = v[:,1]-point[1]
                #
                if Lx is not None and Lx > 0.:
                    dx -= Lx*np.round(dx/Lx)
                #
                if Ly is not None and Ly > 0.:
                    dy -= Ly*np.round(dy/Ly)
                #
                d = float(np.min(np.sqrt(dx*dx+dy*dy)))
            #
            if d < best_d:
                best_d,best = d,iregion
        #
        return best
    #
    for physical_id,info in tracked_now.items():
        #
        points = info.get("points") or [(info["x"],info["y"])]
        #
        for point in points:
            #
            hit = None
            contested = False
            #
            for iregion in candidates(point):
                #
                owner = assignment.get(iregion)
                #
                if owner is None or owner == physical_id:
                    hit = iregion
                    break
                #
                contested = True
            #
            if hit is not None:
                #
                stats["contained"] += 1
                assignment[hit] = physical_id
                continue
            #
            if contested:
                stats["conflict"] += 1
            #
            # Either nothing contained the point, or everything
            # that did was already taken. Never drop the bubble:
            # claim the nearest free region instead.
            #
            stats["missed"].append((physical_id,tuple(point)))
            #
            if not fallback_nearest:
                continue
            #
            hit = nearest_free(point)
            #
            if hit is not None:
                stats["fallback"] += 1
                assignment[hit] = physical_id
    #
    # --------------------------------------------------------
    # Area deficit
    #
    # VOF tagging works on connected CELLS, so two blobs in
    # near-contact are one component: one ID, one volume, one
    # centroid. The facet reconstruction still draws them as
    # two separate closed loops, and the loop the centroid does
    # not land in ends up with no owner.
    #
    # The give-away is arithmetic: the ID owns less enclosed
    # area than its tracked volume. Keep handing it the nearest
    # free loops until the areas account for the volume. The
    # volume is the stopping criterion, so this does not need a
    # tuned threshold -- it stops when the books balance.
    # --------------------------------------------------------
    #
    if fill_area_deficit:
        #
        def area_of(iregion):
            #
            A,cx,cy,ok = region_area_centroid(regions[iregion],Lx,Ly)
            #
            return (A,cx,cy) if ok else (0.,0.,0.)
        #
        owner_of = {}
        #
        for iregion,pid in assignment.items():
            owner_of.setdefault(pid,[]).append(iregion)
        #
        for physical_id,info in tracked_now.items():
            #
            volume = float(info.get("volume",0.))
            #
            if volume <= 0.:
                continue
            #
            claimed = sum(area_of(i)[0] for i in owner_of.get(physical_id,[]))
            #
            deficit = volume-claimed
            #
            # Length scale of this bubble, for the reach limit.
            #
            D_ref = max(float(info.get("diameter",0.)),
                        2.*np.sqrt(max(volume,0.)/np.pi))
            #
            n_extra = 0
            #
            while (deficit > area_tol*volume
                   and n_extra < max_extra_regions):
                #
                # Pick the loop that best ACCOUNTS FOR the
                # shortfall, not the nearest one.
                #
                # Nearest-first grabs whatever speck happens to
                # lie closest, shaves a few percent off the
                # deficit, and repeats until it runs out of
                # tries -- while the loop that actually belongs
                # to the bubble, slightly further away, is
                # never taken.
                #
                best,best_cost,best_A = None,np.inf,0.
                #
                for iregion in range(len(regions)):
                    #
                    if iregion in assignment:
                        continue
                    #
                    A,cx,cy = area_of(iregion)
                    #
                    if A <= 0.:
                        continue
                    #
                    # Never absorb a loop bigger than the
                    # shortfall it is meant to explain.
                    #
                    if A > deficit*1.5:
                        continue
                    #
                    dx,dy = cx-info["x"],cy-info["y"]
                    #
                    if Lx is not None and Lx > 0.:
                        dx -= Lx*np.round(dx/Lx)
                    #
                    if Ly is not None and Ly > 0.:
                        dy -= Ly*np.round(dy/Ly)
                    #
                    dist = float(np.sqrt(dx*dx+dy*dy))
                    #
                    if dist > extra_reach*D_ref:
                        continue
                    #
                    # How well it closes the books, with
                    # distance only as a tie breaker.
                    #
                    cost = (abs(deficit-A)/max(deficit,1.e-30)
                            +0.25*dist/max(D_ref,1.e-30))
                    #
                    if cost < best_cost:
                        best_cost,best,best_A = cost,iregion,A
                #
                if best is None:
                    break
                #
                assignment[best] = physical_id
                stats["deficit_filled"] = stats.get("deficit_filled",0)+1
                #
                deficit -= best_A
                n_extra += 1
    #
    stats.setdefault("deficit_filled",0)
    #
    stats["unclaimed"] = len(regions)-len(assignment)
    #
    return assignment,stats
#
# ============================================================
# ALIGNMENT CHECK
#
# Run this ONCE before generating a movie.
#
# It verifies that facet coordinates and tracker centroids
# live in the same frame and that the facet dump steps exist
# in the tracking history. Almost every colour mismatch that
# is not an algorithmic one shows up here.
# ============================================================
#
def check_tracker_alignment(
    regions,
    tracked_now,
    istep,
    Lx=0.,
    Ly=0.,
    xmin=0.,
    ymin=0.,
    max_rows=15,
):
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
    centroids,perimeters = region_geometry(regions,Lx,Ly,xmin,ymin)
    #
    ids = list(tracked_now.keys())
    #
    tracked_xy = np.asarray(
        [[tracked_now[pid]["x"],tracked_now[pid]["y"]] for pid in ids],
        dtype=float,
    )
    #
    D = periodic_distance_matrix(centroids,tracked_xy,Lx,Ly)
    #
    print("")
    print(
        f"{'region':>7}{'xc':>12}{'yc':>12}"
        f"{'D_facet':>12}{'nearest id':>16}"
        f"{'dist':>12}{'D_track':>12}"
    )
    #
    shown = np.argsort(perimeters)[::-1][:max_rows]
    #
    for iregion in shown:
        #
        iregion = int(iregion)
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
    if len(regions) > max_rows:
        print(f"   ... {len(regions)-max_rows} smaller regions not shown")
    #
    print("")
    print("Distances should be a small fraction of the bubble")
    print("diameter. Large values mean the facet frame and the")
    print("tracker frame do not coincide.")
    print("==============================================")
#

# ============================================================
# ROBUST FACET CONNECTIVITY
#
# Basilisk writes one PLIC segment per interfacial cell. The
# segments of two neighbouring cells meet on the shared cell
# face only APPROXIMATELY: the two reconstructed planes have
# different constants, so the intersection points differ by a
# small fraction of the cell size.
#
# Snapping endpoints onto a 1e-6 lattice therefore shatters a
# single closed interface into hundreds of pieces.
#
# Here endpoints are merged by PROXIMITY (KD-tree + union
# find), with a tolerance measured in units of the local facet
# length, i.e. of the grid spacing.
# ============================================================
#
def build_connected_regions(x,y,tol=None,tol_factor=4.0,Lx=0.,Ly=0.):
    #
    # tol_factor is the endpoint search radius in units of the
    # median facet length.
    #
    # 2.0 was too small. Measured on real frames, the largest
    # gap the contour needs bridged is about 4.2 facet lengths
    # -- the interface is not watertight where it is poorly
    # resolved. At 2.0 a bubble was severed into two open
    # chains, the parity test leaked through the break, and the
    # frame's measured area came out 16% above the tracked
    # volume. At 3.0 and above the same frame closes and lands
    # within 0.4%.
    #
    # Checked across three frames at 2, 3, 4, 5, 6: results are
    # unchanged from 3 upward, so 4 sits in the middle of the
    # plateau rather than on its edge.
    #
    from scipy.spatial import cKDTree
    #
    pts = np.stack((np.asarray(x,dtype=float),
                    np.asarray(y,dtype=float)),axis=1)
    #
    nseg = len(pts)//2
    #
    if nseg == 0:
        return []
    #
    p0 = pts[0:2*nseg:2]
    p1 = pts[1:2*nseg:2]
    #
    # --------------------------------------------------------
    # Automatic tolerance
    #
    # The median facet length is the grid spacing of the finest
    # interfacial level. tol is only a SEARCH RADIUS here, not
    # a merge distance: the greedy pairing below decides what
    # actually joins, so a generous value is safe.
    # --------------------------------------------------------
    #
    lengths = np.sqrt(np.sum((p1-p0)**2,axis=1))
    #
    positive = lengths[lengths > 0.]
    #
    h = float(np.median(positive)) if len(positive) > 0 else 1.
    #
    if tol is None:
        tol = tol_factor*h
    #
    # --------------------------------------------------------
    # Endpoint proximity candidates
    # --------------------------------------------------------
    #
    endpoints = np.concatenate([p0,p1],axis=0)
    #
    if Lx > 0. and Ly > 0.:
        #
        shifted = np.empty_like(endpoints)
        shifted[:,0] = np.mod(endpoints[:,0]-endpoints[:,0].min(),Lx)
        shifted[:,1] = np.mod(endpoints[:,1]-endpoints[:,1].min(),Ly)
        #
        tree = cKDTree(shifted,boxsize=[Lx,Ly])
        coords = shifted
    #
    else:
        #
        tree = cKDTree(endpoints)
        coords = endpoints
    #
    pairs = tree.query_pairs(tol,output_type="ndarray")
    #
    if len(pairs) == 0:
        pairs = np.empty((0,2),dtype=int)
    #
    # --------------------------------------------------------
    # Greedy endpoint pairing
    #
    # Every facet endpoint has at most ONE true continuation:
    # the endpoint of the neighbouring cell's facet on the
    # shared face. Consuming endpoints shortest-pair-first
    # means each endpoint is taken by its genuine partner
    # before any longer, spurious link can be considered.
    #
    # This is what keeps two children that pinch off a few
    # cells apart from being welded into one region, while
    # still bridging the gaps WITHIN each interface.
    # --------------------------------------------------------
    #
    parent = np.arange(nseg,dtype=int)
    #
    partner = -np.ones(2*nseg,dtype=int)
    #
    def find(i):
        #
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        #
        return i
    #
    def union(i,j):
        #
        ri,rj = find(i),find(j)
        #
        if ri != rj:
            parent[rj] = ri
    #
    if len(pairs) > 0:
        #
        d = np.sqrt(np.sum(
            (coords[pairs[:,0]]-coords[pairs[:,1]])**2,
            axis=1,
        ))
        #
        order = np.argsort(d)
        #
        used = np.zeros(len(endpoints),dtype=bool)
        #
        for ipair in order:
            #
            ia = int(pairs[ipair,0])
            ib = int(pairs[ipair,1])
            #
            if used[ia] or used[ib]:
                continue
            #
            # Two endpoints of the same segment are not a link.
            #
            if ia%nseg == ib%nseg:
                continue
            #
            used[ia] = True
            used[ib] = True
            #
            partner[ia] = ib
            partner[ib] = ia
            #
            union(ia%nseg,ib%nseg)
    #
    # --------------------------------------------------------
    # Collect regions
    # --------------------------------------------------------
    #
    roots = {}
    #
    for k in range(nseg):
        roots.setdefault(find(k),[]).append(k)
    #
    regions = []
    #
    for seg_ids in roots.values():
        #
        seg_ids = np.asarray(seg_ids,dtype=int)
        #
        segs = np.stack([p0[seg_ids],p1[seg_ids]],axis=1)
        verts = segs.reshape(-1,2)
        #
        dx = float(verts[:,0].max()-verts[:,0].min())
        dy = float(verts[:,1].max()-verts[:,1].min())
        #
        # ----------------------------------------------------
        # Ordered traversal -> ALL cycles
        #
        # A region can hold more than one closed curve: a
        # crescent and the droplet nestled against it end up
        # in the same connected region whenever they touch
        # within the pairing tolerance. Walking only the cycle
        # through the first segment then describes part of the
        # region, and if the walk doubles back the signed area
        # nearly cancels -- which produced areas of 0.002 on a
        # bubble of D_eq 1.64 and centroids outside the domain.
        #
        # Every cycle is extracted here; the caller picks.
        # ----------------------------------------------------
        #
        loops = []
        visited = set()
        #
        for s_start in seg_ids:
            #
            s_start = int(s_start)
            #
            if s_start in visited:
                continue
            #
            pts_loop = [p0[s_start].copy()]
            prev = p0[s_start]
            cur_end = s_start+nseg
            chain = [s_start]
            closed_here = False
            #
            for _ in range(len(seg_ids)+1):
                #
                nxt = p1[cur_end-nseg] if cur_end >= nseg else p0[cur_end]
                #
                step = nxt-prev
                #
                if Lx > 0.:
                    step[0] -= Lx*np.round(step[0]/Lx)
                #
                if Ly > 0.:
                    step[1] -= Ly*np.round(step[1]/Ly)
                #
                prev = prev+step
                pts_loop.append(prev.copy())
                #
                nb = partner[cur_end]
                #
                if nb < 0:
                    break
                #
                seg_next = int(nb)%nseg
                #
                if seg_next == s_start:
                    closed_here = True
                    break
                #
                if seg_next in visited:
                    break
                #
                visited.add(seg_next)
                chain.append(seg_next)
                #
                cur_end = seg_next if nb >= nseg else seg_next+nseg
            #
            visited.add(s_start)
            visited.update(chain)
            #
            if closed_here and len(pts_loop) >= 4:
                loops.append(np.asarray(pts_loop,dtype=float))
        #
        closed = len(loops) > 0
        loop = loops[0] if closed else None
        #
        regions.append({
            "seg_ids":[int(s) for s in seg_ids],
            "segments":segs,
            "vertices":verts,
            "dx":dx,
            "dy":dy,
            "D_eq":max(dx,dy),
            "nseg":len(seg_ids),
            "loop":loop,
            "loops":loops,
            "closed":closed,
        })
    #
    regions = sorted(regions,key=lambda reg:reg["D_eq"],reverse=True)
    #
    return regions
#
# ============================================================
# CONNECTIVITY REPORT
# ============================================================
#
def report_connectivity(regions,n_expected=None):
    #
    print("")
    print("==============================================")
    print("FACET CONNECTIVITY")
    print("==============================================")
    print("regions found    =",len(regions))
    #
    if n_expected is not None:
        print("bubbles tracked  =",n_expected)
    #
    perims = sorted(
        [region_perimeter(r) for r in regions],
        reverse=True,
    )
    #
    print("largest perimeters =",
          [round(p,4) for p in perims[:6]])
    #
    print("total perimeter    =",round(float(np.sum(perims)),4))
    #
    if n_expected is not None and len(regions) > 4*n_expected:
        print("")
        print("WARNING: far more regions than bubbles.")
        print("The interface is fragmented: increase tol_factor.")
    #
    print("==============================================")
#

# ============================================================
# FRAGMENTED FRAME REPORT
#
# Printed only for frames where containment failed, i.e. where
# no closed region encloses the bubble's centre of mass. Shows
# how far apart the pieces are, measured in units of the LOCAL
# facet length rather than the global median, which is what
# distinguishes an AMR level jump from a uniform near miss.
# ============================================================
#
def report_fragmented_frame(regions,istep,Lx=0.,Ly=0.,max_pieces=6):
    #
    if len(regions) == 0:
        print(f"  step {istep:>9}: no regions at all")
        return
    #
    all_len = []
    #
    for region in regions:
        #
        segs = np.asarray(region["segments"],dtype=float)
        #
        if len(segs) == 0:
            continue
        #
        d = minimum_image(segs[:,1,:]-segs[:,0,:],Lx,Ly)
        all_len.append(np.sqrt(np.sum(d*d,axis=1)))
    #
    if len(all_len) == 0:
        return
    #
    all_len = np.concatenate(all_len)
    positive = all_len[all_len > 0.]
    #
    h_med = float(np.median(positive)) if len(positive) else 1.
    h_min = float(np.min(positive)) if len(positive) else 0.
    h_max = float(np.max(positive)) if len(positive) else 0.
    #
    order = sorted(
        range(len(regions)),
        key=lambda i:region_perimeter(regions[i],Lx,Ly),
        reverse=True,
    )
    #
    big = regions[order[0]]
    vb = np.asarray(big["vertices"],dtype=float)
    #
    gaps = []
    #
    for i in order[1:max_pieces]:
        #
        vo = np.asarray(regions[i]["vertices"],dtype=float)
        #
        dx = vb[:,None,0]-vo[None,:,0]
        dy = vb[:,None,1]-vo[None,:,1]
        #
        if Lx is not None and Lx > 0.:
            dx -= Lx*np.round(dx/Lx)
        #
        if Ly is not None and Ly > 0.:
            dy -= Ly*np.round(dy/Ly)
        #
        gaps.append(float(np.min(np.sqrt(dx*dx+dy*dy))))
    #
    print(
        f"  step {istep:>9}: {len(regions):>4} regions"
        f"  h_med={h_med:.5f} h_min={h_min:.5f} h_max={h_max:.5f}"
        f"  ratio={h_max/max(h_min,1.e-30):.1f}"
    )
    #
    if len(gaps) > 0:
        #
        print(
            "               gaps to largest piece, in h_med ="
            +" ".join(f"{g/h_med:.1f}" for g in gaps)
        )
        #
        print(
            "               gaps to largest piece, in h_max ="
            +" ".join(f"{g/max(h_max,1.e-30):.2f}" for g in gaps)
        )
#

# ============================================================
# CONTAINMENT FAILURE REPORT
#
# When a bubble centre is reported outside its own interface,
# the question is whether the point really is outside or the
# parity test is being corrupted. Per-ray crossing counts
# answer it: consistent even counts mean the point is outside,
# disagreement between rays means the interface is leaking.
# ============================================================
#
def report_containment_failure(
    regions,
    point,
    physical_id,
    istep,
    Lx=0.,
    Ly=0.,
    n_rays=5,
):
    #
    # Reports the NEAREST region, not the largest. With dozens
    # of regions in a frame, "outside the biggest one" says
    # nothing about whether the point belongs to a small blob
    # beside it.
    #
    if len(regions) == 0:
        return
    #
    def min_dist(region):
        #
        v = np.asarray(region["vertices"],dtype=float)
        #
        dx = v[:,0]-point[0]
        dy = v[:,1]-point[1]
        #
        if Lx is not None and Lx > 0.:
            dx -= Lx*np.round(dx/Lx)
        #
        if Ly is not None and Ly > 0.:
            dy -= Ly*np.round(dy/Ly)
        #
        return float(np.min(np.sqrt(dx*dx+dy*dy)))
    #
    dists = np.asarray([min_dist(r) for r in regions],dtype=float)
    #
    inear = int(np.argmin(dists))
    region = regions[inear]
    #
    segs = np.asarray(region["segments"],dtype=float)
    #
    segs_img = region_in_image_of(region,point,Lx,Ly)
    #
    counts = [
        count_crossings(segs_img,point,2.*np.pi*i/n_rays+0.37)
        for i in range(n_rays)
    ]
    #
    d_region = region_perimeter(region,Lx,Ly)/np.pi
    #
    print(
        f"  step {istep:>9} id={physical_id}"
        f"  point=({point[0]:.4f},{point[1]:.4f})"
    )
    #
    print(
        f"      nearest region #{inear}: dist={dists[inear]:.4f}"
        f"  nseg={len(segs)}  D_eq={d_region:.4f}"
    )
    #
    print(
        f"      crossings per ray (nearest) = {counts}"
        f"   (odd = inside)"
    )
    #
    # How many regions sit within one region-diameter: tells a
    # lone thin ligament apart from a point lost in a cloud of
    # neighbouring droplets.
    #
    close = int(np.count_nonzero(dists < max(d_region,1.e-12)))
    #
    print(
        f"      regions within one D_eq = {close}"
        f"   of {len(regions)}"
    )
#

# ============================================================
# CYCLE REPORT
#
# region_area_centroid returns zeros when it rejects a cycle,
# which hides WHY. This prints the raw numbers for every cycle
# in a region so the rejection can be judged.
# ============================================================
#
def report_region_cycles(region,Lx=0.,Ly=0.,label=""):
    #
    loops = region.get("loops")
    #
    if not loops:
        loop = region.get("loop")
        loops = [loop] if loop is not None else []
    #
    P_seg = region_perimeter(region,Lx,Ly)
    #
    print(
        f"  region {label}: nseg={region.get('nseg')}"
        f"  seg-perimeter={P_seg:.4f}  D_eq={P_seg/np.pi:.4f}"
        f"  cycles={len(loops)}"
    )
    #
    if not loops:
        print("     no closed cycle at all (open chain)")
        return
    #
    for k,loop in enumerate(loops):
        #
        if loop is None or len(loop) < 4:
            print(f"     cycle {k}: too short ({0 if loop is None else len(loop)} pts)")
            continue
        #
        A,cx,cy = polygon_area_centroid(loop)
        P = loop_perimeter(loop)
        #
        A_max = P*P/(4.*np.pi) if P > 0. else 0.
        ratio = A/A_max if A_max > 0. else 0.
        #
        span = np.ptp(loop,axis=0)
        #
        print(
            f"     cycle {k}: pts={len(loop):>5}"
            f"  perimeter={P:.4f}  area={A:.6f}"
            f"  A/Amax={ratio:.4f}"
        )
        #
        print(
            f"               centroid=({cx:.4f},{cy:.4f})"
            f"  bbox span=({span[0]:.4f},{span[1]:.4f})"
            f"  pts/seg={len(loop)/max(region.get('nseg',1),1):.2f}"
        )
#

# ============================================================
# RAW HISTORY VALIDATION
#
# Every record must carry one position per volume. When it does
# not, volumes and coordinates stop referring to the same
# objects, and everything downstream is matched on scrambled
# data: the genealogy assigns duplicate labels, bubbles swap
# identity for a step, and the movie faithfully paints the
# result.
#
# This is cheap and catches it at load time instead of leaving
# it to be inferred from a colour swap forty steps later.
# ============================================================
#
def validate_raw_history(event_history_raw,verbose=True,max_report=10):
    #
    bad = []
    #
    keys = (
        ("voll0","xpos0","ypos0","istep_p0"),
        ("voll1","xpos1","ypos1","istep_p1"),
    )
    #
    for ih,rec in enumerate(event_history_raw):
        #
        for kv,kx,ky,ks in keys:
            #
            if kv not in rec or kx not in rec:
                continue
            #
            nv = len(rec[kv])
            nx = len(rec[kx])
            ny = len(rec[ky]) if ky in rec else nx
            #
            if nx != nv or ny != nv:
                bad.append((ih,int(rec.get(ks,-1)),kv,nv,nx,ny))
    #
    if verbose and bad:
        #
        print("")
        print("==============================================")
        print("RAW HISTORY: ARRAY LENGTH MISMATCH")
        print("==============================================")
        print(len(bad),"record/side(s) where the number of positions")
        print("does not match the number of volumes.")
        print("")
        print(f"{'record':>8}{'istep':>9}{'side':>8}"
              f"{'nvol':>7}{'nx':>7}{'ny':>7}")
        #
        for row in bad[:max_report]:
            print(f"{row[0]:>8}{row[1]:>9}{row[2]:>8}"
                  f"{row[3]:>7}{row[4]:>7}{row[5]:>7}")
        #
        if len(bad) > max_report:
            print(f"   ... and {len(bad)-max_report} more")
        #
        print("")
        print("Volumes and positions no longer refer to the same")
        print("components at these steps. The genealogy matches on")
        print("position, so its labels there are unreliable -- this")
        print("is upstream of anything in the movie.")
        print("==============================================")
    #
    elif verbose:
        #
        print("raw history: array lengths consistent in all",
              len(event_history_raw),"records")
    #
    return bad

# ============================================================
# HISTORY ALIGNMENT
#
# clean_genealogy_from_raw_history seeds idx_to_id_history with
# the INITIAL state and then appends once per record:
#
#     idx_to_id_history[0]      state at istep_p0 of record 0
#     idx_to_id_history[ih+1]   state at istep_p1 of record ih
#
# so it holds len(records)+1 entries. Anything that pairs entry
# ih with record ih is reading the PREVIOUS step's labels.
#
# Use this instead of indexing the list directly.
# ============================================================
#
def history_offset(idx_to_id_history,event_history_raw):
    #
    n_rec = len(event_history_raw)
    #
    if len(idx_to_id_history) == n_rec+1:
        return 1
    #
    return 0
#
def ids_at_record(idx_to_id_history,event_history_raw,ih):
    #
    # The ID mapping in force at record ih's istep_p1.
    #
    k = ih+history_offset(idx_to_id_history,event_history_raw)
    #
    if k < 0 or k >= len(idx_to_id_history):
        return {}
    #
    return idx_to_id_history[k]
