//
// Relative tolerance for the untagged (orphan) volume/area warning in
// countDropsBubble_app_v2. Can be overridden before including this file.
#ifndef ORPHAN_TOL
# define ORPHAN_TOL 1e-6
#endif
//
static void restriction_tag_max (Point point, scalar s) {
  double m = 0.;
  foreach_child()
    if (s[] > m) m = s[];
  s[] = m;
}
//
// Returns the (0-based) index of the tag of m carried by the closest
// tagged cell in the 3x3(x3) neighbourhood, or -1 if there is none.
static int nearest_tag (Point point, scalar m, int n)
{
  coord o = {x, y, z};
  double dmin = HUGE;
  int tag = 0;
  foreach_neighbor(1)
    if (m[] > 0 && (int)m[] <= n) {
      double d2 = sq(x - o.x) + sq(y - o.y) + sq(z - o.z);
      if (d2 < dmin) {
        dmin = d2;
        tag = (int)m[];
      }
    }
  return tag - 1;
}
//
// ============================================================
// countDropsBubble_app_v1
// Original formulation:
//  - volume, centroid and velocity summed over the tagged cells only
//    (c >= 0.5 for droplets, c < 0.5 for bubbles; no nearest_tag)
//  - first-order interfacial volume and momentum error estimates only
//  - collective (MPI-safe) allocation check
// Output columns (0-based) in name_2 / name_3:
//   0 t, 1 istep, 2 j, 3 V, 4 A, 5-7 centroid, 8-10 velocity,
//   11-13 dx min/max/avg, 14-16 level min/max/avg, 17 ncells,
//   18 dV_err (1st order), 19 dP_err (1st order)
// ============================================================
//
void countDropsBubble_app_v1 (char *name_1, char *name_2, char *name_3,
                              int istep, scalar c, vector u) {

  scalar m1 = new scalar; // droplets
  scalar m2 = new scalar; // bubbles

  foreach() {
    m1[] = (c[] >= 0.5);
    m2[] = (c[] <  0.5);
  }

  int n1 = tag(m1);
  int n2 = tag(m2);

  for (scalar s in {m1, m2}) {
    s.restriction  = restriction_tag_max; // parent takes a (nonzero) child tag
    s.prolongation = refine_injection;    // halo children inherit the coarse leaf's tag
  }
  boundary ({m1, m2}); // redo restriction + halos with the new operators (also fills MPI ghosts)

  // Volume information
  double *v1 = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *v2 = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // centroid information
  double *c1x = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *c1y = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *c1z = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;

  double *c2x = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *c2y = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *c2z = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Velocity information
  double *u1x = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *u1y = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *u1z = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;

  double *u2x = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *u2y = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *u2z = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Interfacial area
  double *area1 = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *area2 = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Interfacial volume errors - first order
  double *dA1 = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *dA2 = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Interfacial momentum errors - first order
  double *dP1 = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *dP2 = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // AMR/grid information
  double *dx1_min = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *dx1_max = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *dx1_avg = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *lev1_min = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *lev1_max = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *lev1_avg = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *ncells1  = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;

  double *dx2_min = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *dx2_max = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *dx2_avg = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *lev2_min = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *lev2_max = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *lev2_avg = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *ncells2  = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Allocation check: decided collectively so all ranks leave together (no MPI deadlock)
  int alloc_ok = !((n1 > 0 && (!v1 || !c1x || !c1y || !c1z || !u1x || !u1y || !u1z || !area1 ||
                  !dA1 || !dP1 || !dx1_min || !dx1_max || !dx1_avg ||
                  !lev1_min || !lev1_max || !lev1_avg || !ncells1)) ||
      (n2 > 0 && (!v2 || !c2x || !c2y || !c2z || !u2x || !u2y || !u2z || !area2 ||
                  !dA2 || !dP2 || !dx2_min || !dx2_max || !dx2_avg ||
                  !lev2_min || !lev2_max || !lev2_avg || !ncells2)));

  if (!alloc_ok) {
    fprintf(stderr, "countDropsBubble_app_v1: allocation failed on pid %d\n", pid());
    fflush(stderr);
  }
#ifdef _MPI
  MPI_Allreduce(MPI_IN_PLACE, &alloc_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif

  if (!alloc_ok) {

    free(v1); free(v2);

    free(c1x); free(c1y); free(c1z);
    free(c2x); free(c2y); free(c2z);

    free(u1x); free(u1y); free(u1z);
    free(u2x); free(u2y); free(u2z);

    free(area1); free(area2);
    free(dA1); free(dA2);
    free(dP1); free(dP2);

    free(dx1_min); free(dx1_max); free(dx1_avg);
    free(lev1_min); free(lev1_max); free(lev1_avg); free(ncells1);

    free(dx2_min); free(dx2_max); free(dx2_avg);
    free(lev2_min); free(lev2_max); free(lev2_avg); free(ncells2);

    delete({m1, m2});
    return;
  }

  // Initialise minima only after the allocation check (no NULL dereference)
  for (int j = 0; j < n1; j++) {
    dx1_min[j] = HUGE;
    lev1_min[j] = HUGE;
  }

  for (int j = 0; j < n2; j++) {
    dx2_min[j] = HUGE;
    lev2_min[j] = HUGE;
  }

  // Loops to compute volumetric quantities (tagged cells only)
  foreach(serial) {
    coord o = {x, y, z};

    if (m1[] > 0) {
      int j = (int)m1[] - 1;
      if (j >= 0 && j < n1) {
        double vol = dv()*c[];

        v1[j]  += vol;

        c1x[j] += vol*o.x;
        c1y[j] += vol*o.y;
        c1z[j] += vol*o.z;

        u1x[j] += vol*u.x[];
        u1y[j] += vol*u.y[];
        u1z[j] += vol*u.z[];

        dx1_min[j] = min(dx1_min[j], Delta);
        dx1_max[j] = max(dx1_max[j], Delta);
        dx1_avg[j] += vol*Delta;

        lev1_min[j] = min(lev1_min[j], (double)level);
        lev1_max[j] = max(lev1_max[j], (double)level);
        lev1_avg[j] += vol*(double)level;

        ncells1[j] += 1.;
      }
    }

    if (m2[] > 0) {
      int j = (int)m2[] - 1;
      if (j >= 0 && j < n2) {
        double vol = dv()*(1. - c[]);

        v2[j]  += vol;

        c2x[j] += vol*o.x;
        c2y[j] += vol*o.y;
        c2z[j] += vol*o.z;

        u2x[j] += vol*u.x[];
        u2y[j] += vol*u.y[];
        u2z[j] += vol*u.z[];

        dx2_min[j] = min(dx2_min[j], Delta);
        dx2_max[j] = max(dx2_max[j], Delta);
        dx2_avg[j] += vol*Delta;

        lev2_min[j] = min(lev2_min[j], (double)level);
        lev2_max[j] = max(lev2_max[j], (double)level);
        lev2_avg[j] += vol*(double)level;

        ncells2[j] += 1.;
      }
    }
  }

  // Interface quantities for bubbles/droplets using existing m1, m2 tags
  foreach(serial) {
    if (c[] > 1e-6 && c[] < 1. - 1e-6) {

      // Calculation of interfacial quantities
      coord n = interface_normal(point, c), p;
      double alpha = plane_alpha(c[], n);
      double Ac = pow(Delta, dimension - 1)*plane_area_center(n, alpha, &p);
      double vel2 = 0.;
      foreach_dimension() {
        vel2 += sq(u.x[]);
      }

      // First-order error contribution
      double dVc = Ac*Delta;
      double dPc = sqrt(vel2)*dVc;

      // Define PLIC coordinates in the interfacial cells
      coord oi = {x,y,z};
      foreach_dimension() {
        oi.x += p.x*Delta;
      }

      // Owner of the interfacial cell: closest tagged cell in the stencil
      int tag1 = 0;
      int tag2 = 0;
      double d1min = HUGE;
      double d2min = HUGE;
      foreach_neighbor(1) { // we use the 3x3x3 stencil to avoid ownership issue
        double dist2 = 0.;
        coord oc = {x,y,z};
        foreach_dimension() {
          dist2 += sq(oc.x - oi.x);
        }
        if (m1[] > 0. && dist2 < d1min) {
          int candidate = (int)m1[];
          if (candidate >= 1 && candidate <= n1) {
            d1min = dist2;
            tag1 = candidate;
          }
        }
        if (m2[] > 0. && dist2 < d2min) {
          int candidate = (int)m2[];
          if (candidate >= 1 && candidate <= n2) {
            d2min = dist2;
            tag2 = candidate;
          }
        }
      }

      // Droplet owners
      if (tag1 >= 1 && tag1 <= n1) {
        int q = tag1 - 1;
        area1[q] += Ac;
        dA1[q]   += dVc;
        dP1[q]   += dPc;
      }

      // Bubble owners
      if (tag2 >= 1 && tag2 <= n2) {
        int q = tag2 - 1;
        area2[q] += Ac;
        dA2[q]   += dVc;
        dP2[q]   += dPc;
      }

    }
  }

  // delete temporary array
  delete({m1, m2});

#ifdef _MPI
  if (n1 > 0) {
    MPI_Allreduce(MPI_IN_PLACE, v1,  n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, c1x, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, c1y, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, c1z, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, u1x, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u1y, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u1z, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, dx1_min, n1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dx1_max, n1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dx1_avg, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, lev1_min, n1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, lev1_max, n1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, lev1_avg, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, ncells1, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, area1, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dA1, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dP1, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  if (n2 > 0) {
    MPI_Allreduce(MPI_IN_PLACE, v2,  n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, c2x, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, c2y, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, c2z, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, u2x, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u2y, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u2z, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, dx2_min, n2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dx2_max, n2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dx2_avg, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, lev2_min, n2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, lev2_max, n2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, lev2_avg, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, ncells2, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, area2, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dA2, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dP2, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
#endif

  if (pid() == 0) {
    FILE *ftot = fopen(name_1, "a");
    if (!ftot) {
      perror(name_1);
    } else {
      fprintf(ftot, "%.10e %d %d %d\n",
              t, istep, (int)(n1 - 1), (int)(n2 - 1));
      fclose(ftot);
    }

    FILE *fdrop = fopen(name_2, "a");
    if (!fdrop) {
      perror(name_2);
    } else {
      for (int j = 0; j < n1; j++) {
        if (v1[j] > 0.) {
          fprintf(fdrop,
                  "%.10e %d %d %.10e %.10e "
                  "%.10e %.10e %.10e "
                  "%.10e %.10e %.10e "
                  "%.10e %.10e %.10e "
                  "%.10e %.10e %.10e %.10e "
                  "%.10e %.10e\n",
                  t, istep, j, v1[j], area1[j],
                  c1x[j]/v1[j], c1y[j]/v1[j], c1z[j]/v1[j],
                  u1x[j]/v1[j], u1y[j]/v1[j], u1z[j]/v1[j],
                  dx1_min[j], dx1_max[j], dx1_avg[j]/v1[j],
                  lev1_min[j], lev1_max[j], lev1_avg[j]/v1[j], ncells1[j],
                  dA1[j], dP1[j]);
        }
      }
      fclose(fdrop);
    }

    FILE *fbubb = fopen(name_3, "a");
    if (!fbubb) {
      perror(name_3);
    } else {
      for (int j = 0; j < n2; j++) {
        if (v2[j] > 0.) {
          fprintf(fbubb,
                  "%.10e %d %d %.10e %.10e "
                  "%.10e %.10e %.10e "
                  "%.10e %.10e %.10e "
                  "%.10e %.10e %.10e "
                  "%.10e %.10e %.10e %.10e "
                  "%.10e %.10e\n",
                  t, istep, j, v2[j], area2[j],
                  c2x[j]/v2[j], c2y[j]/v2[j], c2z[j]/v2[j],
                  u2x[j]/v2[j], u2y[j]/v2[j], u2z[j]/v2[j],
                  dx2_min[j], dx2_max[j], dx2_avg[j]/v2[j],
                  lev2_min[j], lev2_max[j], lev2_avg[j]/v2[j], ncells2[j],
                  dA2[j], dP2[j]);
        }
      }
      fclose(fbubb);
    }
  }

  free(v1); free(v2);
  free(area1); free(area2);

  free(c1x); free(c1y); free(c1z);
  free(c2x); free(c2y); free(c2z);

  free(u1x); free(u1y); free(u1z);
  free(u2x); free(u2y); free(u2z);

  free(dx1_min); free(dx1_max); free(dx1_avg);
  free(lev1_min); free(lev1_max); free(lev1_avg); free(ncells1);
  free(dA1); free(dP1);

  free(dx2_min); free(dx2_max); free(dx2_avg);
  free(lev2_min); free(lev2_max); free(lev2_avg); free(ncells2);
  free(dA2); free(dP2);
}
//
// ============================================================
// countDropsBubble_app_v2
// Current formulation:
//  - sub-threshold interfacial volume attributed with nearest_tag
//  - first-order and blended (Eq. 5) volume error estimates
//  - centroid error estimate: first-moment bound (paper, Eq. 9)
// All error estimates are written WITHOUT calibration constants, i.e.
// they are the hatted quantities of the paper: multiply by C_V (volume)
// or C_X (position) in post-processing.
// Local interface error (paper, Eq. 5 and Eq. 9):
//   |delta_c| = Delta*min(|kappa_c| Delta, D - 1),
//   with |kappa_c| Delta replaced by D - 1 where kappa is nodata.
// Output columns (0-based) in name_2 / name_3:
//   0 t, 1 istep, 2 j, 3 V, 4 A, 5-7 centroid, 8-10 velocity,
//   11-13 dx min/max/avg, 14-16 level min/max/avg, 17 ncells,
//   18 dV_err (1st, Eq. 3), 19 dV_err (blended, Eq. 5),
//   20 dX_err = (1/V) sum A_c |delta_c| |x_c - x_i|   (Eq. 9)
// ============================================================
//
void countDropsBubble_app_v2 (char *name_1, char *name_2, char *name_3,
                              int istep, scalar c, vector u) {

  scalar m1 = new scalar; // droplets
  scalar m2 = new scalar; // bubbles

  foreach() {
    m1[] = (c[] >= 0.5);
    m2[] = (c[] <  0.5);
  }

  int n1 = tag(m1);
  int n2 = tag(m2);

  for (scalar s in {m1, m2}) {
    s.restriction  = restriction_tag_max; // parent takes a (nonzero) child tag
    s.prolongation = refine_injection;    // halo children inherit the coarse leaf's tag
  }
  boundary ({m1, m2}); // redo restriction + halos with the new operators (also fills MPI ghosts)

  // Volume information
  double *v1 = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *v2 = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // centroid information
  double *c1x = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *c1y = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *c1z = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;

  double *c2x = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *c2y = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *c2z = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Velocity information
  double *u1x = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *u1y = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *u1z = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;

  double *u2x = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *u2y = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *u2z = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Interfacial area
  double *area1 = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *area2 = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  
  // Volume errors - first order (Eq. 3)
  double *dA1_1 = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *dA2_1 = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Volume errors - blended (Eq. 5)
  double *dA1_2 = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *dA2_2 = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Centroid position error (paper, Eq. 9): sum A_c |delta_c| |x_c - x_i|
  double *xe1 = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *xe2 = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // AMR/grid information
  double *dx1_min = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *dx1_max = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *dx1_avg = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *lev1_min = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *lev1_max = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *lev1_avg = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *ncells1  = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;

  double *dx2_min = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *dx2_max = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *dx2_avg = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *lev2_min = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *lev2_max = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *lev2_avg = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *ncells2  = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Allocation check: decided collectively so all ranks leave together (no MPI deadlock)
  int alloc_ok = !((n1 > 0 && (!v1 || !c1x || !c1y || !c1z || !u1x || !u1y || !u1z || !area1 ||
                  !dA1_1 || !dA1_2 || !dx1_min || !dx1_max || !dx1_avg ||
                  !lev1_min || !lev1_max || !lev1_avg || !ncells1 || !xe1)) ||
      (n2 > 0 && (!v2 || !c2x || !c2y || !c2z || !u2x || !u2y || !u2z || !area2 ||
                  !dA2_1 || !dA2_2 || !dx2_min || !dx2_max || !dx2_avg ||
                  !lev2_min || !lev2_max || !lev2_avg || !ncells2 || !xe2)));

  if (!alloc_ok) {
    fprintf(stderr, "countDropsBubble_app_v2: allocation failed on pid %d\n", pid());
    fflush(stderr);
  }
#ifdef _MPI
  MPI_Allreduce(MPI_IN_PLACE, &alloc_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif

  if (!alloc_ok) {

    free(v1); free(v2);

    free(c1x); free(c1y); free(c1z);
    free(c2x); free(c2y); free(c2z);

    free(u1x); free(u1y); free(u1z);
    free(u2x); free(u2y); free(u2z);
    
    free(area1); free(area2);
    free(dA1_1); free(dA2_1);
    free(dA1_2); free(dA2_2);

    free(dx1_min); free(dx1_max); free(dx1_avg);
    free(lev1_min); free(lev1_max); free(lev1_avg); free(ncells1);

    free(dx2_min); free(dx2_max); free(dx2_avg);
    free(lev2_min); free(lev2_max); free(lev2_avg); free(ncells2);

    free(xe1); free(xe2);

    delete({m1, m2});
    return;
  }

  // Initialise minima only after the allocation check (no NULL dereference)
  for (int j = 0; j < n1; j++) {
    dx1_min[j] = HUGE;
    lev1_min[j] = HUGE;
  }

  for (int j = 0; j < n2; j++) {
    dx2_min[j] = HUGE;
    lev2_min[j] = HUGE;
  }

  // Loops to compute volumetric quantities.
  // The tags are built with the c = 0.5 threshold (robust topology), but
  // interfacial cells on the "wrong" side of 0.5 still hold VOF volume of
  // the other phase. Their contribution is given to the closest tagged
  // neighbour so that the full VOF volume is accounted for. Without this,
  // the missing volume scales like A*Delta, i.e. a first-order error.
  double orphan1 = 0., orphan2 = 0.; // volume with no tagged neighbour
  foreach(serial) {
    coord o = {x, y, z};

    // droplets: every cell with c > 0
    if (c[] > 0.) {
      int j = (m1[] > 0) ? (int)m1[] - 1 : nearest_tag (point, m1, n1);
      double vol = dv()*c[];
      if (j >= 0 && j < n1) {
        v1[j]  += vol;

        c1x[j] += vol*o.x;
        c1y[j] += vol*o.y;
        c1z[j] += vol*o.z;

        u1x[j] += vol*u.x[];
        u1y[j] += vol*u.y[];
        u1z[j] += vol*u.z[];

        if (m1[] > 0) { // grid statistics only over the tagged cells
          dx1_min[j] = min(dx1_min[j], Delta);
          dx1_max[j] = max(dx1_max[j], Delta);
          lev1_min[j] = min(lev1_min[j], (double)level);
          lev1_max[j] = max(lev1_max[j], (double)level);
          ncells1[j] += 1.;
        }
        dx1_avg[j]  += vol*Delta;
        lev1_avg[j] += vol*(double)level;
      }
      else
        orphan1 += vol;
    }

    // bubbles: every cell with c < 1
    if (c[] < 1.) {
      int j = (m2[] > 0) ? (int)m2[] - 1 : nearest_tag (point, m2, n2);
      double vol = dv()*(1. - c[]);
      if (j >= 0 && j < n2) {
        v2[j]  += vol;

        c2x[j] += vol*o.x;
        c2y[j] += vol*o.y;
        c2z[j] += vol*o.z;

        u2x[j] += vol*u.x[];
        u2y[j] += vol*u.y[];
        u2z[j] += vol*u.z[];

        if (m2[] > 0) {
          dx2_min[j] = min(dx2_min[j], Delta);
          dx2_max[j] = max(dx2_max[j], Delta);
          lev2_min[j] = min(lev2_min[j], (double)level);
          lev2_max[j] = max(lev2_max[j], (double)level);
          ncells2[j] += 1.;
        }
        dx2_avg[j]  += vol*Delta;
        lev2_avg[j] += vol*(double)level;
      }
      else
        orphan2 += vol;
    }
  }
#ifdef _MPI
  MPI_Allreduce(MPI_IN_PLACE, &orphan1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &orphan2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

  // The centroid error estimates need each droplet's centroid inside the
  // interface loop, so volume and centroid moments are reduced here.
#ifdef _MPI
  if (n1 > 0) {
    MPI_Allreduce(MPI_IN_PLACE, v1,  n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, c1x, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, c1y, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, c1z, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
  if (n2 > 0) {
    MPI_Allreduce(MPI_IN_PLACE, v2,  n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, c2x, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, c2y, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, c2z, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
#endif

  // Volume conservation check: orphan/(tagged + orphan) is the fraction of
  // the VOF volume that could not be assigned to any tag (debris and
  // droplets/bubbles too small to be tagged with the 0.5 threshold).
  {
    double tot1 = 0., tot2 = 0.;
    for (int j = 0; j < n1; j++) tot1 += v1[j];
    for (int j = 0; j < n2; j++) tot2 += v2[j];
    double frac1 = (tot1 + orphan1 > 0.) ? orphan1/(tot1 + orphan1) : 0.;
    double frac2 = (tot2 + orphan2 > 0.) ? orphan2/(tot2 + orphan2) : 0.;
    if (pid() == 0 && (frac1 > ORPHAN_TOL || frac2 > ORPHAN_TOL))
      fprintf(stderr, "countDropsBubble_app_v2: t = %g, untagged volume "
              "%g (fraction %g) drops, %g (fraction %g) bubbles\n",
              t, orphan1, frac1, orphan2, frac2);
  }

  // Interface quantities for bubbles/droplets using existing m1, m2 tags
  scalar kappa[];
  curvature (c, kappa);
  double orphanA1 = 0., orphanA2 = 0.; // interfacial area with no tagged neighbour
  foreach(serial) {
    if (c[] > 1e-6 && c[] < 1. - 1e-6) {
 
      // Calculation of interfacial quantities 
      coord n = interface_normal(point, c), p;
      double alpha = plane_alpha(c[], n);
      double Ac = pow(Delta, dimension - 1)*plane_area_center(n, alpha, &p);
      // First-order error contribution (paper, Eq. 3, without C_V)
      double dVc_1 = Ac*Delta;

      // Blended error contribution (paper, Eq. 5, without C_V):
      //   |delta_c| = Delta*min(|kappa| Delta, D - 1).
      // Where kappa is undefined, |kappa| Delta is replaced by D - 1, i.e.
      // the saturated value of the clipped branch, so the estimate is
      // continuous between the two branches.
      double kD = (kappa[] != nodata) ?
        min(fabs(kappa[])*Delta, dimension - 1.) : dimension - 1.;
      double delta_c = Delta*kD;
      double dVc_2 = Ac*delta_c;
      
      // Define PLIC coordinates in the interfacial cells 
      coord oi = {x,y,z};
      foreach_dimension() {
	oi.x += p.x*Delta;
      }
  
      // Define the points not at the PLIC interface where 0<=c<=1  
      int tag1 = 0;
      int tag2 = 0;
      double d1min = HUGE;
      double d2min = HUGE;
      foreach_neighbor(1) { // we use the 3x3x3 stencil to avoid ownership issue
        double dist2 = 0.;
        coord oc = {x,y,z};
	foreach_dimension() {
	  dist2 += sq(oc.x - oi.x);
	}
        if (m1[] > 0. && dist2 < d1min) {
          int candidate = (int)m1[];
          if (candidate >= 1 && candidate <= n1) {
            d1min = dist2;
            tag1 = candidate;
          }
        }
        if (m2[] > 0. && dist2 < d2min) {
          int candidate = (int)m2[];
          if (candidate >= 1 && candidate <= n2) {
            d2min = dist2;
            tag2 = candidate;
          }
        }
      }
  
      // Droplet owners 
      if (tag1 >= 1 && tag1 <= n1) {
        int q = tag1 - 1;
        area1[q]  += Ac;
        dA1_1[q]  += dVc_1;
        dA1_2[q]  += dVc_2;
        if (v1[q] > 0.) { // paper, Eq. 9: A_c |delta_c| |x_c - x_i|
          coord xc = {c1x[q]/v1[q], c1y[q]/v1[q], c1z[q]/v1[q]};
          double r = sqrt(sq(oi.x - xc.x) + sq(oi.y - xc.y) + sq(oi.z - xc.z));
          xe1[q] += dVc_2*r;
        }
      }
      else
        orphanA1 += Ac;
  
      // Bubble owners 
      if (tag2 >= 1 && tag2 <= n2) {
        int q = tag2 - 1;
        area2[q]  += Ac;
        dA2_1[q]  += dVc_1;
        dA2_2[q]  += dVc_2;
        if (v2[q] > 0.) { // paper, Eq. 9: A_c |delta_c| |x_c - x_i|
          coord xc = {c2x[q]/v2[q], c2y[q]/v2[q], c2z[q]/v2[q]};
          double r = sqrt(sq(oi.x - xc.x) + sq(oi.y - xc.y) + sq(oi.z - xc.z));
          xe2[q] += dVc_2*r;
        }
      }
      else
        orphanA2 += Ac;

    }
  }

  // delete temporary array 
  delete({m1, m2});

#ifdef _MPI
  if (n1 > 0) {
    MPI_Allreduce(MPI_IN_PLACE, u1x, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u1y, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u1z, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, dx1_min, n1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dx1_max, n1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dx1_avg, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, lev1_min, n1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, lev1_max, n1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, lev1_avg, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, ncells1, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, area1, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dA1_1, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dA1_2, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, xe1, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  if (n2 > 0) {
    MPI_Allreduce(MPI_IN_PLACE, u2x, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u2y, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u2z, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, dx2_min, n2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dx2_max, n2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dx2_avg, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, lev2_min, n2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, lev2_max, n2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, lev2_avg, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, ncells2, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, area2, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dA2_1, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dA2_2, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, xe2, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
#endif

  // Area conservation check, same idea as for the volume
#ifdef _MPI
  MPI_Allreduce(MPI_IN_PLACE, &orphanA1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &orphanA2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  {
    double totA1 = 0., totA2 = 0.;
    for (int j = 0; j < n1; j++) totA1 += area1[j];
    for (int j = 0; j < n2; j++) totA2 += area2[j];
    double fracA1 = (totA1 + orphanA1 > 0.) ? orphanA1/(totA1 + orphanA1) : 0.;
    double fracA2 = (totA2 + orphanA2 > 0.) ? orphanA2/(totA2 + orphanA2) : 0.;
    if (pid() == 0 && (fracA1 > ORPHAN_TOL || fracA2 > ORPHAN_TOL))
      fprintf(stderr, "countDropsBubble_app_v2: t = %g, untagged area "
              "%g (fraction %g) drops, %g (fraction %g) bubbles\n",
              t, orphanA1, fracA1, orphanA2, fracA2);
  }
  

  if (pid() == 0) {
    FILE *ftot = fopen(name_1, "a");
    if (!ftot) {
      perror(name_1);
    } else {
      fprintf(ftot, "%.10e %d %d %d\n",
              t, istep, (int)(n1 - 1), (int)(n2 - 1));
      fclose(ftot);
    }

    FILE *fdrop = fopen(name_2, "a");
    if (!fdrop) {
      perror(name_2);
    } else {
      for (int j = 0; j < n1; j++) {
        if (v1[j] > 0.) {
          fprintf(fdrop,
                  "%.10e %d %d %.10e %.10e "
                  "%.10e %.10e %.10e "
                  "%.10e %.10e %.10e "
                  "%.10e %.10e %.10e "
                  "%.10e %.10e %.10e %.10e "
                  "%.10e %.10e "
                  "%.10e\n",
                  t, istep, j, v1[j], area1[j], 
                  c1x[j]/v1[j], c1y[j]/v1[j], c1z[j]/v1[j],
                  u1x[j]/v1[j], u1y[j]/v1[j], u1z[j]/v1[j],
                  dx1_min[j], dx1_max[j], dx1_avg[j]/v1[j],
                  lev1_min[j], lev1_max[j], lev1_avg[j]/v1[j], ncells1[j],
                  dA1_1[j], dA1_2[j],
                  xe1[j]/v1[j]);
        }
      }
      fclose(fdrop);
    }

    FILE *fbubb = fopen(name_3, "a");
    if (!fbubb) {
      perror(name_3);
    } else {
      for (int j = 0; j < n2; j++) {
        if (v2[j] > 0.) {
          fprintf(fbubb,
                  "%.10e %d %d %.10e %.10e "
                  "%.10e %.10e %.10e "
                  "%.10e %.10e %.10e "
                  "%.10e %.10e %.10e "
                  "%.10e %.10e %.10e %.10e "
                  "%.10e %.10e "
                  "%.10e\n",
                  t, istep, j, v2[j], area2[j],
                  c2x[j]/v2[j], c2y[j]/v2[j], c2z[j]/v2[j],
                  u2x[j]/v2[j], u2y[j]/v2[j], u2z[j]/v2[j],
                  dx2_min[j], dx2_max[j], dx2_avg[j]/v2[j],
                  lev2_min[j], lev2_max[j], lev2_avg[j]/v2[j], ncells2[j],
                  dA2_1[j], dA2_2[j],
                  xe2[j]/v2[j]);
        }
      }
      fclose(fbubb);
    }
  }

  /*
  for (int j = 0; j < n2; j++) {
    if (v2[j] > 0.) {
      fprintf(stderr, "Area %.10e\n", area2[j]);
    }
  }
  */

  free(v1); free(v2);
  free(area1); free(area2);

  free(c1x); free(c1y); free(c1z);
  free(c2x); free(c2y); free(c2z);

  free(u1x); free(u1y); free(u1z);
  free(u2x); free(u2y); free(u2z);

  free(dx1_min); free(dx1_max); free(dx1_avg);
  free(lev1_min); free(lev1_max); free(lev1_avg); free(ncells1);
  free(dA1_1); free(dA1_2);

  free(dx2_min); free(dx2_max); free(dx2_avg);
  free(lev2_min); free(lev2_max); free(lev2_avg); free(ncells2);
  free(dA2_1); free(dA2_2);
  free(xe1); free(xe2);
  
}

