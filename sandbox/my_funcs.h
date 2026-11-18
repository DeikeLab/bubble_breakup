//
// Relative tolerance for the untagged (orphan) volume/area warning in
// countDropsBubble_app_v2. Can be overridden before including this file.
#ifndef ORPHAN_TOL
# define ORPHAN_TOL 1e-6
#endif
//
// Axisymmetric runs (axi.h), countDropsBubble_app_v2 only: dv() is the
// volume per radian, and the interface area and the error estimates are
// weighted in the same way. Volume, area and volume errors are written
// multiplied by 2 pi (values of the 3D DPS), the centroid and the velocity
// of each DPS lie on the axis (y = 0), and the curvature of an interface
// cell is bounded with the 3D value D - 1 = 2.
#if AXI
# define AXI_FACTOR (2.*pi)
# define D_PHYS 3.
#else
# define AXI_FACTOR 1.
# define D_PHYS dimension
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
// Per-DPS interface curvature statistics (height-function kappa of
// curvature.h), accumulated over the interfacial cells owned by each DPS.
// Sign: positive for a convex DPS (sphere of radius R: kappa = 2/R for both
// droplets and bubbles; bubble values are -kappa(c) since c is the liquid).
// Output, 8 columns per DPS:
//   kmax, x, y, z of the cell with kmax  (thin neck: kmax ~ 1/r_neck),
//   kmin, area-weighted mean kappa, area-weighted rms kappa,
//   fraction of the interface area with |kappa| Delta > KAPPA_HI
//   (only cells where kappa is defined enter the means; kmax = kmin = 0
//    and the means are 0 if no cell of the DPS has a defined kappa).
// ============================================================
#ifndef KAPPA_HI
# define KAPPA_HI 0.3
#endif
typedef struct {
  double *kmax, *kmin, *ks, *kq, *akap, *ahi, *px, *py, *pz;
} KStats;

static int kstats_alloc (KStats *s, int n)
{
  s->kmax = s->kmin = s->ks = s->kq = s->akap = s->ahi = s->px = s->py = s->pz = NULL;
  if (n <= 0)
    return 1;
  s->kmax = malloc(n*sizeof(double)); s->kmin = malloc(n*sizeof(double));
  s->ks   = calloc(n, sizeof(double)); s->kq   = calloc(n, sizeof(double));
  s->akap = calloc(n, sizeof(double)); s->ahi  = calloc(n, sizeof(double));
  s->px   = calloc(n, sizeof(double)); s->py   = calloc(n, sizeof(double));
  s->pz   = calloc(n, sizeof(double));
  if (!s->kmax || !s->kmin || !s->ks || !s->kq || !s->akap || !s->ahi ||
      !s->px || !s->py || !s->pz)
    return 0;
  for (int j = 0; j < n; j++) {
    s->kmax[j] = -HUGE;
    s->kmin[j] =  HUGE;
  }
  return 1;
}

static void kstats_free (KStats *s)
{
  free(s->kmax); free(s->kmin); free(s->ks); free(s->kq);
  free(s->akap); free(s->ahi); free(s->px); free(s->py); free(s->pz);
}

static void kstats_add (KStats *s, int q, double k, double Ac, double Delta_, coord oi)
{
  if (k > s->kmax[q]) {
    s->kmax[q] = k;
    s->px[q] = oi.x; s->py[q] = oi.y; s->pz[q] = oi.z;
  }
  if (k < s->kmin[q])
    s->kmin[q] = k;
  s->ks[q]   += Ac*k;
  s->kq[q]   += Ac*k*k;
  s->akap[q] += Ac;
  if (fabs(k)*Delta_ > KAPPA_HI)
    s->ahi[q] += Ac;
}

static void kstats_reduce (KStats *s, int n)
{
#ifdef _MPI
  if (n <= 0)
    return;
  // location of the maximum: the owning rank keeps it, the others zero it
  struct { double v; int r; } *loc = malloc(n*sizeof(*loc));
  for (int j = 0; j < n; j++) {
    loc[j].v = s->kmax[j];
    loc[j].r = pid();
  }
  MPI_Allreduce(MPI_IN_PLACE, loc, n, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  for (int j = 0; j < n; j++) {
    s->kmax[j] = loc[j].v;
    if (loc[j].r != pid())
      s->px[j] = s->py[j] = s->pz[j] = 0.;
  }
  free(loc);
  MPI_Allreduce(MPI_IN_PLACE, s->px,   n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, s->py,   n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, s->pz,   n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, s->kmin, n, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, s->ks,   n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, s->kq,   n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, s->akap, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, s->ahi,  n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}

// writes the 8 columns (leading space, no newline); area = total DPS area
static void kstats_fprint (FILE *fp, KStats *s, int j, double area)
{
  int ok = s->akap[j] > 0.;
  fprintf(fp, " %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e",
          ok ? s->kmax[j] : 0., s->px[j], s->py[j], s->pz[j],
          ok ? s->kmin[j] : 0.,
          ok ? s->ks[j]/s->akap[j] : 0.,
          ok ? sqrt(s->kq[j]/s->akap[j]) : 0.,
          area > 0. ? s->ahi[j]/area : 0.);
}
//
//
// ============================================================
// countDropsBubble_app_v1
// Original formulation:
//  - volume, centroid and velocity summed over the tagged cells only
//    (c >= 0.5 for droplets, c < 0.5 for bubbles; no nearest_tag)
//  - first-order interfacial volume error estimate only
//  - collective (MPI-safe) allocation check
// Output columns (0-based) in name_2 / name_3 (only what the tracking uses):
//   0 t, 1 istep, 2 j, 3 V, 4 A, 5-7 centroid, 8-10 velocity,
//   11 dx max, 12 dV_err (1st order),
//   13 extent = max over the interfacial cells of |x_c - x_i| + sqrt(D)/2 Delta,
//      i.e. the radius of the smallest sphere centred at the centroid
//      that contains the DPS (contact test of the tracking)
//   14-19 central second moments per unit volume, (1/V) sum vol (x_i - xc_i)(x_j - xc_j):
//      xx, yy, zz, xy, xz, yz (sphere of radius R: xx = yy = zz = R^2/5)
//   20-27 curvature statistics (see KStats above): kmax, x, y, z of kmax,
//      kmin, mean kappa, rms kappa, area fraction with |kappa| Delta > KAPPA_HI
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

  // Extent: max |x_c - x_i| + sqrt(D)/2 Delta over the interfacial cells
  double *ext1 = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *ext2 = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // AMR/grid information
  double *dx1_max = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;

  double *dx2_max = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Central second moments (volume weighted): sum vol (x_i - xc_i)(x_j - xc_j)
  double *s1xx = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *s1yy = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *s1zz = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *s1xy = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *s1xz = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *s1yz = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;

  double *s2xx = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *s2yy = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *s2zz = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *s2xy = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *s2xz = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *s2yz = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Interface curvature statistics
  KStats k1s, k2s;
  int kok = kstats_alloc(&k1s, n1) & kstats_alloc(&k2s, n2);

  // Allocation check: decided collectively so all ranks leave together (no MPI deadlock)
  int alloc_ok = kok && !((n1 > 0 && (!v1 || !c1x || !c1y || !c1z || !u1x || !u1y || !u1z || !area1 ||
                  !dA1 || !dx1_max || !ext1 ||
                  !s1xx || !s1yy || !s1zz || !s1xy || !s1xz || !s1yz)) ||
      (n2 > 0 && (!v2 || !c2x || !c2y || !c2z || !u2x || !u2y || !u2z || !area2 ||
                  !dA2 || !dx2_max || !ext2 ||
                  !s2xx || !s2yy || !s2zz || !s2xy || !s2xz || !s2yz)));

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
    free(ext1); free(ext2);

    free(dx1_max);

    free(dx2_max);

    free(s1xx); free(s1yy); free(s1zz); free(s1xy); free(s1xz); free(s1yz);
    free(s2xx); free(s2yy); free(s2zz); free(s2xy); free(s2xz); free(s2yz);
    kstats_free(&k1s); kstats_free(&k2s);

    delete({m1, m2});
    return;
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

        dx1_max[j] = max(dx1_max[j], Delta);
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

        dx2_max[j] = max(dx2_max[j], Delta);
      }
    }
  }

  // The extent needs each DPS's centroid inside the interface loop, so
  // volume and centroid moments are reduced here.
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

  // Central second moments about the (now global) centroid, same cells and
  // weights as the volume loop above. Not periodic-aware, like the centroid.
  foreach(serial) {
    coord o = {x, y, z};

    if (m1[] > 0) {
      int j = (int)m1[] - 1;
      if (j >= 0 && j < n1 && v1[j] > 0.) {
        double vol = dv()*c[];
        double dx_ = o.x - c1x[j]/v1[j], dy_ = o.y - c1y[j]/v1[j], dz_ = o.z - c1z[j]/v1[j];
        s1xx[j] += vol*dx_*dx_; s1yy[j] += vol*dy_*dy_; s1zz[j] += vol*dz_*dz_;
        s1xy[j] += vol*dx_*dy_; s1xz[j] += vol*dx_*dz_; s1yz[j] += vol*dy_*dz_;
      }
    }

    if (m2[] > 0) {
      int j = (int)m2[] - 1;
      if (j >= 0 && j < n2 && v2[j] > 0.) {
        double vol = dv()*(1. - c[]);
        double dx_ = o.x - c2x[j]/v2[j], dy_ = o.y - c2y[j]/v2[j], dz_ = o.z - c2z[j]/v2[j];
        s2xx[j] += vol*dx_*dx_; s2yy[j] += vol*dy_*dy_; s2zz[j] += vol*dz_*dz_;
        s2xy[j] += vol*dx_*dy_; s2xz[j] += vol*dx_*dz_; s2yz[j] += vol*dy_*dz_;
      }
    }
  }

  // Interface quantities for bubbles/droplets using existing m1, m2 tags
  scalar kappa[];
  curvature (c, kappa);
  foreach(serial) {
    if (c[] > 1e-6 && c[] < 1. - 1e-6) {

      // Calculation of interfacial quantities
      coord n = interface_normal(point, c), p;
      double alpha = plane_alpha(c[], n);
      double Ac = pow(Delta, dimension - 1)*plane_area_center(n, alpha, &p);
      // cell centre and half diagonal: bound of the distance of any point
      // of the cell (and of its interface) from that centre
      coord oc0 = {x, y, z};
      double half_diag = 0.5*sqrt((double)dimension)*Delta;
      // First-order error contribution
      double dVc = Ac*Delta;

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
        if (kappa[] != nodata)
          kstats_add(&k1s, q, kappa[], Ac, Delta, oi);
        if (v1[q] > 0.) {
          double r = sqrt(sq(oc0.x - c1x[q]/v1[q]) + sq(oc0.y - c1y[q]/v1[q]) +
                          sq(oc0.z - c1z[q]/v1[q])) + half_diag;
          ext1[q] = max(ext1[q], r);
        }
      }

      // Bubble owners
      if (tag2 >= 1 && tag2 <= n2) {
        int q = tag2 - 1;
        area2[q] += Ac;
        dA2[q]   += dVc;
        if (kappa[] != nodata)
          kstats_add(&k2s, q, - kappa[], Ac, Delta, oi);
        if (v2[q] > 0.) {
          double r = sqrt(sq(oc0.x - c2x[q]/v2[q]) + sq(oc0.y - c2y[q]/v2[q]) +
                          sq(oc0.z - c2z[q]/v2[q])) + half_diag;
          ext2[q] = max(ext2[q], r);
        }
      }

    }
  }

  // delete temporary array
  delete({m1, m2});
  kstats_reduce(&k1s, n1);
  kstats_reduce(&k2s, n2);

#ifdef _MPI
  if (n1 > 0) {
    MPI_Allreduce(MPI_IN_PLACE, u1x, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u1y, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u1z, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, dx1_max, n1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, area1, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dA1, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ext1, n1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, s1xx, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s1yy, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s1zz, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s1xy, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s1xz, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s1yz, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  if (n2 > 0) {
    MPI_Allreduce(MPI_IN_PLACE, u2x, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u2y, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u2z, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, dx2_max, n2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, area2, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dA2, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ext2, n2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, s2xx, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s2yy, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s2zz, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s2xy, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s2xz, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s2yz, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
                  "%.10e %.10e %.10e %.10e %.10e %.10e",
                  t, istep, j, v1[j], area1[j],
                  c1x[j]/v1[j], c1y[j]/v1[j], c1z[j]/v1[j],
                  u1x[j]/v1[j], u1y[j]/v1[j], u1z[j]/v1[j],
                  dx1_max[j], dA1[j], ext1[j],
                  s1xx[j]/v1[j], s1yy[j]/v1[j], s1zz[j]/v1[j],
                  s1xy[j]/v1[j], s1xz[j]/v1[j], s1yz[j]/v1[j]);
          kstats_fprint(fdrop, &k1s, j, area1[j]);
          fputc('\n', fdrop);
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
                  "%.10e %.10e %.10e %.10e %.10e %.10e",
                  t, istep, j, v2[j], area2[j],
                  c2x[j]/v2[j], c2y[j]/v2[j], c2z[j]/v2[j],
                  u2x[j]/v2[j], u2y[j]/v2[j], u2z[j]/v2[j],
                  dx2_max[j], dA2[j], ext2[j],
                  s2xx[j]/v2[j], s2yy[j]/v2[j], s2zz[j]/v2[j],
                  s2xy[j]/v2[j], s2xz[j]/v2[j], s2yz[j]/v2[j]);
          kstats_fprint(fbubb, &k2s, j, area2[j]);
          fputc('\n', fbubb);
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

  free(dx1_max);
  free(dA1); free(ext1);

  free(dx2_max);
  free(dA2); free(ext2);

  free(s1xx); free(s1yy); free(s1zz); free(s1xy); free(s1xz); free(s1yz);
  free(s2xx); free(s2yy); free(s2zz); free(s2xy); free(s2xz); free(s2yz);
  kstats_free(&k1s); kstats_free(&k2s);
}
//
// ============================================================
// countDropsBubble_app_v2
// Current formulation:
//  - sub-threshold interfacial volume attributed with nearest_tag
//  - first-order and blended (curvature-based) volume error estimates
//  - centroid error estimate: first-moment bound
// All error estimates are written WITHOUT calibration constants, i.e.
// they are uncalibrated estimates: multiply by C_V (volume)
// or C_X (position) in post-processing.
// Local interface error:
//   |delta_c| = Delta*min(|kappa_c| Delta, D - 1),
//   with |kappa_c| Delta replaced by D - 1 where kappa is nodata.
// Output columns (0-based) in name_2 / name_3 (only what the tracking uses):
//   0 t, 1 istep, 2 j, 3 V, 4 A, 5-7 centroid, 8-10 velocity,
//   11 dx max, 12 dV_err (first order), 13 dV_err (blended),
//   14 dX_err = (1/V) sum A_c |delta_c| |x_c - x_i|,
//   15 extent = max over the interfacial cells of |x_c - x_i| + sqrt(D)/2 Delta,
//      i.e. the radius of the smallest sphere centred at the centroid
//      that contains the DPS (contact test of the tracking)
//   16-21 central second moments per unit volume, (1/V) sum vol (x_i - xc_i)(x_j - xc_j):
//      xx, yy, zz, xy, xz, yz (sphere of radius R: xx = yy = zz = R^2/5)
//   22-29 curvature statistics (see KStats above): kmax, x, y, z of kmax,
//      kmin, mean kappa, rms kappa, area fraction with |kappa| Delta > KAPPA_HI
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
  
  // Volume errors - first order
  double *dA1_1 = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *dA2_1 = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Volume errors - blended
  double *dA1_2 = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *dA2_2 = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Centroid position error: sum A_c |delta_c| |x_c - x_i|
  double *xe1 = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *xe2 = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Extent: max |x_c - x_i| + sqrt(D)/2 Delta over the interfacial cells
  double *ext1 = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *ext2 = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // AMR/grid information
  double *dx1_max = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;

  double *dx2_max = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Central second moments (volume weighted): sum vol (x_i - xc_i)(x_j - xc_j)
  double *s1xx = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *s1yy = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *s1zz = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *s1xy = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *s1xz = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;
  double *s1yz = (n1 > 0) ? calloc(n1, sizeof(double)) : NULL;

  double *s2xx = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *s2yy = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *s2zz = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *s2xy = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *s2xz = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;
  double *s2yz = (n2 > 0) ? calloc(n2, sizeof(double)) : NULL;

  // Interface curvature statistics
  KStats k1s, k2s;
  int kok = kstats_alloc(&k1s, n1) & kstats_alloc(&k2s, n2);

  // Allocation check: decided collectively so all ranks leave together (no MPI deadlock)
  int alloc_ok = kok && !((n1 > 0 && (!v1 || !c1x || !c1y || !c1z || !u1x || !u1y || !u1z || !area1 ||
                  !dA1_1 || !dA1_2 || !dx1_max || !xe1 || !ext1 ||
                  !s1xx || !s1yy || !s1zz || !s1xy || !s1xz || !s1yz)) ||
      (n2 > 0 && (!v2 || !c2x || !c2y || !c2z || !u2x || !u2y || !u2z || !area2 ||
                  !dA2_1 || !dA2_2 || !dx2_max || !xe2 || !ext2 ||
                  !s2xx || !s2yy || !s2zz || !s2xy || !s2xz || !s2yz)));

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

    free(dx1_max);

    free(dx2_max);

    free(xe1); free(xe2);
    free(ext1); free(ext2);

    free(s1xx); free(s1yy); free(s1zz); free(s1xy); free(s1xz); free(s1yz);
    free(s2xx); free(s2yy); free(s2zz); free(s2xy); free(s2xz); free(s2yz);
    kstats_free(&k1s); kstats_free(&k2s);

    delete({m1, m2});
    return;
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

        if (m1[] > 0) // largest cell size over the tagged cells
          dx1_max[j] = max(dx1_max[j], Delta);
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

        if (m2[] > 0) // largest cell size over the tagged cells
          dx2_max[j] = max(dx2_max[j], Delta);
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

#if AXI
  // the centroid of an axisymmetric DPS is on the axis
  for (int j = 0; j < n1; j++) c1y[j] = 0.;
  for (int j = 0; j < n2; j++) c2y[j] = 0.;
#endif

  // Central second moments about the (now global) centroid, with the same
  // cell-to-tag assignment (nearest_tag for sub-threshold interfacial cells)
  // and weights as the volume loop above. Not periodic-aware, like the
  // centroid. In AXI runs only the axial component (xx) is meaningful.
  foreach(serial) {
    coord o = {x, y, z};

    if (c[] > 0.) {
      int j = (m1[] > 0) ? (int)m1[] - 1 : nearest_tag (point, m1, n1);
      if (j >= 0 && j < n1 && v1[j] > 0.) {
        double vol = dv()*c[];
        double dx_ = o.x - c1x[j]/v1[j], dy_ = o.y - c1y[j]/v1[j], dz_ = o.z - c1z[j]/v1[j];
        s1xx[j] += vol*dx_*dx_; s1yy[j] += vol*dy_*dy_; s1zz[j] += vol*dz_*dz_;
        s1xy[j] += vol*dx_*dy_; s1xz[j] += vol*dx_*dz_; s1yz[j] += vol*dy_*dz_;
      }
    }

    if (c[] < 1.) {
      int j = (m2[] > 0) ? (int)m2[] - 1 : nearest_tag (point, m2, n2);
      if (j >= 0 && j < n2 && v2[j] > 0.) {
        double vol = dv()*(1. - c[]);
        double dx_ = o.x - c2x[j]/v2[j], dy_ = o.y - c2y[j]/v2[j], dz_ = o.z - c2z[j]/v2[j];
        s2xx[j] += vol*dx_*dx_; s2yy[j] += vol*dy_*dy_; s2zz[j] += vol*dz_*dz_;
        s2xy[j] += vol*dx_*dy_; s2xz[j] += vol*dx_*dz_; s2yz[j] += vol*dy_*dz_;
      }
    }
  }

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
      // cell centre and half diagonal: bound of the distance of any point
      // of the cell (and of its interface) from that centre
      coord oc0 = {x, y, z};
      double half_diag = 0.5*sqrt((double)dimension)*Delta;
      // First-order error contribution (without C_V)
      double dVc_1 = Ac*Delta;

      // Blended error contribution (without C_V):
      //   |delta_c| = Delta*min(|kappa| Delta, D - 1).
      // Where kappa is undefined, |kappa| Delta is replaced by D - 1, i.e.
      // the saturated value of the clipped branch, so the estimate is
      // continuous between the two branches.
      double kD = (kappa[] != nodata) ?
        min(fabs(kappa[])*Delta, D_PHYS - 1.) : D_PHYS - 1.;
      double delta_c = Delta*kD;
      double dVc_2 = Ac*delta_c;
      
      // Define PLIC coordinates in the interfacial cells 
      coord oi = {x,y,z};
      foreach_dimension() {
	oi.x += p.x*Delta;
      }
#if AXI
      // per radian, as dv() = y Delta^2
      Ac *= oi.y;
      dVc_1 *= oi.y;
      dVc_2 *= oi.y;
#endif
  
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
        if (kappa[] != nodata)
          kstats_add(&k1s, q, kappa[], Ac, Delta, oi);
        dA1_2[q]  += dVc_2;
        if (v1[q] > 0.) { // A_c |delta_c| |x_c - x_i|
          coord xc = {c1x[q]/v1[q], c1y[q]/v1[q], c1z[q]/v1[q]};
          double r = sqrt(sq(oi.x - xc.x) + sq(oi.y - xc.y) + sq(oi.z - xc.z));
          xe1[q] += dVc_2*r;
          double re = sqrt(sq(oc0.x - xc.x) + sq(oc0.y - xc.y) + sq(oc0.z - xc.z)) + half_diag;
          ext1[q] = max(ext1[q], re);
        }
      }
      else
        orphanA1 += Ac;
  
      // Bubble owners 
      if (tag2 >= 1 && tag2 <= n2) {
        int q = tag2 - 1;
        area2[q]  += Ac;
        dA2_1[q]  += dVc_1;
        if (kappa[] != nodata)
          kstats_add(&k2s, q, - kappa[], Ac, Delta, oi);
        dA2_2[q]  += dVc_2;
        if (v2[q] > 0.) { // A_c |delta_c| |x_c - x_i|
          coord xc = {c2x[q]/v2[q], c2y[q]/v2[q], c2z[q]/v2[q]};
          double r = sqrt(sq(oi.x - xc.x) + sq(oi.y - xc.y) + sq(oi.z - xc.z));
          xe2[q] += dVc_2*r;
          double re = sqrt(sq(oc0.x - xc.x) + sq(oc0.y - xc.y) + sq(oc0.z - xc.z)) + half_diag;
          ext2[q] = max(ext2[q], re);
        }
      }
      else
        orphanA2 += Ac;

    }
  }

  // delete temporary array 
  delete({m1, m2});
  kstats_reduce(&k1s, n1);
  kstats_reduce(&k2s, n2);

#ifdef _MPI
  if (n1 > 0) {
    MPI_Allreduce(MPI_IN_PLACE, u1x, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u1y, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u1z, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, dx1_max, n1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, area1, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dA1_1, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dA1_2, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, xe1, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ext1, n1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, s1xx, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s1yy, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s1zz, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s1xy, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s1xz, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s1yz, n1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }

  if (n2 > 0) {
    MPI_Allreduce(MPI_IN_PLACE, u2x, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u2y, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, u2z, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, dx2_max, n2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, area2, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dA2_1, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, dA2_2, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, xe2, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ext2, n2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    MPI_Allreduce(MPI_IN_PLACE, s2xx, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s2yy, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s2zz, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s2xy, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s2xz, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, s2yz, n2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
  

#if AXI
  for (int j = 0; j < n1; j++) u1y[j] = 0.;
  for (int j = 0; j < n2; j++) u2y[j] = 0.;
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
                  "%.10e %.10e "
                  "%.10e %.10e %.10e %.10e %.10e %.10e",
                  t, istep, j, AXI_FACTOR*v1[j], AXI_FACTOR*area1[j],
                  c1x[j]/v1[j], c1y[j]/v1[j], c1z[j]/v1[j],
                  u1x[j]/v1[j], u1y[j]/v1[j], u1z[j]/v1[j],
                  dx1_max[j],
                  AXI_FACTOR*dA1_1[j], AXI_FACTOR*dA1_2[j],
                  xe1[j]/v1[j], ext1[j],
                  s1xx[j]/v1[j], s1yy[j]/v1[j], s1zz[j]/v1[j],
                  s1xy[j]/v1[j], s1xz[j]/v1[j], s1yz[j]/v1[j]);
          kstats_fprint(fdrop, &k1s, j, AXI_FACTOR*area1[j]);
          fputc('\n', fdrop);
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
                  "%.10e %.10e "
                  "%.10e %.10e %.10e %.10e %.10e %.10e",
                  t, istep, j, AXI_FACTOR*v2[j], AXI_FACTOR*area2[j],
                  c2x[j]/v2[j], c2y[j]/v2[j], c2z[j]/v2[j],
                  u2x[j]/v2[j], u2y[j]/v2[j], u2z[j]/v2[j],
                  dx2_max[j],
                  AXI_FACTOR*dA2_1[j], AXI_FACTOR*dA2_2[j],
                  xe2[j]/v2[j], ext2[j],
                  s2xx[j]/v2[j], s2yy[j]/v2[j], s2zz[j]/v2[j],
                  s2xy[j]/v2[j], s2xz[j]/v2[j], s2yz[j]/v2[j]);
          kstats_fprint(fbubb, &k2s, j, AXI_FACTOR*area2[j]);
          fputc('\n', fbubb);
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

  free(dx1_max);
  free(dA1_1); free(dA1_2);

  free(dx2_max);
  free(dA2_1); free(dA2_2);
  free(xe1); free(xe2);
  free(ext1); free(ext2);

  free(s1xx); free(s1yy); free(s1zz); free(s1xy); free(s1xz); free(s1yz);
  free(s2xx); free(s2yy); free(s2zz); free(s2xy); free(s2xz); free(s2yz);
  kstats_free(&k1s); kstats_free(&k2s);
}

