// facets_output.h
// Compact binary output of the PLIC interface reconstruction, written every
// few steps for the 3D shape/curvature/neck analysis in post-processing.
// Needs fractions.h (included by two-phase.h) for interface_normal,
// plane_alpha and plane_area_center.
//
// ============================================================
// Compact binary PLIC output (one record per call, appended to fp):
//   int32 istep, float64 t, int32 nfacets,
//   then nfacets x 7 float32: facet centroid (x,y,z), unit normal (nx,ny,nz)
//   pointing from the c = 0 phase (gas) towards the c = 1 phase (liquid),
//   facet area.
// Exactly 28 bytes per interfacial cell (measured ~3.8x smaller than the
// output_facets text, which writes every polygon vertex as text), and
// sufficient for volume, area, curvature and neck reconstruction.
// output_plic_compact: each rank writes its own stream (serial/debug use).
// output_plic_mpiio:   all ranks write ONE shared file (production use).
// ============================================================
// fills a buffer with the 7 floats of every interfacial cell of this rank
static float * plic_buffer (scalar c, int *nk)
{
  int nf = 0;
  foreach(serial)
    if (c[] > 1e-6 && c[] < 1. - 1e-6)
      nf++;
  float *buf = (nf > 0) ? malloc(7*nf*sizeof(float)) : NULL;
  int k = 0;
  foreach(serial)
    if (c[] > 1e-6 && c[] < 1. - 1e-6 && buf) {
      coord n = interface_normal(point, c), p;
      double alpha = plane_alpha(c[], n);
      double Ac = pow(Delta, dimension - 1)*plane_area_center(n, alpha, &p);
      double nn = sqrt(sq(n.x) + sq(n.y) + sq(n.z));
      // Basilisk's interface normal points out of the c = 1 phase: flip it
      buf[7*k + 0] = x + p.x*Delta;
      buf[7*k + 1] = y + p.y*Delta;
      buf[7*k + 2] = z + p.z*Delta;
      buf[7*k + 3] = nn > 0. ? -n.x/nn : 0.;
      buf[7*k + 4] = nn > 0. ? -n.y/nn : 0.;
      buf[7*k + 5] = nn > 0. ? -n.z/nn : 0.;
      buf[7*k + 6] = Ac;
      k++;
    }
  *nk = k;
  return buf;
}

void output_plic_compact (scalar c, FILE *fp, int istep)
{
  int k;
  float *buf = plic_buffer (c, &k);
  fwrite(&istep, sizeof(int), 1, fp);
  fwrite(&t, sizeof(double), 1, fp);
  fwrite(&k, sizeof(int), 1, fp);
  if (k > 0)
    fwrite(buf, sizeof(float), 7*k, fp);
  fflush(fp);
  free(buf);
}
//
// ============================================================
// Same record, but ONE file for the whole run and all ranks (MPI-IO):
//   record = int32 istep, float64 t, int32 nfacets (total over ranks),
//            then nfacets x 7 float32, the ranks' blocks one after the other
//   (same format as output_plic_compact: the same reader works).
// Usage (collective: every rank calls it; file opened once):
//   static MPI_File fh; static long long off = 0; static int open_ = 0;
//   if (!open_) { MPI_File_open (MPI_COMM_WORLD, name,
//                   MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
//                 open_ = 1; }
//   output_plic_mpiio (f, &fh, &off, i);
//   ... MPI_File_close (&fh) at the end.
// qcc does not parse MPI types: the handle is passed as void * and the MPI
// calls live in a helper without Basilisk constructs.
// ============================================================
#ifdef _MPI
static void plic_mpiio_write (void *fhp, long long *off, int istep, double tt,
                              float *buf, int k)
{
  MPI_File *fh = fhp;
  long long kl = k, before = 0, total = 0;
  MPI_Exscan(&kl, &before, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) before = 0;
  MPI_Allreduce(&kl, &total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  const long long head = sizeof(int) + sizeof(double) + sizeof(int);
  if (rank == 0) {
    char h[sizeof(int) + sizeof(double) + sizeof(int)];
    int tot = (int) total;
    memcpy(h, &istep, sizeof(int));
    memcpy(h + sizeof(int), &tt, sizeof(double));
    memcpy(h + sizeof(int) + sizeof(double), &tot, sizeof(int));
    MPI_File_write_at(*fh, *off, h, (int) head, MPI_BYTE, MPI_STATUS_IGNORE);
  }
  MPI_File_write_at_all(*fh, *off + head + before*7*(long long)sizeof(float),
                        buf, 7*k, MPI_FLOAT, MPI_STATUS_IGNORE);
  *off += head + total*7*(long long)sizeof(float);
}

void output_plic_mpiio (scalar c, void *fhp, long long *off, int istep)
{
  int k;
  float *buf = plic_buffer (c, &k);
  plic_mpiio_write (fhp, off, istep, t, buf, k);
  free(buf);
}

// Convenience layer keeping the MPI handle inside this header, so that the
// calling .c file needs no MPI types (which qcc does not parse):
//   plic_shared_open (name);  output_plic_shared (f, i);  plic_shared_close ();
static MPI_File plic_fh_;
static long long plic_off_ = 0;
static int plic_is_open_ = 0;

static void plic_shared_open_ (const char *name)
{
  MPI_File_open(MPI_COMM_WORLD, (char *) name, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &plic_fh_);
  plic_off_ = 0;
  plic_is_open_ = 1;
}

int plic_shared_is_open (void) { return plic_is_open_; }

void plic_shared_open (const char *name)
{
  if (!plic_is_open_)
    plic_shared_open_ (name);
}

void output_plic_shared (scalar c, int istep)
{
  output_plic_mpiio (c, &plic_fh_, &plic_off_, istep);
}

void plic_shared_close (void)
{
  if (plic_is_open_) {
    MPI_File_close(&plic_fh_);
    plic_is_open_ = 0;
  }
}
#endif
