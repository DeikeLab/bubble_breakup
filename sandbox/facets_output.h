// facets_output.h
// Compact binary output of the PLIC interface reconstruction, for the 3D
// shape / curvature / neck analysis in post-processing.
// Needs fractions.h (included by two-phase.h) for interface_normal,
// plane_alpha and plane_area_center.
//
// ============================================================
// output_plic (c, name, istep): writes ONE record into the file `name`,
// created (or overwritten) at every call; no static state. Intended use: one
// file per output step, e.g. name = "interface/plic_000909250.bin".
//   record = int32 istep, float64 t, int32 nfacets,
//            then nfacets x 7 float32 per interfacial cell:
//            PLIC centroid (x, y, z), unit normal (nx, ny, nz) pointing from
//            the c = 0 phase (gas) towards the c = 1 phase (liquid), area.
// Serial: written with stdio. MPI: all ranks write collectively into the
// same file (MPI-IO), each rank's block after the previous ranks' ones, the
// header written by rank 0 (one file per step, not one per rank).
// 28 bytes per interfacial cell (~3.8x smaller than output_facets text).
// Reader: postprocessing/.../read_plic.py (read_plic_bin).
// ============================================================
//
// the 7 floats of every interfacial cell of this process
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

#ifdef _MPI
// Plain C (no Basilisk constructs): qcc does not parse MPI types and warns
// here; the warning is harmless.
static void plic_write_mpi (const char *name, int istep, double tt,
                            float *buf, int k)
{
  MPI_File fh;
  const MPI_Offset end = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_File_open(MPI_COMM_WORLD, (char *) name, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fh);
  MPI_File_set_size(fh, 0);                    // overwrite an existing file
  long long kl = k, before = 0, total = 0;
  MPI_Exscan(&kl, &before, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    before = 0;                                // MPI_Exscan leaves rank 0 undefined
  MPI_Allreduce(&kl, &total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  const long long head = sizeof(int) + sizeof(double) + sizeof(int);
  if (rank == 0) {
    char h[sizeof(int) + sizeof(double) + sizeof(int)];
    int tot = (int) total;
    memcpy(h, &istep, sizeof(int));
    memcpy(h + sizeof(int), &tt, sizeof(double));
    memcpy(h + sizeof(int) + sizeof(double), &tot, sizeof(int));
    MPI_File_write_at(fh, end, h, (int) head, MPI_BYTE, MPI_STATUS_IGNORE);
  }
  MPI_File_write_at_all(fh, end + head + before*7*(long long)sizeof(float),
                        buf, 7*k, MPI_FLOAT, MPI_STATUS_IGNORE);
  MPI_File_close(&fh);
}
#endif

void output_plic (scalar c, const char *name, int istep)
{
  int k;
  float *buf = plic_buffer (c, &k);
#ifdef _MPI
  plic_write_mpi (name, istep, t, buf, k);
#else
  FILE *fp = fopen(name, "wb");
  if (!fp)
    perror(name);
  else {
    fwrite(&istep, sizeof(int), 1, fp);
    fwrite(&t, sizeof(double), 1, fp);
    fwrite(&k, sizeof(int), 1, fp);
    if (k > 0)
      fwrite(buf, sizeof(float), 7*k, fp);
    fclose(fp);
  }
#endif
  free(buf);
}
