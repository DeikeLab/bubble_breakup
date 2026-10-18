/**
# Performance monitoring (for the Navier--Stokes solvers)

This logs simple statistics available for the various [Navier--Stokes
solvers](/src/README#navierstokes). */

event perfs (i += 1) {
  /*
  static FILE * fp = fopen ("perfs", "w");
  if (i == 0)
    fprintf (fp,
	     "t dt mgp.i mgp.nrelax mgpf.i mgpf.nrelax mgu.i mgu.nrelax "
	     "grid->tn perf.t perf.speed npe\n");
  fprintf (fp, "%.10e %.10e %d %d %d %d %d %d %ld %.10e %.10e %d\n", 
	   t, dt, mgp.i, mgp.nrelax, mgpf.i, mgpf.nrelax, mgu.i, mgu.nrelax,
	   grid->tn, perf.t, perf.speed, npe());
  fflush (fp);
  */
  if( pid() == 0) {
    fflush(stderr);
    char file[80];
    sprintf (file,"perfs.out");
    FILE * fp = fopen(file,"a");
    if (i == 0)
      fprintf (fp,
               "t i dt mgp.i mgp.nrelax mgpf.i mgpf.nrelax mgu.i mgu.nrelax "
               "grid->tn perf.t perf.speed npe\n");
    fprintf (fp, "%.10e %09d %.10e %d %d %d %d %d %d %ld %.10e %.10e %d\n", 
             t, i, dt, mgp.i, mgp.nrelax, mgpf.i, mgpf.nrelax, mgu.i, mgu.nrelax,
             grid->tn, perf.t, perf.speed, npe());
    fclose(fp);
  }
}
