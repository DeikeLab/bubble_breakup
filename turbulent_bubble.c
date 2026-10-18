/**
   Set-up of HIT without and with emulsions/droplets/bubbles */

#include "navier-stokes/centered.h"
#define mu(f)  (1./(clamp(f,0,1)*(1./mu1 - 1./mu2) + 1./mu2))
#include "two-phase.h"
#include "navier-stokes/conserving.h"
#include "tension.h"
#include "tag.h"
#include "view.h"
#include "sandbox/maxruntime.h"
#include "sandbox/perfs.h"
#include "sandbox/my_funcs.h"

/**
   Include density and viscosity ratios.*/

#define RHOR 850.0
#define MUR 25.0

/**
   Define domain size. */

#define WIDTH 120.0

/**
The code takes the level of refinement as optional command-line
argument (as well as an optional maximum runtime). */

double R0 = 1.0;
double SIGMA = 1.0;
double amp_force = 1.0;
double mu1_in = 1.0;
int FORCED = 1;
int MAX_LEVEL = 1;
int MIN_LEVEL = 1;
double end_sim = 1.0;
double t_phy = 1.0;
int single_ph = 1;
int do_restart_tw = 1;
int do_ens = 1;
double t_ie = 1.0;
int input_ens = 1;

/**
   Define some output frequencies. */

double t_out_stat = 64.00;  // output frequency of global observables 
double t_out_tag  = 16.00;  // output frequency of tagging
double t_out_mov  = 8.000;  // output frequency of movie
double t_out_dpst = 8.000;  // output frequency for dump restarting files 
double t_out_dpbk = 0.500;  // output frequency back-up restarting files

/**
   For the restarting step. */

int counter_max = 2;
int counter = 0;
int counter_max_ens = 40;
int counter_ens = 0;

int main (int argc, char * argv[]) {
    
  /**
     Provide the inputs */
    
  maxruntime (&argc, argv);

  if (argc > 1)
    R0 = atof(argv[1]);
  if (argc > 2)
    SIGMA = atof(argv[2]);
  if (argc > 3)
    amp_force = atof(argv[3]);
  if (argc > 4)
    mu1_in = atof(argv[4]);
  if (argc > 5)
    FORCED = atof(argv[5]);
  if (argc > 6)
    MAX_LEVEL = atoi(argv[6]);
  if (argc > 7)
    MIN_LEVEL = atoi(argv[7]);
  if (argc > 8)
    end_sim  = atof(argv[8]);
  if (argc > 9)
    t_phy = atof(argv[9]);
  if (argc > 10)
    single_ph = atoi(argv[10]);
  if (argc > 11)
    do_restart_tw = atoi(argv[11]);
  if (argc > 12)
    do_ens = atoi(argv[12]);
  if (argc > 13)
    t_ie = atof(argv[13]);
  if (argc > 14)
    input_ens = atoi(argv[14]);
  
  if (argc < 15) {

    fprintf(ferr, "Lack of command line arguments. Check! Need %d more arguments\n", 15-argc);
    return 1;

  }
  
  fprintf(stderr, "************************\n"), fflush (stderr);
  fprintf(stderr, "maximum runtime = %.10e seconds\n", _maxruntime), fflush (stderr);
  fprintf(stderr, "Check of input parameters only\n"), fflush (stderr);
  fprintf(stderr, " R0 = %.10e\n ", R0), fflush (stderr);
  fprintf(stderr, " sigma  = %.10e\n ", SIGMA), fflush (stderr);
  fprintf(stderr, " amp_force = %.10e\n ", amp_force), fflush (stderr);
  fprintf(stderr, " mu1_in = %.10e\n ", mu1_in), fflush (stderr);
  fprintf(stderr, " FORCED = %d\n ", FORCED), fflush (stderr);
  fprintf(stderr, " MAXLEVEL = %d\n ", MAX_LEVEL), fflush (stderr);
  fprintf(stderr, " MINLEVEL = %d\n ", MIN_LEVEL), fflush (stderr);
  fprintf(stderr, " tend  = %.10e\n ", end_sim), fflush (stderr);
  fprintf(stderr, " t_phy = %.10e\n ", t_phy), fflush (stderr);
  fprintf(stderr, " single_ph = %d\n ", single_ph), fflush (stderr);
  fprintf(stderr, " do_restart_tw = %d\n ", do_restart_tw), fflush (stderr);
  fprintf(stderr, " do_ens = %d\n ", do_ens), fflush (stderr);
  fprintf(stderr, " t_ie = %.10e\n ", t_ie), fflush (stderr);
  fprintf(stderr, " input_ens = %d\n ", input_ens), fflush (stderr);
  fprintf(stderr, "************************\n"), fflush (stderr);
  
  /**
    Set box size and boundary conditions */

  size (WIDTH);
  foreach_dimension() {
    periodic (right);
  }
   
  /**
    Use reference density for fluid, and define gas by density ratio.*/
  
  //mu1 = 0.01*sq(WIDTH/(2.0*pi))/(2.0)/2.0;  
  mu1 = mu1_in;  
  mu2 = mu1/MUR;
  rho1 = 1.0;
  rho2 = rho1/RHOR;

  /**
    And the surface tension. */

  f.sigma = SIGMA;
  
  /**
    We initialize the grid. */
  
#if TREE
  N = 1 << (MAX_LEVEL-2);
#else
  N = 1 << MAX_LEVEL;
#endif

  /**
    Set tolerance and number of iterations. */

  if(single_ph == 1) {
    TOLERANCE = 1e-4;
  }
  else {
    //mgu.nrelax = 20;
    NITERMAX = 50;
    NITERMIN = 5;
  }

  /**
    Set the input counter of ensembles */

  counter_ens = input_ens;

  /**
    We run! */

  run();

}

void vorticity3D (vector u, vector omg, scalar omg_mod) {

  foreach() {
    omg.x[] = ( (u.z[0,1,0]-u.z[0,-1,0]) - (u.y[0,0,1]-u.y[0,0,-1]) )/(2.0*Delta);
    omg.y[] = ( (u.x[0,0,1]-u.x[0,0,-1]) - (u.z[1,0,0]-u.z[-1,0,0]) )/(2.0*Delta);
    omg.z[] = ( (u.y[1,0,0]-u.y[-1,0,0]) - (u.x[0,1,0]-u.x[0,-1,0]) )/(2.0*Delta);
    omg_mod[] = 0.0;
    foreach_dimension() {
      omg_mod[] += sq(omg.x[]);
    }
    omg_mod[] = sqrt(omg_mod[]);
  }

}

event init (i = 0) {

  /**
     Print relevant simulation parameters */

  fprintf(stderr, "************************\n"), fflush (stderr);
  fprintf(stderr, "A-posteriori check of simulation parameters\n"), fflush (stderr);
  fprintf(stderr, " R0 = %.10e\n ", R0), fflush (stderr);
  fprintf(stderr, " sigma = %.10e\n ", SIGMA), fflush (stderr);
  fprintf(stderr, " amp_force = %.10e\n ", amp_force), fflush (stderr);
  fprintf(stderr, " mu1_in = %.10e\n ", mu1_in), fflush (stderr);
  fprintf(stderr, " FORCED = %d\n ", FORCED), fflush (stderr);
  fprintf(stderr, " MAXLEVEL = %d\n ", MAX_LEVEL), fflush (stderr);
  fprintf(stderr, " MINLEVEL = %d\n ", MIN_LEVEL), fflush (stderr);
  fprintf(stderr, " tend  = %.10e\n ", end_sim), fflush (stderr);
  fprintf(stderr, " t_phy = %.10e\n ", t_phy), fflush (stderr);
  fprintf(stderr, " single_ph = %d\n ", single_ph), fflush (stderr);
  fprintf(stderr, " do_restart_tw = %d\n ", do_restart_tw), fflush (stderr);
  fprintf(stderr, " do_ens = %d\n ", do_ens), fflush (stderr);
  fprintf(stderr, " t_ie = %.10e\n ", t_ie), fflush (stderr);
  fprintf(stderr, " input_ens = %d\n ", input_ens), fflush (stderr);
  fprintf(stderr, " Oh = %.10e\n ", mu1/sqrt(rho1*SIGMA*(2.*R0))), fflush (stderr);
  fprintf(stderr, "************************\n"), fflush (stderr);

  /**
     Create directories for better organization of the output */
  
  fprintf(stderr, "Create directories if needed\n"), fflush (stderr);

  char comm[80];
  sprintf (comm, "mkdir -p restart_bk");
  system(comm);
  sprintf (comm, "mkdir -p restart_ens");
  system(comm);
  sprintf (comm, "mkdir -p interface");
  system(comm);
  sprintf (comm, "mkdir -p tagging_v1");
  system(comm);
  sprintf (comm, "mkdir -p tagging_v2");
  system(comm);

  /**
     Start a new simulation or restore a previous one */

  if (single_ph == 1) {
    fprintf(stderr, "Single-phase simulation\n"), fflush (stderr);
    if (!restore (file = "restart.bin")) {
      fprintf(stderr, "Start a new simulation to generate sustained HIT\n"), fflush (stderr);
     
      double waveno = L0/(2.0*pi);
      double amp = WIDTH/(2.0*pi);
      
      foreach() {
        f[] = 1.; // only liquid at the beginning
    
        double xw = x/waveno;
        double yw = y/waveno;
        double zw = z/waveno;
    
        u.x[] = amp*( cos(yw) + sin(zw) );
        u.y[] = amp*( sin(xw) + cos(zw) );
        u.z[] = amp*( cos(xw) + sin(yw) );
    
      }
    }
    else {
      fprintf(stderr, "We restart from a single-phase precursor\n"), fflush (stderr);
    }
  }
  else {
    fprintf(stderr, "Two-phase simulation\n"), fflush (stderr);
    if (do_restart_tw == 0 && restore (file = "restart_sp.bin")) { // a precursor
      fprintf(stderr, "Restart from statistically stationary single-phase HIT\n"), fflush (stderr);
      fprintf(stderr, "We add the bubble\n"), fflush (stderr);
    
      /* 
      refine(sq(2.0*R0)-sq(x-WIDTH/2.0) - sq(y-WIDTH/2.0) - sq(z-WIDTH/2.0)>0 && level < MAX_LEVEL );
      fraction (f, sq(x-WIDTH/2.0) + sq(y-WIDTH/2.0) + sq(z-WIDTH/2.0) - sq(R0));
      foreach() {      
        foreach_dimension() {
          u.x[] = f[]*u.x[]; // we set to 0 the velocity inside the bubble
        }
      }
      */

      int iter = 0;
      double uemax = 0.2*normf(u.x).avg;
      double femax = 1.0e-2;
      do {
        iter++;
        fprintf(stderr, "Current number of iterations: %d\n", iter);
        fraction (f, sq(x-WIDTH/2.0) + sq(y-WIDTH/2.0) + sq(z-WIDTH/2.0) - sq(R0));
        foreach() {
          foreach_dimension()
            u.x[] *= f[]; // zero velocity inside bubble only
        }
      } while (adapt_wavelet ((scalar *){f, u},
                             (double[]){femax, uemax, uemax, uemax},
                             MAX_LEVEL).nf && iter <= 150);

      // print an image for checking
      vector omg[];
      scalar omg_mod[];
      vorticity3D(u, omg, omg_mod);

      coord ubar;
      foreach_dimension() {
        stats s = statsf(u.x);
        ubar.x = s.sum/s.volume;
      }
      
      double ke = 0., vd = 0., vol = 0.;
      foreach(reduction(+:ke) reduction(+:vd) reduction(+:vol)) {
        vol += dv();
        foreach_dimension() {
          ke += dv()*sq(u.x[] - ubar.x);
          vd += dv()*(sq(u.x[1] - u.x[-1]) +
            	  sq(u.x[0,1] - u.x[0,-1]) +
            	  sq(u.x[0,0,1] - u.x[0,0,-1]))/sq(2.*Delta);
        }
      }
      ke /= 2.*vol;
      vd *= mu1/vol; 
      double vel_rms = sqrt(2.0/3.0*ke); 
      double lambda  = vel_rms*sqrt(15.*(mu1/rho1)*(1.0/vd));
     
      foreach() {
        omg_mod[] /= vel_rms/lambda;
      }
      
      boundary({omg_mod});
      clear();
      view (fov = 44, camera = "iso", ty = .2,
            width = 1200, height = 1200, bg = {1,1,1}, samples = 4);
      box(false, lc = {1,1,1}, lw = 0.1);
      draw_vof("f");
      squares ("omg_mod", linear = true, map = cool_warm, alpha = 1.e-6, n = {1, 0, 0});
      squares ("omg_mod", linear = true, map = cool_warm, alpha = 1.e-6, n = {0, 1, 0});
      squares ("omg_mod", linear = true, map = cool_warm, alpha = 1.e-6, n = {0, 0, 1});
      char namedump[200];
      sprintf (namedump, "./3D_tw_%09d.ppm", i);
      save(namedump);
      //return 1;

    }
    else {
      FILE * fp = fopen("restart.bin", "r");
      if( fp == NULL ) {
        fprintf(stderr, "No restarting file available. Please provide it!\n"), fflush (stderr);
        return 1; 
      }
      else {
        fprintf(stderr, "We restart from a two-phase simulation\n"), fflush (stderr);
        restore (file = "restart.bin");
      }
    }
  }
  
}

/**
   We modify the acceleration event to sustain turbulence. */

event acceleration (i++) {

  face vector av = a;

  if (FORCED) {
    coord ubar;
    foreach_dimension() {
      stats s = statsf(u.x);
      ubar.x = s.sum/s.volume;
    }
    foreach_face() {
      av.x[] += ((f[] + f[-1])/2.)*amp_force*((u.x[] + u.x[-1])/2. - ubar.x);
    }
  }

}

/**
## Outputs

We log the evolution of viscous dissipation, kinetic energy, and
  microscale Reynolds number. */

event logfile (i += 10) {

  coord ubar;
  foreach_dimension() {
    stats s = statsf(u.x);
    ubar.x = s.sum/s.volume;
  }

  double ke = 0., vd = 0., vol = 0.;
  foreach(reduction(+:ke) reduction(+:vd) reduction(+:vol)) {
    vol += dv();
    foreach_dimension() {
      ke += dv()*sq(u.x[] - ubar.x);
      vd += dv()*(sq(u.x[1] - u.x[-1]) +
		  sq(u.x[0,1] - u.x[0,-1]) +
		  sq(u.x[0,0,1] - u.x[0,0,-1]))/sq(2.*Delta);
    }
  }
  ke /= 2.*vol;
  vd *= (mu1/rho1)/vol; 
  double vel_rms   = sqrt(2.0/3.0*ke); 
  double lambda    = vel_rms*sqrt(15.*(mu1/rho1)*(1.0/vd));
  double len_ls    = cube(vel_rms)/vd;
  double re_lambda = rho1*vel_rms*lambda/mu1;
  double re_ls     = rho1*vel_rms*len_ls/mu1;
  double we_est    = 2.0*rho1*pow(vd,2.0/3.0)*pow(2.*R0,5.0/3.0)/SIGMA; // estimated weber based on the input sigma

  if (pid() == 0) {

    fflush(stderr);

    FILE * fd = fopen("stats.out","a");

    fprintf (fd, "%8E %09d %8E %8E %8E %8E %8E %8E %8E %8E\n",
                  t, i, vd, ke, vel_rms, vd/(3.0*sq(vel_rms)), lambda, re_lambda, re_ls, we_est);
    fclose (fd);

  }

  double tot_area = interface_area(f);
  
  if (pid() == 0) {

    fflush(stderr);
    char name_1[80];
    sprintf (name_1,"int_qtn.out");
    FILE * log_sim = fopen(name_1,"a");
    fprintf (log_sim, "%8E %8E %8E\n", t, 1.0*i, tot_area);
    fclose(log_sim);

  }

}

/**
We use adaptivity. */

#if TREE
event adapt (i++) {
  double uemax = 0.2*normf(u.x).avg;
  double femax = 1.0e-2;
  adapt_wavelet ((scalar *){f, u}, 
		 (double[]){femax, uemax, uemax, uemax}, MAX_LEVEL);
  /*
  adapt_wavelet ((scalar *){u}, 
		 (double[]){uemax, uemax, uemax}, MAX_LEVEL);

  */
}
#endif

/** 
   Dump every fraction of t_phy */

event dumpstep (t = 0.0; t <= end_sim; t += t_phy/t_out_dpst) {
  
  if(counter < counter_max) {
    counter++;
  }
  else {
    counter = 1;
  }

  char dname[100];
  sprintf (dname, "dump_%d.bin", counter);
  dump (dname);

  /** 
     Add a symbolic link, log restarting info and size of the bin */

  if (pid () == 0) {

    char comm_2[80];
    sprintf (comm_2, "ln -sf dump_%d.bin restart.bin", counter);
    system(comm_2);

    fflush(stderr);
    char name_0[80];
    sprintf (name_0, "dump_%d.bin", counter);
    FILE * fp = fopen(name_0, "r");
    fseek(fp, 0L, SEEK_END);
    long int res = ftell(fp);
    fclose(fp);

    fflush(stderr);
    char name_1[80];
    sprintf (name_1,"log_restart.out");
    FILE * log_sim = fopen(name_1,"a");
    fprintf (log_sim, "%.10e %.10e %.10e %.10e\n", t, 1.0*i, 1.0*counter, 1.0*res);
    fclose(log_sim);

  }

}

/** 
   Dump every t_ie to perform ensemble simulations */

event dump_ens (t = 0.0; t <= end_sim; t += t_ie) {

  if(do_ens == 1) {	
    
    if(counter_ens <= counter_max_ens) {
      counter_ens++;
    }
    else {
      return 1;
    }
    
    if (pid() == 0) {
      fflush(stderr);
      char name_1[80];
      sprintf (name_1,"./restart_ens/log_ens_forcing.out");
      FILE * log_sim = fopen(name_1,"a");
      fprintf (log_sim, "%8E %8E %02d\n", 
          	         t, 1.0*i, counter_ens);
      fclose(log_sim);
    }
    
    char dname[100];
    sprintf (dname, "./restart_ens/dump_s%02d_A%08E_t%08E_L%02d.bin", 
		    counter_ens, amp_force, t, MAX_LEVEL);
    dump (dname);

  }

}

/** 
   Dump every fraction of t_phy the restart bk */

event dump_backup (t = 0.0; t <= end_sim; t += t_phy/t_out_dpbk) {

  char dname[100];
  sprintf (dname, "./restart_bk/dump_%09d.bin", i);
  dump (dname);

}

/** 
   Tagging */

//event tagging (t = 0.0; t <= end_sim; t += t_phy/t_out_tag) {
event tagging (i++) {

  if (single_ph == 0) {

    char name_1[100], name_2[100], name_3[100];
    sprintf (name_1, "./tagging_v1/num_bubbles_droplets_f1.out");
    sprintf (name_2, "./tagging_v1/droplets_f1.out");
    sprintf (name_3, "./tagging_v1/bubbles_f1.out");
    countDropsBubble_app_v1 (name_1, name_2, name_3, i, f, u);
    
    sprintf (name_1, "./tagging_v2/num_bubbles_droplets_f1.out");
    sprintf (name_2, "./tagging_v2/droplets_f1.out");
    sprintf (name_3, "./tagging_v2/bubbles_f1.out");
    countDropsBubble_app_v2 (name_1, name_2, name_3, i, f, u);

  }
  
}

/** 
   Interface */

//event interface (t = 0.0; t <= end_sim; t += t_phy/t_out_tag) {
event interface (i += 250) {

  if (single_ph == 0) {

    char names[80];
    sprintf (names, "interfaces_%d", pid());
    FILE * fpf = fopen (names,"w");
    output_facets (f, fpf);
    fclose (fpf);
    MPI_Barrier(MPI_COMM_WORLD);

    if (pid() == 0) {
      char command[512]; // bigger buffer!
      sprintf(command, "LC_ALL=C cat interfaces_* > ./interface/facets_%09d.out", i);
      system(command);
      sprintf(command, "rm -f interfaces_*");
      system(command);
    }

    if (pid() == 0) {
      fflush(stderr);
      char name_1[80];
      sprintf (name_1,"./interface/log_interface.out");
      FILE * log_sim = fopen(name_1,"a");
      fprintf (log_sim, "%8E %8E\n", t, 1.0*i);
      fclose(log_sim);
    }
  
  }
  
}

/**
   We generate a movie of the vortices. 
   Commented lines are optional alternative outputs.*/

#define POPEN(name, mode) fopen (name ".ppm", mode)

event movie (t = 0.0; t <= end_sim; t += t_phy/t_out_mov) {

  vector omg[];
  scalar omg_mod[];
  vorticity3D(u, omg, omg_mod);

  coord ubar;
  foreach_dimension() {
    stats s = statsf(u.x);
    ubar.x = s.sum/s.volume;
  }

  double ke = 0., vd = 0., vol = 0.;
  foreach(reduction(+:ke) reduction(+:vd) reduction(+:vol)) {
    vol += dv();
    foreach_dimension() {
      ke += dv()*sq(u.x[] - ubar.x);
      vd += dv()*(sq(u.x[1] - u.x[-1]) +
		  sq(u.x[0,1] - u.x[0,-1]) +
		  sq(u.x[0,0,1] - u.x[0,0,-1]))/sq(2.*Delta);
    }
  }
  ke /= 2.*vol;
  vd *= mu1/vol; 
  double vel_rms  = sqrt(2.0/3.0*ke); 
  double lambda   = vel_rms*sqrt(15.*(mu1/rho1)*(1.0/vd));
 
  foreach() {
    omg_mod[] /= vel_rms/lambda;
  }
  boundary({omg_mod});

  clear();
  view (fov = 44, camera = "iso", ty = .2,
	width = 1200, height = 1200, bg = {1,1,1}, samples = 4);
  box(false, lc = {1,1,1}, lw = 0.1);
  draw_vof("f");
  squares ("omg_mod", linear = true, map = cool_warm, alpha = 1.e-6, n = {1, 0, 0});
  squares ("omg_mod", linear = true, map = cool_warm, alpha = 1.e-6, n = {0, 1, 0});
  squares ("omg_mod", linear = true, map = cool_warm, alpha = 1.e-6, n = {0, 0, 1});
  {
    static FILE * fp = POPEN ("movie", "a");
    save (fp = fp);
    fflush (fp);
  }

}

/**
## End 

   We want to run up "end_sim" physical time. */

event end (t = end_sim) {

  dump ("end.bin");

  if ( pid() == 0 ) {

    char comm[80];
    sprintf (comm, "ln -sf end.bin restart.bin");
    system(comm);

  }

  return 1; // exit

}
