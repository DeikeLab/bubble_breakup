#!/bin/bash
#SBATCH --job-name=08_mu0           # create a short name for your job
#SBATCH --nodes=4                   # node count
#SBATCH --ntasks-per-node=96        # number of tasks per node
#SBATCH --cpus-per-task=1           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G            # memory per cpu-core (4G is default)
#SBATCH --time=23:59:00             # total run time limit (HH:MM:SS)
#
module load intel-mpi/gcc/2021.3.1
#
ulimit -s unlimited
#
# Input parameters
#
R0=8.0;                 
SIGMA=123;                       
amp_force=0.10;                       
mu1_in=0.456407967868879;                       
FORCED=1;
MAX_LEVEL=8;                   
MIN_LEVEL=5;                  
end_sim=10000.;
t_phy=1.;
single_ph=1;
do_restart_tw=0;
do_ens=0;
t_ie=2.;
#
# Extract SBATCH time
#
SBATCH_TIME=$(grep -oP '(?<=#SBATCH --time=)[0-9:]*' $0)
#
srun turbulent_bubble -m $SBATCH_TIME $R0 $SIGMA $amp_force $mu1_in $FORCED $MAX_LEVEL $MIN_LEVEL \
	                 $end_sim $t_phy $single_ph $do_restart_tw $do_ens $t_ie > out.log 2>&1
#
