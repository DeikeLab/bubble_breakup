#
module load intel-mpi/gcc/2021.3.1
rm -rf turbulent_bubble && mkdir -p turbulent_bubble
#
# compile
#
CC99='mpicc -std=c99' qcc -grid=octree -events -autolink -O2 -Wall -pipe -D_FORTIFY_SOURCE=2 -D_MPI=1 -o turbulent_bubble/turbulent_bubble turbulent_bubble.c -L$BASILISK/gl -lfb_tiny -lm
