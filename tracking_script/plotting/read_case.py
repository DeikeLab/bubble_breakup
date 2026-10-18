#
import numpy as np
from param_fig import *
from def_cases import Case_mp, cmpt_param_mp
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import h5py
import scipy.io as sio
import matplotlib.colors as mcolors
from matplotlib import cm
#
main_dir_mp = "/scratch/gpfs/DEIKE/ns8802/bubbles/multiphase/"
#
# Cases generation - multiphase
#
def make_cases_mp(ReL, WeD, d0l, R0, SIGMA, amp, mu0_in, rho_r,
                  eps_sp, n_cases, Leq):
    d0l_str = f"{d0l:.2f}".replace(".", "p")
    return [
        Case_mp(
            ReL, WeD, d0l, R0, SIGMA, amp, mu0_in, rho_r,
            Leq, s,
            main_dir_mp + f"ReL_{ReL:03d}_WeD{WeD:02d}_d0l{d0l_str}_Leq{Leq:02d}/s{s:02d}/",
            eps_sp
        )
        for s in range(n_cases)
    ]
#
cases_ReL_038_We06p0_dL1p49 = make_cases_mp(
    ReL=38,
    WeD=6,
    d0l=1.49,
    R0=8.318876,
    SIGMA=1.554954e2,
    amp=0.10,
    mu0_in=1.84,
    rho_r=1.00,
    eps_sp=8.9232e0,
    n_cases=20,
    Leq=10,
)
#
cases_ReL_038_We15p0_dL1p49 = make_cases_mp(
    ReL=38,
    WeD=15,
    d0l=1.49,
    R0=8.318876,
    SIGMA=6.219818e1,
    amp=0.10,
    mu0_in=1.84,
    rho_r=1.00,
    eps_sp=8.9232e0,
    n_cases=20,
    Leq=10,
)
#
cases_ReL_038_We30p0_dL1p49 = make_cases_mp(
    ReL=38,
    WeD=30,
    d0l=1.49,
    R0=8.318876,
    SIGMA=3.1067150000e+01,
    amp=0.10,
    mu0_in=1.84,
    rho_r=1.00,
    eps_sp=8.9232e0,
    n_cases=20,
    Leq=10,
)
#
cases_ReL_038_We45p0_dL1p49 = make_cases_mp(
    ReL=38,
    WeD=45,
    d0l=1.49,
    R0=8.318876,
    SIGMA=2.073273e+01,
    amp=0.10,
    mu0_in=1.84,
    rho_r=1.00,
    eps_sp=8.9232e0,
    n_cases=20,
    Leq=10,
)
#
tot_cases_mp = (  cases_ReL_038_We06p0_dL1p49 
                + cases_ReL_038_We15p0_dL1p49 
                + cases_ReL_038_We30p0_dL1p49 
                + cases_ReL_038_We45p0_dL1p49 ) 
#
# compute parameters and load some observables
#
for case in tot_cases_mp:
    #
    # define the parameters
    #
    case = cmpt_param_mp(case);
    #
