#
# This is to define the class Case 
#
import numpy as np
#
rho_ratio = 850;
mu_ratio  = 25;
Lx = 120;
#
class Case_mp:
  #
  def __init__(self, ReL, WeD, d0l, R0, SIGMA, amp, mu1_in, rho_r, res, nens, work_dir, mean_eps):
      self.ReL   = ReL         # Taylor-microscale Reynolds number
      self.WeD   = WeD         # initial d0-based Weber number
      self.d0l   = d0l         # ratio between initial diameter and Taylor microscale
      self.R0    = R0          # initial radius
      self.sigma = SIGMA       # surface tension
      self.amp = amp           # forcing amplitude
      self.mu1_in = mu1_in     # liquid viscosity
      self.rho_r = rho_r       # Reference density
      self.res = res           # Numerical resolution
      self.nens = nens         # run number
      self.work_dir = work_dir # Directory where simulation data is stored
      self.mean_eps = mean_eps # mean dissipation (from single phase simulations)
  #
  def __repr__(self):
    attrs = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
    return f"{self.__class__.__name__}({attrs})"
  #
#
def cmpt_param_mp(case):
  #
  # Below are the quantities that vary from case to case
  #
  Wec = 3;
  #
  case.rho1 = case.rho_r;
  case.mu1  = case.mu1_in;
  case.rho2 = case.rho1/rho_ratio;
  case.mu2  = case.mu1/mu_ratio;
  case.ted  = ((case.R0*2)**(2./3.))*(case.mean_eps)**(-1./3.);
  case.d_h  = (Wec/2.)**(3.0/5.0)*(case.sigma/case.rho1)**(3.0/5.0)*case.mean_eps**(-2.0/5.0);
  case.Oh_l = case.mu1/np.sqrt(case.rho1*case.sigma*2.*case.R0);
  #
  return case;
#