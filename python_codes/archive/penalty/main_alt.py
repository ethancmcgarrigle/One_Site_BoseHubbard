import numpy as np
import math 
import matplotlib
#matplotlib.use('Tkagg')
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt 
import pdb


# Code to run auxiliary field boson hubbard model for 1 site  
def tstep_EM(_w, wforce, dt, mobility, applynoise):
  # Step
  Ntau = len(_w)
  #print(wforce[1])
  _w -= (wforce * dt * mobility) # += op will modify _w as intended   
  dV = 1. 
  scale = np.sqrt(2. * mobility * dt / dV) 
  #scale = np.sqrt(mobility * dt / Ntau ) 
  noise_pcnt = 1.00

 #  for i, w in enumerate(_w): 
 #    if( -1j*_w[i] < 1.6 ):
 #      #print('Warning! Positivity condition violated')
 #      _w[i] = np.conj(_w[i]) 

  # Mean field or CL? 
  if(applynoise):
    # Generate noise 
    #w_noise = np.random.normal(0., np.sqrt(2. * dt * _mobility), ntau) # real noise
    #w_noise = np.random.normal(0., np.sqrt(2. * dt * _mobility), Ntau) # real noise
    w_noise = np.zeros(len(_w), dtype=np.complex_)
    w_noise += np.random.normal(0., 1., Ntau) # real noise
    w_noise *= scale * noise_pcnt 
    _w += w_noise

  return _w


 #def tstep_ETD(_w, lincoef, nonlinforce, nonlinforce_coef, dt, mobility, applynoise):
 #  # Step
 #  _w *= lincoef # e^{-dt * mobility * beta U / ntau)
 #  _w += (-wforce * nonlinforce_coef) # += op will modify _w as intended   
 #
 #  dV = 1.
 #  scale = np.sqrt(2. * mobility * dt / dV)
 #  # Mean field or CL? 
 #  Ntau = len(_w)
 #  if(applynoise):
 #    # Generate noise 
 #    #w_noise = np.random.normal(0., np.sqrt(2. * dt * _mobility), ntau) # real noise
 #    #w_noise = np.random.normal(0., np.sqrt(2. * dt * _mobility), Ntau) # real noise
 #    w_noise = np.random.normal(0., 1., Ntau) # real noise
 #    w_noise *= scale
 #    _w += w_noise
 #
 #  return _w
 #

def Nk_Bose(beta, mu):
  return 1./(np.exp(-beta * mu) - 1.)


def calc_det_fxns(beta, ntau, mu, U, w_field):
  # Matrix has PBC structure, so just fill a CSfield (vector for single site) of 
  # fill the vector  
  offdiag_vec = np.zeros(ntau, dtype=np.complex_)
  offdiag_vec += w_field
  #E_tot = beta * 1j * U * np.sum(offdiag_vec) / ntau
  E_tot = 1j * np.sum(offdiag_vec) / ntau
  E_tot += 0.5 * U * beta 
  E_tot += mu * beta

  exp_factor = np.exp(E_tot) # e^{-\Delta_{\tau} \sum_{j} E_{j} }
  det = 1. - exp_factor
  exp_minus_factor = 1./exp_factor 

  # Calc N_operator  
  N_operator = 0. + 1j*0.
  N_operator = 1./(exp_minus_factor - 1.)

  # Linear part 
  dS_dw = np.zeros(ntau, dtype=np.complex_) 
  dS_dw += w_field  
  dS_dw /= (U * beta / ntau )
  #dS_dw *= U * beta / ntau

  # nonlinear part  
  dS_dw += -N_operator * 1j / ntau
  #dS_dw += -N_operator * 1j * U * beta / ntau

  k = 500
  #k = 0
  # Add a penalty
  penalty = np.zeros(ntau, dtype=np.complex_) 
  #penalty += -2 * k * 1j * beta * U / ntau
  penalty += -2 * k * 1j / ntau
  penalty /= (1. + np.exp(-2*k * E_tot.real)) 
  #penalty /= (1. + np.exp(-2*k * np.abs(E_tot))) 
  shift = 0.0001
  #analytical_lim = mu/U + shift
  #analytical_lim = mu/U + 0.5 + shift
  #penalty /= (1. + np.exp(2*k * (np.sum(-w_field*1j)/ntau - analytical_lim) ))
  #print(np.mean(penalty))
  dS_dw += penalty

  #N_operator += (2*k) / (1. + np.exp(2*k * (np.sum(-w_field)*1j/ntau - analytical_lim) ) ) 
  N_operator += (2*k) / (1. + np.exp(-2*k * E_tot.real))
  #N_operator += (2*k) / (1. + np.exp(-2*k * np.abs(E_tot)))

  N_operator_sq = 0. + 1j*0.
  N_operator_sq = N_operator * N_operator

  # output 
  return (dS_dw, N_operator, N_operator_sq)


## System ## 
_U = 1.0
#_U = 0.0
_beta = 50.00
_mu = 1.10
#_mu = -0.10
ntau = 1
_T = 1./_beta
print(' Temperature: ' + str(1./_beta))
print(' Imaginary time discertization: ' + str(_beta / ntau) + '\n')

#print(' mu/U: ' + str(_mu/_U) + '\n')

#(D, dS_dD) = calc_det(_beta, ntau, _mu, _U)
# Create and initialize w fields 
_w = np.zeros(ntau, dtype=np.complex_) 
_wforce = np.zeros(ntau, dtype=np.complex_)  

# initialize w field 
#_w += -(_mu) * 1j
_shift = +1.00
_w += (_mu/_U) + 0.5 + _shift 
_w *= 1j

## Numerics ## 
_dt = 0.001
#numtsteps = int(1E7)
numtsteps = int(1E6)
#numtsteps = int(10000)
#iointerval = 1000
iointerval = 2000
#iointerval = 10
_isEM = True
#_mobility = 1.0
_mobility = 1.0 * ntau 
_applynoise = True
_MF_tol = 1E-6

## Plotting/Output ## 
_isPlotting = True


# Operators 
N_avg = 0. + 1j*0.
N2_avg = 0. + 1j*0.
assert((numtsteps/iointerval).is_integer())
Num_samples = int(numtsteps/iointerval)
Partnum_per_site_samples = np.zeros(Num_samples, dtype=np.complex_) 
N2_per_site_samples = np.zeros(Num_samples, dtype=np.complex_) 
_w_samples = np.zeros(Num_samples, dtype=np.complex_) 
ctr = 0

print('Starting simulation')
# main loop 
for i in range(0, numtsteps):
  # Calculate force, det, and N field operator 
  _wforce.fill(0.) 
  #detS, _wforce, N_operator += compute_w_force(_U, _mu, _beta, _w, ntau) 

  N_sample = 0. + 1j*0.
  N2_sample = 0. + 1j*0.
  _wforce, N_sample, N2_sample = calc_det_fxns(_beta, ntau, _mu, _U, _w)

  if(np.isnan(N_sample)):
    print('Trajectory diverged. Particle number is nan, ending simulation')
    break
  #print(_wforce[0])
  #print(N_sample)
  
  # step/propagate the field 
  #print('w before' + str(_w.real) + ' complex ' + str(_w.imag))
  if(_isEM):
    _w = tstep_EM(_w, _wforce, _dt, _mobility, _applynoise)
  #print('w after ' + str(_w.real) + 'w complex ' + str(_w.imag))

  # Calculate operators and add to average 
  if(_applynoise):
    N_avg += N_sample 
    #N2_avg += (np.mean(_w.imag) ** 2)
    N2_avg += N2_sample 
    if(i % iointerval == 0): 
      _w_samples[ctr] = np.mean(_w) # integrate over tau 
      N_avg /= iointerval
      N2_avg /= iointerval
      print('Completed ' + str(i) + ' steps. Particle number block avg = ' + str(N_avg) )
      Partnum_per_site_samples[ctr] = N_avg
      N2_per_site_samples[ctr] = N2_avg
      # reset N_avg 
      N_avg = 0. + 1j*0. 
      N2_avg = 0. + 1j*0. 
      ctr += 1

  # For mean-field, keep going until the tolerance is reached 
  if(not _applynoise):
    if(i % iointerval == 0): 
      _w_samples[ctr] = np.mean(_w)
      Partnum_per_site_samples[ctr] = N_sample 
      N2_per_site_samples[ctr] = N2_sample 
      ctr += 1

    if(np.max(_wforce).real < _MF_tol):
      print('You have reached the force tolerance. \n')
      print('The mean-field particle number (per site) is  ' + str(N_sample))
      break


thermal_avg_N = np.mean(Partnum_per_site_samples)
thermal_avg_N2 = np.mean(N2_per_site_samples)
thermal_avg_w = np.mean(_w_samples)

if(_applynoise):
  print('Average particle number (real) : ' + str(thermal_avg_N.real) + '\n')
  print('Average particle number (imag) : ' + str(thermal_avg_N.imag) + '\n')
  print('Average particle number sq (real) : ' + str(thermal_avg_N2.real) + '\n')
  print('Average particle number sq (imag) : ' + str(thermal_avg_N2.imag) + '\n')
  print('Average _w imaginary value: ' + str(thermal_avg_w.imag) + '\n')
  print('Average _w real value: ' + str(thermal_avg_w.real) + '\n')



plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_orderparams.txt')

if(_isPlotting):
  plt.figure(figsize=(6., 6.))
  plt.plot(np.array(range(0, Num_samples)) * float(iointerval), Partnum_per_site_samples.real, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Re[N]')
  plt.plot(np.array(range(0, Num_samples)) * float(iointerval), Partnum_per_site_samples.imag, marker='x', color = 'r', markersize = 4, linewidth = 2., label = 'Im[N]')
  if(_U == 0):
    plt.axhline(y = Nk_Bose(_beta, _mu), color = 'k', linestyle = 'dashed', label = r'Exact, Ideal gas ($N_{\tau} \to \infty$)') 
  else:
    plt.axhline(y = 1.84579, color = 'k', linestyle = 'dashed', label = r'Exact, Sum over states') 
  plt.title('$T = ' + str(_T) + '$, $\mu = $ ' + str(_mu) + ', $U = ' + str(_U) + '$',fontsize = 16)
  plt.xlabel('Iterations', fontsize = 28)
  plt.ylabel('$N$', fontsize = 28)
  plt.legend()
  plt.show()


  plt.figure(figsize=(6., 6.))
  plt.plot(np.array(range(0, Num_samples)) * float(iointerval), N2_per_site_samples.real, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Re$[N^2]$')
  plt.plot(np.array(range(0, Num_samples)) * float(iointerval), N2_per_site_samples.imag, marker='x', color = 'r', markersize = 4, linewidth = 2., label = 'Im$[N^2]$')
  if(_U == 0):
    plt.axhline(y = Nk_Bose(_beta, _mu) * Nk_Bose, color = 'k', linestyle = 'dashed', label = r'Exact, Ideal gas') 
  else:
    plt.axhline(y = 4.05073, color = 'k', linestyle = 'dashed', label = r'Exact, Sum over states') 
  plt.title('$T = ' + str(_T) + '$, $\mu = $ ' + str(_mu) + ', $U = ' + str(_U) + '$',fontsize = 16)
  plt.xlabel('Iterations', fontsize = 28)
  plt.ylabel('$N^2$', fontsize = 28)
  plt.legend()
  plt.show()

  plt.figure(figsize=(6., 6.))
  plt.plot(np.array(range(0, Num_samples)) * float(iointerval), _w_samples.real, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Re[w]')
  plt.plot(np.array(range(0, Num_samples)) * float(iointerval), _w_samples.imag, marker='p', color = 'b', markersize = 4, linewidth = 2., label = 'Im[w]')
  analytical_lim = _mu/_U + 0.5 
  plt.axhline(y = analytical_lim, color = 'k', linestyle = 'dashed', label = r'analyiticity limit') 
  plt.xlabel('Iterations', fontsize = 28)
  plt.ylabel('$w$', fontsize = 28)
  plt.legend()
  plt.show()
  
  plt.figure(figsize=(6., 6.))
  plt.plot(_w_samples.real, _w_samples.imag, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Aux. Field Theory')
  plt.axhline(y = analytical_lim, color = 'k', linestyle = 'dashed', label = r'analyiticity limit') 
  plt.xlabel('Re[$w$]', fontsize = 28)
  plt.ylabel('Im[$w$]', fontsize = 28)
  plt.legend()
  plt.show()
  
 


 
