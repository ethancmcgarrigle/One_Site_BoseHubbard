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
  _w += (-wforce * dt * mobility) # += op will modify _w as intended   
  dV = 1. 
  scale = np.sqrt(2. * mobility * dt / dV ) 

  Ntau = len(_w)
  # Mean field or CL? 
  if(applynoise):
    # Generate noise 
    #w_noise = np.random.normal(0., np.sqrt(2. * dt * _mobility), ntau) # real noise
    #w_noise = np.random.normal(0., np.sqrt(2. * dt * _mobility), Ntau) # real noise
    w_noise = np.zeros(len(_w), dtype=np.complex_)
    w_noise += np.random.normal(0., 1., Ntau) # real noise
    w_noise *= scale 
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
  #offdiag_vec = np.zeros(ntau, dtype=np.complex_)
  # fill the vector  
  offdiag_vec = np.zeros(ntau, dtype=np.complex_)
 #  print('w_field: ')
 #  print(w_field)
 #  print()
  offdiag_vec += w_field
  offdiag_vec *= +1j * U 
  #offdiag_vec *= 1j * .sqrt(U) / (np.sqrt(beta/ntau)) 
  #offdiag_vec += 0.5 * U 
  offdiag_vec += mu 
  offdiag_vec *= beta / ntau
  #offdiag_vec = np.exp(offdiag_vec)
  offdiag_vec += 1.  # a_j 

  #print(offdiag_vec)
  # calculate the determinant as 1 -  prod(offdiag_vec_j) for all j 
  #offdiag_vec = 1./offdiag_vec
  prod = offdiag_vec.prod()
  #if(ntau / 2 == 0): # e.g. any integer ntau 
 #  if(float(ntau).is_integer()): # e.g. any integer ntau 
 #    sign = 1. 
 #  else:
 #    sign = -1. 
  detS = 1. - prod

  # check the determinant via a matrix operation 
 #  print()
 #  S = np.diag(np.ones(ntau, dtype=complex)) 
 #  subdiag = np.diag(offdiag_vec[0:-1], k=-1) 
 #  S += subdiag
 #  S[0,-1] = offdiag_vec[-1]
 #
 #  print(S)
 #  print()
 #  print('Calculating determinant: ' + str(np.linalg.det(S)))
 #  print('Manually: ' + str(det))
  #det = np.exp(-prod) 

  # Now that we've gotten det(S) and prod(aj); calc observables and forces 
  N_operator = 0. + 1j*0.
  ddet_dw = np.zeros(ntau,dtype=np.complex_)
  ddet_dw += prod
  ddet_dw /= offdiag_vec
  N_operator += np.sum(ddet_dw)
  ddet_dw *= 1j * beta * U  / float(ntau)
  #ddet_dw *= 1j * np.sqrt(beta * U  / float(ntau))
  ddet_dw /= detS

  # Linear part 
  dS_dw = np.zeros(ntau, dtype=np.complex_) 
  dS_dw += w_field 
  dS_dw *= U * beta / ntau

  # nonlinear part  
  dS_dw += ddet_dw # correct 
  #dS_dw -= ddet_dw 

  N_operator /= ntau
  N_operator /= detS

 #  print()
 #  print('Det(S): \n' )
 #  print(det)
 #  print()
 #
 #  print('dS_dw_j): \n' )
 #  print(dS_d_det)
 #  print()
  #pdb.set_trace() 
  return (dS_dw, N_operator)


## System ## 
_U = 1.0
_beta = 1.00
_mu = 1.10
#ntau = 56
ntau = 40
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
#_w += (_mu / _U) * 1j

## Numerics ## 
_dt = 0.01
#numtsteps = int(1E6)
numtsteps = int(50000)
#numtsteps = int(2)
#numtsteps = int(100)
iointerval = 500
#iointerval = 1
_isEM = True
_mobility = 1.0 * ntau 
_applynoise = True
_MF_tol = 1E-6

## Plotting/Output ## 
_isPlotting = True


# Operators 
N_avg = 0. + 1j*0.
assert((numtsteps/iointerval).is_integer())
N_samples = int(numtsteps/iointerval)
Partnum_per_site_samples = np.zeros(N_samples, dtype=np.complex_) 
_w_samples = np.zeros(N_samples, dtype=np.complex_) 
ctr = 0

print('Starting simulation')
# main loop 
for i in range(0, numtsteps):
  # Calculate force, det, and N field operator 
  _wforce.fill(0.) 
  #detS, _wforce, N_operator += compute_w_force(_U, _mu, _beta, _w, ntau) 

  N_sample = 0. + 1j*0.
  _wforce, N_sample = calc_det_fxns(_beta, ntau, _mu, _U, _w)
  #print(_wforce)
  
  # step/propagate the field 
  #print('w before' + str(_w.real) + ' complex ' + str(_w.imag))
  if(_isEM):
    _w = tstep_EM(_w, _wforce, _dt, _mobility, _applynoise)
  #print('w after ' + str(_w.real) + 'w complex ' + str(_w.imag))

  # Calculate operators and add to average 
  if(_applynoise):
    N_avg += N_sample 
    if(i % iointerval == 0): 
      _w_samples[ctr] = np.mean(_w) # integrate over tau 
      N_avg /= iointerval
      print('Completed ' + str(i) + ' steps. Particle number block avg = ' + str(N_avg) )
      Partnum_per_site_samples[ctr] = N_avg
      # reset N_avg 
      N_avg = 0. + 1j*0. 
      ctr += 1

  # For mean-field, keep going until the tolerance is reached 
  if(not _applynoise):
    if(i % iointerval == 0): 
      _w_samples[ctr] = np.mean(_w)
      Partnum_per_site_samples[ctr] = N_sample 
      ctr += 1

    if(np.max(_wforce).real < _MF_tol):
      print('You have reached the force tolerance. \n')
      print('The mean-field particle number (per site) is  ' + str(N_sample))
      break


thermal_avg_N = np.mean(Partnum_per_site_samples)
thermal_avg_w = np.mean(_w_samples)

if(_applynoise):
  print('Average particle number (real) : ' + str(thermal_avg_N.real) + '\n')
  print('Average particle number (imag) : ' + str(thermal_avg_N.imag) + '\n')
  print('Average _w imaginary value: ' + str(thermal_avg_w.imag) + '\n')
  print('Average _w real value: ' + str(thermal_avg_w.real) + '\n')



plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_orderparams.txt')

if(_isPlotting):
  plt.figure(figsize=(6., 6.))
  plt.plot(np.array(range(0, N_samples)) * float(iointerval), Partnum_per_site_samples.real, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Re[N]')
  plt.plot(np.array(range(0, N_samples)) * float(iointerval), Partnum_per_site_samples.imag, marker='x', color = 'r', markersize = 4, linewidth = 2., label = 'Im[N]')
  if(_U == 0):
    plt.axhline(y = Nk_Bose(_beta, _mu), color = 'k', linestyle = 'dashed', label = r'Exact, Ideal gas ($N_{\tau} \to \infty$)') 
  else:
    plt.axhline(y = 1.85, color = 'k', linestyle = 'dashed', label = r'Exact, Sum over states') 
  plt.title('$T = ' + str(_T) + '$, $\mu = $ ' + str(_mu) + ', $U = ' + str(_U) + '$',fontsize = 16)
  plt.xlabel('Iterations', fontsize = 28)
  plt.ylabel('$N$', fontsize = 28)
  plt.legend()
  plt.show()

  plt.figure(figsize=(6., 6.))
  plt.plot(np.array(range(0, N_samples)) * float(iointerval), _w_samples.real, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Re[w]')
  plt.plot(np.array(range(0, N_samples)) * float(iointerval), _w_samples.imag, marker='p', color = 'b', markersize = 4, linewidth = 2., label = 'Im[w]')
  plt.xlabel('Iterations', fontsize = 28)
  plt.ylabel('$w$', fontsize = 28)
  plt.legend()
  plt.show()
  
  plt.figure(figsize=(6., 6.))
  plt.plot(_w_samples.real, _w_samples.imag, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Aux. Field Theory')
  plt.xlabel('Re[$w$]', fontsize = 28)
  plt.ylabel('Im[$w$]', fontsize = 28)
  plt.legend()
  plt.show()
  
 


 
