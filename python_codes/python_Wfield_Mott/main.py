import numpy as np
import math 
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt 


# Code to run auxiliary field boson hubbard model for 1 site  

 #def compute_exp(U, mu, beta, w):
 #  x = beta * (mu + 0.5*U - w*1j)
 #  #x = beta * (mu - w*1j)
 #  return np.exp(-x)
 #
 #
 #def compute_N(U, mu, beta, w):
 #  N = 0. + 1j * 0
 #  N = compute_exp(U, mu, beta, w) 
 #  N -= 1.  
 #  N = 1./N
 #  return N


 #def compute_w_force(U, mu, beta, w, ntau):
 #  F = np.zeros(ntau, dtype=np.complex_) # dS/dw 
 #  #F = compute_N(U, mu, beta, w) 
 #  #F *= 1j*beta 
 #  #F += beta*U*w
 #  det, F = calc_det(beta, ntau, mu, U, w)
 #  return F


def tstep_EM(_w, wforce, dt, mobility, applynoise):
  # Step
  _w += (-wforce * dt * mobility) # += op will modify _w as intended   

  # Mean field or CL? 
  if(applynoise):
    # Generate noise 
    w_noise = np.random.normal(0., np.sqrt(2. * dt * _mobility), ntau) # real noise
    _w += w_noise

  return _w



def calc_det_fxns(beta, ntau, mu, U, w_field):
  # Matrix has PBC structure, so just fill a CSfield (vector for single site) of 
  #offdiag_vec = np.zeros(ntau, dtype=np.complex_)
  # fill the vector  
  offdiag_vec = np.zeros(ntau, dtype=np.complex_)
  offdiag_vec += w_field*1j * U
  offdiag_vec += -0.5*U
  offdiag_vec += -mu 
  offdiag_vec *= -beta / ntau
  offdiag_vec += 1.  # a_j 

  # calculate the determinant as 1 -  prod(offdiag_vec_j) for all j 
  prod = offdiag_vec.prod()
  if(ntau % 2 == 0):
    sign = 1. 
  else:
    sign = -1. 

  det = 1. - prod*sign  # +??  (-1) ** (2*ntau - 1) , therefore a minus sign overall for even ntau 

  # Now that we've gotten det(S) and prod(aj); calc observables and forces 
  N_operator = 0. + 1j*0.
  dS_d_det = np.zeros(ntau,dtype=np.complex_)
  dS_d_det += prod
  dS_d_det /= offdiag_vec  # prod(aj)prime 
  N_operator += np.sum(dS_d_det)
  dS_d_det *= 1j * beta * U  * sign /ntau
  dS_d_det /= det

  # Linear part 
  dS_dw = np.zeros(ntau, dtype=np.complex_) 
  dS_dw += w_field 
  dS_dw *= beta * U / ntau 

  # nonlinear part  
  #dS_dw -= dS_d_det 
  dS_dw += dS_d_det 

  #N_operator *= sign * -1
  N_operator *= sign 
  N_operator /= ntau
  N_operator /= det
 
 #  print()
 #  print('Det(S): \n' )
 #  print(det)
 #  print()
 #
 #  print('dS_dw_j): \n' )
 #  print(dS_d_det)
 #  print()
  return (dS_dw, N_operator)


## System ## 
_U = 0.00
_beta = 0.5
_mu = -0.0001
ntau = 5000
print(' Temperature: ' + str(1./_beta) + '\n')
#print(' mu/U: ' + str(_mu/_U) + '\n')

#(D, dS_dD) = calc_det(_beta, ntau, _mu, _U)
# Create and initialize w fields 
_w = np.zeros(ntau, dtype=np.complex_) 
_wforce = np.zeros(ntau, dtype=np.complex_)  

# initialize w field 
#_w += (_mu / _U) * 1j

## Numerics ## 
_dt = 0.025
#numtsteps = int(1E6)
numtsteps = int(20000)
iointerval = 500
#iointerval = 1
_isEM = True
_mobility = 25.0 
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
  print('Average particle number : ' + str(thermal_avg_N.real) + '\n')
  print('Average _w imaginary value: ' + str(thermal_avg_w.imag) + '\n')
  print('Average _w real value: ' + str(thermal_avg_w.real) + '\n')



plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_orderparams.txt')

if(_isPlotting):
  plt.figure(figsize=(8., 8.))
  plt.plot(range(0, N_samples), Partnum_per_site_samples.real, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Re[N]')
  plt.plot(range(0, N_samples), Partnum_per_site_samples.imag, marker='x', color = 'r', markersize = 4, linewidth = 2., label = 'Im[N]')
  plt.xlabel('Iterations', fontsize = 22)
  plt.ylabel('$N$', fontsize = 22)
  plt.legend()
  plt.show()

  plt.figure(figsize=(8., 8.))
  plt.plot(range(0, N_samples), _w_samples.real, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Re[w]')
  plt.plot(range(0, N_samples), _w_samples.imag, marker='p', color = 'b', markersize = 4, linewidth = 2., label = 'Im[w]')
  plt.xlabel('Iterations', fontsize = 22)
  plt.ylabel('$w$', fontsize = 22)
  plt.legend()
  plt.show()
  
  plt.figure(figsize=(8., 8.))
  plt.plot(_w_samples.real, _w_samples.imag, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Aux. Field Theory')
  plt.xlabel('Re[$w$]', fontsize = 22)
  plt.ylabel('Im[$w$]', fontsize = 22)
  plt.legend()
  plt.show()
  
 


 
