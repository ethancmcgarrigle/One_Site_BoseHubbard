import numpy as np
import math 
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt 


def compute_exp(U, mu, beta, w):
  x = beta * (mu + 0.5*U - w*1j)
  #x = beta * (mu - w*1j)
  return np.exp(-x)


def compute_N(U, mu, beta, w):
  N = 0. + 1j * 0
  N = compute_exp(U, mu, beta, w) 
  N -= 1.  
  N = 1./N
  return N


def compute_w_force(U, mu, beta, w):
  F = 0. + 1j*0 # dS/dw 
  F = compute_N(U, mu, beta, w) 
  F *= 1j*beta 
  F += beta*U*w
  return F


def tstep_EM(_w, wforce, dt, mobility, applynoise):
  # Step
  _w += (-wforce * dt * mobility) # += op will modify _w as intended   

  # Mean field or CL? 
  if(applynoise):
    # Generate noise 
    w_noise = np.random.normal(0., np.sqrt(2. * dt * _mobility), size=None) # real noise
    _w += w_noise

  return _w


# Create and initialize w fields 
_w = 0. + 1j*0.
_wforce = 0. + 1j*0.

# TODO add parser 
## System ## 
_U = 0.5
_beta = 50.0
_mu = 2.6
print(' Temperature: ' + str(1./_beta) + '\n')
print(' mu/U: ' + str(_mu/_U) + '\n')

## Numerics ## 
_dt = 0.001
numtsteps = int(1E6)
iointerval = 500
_isEM = True
_mobility = 1.0
_applynoise = True
_MF_tol = 1E-6

## Plotting/Output ## 
_isPlotting = False


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
  # Calculate force
  _wforce = 0.
  _wforce += compute_w_force(_U, _mu, _beta, _w) 
  #print(_wforce)
  
  # step/propagate the field 
  #print('w before' + str(_w.real) + ' complex ' + str(_w.imag))
  if(_isEM):
    _w = tstep_EM(_w, _wforce, _dt, _mobility, _applynoise)
  #print('w after ' + str(_w.real) + 'w complex ' + str(_w.imag))

  # Calculate operators and add to average 
  if(_applynoise):
    N_avg += compute_N(_U, _mu, _beta, _w) 
    if(i % iointerval == 0): 
      _w_samples[ctr] = _w
      N_avg /= iointerval
      Partnum_per_site_samples[ctr] = N_avg
      # reset N_avg 
      N_avg = 0. + 1j*0. 
      ctr += 1

  # For mean-field, keep going until the tolerance is reached 
  if(not _applynoise):
    if(i % iointerval == 0): 
      _w_samples[ctr] = _w
      Partnum_per_site_samples[ctr] = compute_N(_U, _mu, _beta, _w) 
      ctr += 1

    if(_wforce < _MF_tol):
      print('You have reached the force tolerance. \n')
      print('The mean-field particle number (per site) is  ' + str(compute_N(_U, _mu, _beta, _w)))
      break


thermal_avg_N = np.mean(Partnum_per_site_samples)
thermal_avg_w = np.mean(_w_samples)

if(_applynoise):
  print('Average particle number : ' + str(thermal_avg_N.real) + '\n')
  print('Average _w imaginary value: ' + str(thermal_avg_w.imag) + '\n')



plt.style.use('~/CSBosonsCpp/tools/Figs_scripts/plot_style_orderparams.txt')

if(_isPlotting):
  plt.figure(figsize=(8., 8.))
  plt.plot(range(0, N_samples), Partnum_per_site_samples.real, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Aux. Field Theory')
  plt.xlabel('Iterations', fontsize = 22)
  plt.ylabel('$N$', fontsize = 22)
  plt.legend()
  plt.show()
  
  
  
  plt.figure(figsize=(8., 8.))
  plt.plot(_w_samples.real, _w_samples.imag, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Aux. Field Theory')
  plt.xlabel('Re[$w$]', fontsize = 22)
  plt.ylabel('Im[$w$]', fontsize = 22)
  plt.legend()
  plt.show()
  
  plt.figure(figsize=(8., 8.))
  plt.plot(range(0, N_samples), _w_samples.imag, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Aux. Field Theory')
  plt.xlabel('Iterations', fontsize = 22)
  plt.ylabel('Im[$w$]', fontsize = 22)
  plt.legend()
  plt.show()
 


 
