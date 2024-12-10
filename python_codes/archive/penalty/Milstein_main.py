import numpy as np
import math 
import matplotlib
#matplotlib.use('Tkagg')
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt 
import pdb


# Code to run auxiliary field boson hubbard model for 1 site  
def tstep_EM(_w, wforce, dt, mobility, applynoise, betaU):
  # Step
  Ntau = len(_w)
  #print(wforce[1])
  #_w -= (wforce * dt * mobility) * 3 * (_w**(2/3)) 
  _w -= (wforce * dt * mobility) * 2 * (_w**(1/2)) 

  shift = 0.005
  _correcting = False
  # Element-wise correction
  if(_correcting):
    for i, w in enumerate(_w): 
      if( _w[i].real < 0): 
        _w[i] += -(_w[i].real - shift) 
       # _w[i] = np.conj(_w[i]) 

  dV = 1. 
  scale = np.sqrt(2. * mobility * dt / dV) 
  noise_pcnt = 1.00

  if(applynoise):
    # Generate noise 
    #w_noise = np.random.normal(0., np.sqrt(2. * dt * _mobility), ntau) # real noise
    #w_noise = np.random.normal(0., np.sqrt(2. * dt * _mobility), Ntau) # real noise
    w_noise = np.zeros(len(_w), dtype=np.complex_)
    w_noise += np.random.normal(0., 1., Ntau)  # pure imaginary noise
    w_noise *= scale * noise_pcnt * 1j
    _w += 2. * (_w**(1/2)) * w_noise 
    #_w += 3. * (_w**(2/3)) * w_noise 
    #_w += 3. * dt * mobility * (_w**(1/3)) # euler maruyuma  
    #_w += 3. * (_w**(2/3)) * w_noise # euler maruyuma  
    #_w += 3. * (_w**(2/3)) * w_noise 

    #_w += 3. * (_w**(1/3)) * (w_noise*np.conj(w_noise)) 
    _w += w_noise**2 
    #_w += 2 * (_w ** (1/2)) * (w_noise**2) 
    #_w += 3. * (_w**(1/3)) * (w_noise**2) 
    #_w += 3. * (_w**(1/3)) * (np.abs(w_noise)**2) 
    #_w += 3. * 1j * (_w**(1/3)) * (np.abs(w_noise)**2 + dt) 
    #_w += 3. * 1j * (_w**(1/3)) * (w_noise**2 + dt*(1 - scale)) 

    #_w += 3. * (_w ** (1/3) ) * (1j*w_noise**2 - dt) # EM part 
    #_w += 3. * (_w ** (1/3) ) * (1j * w_noise**2 - dt) # EM part 

    #if( 1j*np.sum(_w)/Ntau > 0 ):
      #print('Warning! Positivity condition violated')
      #_w += -2j * (_w.imag) 
      #_w += -2j * (_w.imag - overall_shift) 

  return _w



def Nk_Bose(beta, mu):
  return 1./(np.exp(-beta * mu) - 1.)


def calc_force_and_ops(beta, ntau, mu, U, w_field, eps_shift):
  # takes rho (transformed x_field) as the input 
  force = np.zeros(ntau, dtype=np.complex_)
  # Term 1
  force += w_field**(1/2)
  #force += w_field**(1/3) 
  force *= beta * U / ntau

  # Term 2
  overall_shift = mu/U + 0.5 + eps_shift
  force += -beta * U * (overall_shift) / ntau 

  # Term 3
  exp_arg = -beta * U * np.sum(w_field**(1/2)) / ntau
  #exp_arg = -beta * U * np.sum(w_field**(1/3)) / ntau
  #exp_arg += beta*U*eps_shift 
  force += (beta * U / ntau) / (np.exp(exp_arg) - 1.)

  #force += -1/(w_field**(1/2))

  # Calc N_operator  
  N_operator = 0. + 1j*0.
  N_operator += overall_shift
  dS_dmu = np.sum(w_field**(1/2)) / ntau # cancelled beta later 
  #dS_dmu = np.sum(w_field**(1/3)) / ntau # cancelled beta later 
  #dS_dmu = 1j * np.sum(w_field) / ntau 
  N_operator += -dS_dmu # omitted beta cancellation  
  N_operator_sq = 0. + 1j*0.
  N_operator_sq = N_operator * N_operator

  # output 
  return (force, N_operator, N_operator_sq)


## System ## 
_U = 1.0
#_U = 0.00001
_beta = 1.00
_mu = 1.10
#_mu = -0.10
epsilon_shift = 0.00
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
_shift = +5
#_w += -(_mu/_U) + 0.5 + _shift 
_w += -1.5 
#_w *= 1j

## Numerics ## 
_dt = 0.005
#_dt = 0.0005
#numtsteps = int(1E7)
#numtsteps = int(200000)
#numtsteps = int(2)
numtsteps = int(150000)
#iointerval = 500
iointerval = 500
_isEM = True
_mobility = 1.0 * ntau 
_applynoise = True
_MF_tol = 1E-6

## Plotting/Output ## 
_isPlotting = True


# Operators 
N_avg = 0. + 1j*0.
N2_avg = 0. + 1j*0.
assert((numtsteps/iointerval).is_integer())
N_samples = int(numtsteps/iointerval)
Partnum_per_site_samples = np.zeros(N_samples, dtype=np.complex_) 
N2_per_site_samples = np.zeros(N_samples, dtype=np.complex_) 
_w_samples = np.zeros(N_samples, dtype=np.complex_) 
ctr = 0

print('Starting simulation')
# main loop 
for i in range(0, numtsteps):
  # Calculate force, det, and N field operator 
  _wforce.fill(0.) 
  #detS, _wforce, N_operator += compute_w_force(_U, _mu, _beta, _w, ntau) 

  N_sample = 0. + 1j*0.
  N2_sample = 0. + 1j*0.
  _wforce, N_sample, N2_sample = calc_force_and_ops(_beta, ntau, _mu, _U, _w, epsilon_shift)

  if(np.isnan(N_sample)):
    print('Trajectory diverged. Particle number is nan, ending simulation')
    break
  #print(_wforce[0])
  #print(N_sample)
  
  # step/propagate the field 
  #print('w before' + str(_w.real) + ' complex ' + str(_w.imag))
  if(_isEM):
    _w = tstep_EM(_w, _wforce, _dt, _mobility, _applynoise, _beta*_U)
  #print('w after ' + str(_w.real) + 'w complex ' + str(_w.imag))

  # Calculate operators and add to average 
  if(_applynoise):
    N_avg += N_sample 
    N2_avg += N2_sample 
    if(i % iointerval == 0): 
      _w_samples[ctr] = np.mean(_w) # integrate over tau 
      N_avg /= iointerval
      N2_avg /= iointerval
      print('Completed ' + str(i) + ' steps. Particle number block avg = ' + str(N_avg) )
      Partnum_per_site_samples[ctr] = N_avg
      N2_per_site_samples[ctr] = N2_sample 
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


thermal_avg_N = np.mean(Partnum_per_site_samples[5:])
#thermal_avg_N = np.mean(Partnum_per_site_samples[5:].real)
thermal_avg_w = np.mean(_w_samples)
thermal_avg_N2 = np.mean(N2_per_site_samples[5:])

if(_applynoise):
  print('Average particle number (real) : ' + str(thermal_avg_N.real) + '\n')
  print('Average particle number (imag) : ' + str(thermal_avg_N.imag) + '\n')
  print('Average _w imaginary value: ' + str(thermal_avg_w.imag) + '\n')
  print('Average _w real value: ' + str(thermal_avg_w.real) + '\n')
  print('Average particle number sq (real) : ' + str(thermal_avg_N2.real) + '\n')
  print('Average particle number sq (imag) : ' + str(thermal_avg_N2.imag) + '\n')



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
  plt.plot(np.array(range(0, N_samples)) * float(iointerval), N2_per_site_samples.real, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Re$[N^2]$')
  plt.plot(np.array(range(0, N_samples)) * float(iointerval), N2_per_site_samples.imag, marker='x', color = 'r', markersize = 4, linewidth = 2., label = 'Im$[N^2]$')
  if(_U == 0):
    plt.axhline(y = Nk_Bose(_beta, _mu) * Nk_Bose, color = 'k', linestyle = 'dashed', label = r'Exact, Ideal gas') 
  else:
    plt.axhline(y = 4.05073, color = 'k', linestyle = 'dashed', label = r'Exact, Sum over states') 
  plt.title('$T = ' + str(_T) + '$, $\mu = $ ' + str(_mu) + ', $U = ' + str(_U) + '$',fontsize = 16)
  plt.ylim(-5, thermal_avg_N**2 + 5)
  plt.xlabel('Iterations', fontsize = 28)
  plt.ylabel('$N^2$', fontsize = 28)
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
  
 


 
