import numpy as np
import math 
import matplotlib
matplotlib.use('Tkagg')
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt 
import pdb
from single_site_BH_reference import *

from main import *


def N_w_op_modified(beta, mu, U, w):
  E_tot = beta * (mu + U*0.5 - 1j*np.sqrt(U / beta)*w) 
  # Calc N_operator  
  N_tmp = 0. + 1j*0.
  N_tmp = np.exp(E_tot) / (1. - np.exp(E_tot)).real 
  return N_tmp



def calc_ops_and_forces_quenched(beta, ntau, mu, U, w_field):
  ''' Calculates and outputs N[w], U[w], and dS/dw '''
  ''' Phase quenched model, S = w^2/2 + Re[ln[1 - exp(..)]] ''' 
  
  # Calc N_operator  
  N_tmp = N_w_op(beta, mu, U, w_field)
  U_tmp = internal_energy_w_op(beta, mu, U, w_field)

  # Force 
  dS_dw = np.zeros_like(w_field)
  dS_dw += w_field  

  # nonlinear part  
  dS_dw += N_tmp * 1j * np.sqrt(U * beta)
  #dS_dw += N_tmp * 1j * np.sqrt(U * beta / ntau)

  _penalty_strength = 1.00
  #_penalty_width = 0.005     # 0.01 worked really well with dt = 0.01 ; penalty = 0.1 was biased 
  #_penalty_width = 0.01     # 0.01 worked really well with dt = 0.01 ; penalty = 0.1 was biased 
  _penalty_width = 0.000045    # 0.01 worked really well with dt = 0.01 ; penalty = 0.1 was biased 
  dS_dw -= smeared_delta(_w, limit, _penalty_width, _penalty_strength)
  #dS_dw -= Gaussian_force(_w, limit, _penalty_width, _penalty_strength)
  return (dS_dw, N_tmp, U_tmp)





if __name__ == "__main__":
  ''' Script to run a CL simulation of the single-site Bose Hubbard model in the auxiliary variable representation'''
  ## System ## 
  _U = 1.00
  _beta = 1.0
  _mu = 1.0
  ntau = 1     # keep at 1 
  _T = 1./_beta
  print(' Temperature: ' + str(1./_beta))
  print(' Imaginary time discertization: ' + str(_beta / ntau) + '\n')
  
  # Create and initialize w fields 
  _w = np.zeros(ntau, dtype=np.complex_) 
  _wforce = np.zeros(ntau, dtype=np.complex_)  
  
  _v = np.zeros(ntau, dtype=np.complex_) # dual variable 
  
  # initialize w field 
  ''' Inequality constraint: Im[w] < -beta * (mu + U/2)/sqrt(betaU) for a convergent partition function ''' 
  _shift = +2.
  
  # constraint is on the imaginary part of _w 
  limit = -_beta * (_mu + 0.5*_U) / np.sqrt(_beta * _U) 
  print('Upper Im[w] limit: ' )
  print(limit)
  _w += limit 
  _w += -_shift # extra shift for conservative starting point 
  _w *= 1j 
  
  ## Numerics ## 
  _dt = 0.005
  _dt_nominal = _dt
  numtsteps = int(100000)
  iointerval = 1000
  _isEM = True
  _mobility = 1./(np.sqrt(_beta * _U)) * ntau 
  _applynoise = True
  _MF_tol = 1E-6
  
  ## Plotting/Output ## 
  _isPlotting = True
  
  
  # Operators 
  N_avg = 0. + 1j*0.
  N2_avg = 0. + 1j*0.
  U_avg = 0. + 1j*0.
  w_avg = 0. + 1j*0.
  v_avg = 0. + 1j*0.
  dt_avg = 0. 
  assert((numtsteps/iointerval).is_integer())
  N_samples = int(numtsteps/iointerval)
  Partnum_per_site_samples = np.zeros(N_samples, dtype=np.complex_) 
  N2_samples = np.zeros(N_samples, dtype=np.complex_) 
  U_samples = np.zeros(N_samples, dtype=np.complex_) 
  _w_samples = np.zeros(N_samples, dtype=np.complex_) 
  _v_samples = np.zeros(N_samples, dtype=np.complex_) 
  _dt_samples = np.zeros(N_samples, dtype=np.complex_) 
  _w_samples[0] += _w
  _v_samples[0] += _v
  _dt_samples[0] = _dt_nominal 
  U_samples[0] = U_avg
  ctr = 0
  
  
  _Kref = 1E-2
  print('Starting simulation')
  adaptive_timestepping = True
  
  
  # main loop 
  for i in range(numtsteps):
    # Calculate force, det, and N field operator 
    _wforce.fill(0.) 
  
    N_sample = 0. + 1j*0.
    #_wforce, N_sample, U_sample =  calc_ops_and_forces(_beta, ntau, _mu, _U, _w)
    _wforce, N_sample, U_sample = calc_ops_and_forces_quenched(_beta, ntau, _mu, _U, _w)
  
    if(adaptive_timestepping):
      _dt = select_dt(_dt, _wforce, _Kref)
      #_dt = select_dt_option2(_dt, _wforce, _Kref)
    
    # step/propagate the field 
    if(_isEM):
      _w, _dt = tstep_EM(_w, _wforce, _dt, _mobility, _applynoise, False)
    
    # project _w s.t. the inequality constraint is obeyed: h<0
    #_w, _v = project_w(_w, _v, _beta, _mu, _U, _dt)
  
    # Calculate operators and add to average 
    if(_applynoise):
      N_avg += N_sample
      N2_avg += (N_sample**2.)
      U_avg += U_sample
      w_avg += _w[0] 
      v_avg += _v[0] 
      dt_avg += _dt
  
      if(i % iointerval == 0): 
        _dt_samples[ctr] = dt_avg/iointerval
        _w_samples[ctr] = w_avg/iointerval 
        _v_samples[ctr] = v_avg/iointerval 
        N_avg /= iointerval
        N2_avg /= iointerval
        U_avg /= iointerval
        print('Completed ' + str(i) + ' steps. Particle number block avg = ' + str(N_avg) )
        Partnum_per_site_samples[ctr] = N_avg
        N2_samples[ctr] = N2_avg
        U_samples[ctr] = U_avg
        # reset N_avg 
        N_avg = 0. + 1j*0. 
        U_avg = 0. + 1j*0. 
        N2_avg = 0. + 1j*0. 
        w_avg = 0. + 1j*0. 
        v_avg = 0. + 1j*0. 
        dt_avg = 0. 
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
  thermal_avg_N2 = np.mean(N2_samples.real)
  thermal_avg_w = np.mean(_w_samples)
  
  if(_applynoise):
    print('Average particle number (real) : ' + str(thermal_avg_N.real) + '\n')
    #print('Average particle number (imag) : ' + str(thermal_avg_N.imag) + '\n')
    print('Average particle number squared (real) : ' + str(thermal_avg_N2.real) + '\n')
    #print('Average particle number squared (imag) : ' + str(thermal_avg_N2.imag) + '\n')
    #print('Average _w imaginary value: ' + str(thermal_avg_w.imag) + '\n')
    #print('Average _w real value: ' + str(thermal_avg_w.real) + '\n')
    print('Average internal energy (real) : ' + str(np.mean(U_samples).real) + '\n')
    #print('Average interal energy (imag) : ' + str(np.mean(U_samples).imag) + '\n')
  
  
  # Calculate and disdplay the exact reference 
  N_exact, U_exact = generate_reference(_beta, _mu, _U, 500, True)
  print('Contour integration references')
  N_contour_wrong, U_contour_wrong = contour_integration_ref(_beta, _mu, _U, 0., False)
  N_contour_correct, U_contour_correct = contour_integration_ref(_beta, _mu, _U, 1j*(limit - 0.25), False)
  
  
  plt.style.use('~/tools_csbosons/python_plot_styles/plot_style_data.txt')  
  
  if(_isPlotting):
    plt.figure(figsize=(6., 6.))
    plt.plot(np.array(range(0, N_samples)) * float(iointerval), Partnum_per_site_samples.real, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Re[N]')
    plt.plot(np.array(range(0, N_samples)) * float(iointerval), Partnum_per_site_samples.imag, marker='x', color = 'r', markersize = 2, linewidth = 0.5, label = 'Im[N]')
    if(_U == 0):
      plt.axhline(y = Nk_Bose(_beta, _mu), color = 'k', linestyle = 'dashed', label = r'Exact, Ideal gas ($N_{\tau} \to \infty$)') 
    else:
      plt.axhline(y = N_exact, color = 'k', linestyle = 'dashed', label = r'Exact, Sum over states') 
      plt.axhline(y = N_contour_wrong, color = 'r', linestyle = 'dashed', label = r'Contour integral, Im$[w] = 0$') 
      #plt.axhline(y = N_contour_correct, color = 'b', linestyle = 'dashed', label = r'Contour integral, $Im[w] < A$') 
    plt.title('$T = ' + str(_T) + '$, $\mu = $ ' + str(_mu) + ', $U = ' + str(_U) + '$',fontsize = 16)
    plt.xlabel('CL Iterations', fontsize = 28)
    plt.ylabel('$N$', fontsize = 28)
    plt.legend()
    plt.show()
  
    plt.figure(figsize=(6., 6.))
    plt.plot(np.array(range(0, N_samples)) * float(iointerval), U_samples.real, marker='o', color = 'b', markersize = 4, linewidth = 2., label = 'Re[U]')
    plt.plot(np.array(range(0, N_samples)) * float(iointerval), U_samples.imag, marker='x', color = 'r', markersize = 2, linewidth = 0.5, label = 'Im[U]')
    plt.axhline(y = U_exact, color = 'k', linestyle = 'dashed', linewidth = 2.0, label = r'Exact, Sum over states') 
    plt.axhline(y = U_contour_wrong, color = 'r', linestyle = 'dashed', linewidth = 2.0, label = r'Contour, Im[$w$]$ = 0$') 
      #plt.axhline(y = N_contour_wrong, color = 'r', linestyle = 'dashed', label = r'Contour integral, $Im[w] = 0$') 
      #plt.axhline(y = N_contour_correct, color = 'b', linestyle = 'dashed', label = r'Contour integral, $Im[w] < A$') 
    plt.title('$T = ' + str(_T) + '$, $\mu = $ ' + str(_mu) + ', $U = ' + str(_U) + '$',fontsize = 16)
    plt.xlabel('CL Iterations', fontsize = 28)
    plt.ylabel('$U$', fontsize = 28)
    plt.legend()
    plt.show()
  
    if(adaptive_timestepping):
      plt.figure(figsize=(6., 6.))
      plt.plot(np.array(range(0, N_samples)) * float(iointerval), _dt_samples, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Timestep')
      plt.axhline(y = _dt_nominal, color = 'k', linestyle = 'dashed', label = r'Nominal') 
      plt.title('$T = ' + str(_T) + '$, $\mu = $ ' + str(_mu) + ', $U = ' + str(_U) + '$',fontsize = 16)
      plt.xlabel('CL Iterations', fontsize = 28)
      plt.ylabel('$\Delta t$', fontsize = 28)
      plt.legend()
      plt.show()
  
    plt.figure(figsize=(6., 6.))
    plt.plot(np.array(range(0, N_samples)) * float(iointerval), _w_samples.real, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Re[w]')
    plt.plot(np.array(range(0, N_samples)) * float(iointerval), _w_samples.imag, marker='p', color = 'b', markersize = 4, linewidth = 2., label = 'Im[w]')
    plt.axhline(y = limit, color = 'r', linestyle='dashed', label = 'Inequality bound', linewidth = 2.) 
    plt.xlabel('Iterations', fontsize = 28)
    plt.ylabel('$w$', fontsize = 28)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(6., 6.))
    plt.plot(_w_samples.real, _w_samples.imag, marker='o', color = 'k', markersize = 4, linewidth = 0., label = 'Aux. Field Theory')
    plt.axhline(y = limit, color = 'r', linestyle='dashed', label = 'Inequality bound', linewidth = 2.) 
    plt.xlabel('Re[$w$]', fontsize = 28)
    plt.ylabel('Im[$w$]', fontsize = 28)
    plt.legend()
    plt.show()
    
   #  plt.figure(figsize=(6., 6.))
   #  plt.plot(_v_samples.real, _v_samples.imag, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Aux. Field Theory')
   #  plt.xlabel('Re[$v$]', fontsize = 28)
   #  plt.ylabel('Im[$v$]', fontsize = 28)
   #  plt.legend()
   #  plt.show()
   # 
  
