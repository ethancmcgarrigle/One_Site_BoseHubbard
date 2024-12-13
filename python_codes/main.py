import numpy as np
import math 
import matplotlib
matplotlib.use('Tkagg')
#matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt 
import pdb
from single_site_BH_reference import *
from mpmath import *
mp.pretty = True


def select_dt(dt, wforce, Kref):
  p = 2
  upper_bound = Kref*p
  lower_bound = Kref/p
  #lower_bound = Kref/1000.

  force_modulus = np.abs(wforce)
  if(force_modulus * dt > upper_bound): 
    dt = upper_bound / force_modulus 

  if(force_modulus * dt < lower_bound):
    dt = lower_bound / force_modulus 

  # Absolute upper bound 
  if(dt > 0.1):
    dt = 0.001 # set to a nominal value 

  return dt  



def select_dt_option2(dt, wforce, Kref):
  ''' Another option for adaptive time stepping .
      if dt > 2 / |F|^2 , then replace 
          dt --> dt = 2 / |F|^2
      Can choose a constant of proportionality too, e.g.
        dt = 2 * Kref / |F|^2 ''' 

  force_modulus = np.abs(wforce)
  upper_bound = 2. / force_modulus 
  if(0.5*dt > (2. / (force_modulus**2.))):
    dt = 2. / (force_modulus**2.) 

  # Absolute upper bound 
  if(dt > 0.1):
    dt = 0.001 # set to a nominal value 

  return dt  


def Gaussian_force(w, w_max, var, amplitude = 1.):
  eps = 1e-12
  force = np.zeros_like(w, dtype=np.complex_)
  force = -0.5 * (w.imag - w_max + eps)**2. / (var**2.)
  force = -np.exp(force) * 1j / (np.sqrt(2.*np.pi) * var ) # should be negative and purely imaginary 
  return force*amplitude 


def smeared_delta(w, w_max, var, amplitude = 1.):
  eps = 1e-12
  force = np.zeros_like(w, dtype=np.complex_)
  force = (w.imag - w_max + eps)**2. 
  force += var**2
  force = 1./force
  force *= -1.*var/np.pi
  return force*amplitude*1j 


def lorentzian(w, w_max, var, amplitude = 1.):
  eps = 1e-12
  force = np.zeros_like(w, dtype=np.complex_)
  force = 2.*(w.imag - w_max + eps)**2. 
  force += 0.5*var**2
  force = 1./force
  force *= -1.*var/np.pi
  return force*amplitude*1j 


# Code to run auxiliary field boson hubbard model for 1 site  
def tstep_EM(_w, wforce, dt, mobility, applynoise, enforce_limit=False):
  # Euler Maruyama Step
  Ntau = len(_w)
  _wref = np.zeros_like(_w)
  _wref += _w
  _w -= (wforce * dt * mobility) 

  # need to check if Im[w] > limit or getting too close, Im[w] < limit must be enforced  
  eps = 1E-16
  #while((_w.imag - limit) > -eps):
  if(enforce_limit):
    while((_w.imag - limit) > 0.):
      print('Im[w] has violated the constraint. Lowering the timestep. ')
      # Return w back to its original configuration and lower the step 
      #dt = dt*0.25
      _w.fill(0.)
      _w += _wref
      # Option 2: determine dt such that Im[w] doesn't pass the boundary  
      # Im[w](l+1) = Im[w](l) - dt*Im[force]*mobility
        # (limit - eps) = Im[w](l) - dt*Im[force]*mobility
        # therefore, dt = (limit - eps - Im[w])/(-Im[force] * mobility) 
      dt = (limit - eps - _w.imag)/(-wforce.imag * mobility)
      _w -= (wforce * dt * mobility) 


  dV = 1. 
  scale = np.sqrt(2. * mobility * dt / dV) 
  noise_pcnt = 1.00

  # Mean field or CL? 
  if(applynoise):
    # Generate noise 
    w_noise = np.zeros(len(_w), dtype=np.complex_)
    w_noise += np.random.normal(0., 1., Ntau) # real noise
    w_noise *= scale * noise_pcnt 
    _w += w_noise

  return _w, dt


def tstep_ETD(_w, wforce, dt, mobility, applynoise, enforce_limit=False):
  ''' ETD timestepper for w field'''
  ''' wforce = w + nonlinear-terms'''
  # Separate nonlinear force 
  nonlinforce = wforce - _w 

  # linear coefficient is 1. 
  linear_coef = np.exp(-dt*mobility*np.ones(len(_w)))
  
  # nonlinear t-step coefficient
  nonlin_coef = -(1. - linear_coef)/1. 

  # noisescl coefficient 
  noisescl = np.sqrt((1. - linear_coef*linear_coef)/1.) 

  # Step
  _w *= linear_coef # e^{-dt * mobility * beta U / ntau)
  _w += (nonlinforce * nonlin_coef) 

  if(applynoise):
    # Generate noise 
    w_noise = np.random.normal(0., 1., len(_w)) # real noise
    w_noise *= noisescl 
    _w += w_noise

  return _w, dt



def tstep_ETDRK2(_w, wforce, dt, mobility, applynoise, beta, mu, U, enforce_limit=False):
  ''' ETDRK2 timestepper for w field'''
  ''' wforce = w + nonlinear-terms'''
  # linear coefficient is 1. 
  linear_coef = np.exp(-dt*mobility*np.ones(len(_w)))

  RK2_coeff = -(linear_coef -1. + dt*mobility*1.)/(dt * mobility * 1.*1.)

  prev_nonlin_part = wforce - _w

  # get predicted w via ETD 
  w_predicted = np.zeros_like(_w, dtype=np.complex_)
  w_predicted, dt = tstep_ETD(_w, wforce, dt, mobility, applynoise, False)

  # Evaluate nonlinear force 
  wforce_predicted, x, y = calc_ops_and_forces(beta, len(_w), mu, U, w_predicted, dt_scale = 1.)
  wforce_predicted -= w_predicted # get nonlinear part  

  # Take difference between current and previous nonlinear force  
  wforce_predicted -= prev_nonlin_part 


  _w = w_predicted + (RK2_coeff*wforce_predicted) 
  
  return _w, dt



def project_w(_w, _v, _beta, _mu, _U, dt):
  #eps = 1E-12
  eps = 0.0001
  # Evaluate h[w]
  h_w = 0. # real 
  h_w = _w.imag + (_beta * (_U*0.5 + _mu) / np.sqrt(_beta * _U))

  if(h_w < 0.):
    _v.fill(0.)
    _print_imaginary = False
  else:
    _print_imaginary = False
    # h(w) => 0. represent divergent contributions  
    # Choice 1: send w back to the boundary + an epsilon 
    _v = (2. /  dt) * (eps + h_w) 
    #_v = (-0.1 /  dt) * (eps + h_w) 
    # Choice 2: Reflect across the boundary  
    _v *= 2.
    # Choice 3: delta-function force (hard-wall) 
    
  # Project w
  _w -= _v * 1j * 0.5 * dt
  #_w += _v * 1j * 0.5 * dt 

  if(_print_imaginary):
    print('Performing projection')
    print('Imaginary part of w from inequality projection')
    print(_w.imag)

  return _w, _v




def Nk_Bose(beta, mu):
  return 1./(np.exp(-beta * mu) - 1.)


def calc_ops_and_forces(beta, ntau, mu, U, w_field, dt_scale = 1.):
  ''' Calculates and outputs N[w], U[w], and dS/dw '''
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
  _penalty_width = 0.0005   # good for mu/U ~ 1 
  #_penalty_width = 0.0005    # good for mu/U >> 1
  dS_dw -= (1./np.sqrt(beta*U)) * np.pi * smeared_delta(_w, limit, _penalty_width, _penalty_strength)

  #dS_dw -= (1./np.sqrt(beta*U)) * np.pi * lorentzian(_w, limit, _penalty_width, _penalty_strength)

  #dS_dw -= (np.sqrt(beta/U)**(-1.)) * np.pi * smeared_delta(_w, limit, _penalty_width, _penalty_strength)
  #dS_dw -= (np.sqrt(beta/U)**(-1.)) * np.pi * Gaussian_force(_w, limit, _penalty_width, _penalty_strength)
  #dS_dw -= (np.sqrt(beta * U)**-1.) * np.pi * Gaussian_force(_w, limit, _penalty_width, _penalty_strength)
  return (dS_dw, N_tmp, U_tmp)


def return_saddle_pt(beta, mu, U, w_init):
  '''Returns the physical saddle point of the w theory''' 
  cost = 1.
  tol = 1E-7 # iteration tolerance 
  w_tmp = 0. 
  w_tmp += w_init
  alpha = 1E-2 # relaxation parameter
  ctr = 1
  while(cost > tol):
    tmp_force, _, _ = calc_ops_and_forces(beta, len(w_init), mu, U, w_tmp)
    w_tmp -= tmp_force*alpha 
    cost = np.abs(tmp_force)
    if(cost < tol):
      print('Converged to a saddle point in ' + str(ctr) + ' iterations.')
      return w_tmp

    ctr += 1
    if ctr > 1000:
      print('We have exceeded the max number of iterations. Quitting') 
      return 0. 
     
  return w_tmp 




if __name__ == "__main__":
  ''' Script to run a CL simulation of the single-site Bose Hubbard model in the auxiliary variable representation'''
  ## System ## 
  _U = 1.00
  _beta = 1.0
  _mu = 1.1
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

  w_saddle = return_saddle_pt(_beta, _mu, _U, _w)
  print('Saddle point : ' + str(w_saddle))
  print('N operator at the saddle point : ' + str(N_w_op(_beta, _mu, _U, w_saddle)))
  
  
  ## Numerics ## 
  _dt = 0.001
  #_dt = 0.01
  _dt_nominal = _dt
  numtsteps = int(500000)
  iointerval = 1000
  _isEM = False
  if((_beta * _U) > 1.):
    _mobility = 1./(_beta * _U) * ntau 
    #_mobility = 1./np.sqrt(_beta * _U) * ntau 
  else:
    _mobility = (np.sqrt(_beta * _U)) * ntau 

  _applynoise = True
  _MF_tol = 1E-6
  
  ## Plotting/Output ## 
  _isPlotting = True
  
  
  # Operators 
  N_avg = 0. + 1j*0.
  N2_avg = 0. + 1j*0.
  U_avg = 0. + 1j*0.
  U2_avg = 0. + 1j*0.
  w_avg = 0. + 1j*0.
  v_avg = 0. + 1j*0.
  dt_avg = 0. 
  log_arg_avg = 0. 
  assert((numtsteps/iointerval).is_integer())
  N_samples = int(numtsteps/iointerval)
  Partnum_per_site_samples = np.zeros(N_samples, dtype=np.complex_) 
  N2_samples = np.zeros(N_samples, dtype=np.complex_) 
  U2_samples = np.zeros(N_samples, dtype=np.complex_) 
  U_samples = np.zeros(N_samples, dtype=np.complex_) 
  _w_samples = np.zeros(N_samples, dtype=np.complex_) 
  _v_samples = np.zeros(N_samples, dtype=np.complex_) 
  _dt_samples = np.zeros(N_samples, dtype=np.complex_) 
  _log_arg_samples = np.zeros(N_samples, dtype=np.complex_) 
  _w_samples[0] += _w
  _v_samples[0] += _v
  _dt_samples[0] = _dt_nominal 
  _log_arg_samples[0] = 1. - np.exp(_beta*_mu + _beta*_U*0.5 - 1j*_w[0]*np.sqrt(_beta*_U))
  U_samples[0] = U_avg
  ctr = 0
  
  
  _Kref = 1E-2
  #_Kref = 5E-4
  print('Starting simulation')
  adaptive_timestepping = True
  
  # main loop 
  for i in range(1, numtsteps+1):
    # Calculate force, det, and N field operator 
    _wforce.fill(0.) 
  
    N_sample = 0. + 1j*0.
    _wforce, N_sample, U_sample =  calc_ops_and_forces(_beta, ntau, _mu, _U, _w, _dt)
  
    if(adaptive_timestepping):
      _dt = select_dt(_dt, _wforce, _Kref)
      #_wforce, N_sample, U_sample =  calc_ops_and_forces(_beta, ntau, _mu, _U, _w, _dt)
      #_dt = select_dt_option2(_dt, _wforce, _Kref)
    
    # step/propagate the field 
    if(_isEM):
      _w, _dt = tstep_EM(_w, _wforce, _dt, _mobility, _applynoise, False)
    else:
      _w, _dt = tstep_ETDRK2(_w, _wforce, _dt, _mobility, _applynoise, _beta, _mu, _U, False)
      #_w, _dt = tstep_ETD(_w, _wforce, _dt, _mobility, _applynoise, False)
    
    # project _w s.t. the inequality constraint is obeyed: h<0
    #_w, _v = project_w(_w, _v, _beta, _mu, _U, _dt)
  
    # Calculate operators and add to average 
    if(_applynoise):
      N_avg += N_sample
      N2_avg += (N_sample**2.)
      U_avg += U_sample
      U2_avg += U_sample**2.
      w_avg += _w[0] 
      v_avg += _v[0] 
      dt_avg += _dt
      log_arg_avg += 1. - np.exp(_beta*_mu + _beta*_U*0.5 - 1j*_w[0]*np.sqrt(_beta*_U))
  
      if(i % iointerval == 0): 
        _dt_samples[ctr] = dt_avg/iointerval
        _w_samples[ctr] = w_avg/iointerval 
        _v_samples[ctr] = v_avg/iointerval 
        _log_arg_samples[ctr] = log_arg_avg/iointerval 
        N_avg /= iointerval
        N2_avg /= iointerval
        U2_avg /= iointerval
        U_avg /= iointerval
        print('Completed ' + str(i) + ' steps. Particle number block avg = ' + str(N_avg) )
        Partnum_per_site_samples[ctr] = N_avg
        N2_samples[ctr] = N2_avg
        U_samples[ctr] = U_avg
        U2_samples[ctr] = U2_avg
        # reset N_avg 
        N_avg = 0. + 1j*0. 
        U_avg = 0. + 1j*0. 
        N2_avg = 0. + 1j*0. 
        w_avg = 0. + 1j*0. 
        v_avg = 0. + 1j*0. 
        dt_avg = 0. 
        log_arg_avg = 0. 
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
  thermal_avg_N2 = np.mean(N2_samples)
  thermal_avg_w = np.mean(_w_samples)
  
  if(_applynoise):
    print('Average particle number (real) : ' + str(thermal_avg_N.real) + '\n')
    #print('Average particle number (imag) : ' + str(thermal_avg_N.imag) + '\n')
    print('Average particle number squared (real) : ' + str(thermal_avg_N2.real) + '\n')
    #print('Average particle number squared (imag) : ' + str(thermal_avg_N2.imag) + '\n')
    #print('Average _w imaginary value: ' + str(thermal_avg_w.imag) + '\n')
    #print('Average _w real value: ' + str(thermal_avg_w.real) + '\n')
    print('Average internal energy (real) : ' + str(np.mean(U_samples).real) + '\n')
    print('Average internal energy squared (real) : ' + str(np.mean(U2_samples).real) + '\n')
    #print('Average interal energy (imag) : ' + str(np.mean(U_samples).imag) + '\n')
  
  
  # Calculate and disdplay the exact reference 
  N_exact, U_exact = generate_reference(_beta, _mu, _U, 5000, True)
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

 #    plt.figure(figsize=(6., 6.))
 #    plt.plot(np.array(range(0, N_samples)) * float(iointerval), _log_arg_samples, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'arg[z]')
 #    #plt.axvline(x = 0., color = 'k', linestyle = 'dashed', label = r'Nominal') 
 #    plt.title('$T = ' + str(_T) + '$, $\mu = $ ' + str(_mu) + ', $U = ' + str(_U) + '$',fontsize = 16)
 #    plt.xlabel('CL Iterations', fontsize = 28)
 #    plt.ylabel('Argument', fontsize = 28)
 #    plt.legend()
 #    plt.show()

    plt.figure(figsize=(6., 6.))
    plt.plot(_log_arg_samples.real, _log_arg_samples.imag, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'z')
    plt.axvline(x = 0., color = 'k', linestyle = 'dashed', label = r'x = 0') 
    plt.title('$T = ' + str(_T) + '$, $\mu = $ ' + str(_mu) + ', $U = ' + str(_U) + '$',fontsize = 16)
    plt.xlabel('Re[z]', fontsize = 28)
    plt.ylabel('Im[z]', fontsize = 28)
    plt.legend()
    plt.show()
  
    plt.figure(figsize=(6., 6.))
    plt.plot(np.array(range(0, N_samples)) * float(iointerval), _w_samples.real, marker='o', color = 'k', markersize = 4, linewidth = 2., label = 'Re[w]')
    plt.plot(np.array(range(0, N_samples)) * float(iointerval), _w_samples.imag, marker='p', color = 'b', markersize = 4, linewidth = 2., label = 'Im[w]')
    plt.axhline(y = limit, color = 'r', linestyle='dashed', label = 'Inequality bound', linewidth = 2.) 
    plt.axhline(y = w_saddle[0].imag, color = 'g', linestyle='dashed', label = 'Saddle point', linewidth = 2.) 
    plt.xlabel('Iterations', fontsize = 28)
    plt.ylabel('$w$', fontsize = 28)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(6., 6.))
    plt.plot(_w_samples.real, _w_samples.imag, marker='o', color = 'k', markersize = 4, linewidth = 0., label = 'Aux. Field Theory')
    plt.axhline(y = limit, color = 'r', linestyle='dashed', label = 'Inequality bound', linewidth = 2.) 
    plt.scatter(np.array([w_saddle[0].real]), np.array([w_saddle[0].imag]), linewidth = 0., color = 'r', label = 'Saddle point') 
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
  
