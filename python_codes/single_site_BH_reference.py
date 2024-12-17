import numpy as np
import math 
import pdb


def calculate_local_Hamiltonian(mu, U, n_vector):
    # H = U/2 n^2 + (-mu + U/2)n
    H_tmp = (U*(n_vector**2.)/2.) - (n_vector * (mu + U/2.)) 
    return H_tmp   # size N_vector


def calculate_Z(beta, mu, U, N_terms):
    occupation_nums = np.linspace(0, N_terms-1, N_terms)
    h_n = calculate_local_Hamiltonian(mu, U, occupation_nums)
    weights = np.exp(-beta * h_n) 
    return np.sum(weights)


def calculate_N_avg(beta, mu, U, Z, N_terms, calc_N_squared):
    occupation_nums = np.arange(0., N_terms, 1.)
    h_n = calculate_local_Hamiltonian(mu, U, occupation_nums)
    # <N> = sum n e^{-beta H}/Z 
    weights = np.exp(-beta * h_n) 
    weights *= occupation_nums
    results = [np.sum(weights)/Z]
    if(calc_N_squared):
      N_sq_weights = np.zeros(N_terms)
      N_sq_weights = weights * occupation_nums 
      results.append(np.sum(N_sq_weights)/Z)

    return results

def calculate_U_avg(beta, mu, U, Z, N_terms, calc_U_squared):
    #occupation_nums = np.arange(0., N_terms, 1.)
    occupation_nums = np.linspace(0, N_terms-1, N_terms) 
    # <U> = sum H e^{-beta H}/Z 
    weights = np.exp(-beta * calculate_local_Hamiltonian(mu, U, occupation_nums))
    weights *= calculate_local_Hamiltonian(mu, U, occupation_nums)
    results = [np.sum(weights)/Z]
    if(calc_U_squared):
      U_sq_weights = weights * calculate_local_Hamiltonian(mu, U, occupation_nums)
      results.append(np.sum(U_sq_weights)/Z)

    return results


def display_averages(N_ops, U_ops, method_str = 'sum over states', suppress_output=False):
  ''' Each argument is a list of the operators '''   
  N_op = N_ops[0]
  U_op = U_ops[0]
  print()
  if(not suppress_output):
    print('Single site Bose Hubbard model, ' + method_str + ' reference results\n')
    print('Average particle number : ' + str(N_op) + '\n')
    print('Average internal energy : ' + str(U_op) + '\n')
 
  if( len(N_ops) > 1):
    N2_avg = N_ops[1]

  if( len(U_ops) > 1):
    U2_avg = U_ops[1]
    if(not suppress_output):
      print('Average particle number squared : ' + str(N2_avg) + '\n')
      print('Average internal energy squared : ' + str(U2_avg) + '\n')




def generate_reference(beta, mu, U, N = 500, calcSquared_ops=False, suppress_output=False):
  Z = calculate_Z(beta, mu, U, N)
  N_list = calculate_N_avg(beta, mu, U, Z, N, calcSquared_ops)
  U_list = calculate_U_avg(beta, mu, U, Z, N, calcSquared_ops) 

  display_averages(N_list, U_list, 'Sum over states', suppress_output)  

  return N_list[0], U_list[0]



def S_eff(beta, mu, U, w):
    action = np.zeros(len(w), dtype=np.complex_)
    action = 0.5 * (w**2.) + np.log(1. - np.exp(beta*mu + beta*U*0.5 - 1j*w*np.sqrt(beta*U)))
    return action


def N_w_op(beta, mu, U, w):
  E_tot = beta * (mu + U*0.5 - 1j*np.sqrt(U / beta)*w) 
  # Calc N_operator  
  N_tmp = 0. + 1j*0.
  N_tmp = np.exp(E_tot) / (1. - np.exp(E_tot)) 
  return N_tmp



def internal_energy_w_op(beta, mu, U, w):
   ''' Internal energy operator U[w] in the auxiliary theory ''' 
   E = -(mu + U*0.5 - 0.5*1j*np.sqrt(U / beta)*w) 
   E *= N_w_op(beta, mu, U, w)
   return E  



def contour_integration_ref(beta, mu, U, w_imag_ref, calcSquared_ops = False, display_results = False):
   ''' Function to perform contour integration for an additional reference ''' 
   #print('Single site Bose Hubbard model, contour integration reference results\n')
   # Make the contour: 
   #N_points = 50000
   discretization = 0.1
   if(beta * U > 1.):
     w_real_max = 250. * beta * U 
     #w_real_max = 3.14 
     N_points = int(w_real_max * 2 / discretization) 
     w = np.linspace(-w_real_max, w_real_max, N_points * int(beta * U), dtype=np.complex_)
   else:
     discretization = 0.01
     w_real_max = 500.
     N_points = int(w_real_max*2 / discretization)
     w = np.linspace(-w_real_max, w_real_max, N_points, dtype=np.complex_)

   print('Using ' + str(N_points) + ' points from -' + str(w_real_max) + ' to +' + str(w_real_max))
   # Real part of w 

   # Integrate the w contour with constant imaginary part: 
   w += w_imag_ref*np.ones_like(w, dtype=np.complex_)

   # Partition function: 
   weights = np.exp(-S_eff(beta, mu, U, w) )
   Z = np.trapz(y = weights, x = w)

   N_w = N_w_op(beta, mu, U, w) 
   U_w = internal_energy_w_op(beta, mu, U, w) 

   N_avg = np.trapz(y = weights*N_w, x = w)/Z   
   U_avg = np.trapz(y = weights*U_w, x = w)/Z   
   N2_tmp = np.trapz(y = weights*N_w*N_w, x = w)/Z
   U2_tmp = np.trapz(y = weights*U_w*U_w, x = w)/Z
   N_results = [N_avg.real]
   U_results = [U_avg.real]

   if(calcSquared_ops):
     N_results.append(N2_tmp.real)
     U_results.append(U2_tmp.real)

   if(display_results):
     display_averages(N_results, U_results, 'contour integration', display_results)  

   return N_avg, U_avg



if __name__ == "__main__":
  ''' Calculate the sum over states reference using N terms '''
  N_terms = 5000
  print('Single site bose hubbard reference calculation, using ' + str(N_terms) + ' terms ')
 
  ''' Single site bose-hubbard model: 
    U: onsite repulsion 
    beta: inverse temperature 
    mu: chemical potential '''

  ''' Outputs / Thermodynamic Observables: 
     average site occupation (N) 
     average interal energy (U) 
  '''
  _beta = 1.0
  _mu = 1.10
  _U = 5.00

  limit = -_beta * (_mu + 0.5*_U) / np.sqrt(_beta * _U) 

  print('Inverse temperature: ' + str(_beta) )
  print('Chemical Potential: ' + str(_mu) )
  print('Interaction strength: ' + str(_U) )

  N_exact = generate_reference(_beta, _mu, _U, N_terms, True)

  #N_contour = contour_integration_ref(_beta, _mu, _U, 0, False)
  N_contour, U_avg = contour_integration_ref(_beta, _mu, _U, 1j*(limit - 0.2), True, True)
  

