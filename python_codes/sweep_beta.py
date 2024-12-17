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
from scipy.stats import sem 
from main import * 

  


if __name__ == "__main__":
  ''' Script to run a CL simulation of the single-site Bose Hubbard model in the auxiliary variable representation'''

  plot_results = False
  U = 1.0
  #beta = np.arange(0.1, 2.0, 0.05) 
  T = np.arange(0.5, 10.0, 0.5) 
  beta = 1./T
  beta = np.sort(beta)
  mu = 1.1

  N = np.zeros_like(beta)
  N_err = np.zeros_like(beta)
  N2 = np.zeros_like(beta)
  N2_err = np.zeros_like(beta)
  U_energy = np.zeros_like(beta)
  U_err = np.zeros_like(beta)
  U2 = np.zeros_like(beta)
  U2_err = np.zeros_like(beta)

  for i, beta_value in enumerate(beta):
    # Run the CL simulation 
    ops_list = auxiliary_field_CL(U, beta_value, mu, plot_results, True)  
    # Extract the results  
    N[i] = ops_list[0]
    N_err[i] = ops_list[1]
    N2[i] = ops_list[2]
    N2_err[i] = ops_list[3]
    U_energy[i] = ops_list[4]
    U_err[i] = ops_list[5]
    U2[i] = ops_list[6]
    U2_err[i] = ops_list[7]
    #print('Finished CL simulation at beta = ' + str(beta_value) )


  #N_ref = np.zeros(1000)
  #N_exact, U_exact = generate_reference(_beta, _mu, _U, 5000, True)
  T_ref = np.linspace(T[0], T[-1], 2000)
  beta_ref = 1./T_ref
  beta_ref = np.sort(beta_ref)
  N_ref = np.zeros_like(beta_ref)
  U_ref = np.zeros_like(beta_ref)
  for i, beta_val in enumerate(beta_ref):
    N_ref[i], U_ref[i] = generate_reference(beta_val, mu, U, 500, True, False) 

  plt.style.use('~/tools_csbosons/python_plot_styles/plot_style_data.txt')  
  plt.figure(figsize=(4., 4.))
  plt.errorbar(1./beta, N, yerr = N_err, marker='o', color = 'b', markersize = 4, linewidth = 0., elinewidth = 2., label = 'Re[N]')
  plt.plot(1./beta_ref, N_ref, color = 'b', markersize = 0, linewidth = 2., label = 'Exact reference')
  plt.xlabel('$T$', fontsize = 28)
  plt.ylabel('$N$', fontsize = 28)
  plt.savefig('N_vs_T.pdf', dpi = 300)
  plt.legend()
  plt.show()

  plt.figure(figsize=(4., 4.))
  plt.errorbar(1./beta, U_energy, yerr = N_err, marker='o', color = 'b', markersize = 4, linewidth = 0., elinewidth = 2., label = 'Re[U]')
  plt.plot(1./beta_ref, U_ref, color = 'b', markersize = 0, linewidth = 2., label = 'Exact reference')
  plt.xlabel('$T$', fontsize = 28)
  plt.ylabel('$U$', fontsize = 28)
  plt.legend()
  plt.savefig('U_vs_T.pdf', dpi = 300)
  plt.show()
  #header_name = '
  #np.savetxt(''

