import numpy as np
from ssa import optimize, gen_toy_data, process_data

toy_data_params = {}
# Specify means of the stationary sources (2 dimensions in this case)
toy_data_params['mu_stationary'] = [0,1]
# Specify number of noisy dimensions
toy_data_params['dimensions_noisy'] = 2
# 10 epochs of varying size
toy_data_params['epoch_sizes'] = [np.random.randint(1000,10000) for x in range(10)]
# Generate the mixed data, and store the true mixing matrix
toy_data_mixed, mixer = gen_toy_data(toy_data_params)

data = process_data(toy_data_mixed, toy_data_params['epoch_sizes'])
demixer = optimize(data, dim_s = 2, restarts = 20)

np.set_printoptions(suppress = True)
print(np.round(demixer.dot(mixer),3))
# A perfect demixer should have zeroes in the upper-right (dim_n x dim_n) block of the demixer x mixer matrix product