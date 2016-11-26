# pyssa
paulbuenau's SSA-Toolbox (written in Java), ported into Python. Original repository can be found at https://github.com/paulbuenau/SSA-Toolbox.

Libraries required: NumPy, SciPy

# Limitations:

1. The epoch labels (or at least the number of epochs) must be specified.  
  (The original has the option of using a heuristic to guess the number of epochs, assuming they are equally sized.)

2. The algorithm always runs SSA with respect to the covariance matrix and the mean.  
  (The original has the option of ignoring the mean.)

3. The returned result is only the estimated mixing matrix.  
  (The original returns the bases, projections, and signals; nevertheless, these can be computed easily once the estimated mixing matrix is known.)

# Usage (Example available in demo.py):

1. Load your data into a numpy array, ordered by their epochs (i.e. the first n1 rows should belong to epoch 1, next n2 to epoch 2, and so on).

2. Pre-process the data with the process_data function; the first argument is the raw data from (1), and the second is either a list of the epoch sizes (so the corresponding list in this case should be [n1,n2,...]) or simply the number of epochs (in which case the epoch sizes are assumed to be equal).

3. Run the optimize function, using the output from (2) as the first argument, and specify the dimension of the stationary sources for the second argument. The function returns a demixing matrix; right-multiplying the data from (1) by this matrix should yield a transformation of the data where the first s columns (as specified in the second argument of the optimize function) are weakly stationary.

# Remarks:

1. The main issue with the original Java program is that it can be *very* slow at loading large amounts of data (on the order of seconds for a 10MB .csv), whereas the NumPy/Pandas parser handles this much more quickly.

2. The optimization algorithm is independent of the number of observations once the data has been pre-processed. On the other hand, the dimensionality of the data *d*, and number of epochs *k*, affect the calculation of the objective function (which is repeatedly called by the optimization algorithm); thus the computational complexity is asymtotically O(k\*d^3) in theory (rotations of *k* *d*-dimensional matrices). However, unless *d* is *very* large, the bottleneck, in practice, is from the overhead of function calls, which scales linearly with *k*.

3. The loss_translation argument in the optimize function should *never* have to be supplied; the objective function is non-negative if the data has been whitened correctly (done by default at the start of the optimize function).
