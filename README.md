# pyssa
paulbuenau's SSA-Toolbox (written in Java), ported into Python. Original repository can be found at https://github.com/paulbuenau/SSA-Toolbox.

Libraries required: NumPy, SciPy

Limitations:

1. The epoch labels (or at least the number of epochs) must be specified.  
  (The original has the option of using a heuristic to guess the number of epochs, assuming they are equally sized.)

2. The algorithm always runs SSA with respect to the covariance matrix and the mean.  
  (The original doesn't require optimizing w.r.t. the mean.)

3. The returned result is only the estimated mixing matrix.  
  (The original returns the bases, projections, and signals; nevertheless, these can be computed using the estimated mixing matrix)
