# pyssa
paulbuenau's SSA-Toolbox (written in Java), ported into Python
Original repository can be found at https://github.com/paulbuenau/SSA-Toolbox.

Libraries required: NumPy, SciPy

Limitations:
1. The epoch labels (or at least the number of epochs) must be specified.
2. The algorithm always runs SSA with respect to the covariance matrix and the mean.
3. The returned result is only the estimated mixing matrix.
