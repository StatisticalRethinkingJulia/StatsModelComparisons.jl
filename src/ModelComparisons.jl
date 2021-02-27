"""ModelComparisons

This module implements several model comparison methods such as PSIS, WAIC and DIC..

Included functions
------------------
psisloo
    Pareto smoothed importance sampling leave-one-out log predictive densities.

psislw
    Pareto smoothed importance sampling.

gpdfitnew
    Estimate the paramaters for the Generalized Pareto Distribution (GPD).

gpinv
    Inverse Generalised Pareto distribution function.

waic
    ...

dic
    ...

deviance
    ...

References
----------
Aki Vehtari, Andrew Gelman and Jonah Gabry (2015). Efficient implementation
of leave-one-out cross-validation and WAIC for evaluating fitted Bayesian
models. arXiv preprint arXiv:1507.04544.

Aki Vehtari and Andrew Gelman (2015). Pareto smoothed importance sampling.
arXiv preprint arXiv:1507.02646.
"""

module ModelComparisons

using StatsFuns, Statistics

mc_path = @__DIR__

include("psisloo.jl")
include("psislw.jl")
include("gpdfitnew.jl")
include("gpinv.jl")
include("waic.jl")
include("pk_utilities.jl")
include("dic.jl")

export
    mc_path

end # module
