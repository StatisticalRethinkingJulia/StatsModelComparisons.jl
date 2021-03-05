# Copyright (c) 2015 Aki Vehtari, Tuomas Sivula
# Original Matlab version by Aki Vehtari. Translation to Python
# by Tuomas Sivula.

# This software is distributed under the GNU General Public
# License (version 3 or later); please refer to the file
# License.txt, included with the software, for details.

"""
    psisloo(log_lik, [wcpp, wtrunc])

PSIS leave-one-out log predictive densities.

Computes the log predictive densities given posterior samples of the log
likelihood terms p(y_i|\theta^s) in input parameter `log_lik`. Returns a
sum of the leave-one-out log predictive densities `loo`, individual
leave-one-out log predictive density terms `loos` and an estimate of Pareto
tail indeces `ks`. If tail index k>0.5, variance of the raw estimate does
not exist and if tail index k>1 the mean of the raw estimate does not exist
and the PSIS estimate is likely to have large variation and some bias.

# Arguments
* `log_lik::AbstractArray`: Array of size n x m containing n posterior samples of the log likelihood terms p(y_i|\theta^s).
* `wcpp::Real`: Percentage of samples used for GPD fit estimate (default is 20).
* `wtrunc::Float64`: Positive parameter for truncating very large weights to n^wtrunc. Providing False or 0 disables truncation. Default values is 3/4.

# Returns
* `loo::Real`: sum of the leave-one-out log predictive densities.
* `loos::AbstractArray`: Individual leave-one-out log predictive density terms.
* `ks::AbstractArray`: Estimated Pareto tail indeces.
"""
function psisloo(log_lik::AbstractArray, wcpp::Int64=20, wtrunc::Float64=3/4)
   
    # log raw weights from log_lik
    lw = copy(log_lik)

    # compute Pareto smoothed log weights given raw log weights
    lwp, ks = psislw(-lw, wcpp, wtrunc)

    lwp += lw
    loos = reshape(logsumexp(lwp; dims=1), size(lwp, 2))
    loo = sum(loos)

    return loo, loos, ks
end
