"""
    dic(loglik::AbstractVector{<:Real})

Compute the Deviance Information Criterion (DIC).

# Arguments
* `loglik::AbstractArray`: A vector of posterior log likelihoods

# Returns
* `dic::Real`: DIC value

Note: DIC assumes that the posterior distribution is approx. multivariate
Gaussian and tends to select overfitted models.
"""
function dic(loglik::AbstractVector{<:Real})
    D = map(deviance, loglik)
    mean_D = mean(D)
    var_D = var(D; mean=mean_D)
    return mean_D + var_D / 2
end

deviance(loglikelihood::Real) = -2 * loglikelihood
