"""
    dic(loglike::AbstractVector{<:Real})

Computes Deviance Information Criterion (DIC).

# Arguments
* `loglike::AbstractArray`: A vector of posterior log likelihoods

# Returns
* `dic::Real`: DIC value
"""
function dic(loglike::AbstractVector{<:Real})
    D = map(deviance, loglike)
    mean_D = mean(D)
    var_D = var(D; mean=mean_D)
    return mean_D + var_D / 2
end

deviance(loglikelihood::Real) = -2 * loglikelihood
