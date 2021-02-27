"""
    dic(loglike::AbstractVector{<:Real})

Computes Deviance Information Criterion (DIC).

# Arguments
* `log_lik::AbstractArray`: A vector of posterior log likelihoods

# Returns
* `dic::Real`: DIC value
"""
function dic(loglike::AbstractVector{<:Real})
    D = deviance.(loglike)
    return mean(D) + 0.5 * var(D)
end

deviance(loglikelihood::Real) = -2 * loglikelihood
