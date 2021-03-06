"""
    aic(max_loglike, k)

Computes the Akaike Information Criterion (AIC).

# Arguments
* `max_loglike`: the log likelihood evaluated at the maximum likelihood estimate
* `k`: the number of parameters in the model

# Returns
* `aic::Real`: AIC value
"""
aic(max_loglike, k) = 2 * (k - max_loglike)

"""
    bic(max_loglike, k, n)

Computes the Bayesian Information Criterion (BIC).

# Arguments
* `max_loglike`: the log likelihood evaluated at the maximum likelihood estimate
* `k`: the number of parameters in the model
* `n`: the number of observations in the data

# Returns
* `bic::Real`: BIC value
"""
bic(max_loglike, k, n) = k * log(n) - 2 * max_loglike
