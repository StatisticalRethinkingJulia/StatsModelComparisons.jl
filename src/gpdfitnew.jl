
"""
    gpdfitnew(x)

Estimate the paramaters for the Generalized Pareto Distribution (GPD). Returns empirical Bayes estimate for the parameters of the two-parameter generalized Parato distribution given the data.

# Arguments
* `x::AbstractArray`: One dimensional data array.

# Returns
* `k::Float64, sigma::Float64`: Estimated parameter values.

# Notes
* This function returns a negative of Zhang and Stephens's k, because it is more common parameterization.
"""
function gpdfitnew(x::AbstractArray)
    if ndims(x) != 1 || length(x) <= 1
        throw(DimensionMismatch())
    end
    sort!(x)

    n = length(x)
    m = 80 + trunc(Int, sqrt(n))
    prior = 3
    
    bs = collect(1:m) .- 0.5  
    bs .= (1 .- sqrt.(m ./ bs)) ./ prior ./ x[trunc(Int64, n/4 + 0.5)] .+ 1 / x[end]

    ks = mean(log1p.(-bs .* x'), dims=2)[:,1]
    L = n * (log.(bs ./ -ks) .- ks .- 1)
    w = similar(L)
    w .= (1 ./ sum(exp.(L .- L'), dims=1)')[:,1]

    # remove negligible weights
    dii = w .>= (10 * eps(Float64))
    if ~all(dii)
        w = w[dii]
        bs = bs[dii]
    end
    # normalise w
    w ./= sum(w)
    # posterior mean for b
    b = sum(bs .* w)
    # Estimate for k, note that we return a negative of Zhang and
    # Stephens's k, because it is more common parameterisation.
    k = mean(log1p.(-b .* x))
    # estimate for sigma
    sigma = -k / b

    return k, sigma

end
