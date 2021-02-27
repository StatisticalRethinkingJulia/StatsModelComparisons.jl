using Statistics

"""
    psislw(lw, [wcpp, wtrunc])

Compute the Pareto smoothed importance sampling (PSIS).

# Arguments
* `lw::AbstractArray`: Array of size n x m containing m sets of n log weights. It is also possible to provide one dimensional array of length n.
* `wcpp::Real`: Percentage of samples used for GPD fit estimate (default is 20).
* `wtrunc::Float64`: Positive parameter for truncating very large weights to n^wtrunc. Providing False or 0 disables truncation. Default values is 3/4.

# Returns
* `lw_out::AbstractArray`: Smoothed log weights
* `kss::AbstractArray`: Pareto tail indices
"""
function psislw(lw::AbstractArray, wcpp::Int64=20, wtrunc::Float64=3/4)

    lw_out = copy(lw)

    if ~(1 <= ndims(lw_out) <= 2)
        throw(DimensionMismatch("Argument `lw` must be 1 or 2 dimensional."))
    end
    if size(lw_out,1) <= 1
        error("More than one log-weight needed.")
    end
    return _psislw(lw_out, wcpp, wtrunc)
end

function _psislw(lw_out::Array{Float64}, wcpp::Int64, wtrunc::Float64)

    if ndims(lw_out) == 2
        n, m = size(lw_out)
    elseif ndims(lw_out) == 1
        n = length(lw_out)
        m = 1
    end
    kss = zeros(Float64, m)

    # precalculate constants
    cutoffmin = log(eps(0.5))
    logn = log(n)

    # loop over sets of log weights
    for i in 1:m
        x = view(lw_out, :, i)
        x .-= maximum(x)
        xcutoff = max(quantile(x, 1 - wcpp/100), cutoffmin)
        expxcutoff = exp(xcutoff)
        x2 = x[x .> xcutoff]
        n2 = length(x2)
        if n2 <= 4
            # not enough tail samples for gpdfitnew
            k = Inf
        else
            # order of tail samples
            x2si = sortperm(x2)
            # fit generalized Pareto distribution to the right tail samples
            x2 .= exp.(x2) .- expxcutoff
            k, sigma = gpdfitnew(x2)
            # compute ordered statistic for the fit
            sti = collect(0.5:n2) / n2 
            qq = gpinv(sti, k, sigma)
            qq .= log.(qq .+ expxcutoff)
            # place the smoothed tail into the output array
            x[x .> xcutoff] = qq[x2si]
        end
        if wtrunc > 0
            # truncate too large weights
            lwtrunc = wtrunc * logn - logn + logsumexp(x)
            x[x .> lwtrunc] .= lwtrunc
        end
        # renormalize weights
        x .-= logsumexp(x)
        # store tail index k
        kss[i] = k
    end
    # If the provided input array is one dimensional, return kss as scalar.
    if ndims(lw_out) == 1
        kss = kss[1]
    end
    return lw_out, kss
end

export
    psislw
