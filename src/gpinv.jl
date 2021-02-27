"""
    gpinv(p, k, sigma)

Inverse Generalised Pareto distribution function.
"""
function gpinv(p, k, sigma)
    x = similar(p)
    if sigma <= 0
        return x
    end
    ok = (p .> 0) .& (p .< 1)
    pv = view(p, ok)
    xv = view(x,ok)
    if abs(k) < eps(Float64)
        xv .= -log1p.(-pv)
    else
        xv .= expm1.(log1p.(-pv) .* (-k)) ./ k
    end
    x .*= sigma
    if ~all(ok)
        x[p .== 0] = 0
        if k >= 0
            x[p .== 1] = Inf
        else
            x[p .== 1] = -sigma / k
        end
    end
    x
end
