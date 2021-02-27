using StatsPlots

function pk_qualify(pk::Vector{Float64})
    pk_good = sum(pk .<= 0.5)
    pk_ok = length(pk[pk .<= 0.7]) - pk_good
    pk_bad = length(pk[pk .<= 1]) - pk_good - pk_ok
    (good=pk_good, ok=pk_ok, bad=pk_bad, very_bad=sum(pk .> 1))
end

function pk_plot(pk::Vector{Float64}; title="PSIS diagnostic plot.",
    leg=:topleft, kwargs...)
    scatter(pk, xlab="Datapoint", ylab="Pareto shape k",
        marker=2.5, lab="Pk points", leg=leg)
  hline!([0.5], lab="pk = 0.5");hline!([0.7], lab="pk = 0.7")
  hline!([1], lab="pk = 1.0")
  title!(title)
end

export
    pk_qualify,
    pk_plot