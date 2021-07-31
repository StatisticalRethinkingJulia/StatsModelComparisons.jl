using Turing, Random
using CSV, DataFrames, StatsBase
using ParetoSmooth

Random.seed!(129111)

df = CSV.read(joinpath("data", "WaffleDivorce.csv"), DataFrame)
df.D = zscore(df.Divorce)
df.M = zscore(df.Marriage)
df.A = zscore(df.MedianAgeMarriage)
data = (D=df.D, A=df.A)

function lin(a, b, c, x...)
    result = @. a + b * c
    for i in 1:2:length(x)
        @. result += x[i] * x[i+1]
    end
    return result
end


@model function m5_1t(A, D)
    a ~ Normal(0, 0.2)
    bA ~ Normal(0, 0.5)
    σ ~ Exponential(1)
    for i in eachindex(D)
        μ = lin(a, A[i], bA)
        D[i] ~ Normal(μ, σ)
    end
end

chn5_1t = sample(m5_1t(df.A, df.D), NUTS(1000, .9), MCMCThreads(), 1000, 4)
chn5_1t |> display

@model function m5_2t(M, D)
    a ~ Normal(0, 0.2)
    bM ~ Normal(0, 0.5)
    σ ~ Exponential(1)
    for i in eachindex(D)
        μ = lin(a, M[i], bM)
        D[i] ~ Normal(μ, σ)
    end
end

chn5_2t = sample(m5_2t(df.M, df.D), NUTS(1000, .9), MCMCThreads(), 1000, 4)
chn5_2t |> display

@model function m5_3t(A, M, D)
    a ~ Normal(0, 0.2)
    bA ~ Normal(0, 0.5)
    bM ~ Normal(0, 0.5)
    σ ~ Exponential(1)
    for i in eachindex(D)
        μ = a + M[i] * bM + A[i] * bA
        D[i] ~ Normal(μ, σ)
    end
end

chn5_3t = sample(m5_3t(df.A, df.M, df.D), NUTS(1000, .9), MCMCThreads(), 1000, 4)
chn5_3t |> display

pw_lls5_1t = pointwise_log_likelihoods(m5_1t(df.A, df.D), chn5_1t)
psis_loo_output5_1t = psis_loo(m5_1t(df.A, df.D), chn5_1t)
psis_loo_output5_1t |> display

pw_lls5_2t = pointwise_log_likelihoods(m5_2t(df.M, df.D), chn5_2t)
psis_loo_output5_2t = psis_loo(m5_2t(df.M, df.D), chn5_2t)
psis_loo_output5_2t |> display

pw_lls5_3t = pointwise_log_likelihoods(m5_3t(df.A, df.M, df.D), chn5_3t)
psis_loo_output5_3t = psis_loo(m5_3t(df.A, df.M, df.D), chn5_3t)
psis_loo_output5_3t |> display

loo = loo_compare([pw_lls5_1t, pw_lls5_2t, pw_lls5_3t];
    model_names=[:m5_1t, :m5_2t, :m5_3t])
for psis in loo_comparison.psis
    psis |> display
end
println()
loo |> display

