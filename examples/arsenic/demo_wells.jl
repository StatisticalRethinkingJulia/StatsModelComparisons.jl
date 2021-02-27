using StatsModelComparisons
using StanSample, StatsFuns, StatsPlots
using JSON
using Printf

ProjDir = @__DIR__

include(joinpath(ProjDir, "cvit.jl"))

# Data
data = JSON.parsefile(joinpath(ProjDir, "wells.data.json"))
y = Float64.(data["switched"])
x = Float64[data["arsenic"]  data["dist"]]
n, m = size(x)

# Model
model_str = read(open(joinpath(ProjDir, "arsenic_logistic.stan")), String)
tmpdir = mktempdir()
sm1 = SampleModel("arsenic_logistic", model_str; tmpdir)
println("\n-----------------------\n")

data1 = (p = m, N = n, y = Int.(y), x = x)
# Fit the model in Stan
rc1 = stan_sample(sm1; data=data1)
if success(rc1)
    nt1 = read_samples(sm1)

    # Compute LOO and standard error
    log_lik = nt1.log_lik'
    loo, loos, pk = psisloo(log_lik)
    elpd_loo = sum(loos)
    se_elpd_loo = std(loos) * sqrt(n)
    @printf(">> elpd_loo = %.1f, SE(elpd_loo) = %.1f\n", elpd_loo, se_elpd_loo)

    # Check the shape parameter k of the generalized Pareto distribution
    if all(pk .< 0.5)
        println("All Pareto k estimates OK (k < 0.5)")
    else
        pkn1 = sum((pk .>= 0.5) & (pk .< 1))
        pkn2 = sum(pk .>= 1)
        @printf(">> %d (%.0f%%) PSIS Pareto k estimates between 0.5 and 1\n", pkn1, pkn1/n*100)
        @printf(">> %d (%.0f%%) PSIS Pareto k estimates greater than 1\n", pkn2, pkn2/n*100)
    end
end
println("\n-----------------------\n")

# Fit a second model, using log(arsenic) instead of arsenic
x2 = Float64[log.(data["arsenic"])  data["dist"]]

# Model
data2 = (p = m, N = n, y = Int.(y), x = x2)
# Fit the model in Stan
rc2 = stan_sample(sm1; data=data2)

if success(rc2)
    nt2 = read_samples(sm1)
    # Compute LOO and standard error
    log_lik = nt2.log_lik'
    loo2, loos2, pk2 = psisloo(log_lik)
    elpd_loo = sum(loos2)
    se_elpd_loo = std(loos2) * sqrt(n)
    @printf(">> elpd_loo = %.1f, SE(elpd_loo) = %.1f\n", elpd_loo, se_elpd_loo)

    # Check the shape parameter k of the generalized Pareto distribution
    if all(pk .< 0.5)
        println("All Pareto k estimates OK (k < 0.5).")
    else
        pk_good = sum(pk .<= 0.5)
        pk_ok = length(pk[pk .<= 0.7]) - pk_good
        pk_bad = length(pk[pk .<= 1]) - pk_good - pk_ok
        println((good=pk_good, ok=pk_ok, bad=pk_bad, very_bad=sum(pk .>= 1)))
    end
end

if success(rc1) && success(rc2)
    ## Compare the models
    loodiff = loos - loos2
    @printf("elpd_diff = %.1f, SE(elpd_diff) = %.1f\n",sum(loodiff), std(loodiff) * sqrt(n))
end
println("\n-----------------------\n")

## k-fold-CV
# k-fold-CV should be used if several khats>0.5
# in this case it is not needed, but provided as an example

model_str = read(open(joinpath(ProjDir, "arsenic_logistic_t.stan")), String)
sm3 = SampleModel("arsenic_logistic_t", model_str; tmpdir=tmpdir);

cvitr, cvitst = cvit(n, 10, true)
kfcvs = similar(loos)
for cvi in 1:10
    standatacv = (p = m, N = length(cvitr[cvi]), Nt = length(cvitst[cvi]),
        x = x[cvitr[cvi],:], y = Int.(y[cvitr[cvi]]),
        xt = x[cvitst[cvi],:], yt = Int.(y[cvitst[cvi]]))

    # Fit the model in Stan
    rc3 = stan_sample(sm3; data=standatacv)
    if success(rc3)
        nt3 = read_samples(sm3)
        # Compute LOO and standard error
        log_likt = nt3.log_likt'
        local n_sam, n_obs = size(log_likt)
        kfcvs[cvitst[cvi]] .=
            reshape(logsumexp(log_likt .- log(n_sam); dims=1), n_obs)
    end
end

# compare PSIS-LOO and k-fold-CV
plot([-3.5, 0], [-3.5, 0], color=:red)
scatter!(loos, kfcvs, xlab = "PSIS-LOO", ylab = "10-fold-CV",
    leg=false, color=:darkblue)
savefig(joinpath(ProjDir, "compare.png"))
