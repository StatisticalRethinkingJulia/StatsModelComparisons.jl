using Test
using StatsModelComparisons
using StanSample, StatsFuns
using Printf
using JSON

@testset "Arsenic"  begin 

    ProjDir = @__DIR__

    #=
    if haskey(ENV, "JULIA_CMDSTAN_HOME")
        include(joinpath(ProjDir, "test_demo_wells.jl"))
    else
      println("\nJULIA_CMDSTAN_HOME not set. Skipping tests")
    end
    =#

    include(joinpath(ProjDir, "cvit.jl"))

    # Data
    data = JSON.parsefile(joinpath(ProjDir, "wells.data.json"))
    y = Float64.(data["switched"])
    x = Float64[data["arsenic"]  data["dist"]]
    n, m = size(x)

    # Model
    model_str = read(open(joinpath(ProjDir, "arsenic_logistic.stan")), String)
    sm1 = SampleModel("arsenic_logistic", model_str)

    data1 = (p = m, N = n, y = Int.(y), x = x)
    # Fit the model in Stan
    rc1 = stan_sample(sm1; data=data1)
    if success(rc1)
        nt1 = read_samples(sm1, :namedtuple)

        # Compute LOO and standard error
        log_lik = nt1.log_lik'
        loo, loos, pk = psisloo(log_lik)
        elpd_loo = sum(loos)
        se_elpd_loo = std(loos) * sqrt(n)

        @test elpd_loo ≈ -1968.3 atol=2.0
        @test se_elpd_loo ≈ 15.5 atol=0.5
        @test all(pk .< 0.5)
    end
    println()

    # Fit a second model, using log(arsenic) instead of arsenic
    x2 = Float64[log.(data["arsenic"])  data["dist"]]

    # Model
    data2 = (p = m, N = n, y = Int.(y), x = x2)
    # Fit the model in Stan
    rc2 = stan_sample(sm1; data=data2)

    if success(rc2)
        nt2 = read_samples(sm1, :namedtuple)
        # Compute LOO and standard error
        log_lik = nt2.log_lik'
        loo2, loos2, pk2 = psisloo(log_lik)
        elpd_loo = sum(loos2)
        se_elpd_loo = std(loos2) * sqrt(n)

        @test elpd_loo ≈ -1952.1 atol=2.0
        @test se_elpd_loo ≈ 16.2 atol=0.5
        @test all(pk .< 0.5)
    end

    if success(rc1) && success(rc2)
        ## Compare the models
        loodiff = loos - loos2
        
        @test sum(loodiff) ≈ -16.3 atol=0.3
        @test std(loodiff) * sqrt(n) ≈ 4.4 atol=0.2
    end
end