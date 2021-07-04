using Test
using StatsModelComparisons, Turing
using StatsFuns, CSV, Random, DataFrames, ParetoSmooth

#@testset "Cars Turing"  begin 

    ProjDir = @__DIR__

    df = CSV.read(joinpath(ProjDir, "..", "..", "data", "cars.csv"), DataFrame)

    @model function model(speed, dist)
        a ~ Normal(0, 100)
        b ~ Normal(0, 10)
        σ ~ Exponential(1) 
        μ = a .+ b * speed
    
        dist ~ MvNormal(μ, σ)
    end

    # temporary method for MCMCChains
    function pointwise_loglikes(chain::Chains, data, ll_fun)
        samples = Array(Chains(chain, :parameters).value)
        pointwise_loglikes(samples, data, ll_fun)
    end

    # temporary generic method for arrays
    function pointwise_loglikes(samples::Array{Float64,3}, data, ll_fun)
        n_data = length(data)
        n_samples, n_chains = size(samples)[[1,3]]
        pointwise_lls = fill(0.0, n_data, n_samples, n_chains)
        for c in 1:n_chains 
            for s in 1:n_samples
                for d in 1:n_data
                    pointwise_lls[d,s,c] = ll_fun(samples[s,:,c], data[d])
                end
            end
        end
        return pointwise_lls
    end

    # temporary function
    function compute_loo(psis_output, pointwise_lls)
        dims = size(pointwise_lls)
        lwp = deepcopy(pointwise_lls)
        lwp += psis_output.weights;
        lwpt = reshape(lwp, dims[1], dims[2] * dims[3])';
        loos = reshape(logsumexp(lwpt; dims=1), size(lwpt, 2));
        return sum(loos)
    end

    function compute_loglike(a, b, σ, data)
        μ = a .+ b * data[1]
        LLs = logpdf(Normal(μ, σ), data[2])
    end

    Random.seed!(1)
    speed,dist = df.Speed, df.Dist
    chain = sample(model(speed, dist), NUTS(1000, .65), MCMCThreads(), 1000, 4)

    data = map((s,d)->(s,d), speed, dist)
    # compute the pointwise log likelihoods where indices correspond to [data, sample, chain]
    pointwise_lls = pointwise_loglikes(chain, data, (p,d)-> compute_loglike(p..., d))

    # compute the psis object
    psis_output = psis(pointwise_lls)

    # return loo based on Rob's example
    loo = compute_loo(psis_output, pointwise_lls)

    # log_lik = nt_cars.log_lik'
    # n_sam, n_obs = size(log_lik)
    # lppd = reshape(logsumexp(log_lik .- log(n_sam); dims=1), n_obs)

    # pwaic = [var(log_lik[:, i]) for i in 1:n_obs]
    # @test -2(sum(lppd) - sum(pwaic)) ≈ 421.0 atol=2.0
    # loo, loos, pk = psisloo(log_lik)
    # @test -2loo ≈ 421.0 atol=2.0

#end

