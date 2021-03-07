using Test

@testset "BIC"  begin 

    using StatsModelComparisons

    v1 = bic(-200.0, 10, 100)
    v2 = bic(-200.0, 2, 100)
    v3 = bic(-200.0, 2, 10)
    v4 = bic(-100.0, 2, 10)
    @test v2 < v1
    @test v3 < v2
    @test v4 < v3
end

@testset "AIC"  begin 

    using StatsModelComparisons

    v1 = aic(-200.0, 10)
    v2 = aic(-200.0, 2)
    v3 = aic(-100.0, 2)
    @test v2 < v1
    @test v3 < v2
end

@testset "DIC"  begin 

    using StatsModelComparisons
    x = randn(100)
    LLs = x .- mean(x)
    v1 = dic(LLs)
    v2 = dic(LLs*2)
    v3 = dic(LLs .+ 1)
    @test v1 < v2
    @test v3 < v1
end


@testset "WAIC and LOO" begin 

    using StatsModelComparisons, StanSample
    using StatsFuns, Random, CSV
    import StanSample: read_csv_files

    cd(@__DIR__)
    Random.seed!(89905)
    # df = RDatasets.dataset("datasets", "cars")

    # stream = open("cars_model.stan", "r")
    # cars_stan = read(stream, String)
    # close(stream)

    # cars_stan_model = SampleModel("cars.model", cars_stan)
    # data = (N = size(df, 1), speed = df.Speed, dist = df.Dist)
    # rc = stan_sample(cars_stan_model; data)

    # stan_summary(cars_stan_model, true)
    # nt_cars = read_samples(cars_stan_model);
    # log_lik = nt_cars.log_lik'

    # files = readdir("test_chains")
    # files = map(f->"test_chains/"*f, files)
    # samples = read_csv_files(files, 2000, :namedtuple; include_internals=false)
    # df = vcat(dfs...)

    files = readdir("test_chains/")

    dfs = map(f->CSV.read("test_chains/"*f, DataFrame), files)
    dfs = map(x->x[1001:end,:], dfs)
    df = vcat(dfs...)
    
    col_idx = findall(x->occursin("log_lik", x), names(df))
    pw_log_liks = Array(df[:,col_idx])
    WAIC = waic(pw_log_liks)
    loo,_,_ = psisloo(pw_log_liks)

    @test WAIC.WAIC ≈ 421.0 rtol = .01
    @test WAIC.std_err ≈ 16.3 rtol = .01
    @test loo ≈ -210.5 rtol = .05  
end
