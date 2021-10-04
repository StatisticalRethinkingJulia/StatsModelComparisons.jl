using Test
using StatsModelComparisons, StanSample
using StatsFuns, CSV, Random

@testset "Cars"  begin 

    ProjDir = @__DIR__

    df = CSV.read(joinpath(ProjDir, "..", "data", "cars.csv"), DataFrame)

    cars_stan = "
    data {
        int N;
        vector[N] speed;
        vector[N] dist;
    }
    parameters {
        real a;
        real b;
        real sigma;
    }
    transformed parameters{
        vector[N] mu;
        mu = a + b * speed;
    }
    model {
        a ~ normal(0, 100);
        b ~ normal(0, 10);
        sigma ~ exponential(1);
        dist ~ normal(mu, sigma)    ;
    }
    generated quantities {
        vector[N] log_lik;
        for (i in 1:N)
            log_lik[i] = normal_lpdf(dist[i] | mu[i], sigma);
    }
    "

    Random.seed!(1)
    cars_stan_model = SampleModel("cars.model", cars_stan)
    data = (N = size(df, 1), speed = df.Speed, dist = df.Dist)
    rc = stan_sample(cars_stan_model; data)

    if success(rc)
        nt_cars = read_samples(cars_stan_model, :namedtuple);
    end

    log_lik = nt_cars.log_lik'
    n_sam, n_obs = size(log_lik)
    lppd = reshape(logsumexp(log_lik .- log(n_sam); dims=1), n_obs)

    pwaic = [var(log_lik[:, i]) for i in 1:n_obs]
    @test -2(sum(lppd) - sum(pwaic)) ≈ 421.0 atol=2.0
    loo, loos, pk = psisloo(log_lik)
    @test -2loo ≈ 421.0 atol=2.0

end
