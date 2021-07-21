#
# This example assumes below packages are available in your environment
#

using StanSample, StatsModelComparisons, ParetoSmooth
using StatsFuns, CSV
#using StatisticalRethinking: pk_plot
#using MCMCChains

ProjDir = @__DIR__

df = CSV.read(joinpath(ProjDir, "..", "..", "data", "cars.csv"), DataFrame)

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

#Random.seed!(1)
cars_stan_model = SampleModel("cars.model", cars_stan)
data = (N = size(df, 1), speed = df.Speed, dist = df.Dist)
rc = stan_sample(cars_stan_model; data)
println()

if success(rc)
    stan_summary(cars_stan_model, true)
    nt_cars = read_samples(cars_stan_model);

    log_lik = nt_cars.log_lik'
    n_sam, n_obs = size(log_lik)
    lppd = reshape(logsumexp(log_lik .- log(n_sam); dims=1), n_obs)

    pwaic = [var(log_lik[:, i]) for i in 1:n_obs]
    @show -2(sum(lppd) - sum(pwaic))
    println()

    @show waic(log_lik)
    println()

    loo, loos, pk = psisloo(log_lik)
    @show -2loo

    println("\nUsing ParetoSmooth' psis()\n")

    ll = reshape(nt_cars.log_lik, 50, 1000, 4);
    cars_loo = ParetoSmooth.loo(ll)
    cars_loo |> display
    println()
    cars_loo.estimates |> display
    println()
    cars_loo.pointwise |> display
    println()

    if isdefined(Main, :StatisticalRethinking)
        pk_plot(cars_loo.psis_object.pareto_k)
        savefig(joinpath(ProjDir, "pareto_k_plot.png"))
        pk_plot(pk)
        savefig(joinpath(ProjDir, "pk_plot.png"))
        closeall()
    end
end

if success(rc) && isdefined(Main, :MCMCChains)
    chn = read_samples(cars_stan_model; output_format=:mcmcchains);
    log_lik2 = Matrix(Array(chn)[:, 54:end]');
    ll2 = reshape(log_lik2, 50, 1000, 4);

    cars_loo2 = ParetoSmooth.loo(ll2)
    println()
    cars_loo2.estimates |> display
    println()
    cars_loo2.pointwise |> display

    if isdefined(Main, :StatisticalRethinking)
        pk_plot(cars_loo2.psis_object.pareto_k)
        savefig(joinpath(ProjDir, "pareto_k_plot_2.png"))
        closeall()
    end
end
