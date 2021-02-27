### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ b9fc511c-6007-11eb-0c6b-1f6871a40710
using Pkg, DrWatson

# ╔═╡ 20d377b2-6008-11eb-364a-617b6934ecb2
begin
	@quickactivate "StatsModelComparisons"
	using StatsModelComparisons
	using StanSample, StatsFuns, StatsPlots
	using DataFrames, CSV, RDatasets
end

# ╔═╡ af6b0b20-6008-11eb-2fa1-2f61145ab7db
md"
!!! note

	This script assumes that RDatasets.jl is loaded. RDatasets.jl is not included in the dependencies of PSIS.jl (as it would require R to be installed)."

# ╔═╡ 20d43cda-6008-11eb-09f0-53489a26110d
df = RDatasets.dataset("datasets", "cars");

# ╔═╡ 20e22570-6008-11eb-1565-f13541b41861
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
";

# ╔═╡ 20e2ebb6-6008-11eb-34f5-61f1ec3d024c
begin
	cars_stan_model = SampleModel("cars.model", cars_stan)
	data = (N = size(df, 1), speed = df.Speed, dist = df.Dist)
	rc = stan_sample(cars_stan_model; data)

	if success(rc)
		cars_df = read_samples(cars_stan_model; output_format=:dataframe)
		isdefined(Main, :StatisticalRethinking) && 
            PRECIS(cars_df[:, [:a, :b, :sigma]])
	end
end

# ╔═╡ 5fc59200-6008-11eb-3e06-1d0bcdf11d7d
if success(rc)
	nt_cars = read_samples(cars_stan_model);
	log_lik = nt_cars.log_lik'
end;

# ╔═╡ 20ed768a-6008-11eb-13f4-458ca1a29592
begin
    ns, n = size(log_lik)
	lppd = reshape(logsumexp(log_lik .- log(ns); dims=1), n)
	pwaic = [var(log_lik[:, i]) for i in 1:n]
	-2(sum(lppd) - sum(pwaic))
end

# ╔═╡ efe960f0-600a-11eb-1df4-5be83899715a
begin
	waic_vec = -2(lppd - pwaic)
	sqrt(n*var(waic_vec))
end

# ╔═╡ 20f5a31e-6008-11eb-2ce1-075893273872
waic(log_lik)

# ╔═╡ 6b67e38e-6009-11eb-3d9f-afc517b7d9fb
begin
	loo, loos, pk = psisloo(log_lik)
	loo
end

# ╔═╡ 82232dfe-6009-11eb-3b32-dbc487c6e4e7
pk_qualify(pk)

# ╔═╡ 8ddf22b0-6009-11eb-08da-5198ba046628
pk_plot(pk)

# ╔═╡ Cell order:
# ╟─af6b0b20-6008-11eb-2fa1-2f61145ab7db
# ╠═b9fc511c-6007-11eb-0c6b-1f6871a40710
# ╠═20d377b2-6008-11eb-364a-617b6934ecb2
# ╠═20d43cda-6008-11eb-09f0-53489a26110d
# ╠═20e22570-6008-11eb-1565-f13541b41861
# ╠═20e2ebb6-6008-11eb-34f5-61f1ec3d024c
# ╠═5fc59200-6008-11eb-3e06-1d0bcdf11d7d
# ╠═20ed768a-6008-11eb-13f4-458ca1a29592
# ╠═efe960f0-600a-11eb-1df4-5be83899715a
# ╠═20f5a31e-6008-11eb-2ce1-075893273872
# ╠═6b67e38e-6009-11eb-3d9f-afc517b7d9fb
# ╠═82232dfe-6009-11eb-3b32-dbc487c6e4e7
# ╠═8ddf22b0-6009-11eb-08da-5198ba046628
