### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ dcb4d418-5ec2-11eb-29d8-214f38b4d3ae
using Pkg, DrWatson

# ╔═╡ d20c24f8-5ec2-11eb-3d45-d97fedebee8e
begin
	@quickactivate "ParetoSmoothedImportanceSamplng"
	using ParetoSmoothedImportanceSampling
	using StanSample, StatsFuns, StatsPlots
	using DataFrames, CSV
end

# ╔═╡ e3552750-5e9f-11eb-324b-8df36d671c79
begin
	ProjDir = joinpath(psis_path, "..", "examples", "roaches")
	df = CSV.read(joinpath(ProjDir, "roachdata.csv"), DataFrame)
	df.roach1 = df.roach1 / 100
end;

# ╔═╡ e3691972-5e9f-11eb-20b2-a766ef562598
roaches1_stan = "
data {
  int<lower=0> N; 
  int<lower=0> K; 
  vector[N] exposure2;
  vector[N] roach1;
  vector[N] senior;
  vector[N] treatment;
  int y[N];
}
transformed data {
  vector[N] log_expo;
  log_expo = log(exposure2);
}
parameters {
  vector[K] beta;
}
transformed parameters {
   vector[N] eta;
   eta = log_expo + beta[1] + beta[2] * roach1 + beta[3] * treatment
                + beta[4] * senior;
}
model {
  y ~ poisson_log(eta);
}
generated quantities {
  vector[N] log_lik;
  for (i in 1:N)
    log_lik[i] = poisson_log_lpmf(y[i] | eta[i]);
}
";

# ╔═╡ e371fde4-5e9f-11eb-39ad-8d25ac3034fc
begin
	data = (N = size(df, 1), K = 4, y = Int.(df.y), roach1=df.roach1,
 		exposure2=df.exposure2, senior=df.senior, treatment=df.treatment)
	n = size(df, 1)
	tmpdir=joinpath(ProjDir, "tmp")
	sm1 = SampleModel("roaches1", roaches1_stan; tmpdir)
	rc1 = stan_sample(sm1; data)
end;

# ╔═╡ e3a796f2-5e9f-11eb-20bf-9de765acb853
if success(rc1)
  read_summary(sm1, true)
  nt1 = read_samples(sm1)

  # Compute LOO and standard error
  log_lik1 = nt1.log_lik'
  loo1, loos1, pk1 = psisloo(log_lik1)
  elpd_loo1 = sum(loos1)
  se_elpd_loo1 = std(loos1) * sqrt(n)
  (elpd_loo = elpd_loo1, se_elpd_loo = se_elpd_loo1)
end

# ╔═╡ abc94b86-5ea9-11eb-1d5d-578926de3257
if success(rc1)
	# Check the shape parameter k of the generalized Pareto distribution
	pk_qualify(pk1) |> display
end

# ╔═╡ e3b82668-5e9f-11eb-1641-a9dfed9eb108
pk_plot(pk1, title="PSIS diagnostic plot for poisson-log model.")

# ╔═╡ e3b8fc58-5e9f-11eb-0607-e5424a04df9c
md" ##### Simple negative binomial regression example using the 2nd parametrization of the negative binomial distribution, see section 40.1-3 in the Stan reference guide."

# ╔═╡ 5824d2ae-5ea2-11eb-3545-dd03c4aa7a69
roaches2_stan = "
data {
  int<lower=0> N; 
  int<lower=0> K;
  vector[N] exposure2;
  vector[N] roach1;
  vector[N] senior;
  vector[N] treatment;
  int y[N];
}
transformed data {
  vector[N] log_expo;
  log_expo = log(exposure2);
}
parameters {
  real phi;
  vector[K] beta;
}
transformed parameters {
   vector[N] eta;
   eta = log_expo + beta[1] + beta[2] * roach1 + beta[3] * treatment
                + beta[4] * senior;
}
model {  
  phi ~ normal(0, 10);
  beta[1] ~ cauchy(0,10);   //prior for the intercept following Gelman 2008
  for(i in 2:K)
   beta[i] ~ cauchy(0,2.5); //prior for the slopes following Gelman 2008
  y ~ neg_binomial_2_log(eta, phi);
}
generated quantities {
 vector[N] log_lik;
 for(i in 1:N){
  log_lik[i] <- neg_binomial_2_log_lpmf(y[i] | eta[i], phi);
 }
}
";

# ╔═╡ e3cab9ca-5e9f-11eb-3b31-c34fabb9e8d1
begin
	sm2 = SampleModel("roaches2", roaches2_stan; tmpdir)
	rc2 = stan_sample(sm2; data)
end;

# ╔═╡ e3ddad0a-5e9f-11eb-0a7d-150f9f398fe0
if success(rc2)
  read_summary(sm2, true)
  nt2 = read_samples(sm2)

  # Compute LOO and standard error
  log_lik2 = nt2.log_lik'
  loo2, loos2, pk2 = psisloo(log_lik2)
  elpd_loo2 = sum(loos2)
  se_elpd_loo2 = std(loos2) * sqrt(n)
  (elpd_loo = elpd_loo2, se_elpd_loo = se_elpd_loo2)
end

# ╔═╡ ce509a0c-5ea8-11eb-3f2e-01023f29a1e3
if success(rc2)
	  # Check the shape parameter k of the generalized Pareto distribution
	pk_qualify(pk2)
end

# ╔═╡ e3f103f0-5e9f-11eb-159a-452961ed2619
pk_plot(pk2; title="PSIS diagnostic plot for neg-binomial model.", leg=:topright)

# ╔═╡ Cell order:
# ╠═dcb4d418-5ec2-11eb-29d8-214f38b4d3ae
# ╠═d20c24f8-5ec2-11eb-3d45-d97fedebee8e
# ╠═e3552750-5e9f-11eb-324b-8df36d671c79
# ╠═e3691972-5e9f-11eb-20b2-a766ef562598
# ╠═e371fde4-5e9f-11eb-39ad-8d25ac3034fc
# ╠═e3a796f2-5e9f-11eb-20bf-9de765acb853
# ╠═abc94b86-5ea9-11eb-1d5d-578926de3257
# ╠═e3b82668-5e9f-11eb-1641-a9dfed9eb108
# ╟─e3b8fc58-5e9f-11eb-0607-e5424a04df9c
# ╠═5824d2ae-5ea2-11eb-3545-dd03c4aa7a69
# ╠═e3cab9ca-5e9f-11eb-3b31-c34fabb9e8d1
# ╠═e3ddad0a-5e9f-11eb-0a7d-150f9f398fe0
# ╠═ce509a0c-5ea8-11eb-3f2e-01023f29a1e3
# ╠═e3f103f0-5e9f-11eb-159a-452961ed2619
