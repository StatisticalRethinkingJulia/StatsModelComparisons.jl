#
# This example assumes below packages are available in your environment
#

using StatsModelComparisons, StanSample
using CSV, Printf, StatsPlots

ProjDir = @__DIR__

df = CSV.read(joinpath(ProjDir, "roachdata.csv"), DataFrame)
df.roach1 = df.roach1 / 100

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

data = (N = size(df, 1), K = 4, y = Int.(df.y), roach1=df.roach1,
  exposure2=df.exposure2, senior=df.senior, treatment=df.treatment)
n = size(df, 1)

tmpdir = mktempdir()
sm1 = SampleModel("roaches1", roaches1_stan; tmpdir)
rc1 = stan_sample(sm1; data)

if success(rc1)
  stan_summary(sm1, true)
  nt1 = read_samples(sm1)

  # Compute LOO and standard error
  log_lik1 = nt1.log_lik'
  loo1, loos1, pk1 = psisloo(log_lik1)
  elpd_loo1 = sum(loos1)
  se_elpd_loo1 = std(loos1) * sqrt(n)
  @printf(">> elpd_loo = %.1f, SE(elpd_loo) = %.1f\n", elpd_loo1, se_elpd_loo1)

  # Check the shape parameter k of the generalized Pareto distribution
  if all(pk1 .< 0.5)
      println("All Pareto k estimates OK (k < 0.5).")
  else
    pk_good = sum(pk1 .<= 0.5)
    pk_ok = length(pk1[pk1 .<= 0.7]) - pk_good
    pk_bad = length(pk1[pk1 .<= 1]) - pk_good - pk_ok
    println((good=pk_good, ok=pk_ok, bad=pk_bad, very_bad=sum(pk1 .> 1)))
  end
end

begin
  scatter(pk1, xlab="Datapoint", ylab="Pareto shape k",
    marker=2.5, lab="Pk points")
  hline!([0.5], lab="pk = 0.5");hline!([0.7], lab="pk = 0.7")
  hline!([1], lab="pk = 1.0")
  title!("PSIS diagnostic plot for poisson-log model.")
  savefig(joinpath(ProjDir, "diag_plot_1.png"))
end


#=
*Simple negative binomial regression example
*using the 2nd parametrization of the negative
*binomial distribution, see section 40.1-3 in the Stan
*reference guide
=#

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

tmpdir=joinpath(ProjDir, "tmp")
sm2 = SampleModel("roaches2", roaches2_stan; tmpdir)
rc2 = stan_sample(sm2; data)

if success(rc2)
  read_summary(sm2, true)
  nt2 = read_samples(sm2)

  # Compute LOO and standard error
  log_lik2 = nt2.log_lik'
  loo2, loos2, pk2 = psisloo(log_lik2)
  elpd_loo2 = sum(loos2)
  se_elpd_loo2 = std(loos2) * sqrt(n)
  @printf(">> elpd_loo = %.1f, SE(elpd_loo) = %.1f\n", elpd_loo2, se_elpd_loo2)

  # Check the shape parameter k of the generalized Pareto distribution
  pk_qualify(pk2) |> display
  
end

begin
  scatter(pk2, xlab="Datapoint", ylab="Pareto shape k",
    marker=2.5, lab="Pk points")
  hline!([0.5], lab="pk = 0.5");hline!([0.7], lab="pk = 0.7")
  hline!([1], lab="pk = 1.0")
  title!("PSIS diagnostic plot for neg-binomial model.")
  savefig(joinpath(ProjDir, "diag_plot_2.png"))
end

