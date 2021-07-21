using StanSample, ParetoSmooth
using StatisticalRethinking

df = CSV.read(sr_datadir("WaffleDivorce.csv"), DataFrame);
scale!(df, [:Marriage, :MedianAgeMarriage, :Divorce])
data = (N=size(df, 1), D=df.Divorce_s, A=df.MedianAgeMarriage_s,
    M=df.Marriage_s)

stan5_1 = "
data {
    int < lower = 1 > N; // Sample size
    vector[N] D; // Outcome
    vector[N] A; // Predictor
}
parameters {
    real a; // Intercept
    real bA; // Slope (regression coefficients)
    real < lower = 0 > sigma;    // Error SD
}
transformed parameters {
    vector[N] mu;               // mu is a vector
    for (i in 1:N)
        mu[i] = a + bA * A[i];
}
model {
    a ~ normal(0, 0.2);         //Priors
    bA ~ normal(0, 0.5);
    sigma ~ exponential(1);
    D ~ normal(mu , sigma);     // Likelihood
}
generated quantities {
    vector[N] log_lik;
    for (i in 1:N)
        log_lik[i] = normal_lpdf(D[i] | mu[i], sigma);
}
";

stan5_2 = "
data {
    int N;
    vector[N] D;
    vector[N] M;
}
parameters {
    real a;
    real bM;
    real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu;
    for (i in 1:N)
        mu[i]= a + bM * M[i];

}
model {
    a ~ normal( 0 , 0.2 );
    bM ~ normal( 0 , 0.5 );
    sigma ~ exponential( 1 );
    D ~ normal( mu , sigma );
}
generated quantities {
    vector[N] log_lik;
    for (i in 1:N)
        log_lik[i] = normal_lpdf(D[i] | mu[i], sigma);
}
";

stan5_3 = "
data {
  int N;
  vector[N] D;
  vector[N] M;
  vector[N] A;
}
parameters {
  real a;
  real bA;
  real bM;
  real<lower=0> sigma;
}
transformed parameters {
    vector[N] mu;
    mu = a + + bA * A + bM * M;
}
model {
  a ~ normal( 0 , 0.2 );
  bA ~ normal( 0 , 0.5 );
  bM ~ normal( 0 , 0.5 );
  sigma ~ exponential( 1 );
  D ~ normal( mu , sigma );
}
generated quantities{
    vector[N] log_lik;
    for (i in 1:N)
        log_lik[i] = normal_lpdf(D[i] | mu[i], sigma);
}
";

m5_1s = SampleModel("m5.1s", stan5_1)
rc5_1s = stan_sample(m5_1s; data)
if success(rc5_1s)
    st5_1s = read_samples(m5_1s; output_format=:table)
    log_lik = matrix(st5_1s, "log_lik")

    ll = reshape(Matrix(log_lik'), 50, 1000, 4);
    m5_1s_loo = ParetoSmooth.loo(ll)
end

m5_2s = SampleModel("m5.2s", stan5_2)
rc5_2s = stan_sample(m5_2s; data)
if success(rc5_2s)
    st5_2s = read_samples(m5_2s; output_format=:table)
    log_lik = matrix(st5_2s, "log_lik")

    ll = reshape(Matrix(log_lik'), 50, 1000, 4);
    m5_2s_loo = ParetoSmooth.loo(ll)
end

m5_3s = SampleModel("m5.3s", stan5_3)
rc5_3s = stan_sample(m5_3s; data)
if success(rc5_3s)
    st5_3s = read_samples(m5_3s; output_format=:table)
    log_lik = matrix(st5_3s, "log_lik")

    ll = reshape(Matrix(log_lik'), 50, 1000, 4);
    m5_3s_loo = ParetoSmooth.loo(ll)
end

if success(rc5_1s) && success(rc5_2s) && success(rc5_3s)
    m5_1s_loo |> display
    m5_3s_loo |> display
    m5_2s_loo |> display
end

#=
With SR/ulam():
```
       PSIS    SE dPSIS  dSE pPSIS weight
m5.1u 126.0 12.83   0.0   NA   3.7   0.67
m5.3u 127.4 12.75   1.4 0.75   4.7   0.33
m5.2u 139.5  9.95  13.6 9.33   3.0   0.00
```
=#
