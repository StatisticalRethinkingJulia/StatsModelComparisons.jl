using StanSample, ParetoSmooth
using StatisticalRethinking
import StatisticalRethinking: lppd


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
    ka5_1s = read_samples(m5_1s; output_format=:keyedarray)
    log_lik5_1s = matrix(ka5_1s, "log_lik")
    m5_1s_loo = psis_loo(log_lik5_1s)
end

m5_2s = SampleModel("m5.2s", stan5_2)
rc5_2s = stan_sample(m5_2s; data)
if success(rc5_2s)
    ka5_2s = read_samples(m5_2s; output_format=:keyedarray)
    log_lik5_2s = matrix(ka5_2s, "log_lik")
    m5_2s_loo = psis_loo(log_lik5_2s)
end

m5_3s = SampleModel("m5.3s", stan5_3)
rc5_3s = stan_sample(m5_3s; data)
if success(rc5_3s)
    ka5_3s = read_samples(m5_3s; output_format=:keyedarray)
    log_lik5_3s = matrix(ka5_3s, "log_lik")
    m5_3s_loo = psis_loo(log_lik5_3s)
end

if success(rc5_1s) && success(rc5_2s) && success(rc5_3s)
    m5_1s_loo |> display
    m5_3s_loo |> display
    m5_2s_loo |> display
end

# Adapt for kar format
lppd(kar) = [logsumexp(kar[i, :]) - log(size(kar, 2)) for i in 1:size(kar, 1)]

#=

    kar = Vector{Matrix{Float64}}(undef, length(nmodels))
    kar[i] = reshape(ka[i].data, nparams, nchains*ndraws)
    lppds[i] = -2sum(lppd(kar[i]))
    pk = Vector{Vector{Float64}}(undef, length(nmodels))
    pk[i] = psis[i].pointwise(:pareto_k)

=#

struct LooCompare
    loglikelihoods::Vector{Array{Float64, 3}}
    psis::Vector{PsisLoo}
    table::KeyedArray
end

function loo_compare(models::Vector{SampleModel}; 
    loglikelihood_name="log_lik", model_names=nothing)

    if isnothing(model_names)
        mnames = [models[i].name for i in 1:length(models)]
    end

    nmodels = length(models)

    ka = Vector{KeyedArray}(undef, nmodels)
    ll = Vector{KeyedArray}(undef, nmodels)

    for i in 1:length(models)
        ka[i] = read_samples(models[i]; output_format=:keyedarray)
        ll[i] = matrix(ka[i], loglikelihood_name)
    end

    #loo_compare(Array.(ka); model_names=mnames)
    (lls=Array.(ll), model_names=mnames)
end

function loo_compare(loglikelihoods::Vector{Array{Float64, 3}};
    model_names=nothing)

    nmodels = length(loglikelihoods)

    if isnothing(model_names)
        mnames = ["model_$i" for i in 1:nmodels]
    else
        mnames = model_names
    end

    psis = Vector{PsisLoo}(undef, nmodels)
    loov = Vector{Float64}(undef, nmodels)
    loos = Vector{Vector{Float64}}(undef, nmodels)

    for i in 1:nmodels
        psis[i] = psis_loo(loglikelihoods[i])
        loov[i] = psis[i].estimates(:loo_score, :total)
        loos[i] = psis[i].pointwise(:loo_score)
    end

    ind = sortperm([-2loov[i][1] for i in 1:nmodels])
    
    lls = loglikelihoods[ind]
    loov = loov[ind]
    loos = loos[ind]
 
    if length(mnames) > 0
        mnames = String.(model_names[ind])
    end

    psis_values = round.([-2loov[i] for i in 1:length(loov)], digits=2)

    se = [sqrt(size(lls[i], 1)*var2(-2loos[i])) for i in 1:nmodels]
    
    dpsis = zeros(nmodels)
    for i in 2:nmodels
        dpsis[i] = psis_values[i] - psis_values[1]
    end

    dse = zeros(nmodels)
    for i in 2:nmodels
        diff = 2(loos[1] - loos[i])
        dse[i] = √(length(loos[i]) * var2(diff))
    end

    ppsis = zeros(nmodels)
    for j in 1:nmodels
        nparams, nsamples, nchains = size(lls[j])
        pd = zeros(nmodels, nchains)
        #pd[j, :] = [var2(lls[j][:,i]) for i in 1:nchains]
        ppsis[j] = sum(pd[j, :]) 
    end

    weights = ones(nmodels)
    sumval = sum([exp(-0.5psis_values[i]) for i in 1:nmodels])
    for i in 1:nmodels
        weights[i] = exp(-0.5psis_values[i])/sumval
    end

    ka = KeyedArray((models=mnames, PSIS=psis_values, SE=se, dPSIS=dpsis,
        dSE=dse, weight=weights))
    LooCompare(lls, psis, ka)

end

models = [m5_1s, m5_2s, m5_3s]
loglikelihood_name = :log_lik
loglikelihoods, model_names = loo_compare(models)
psis = loo_compare(loglikelihoods; model_names)
psis.table |> display

#=
With SR/ulam():
```
       PSIS    SE dPSIS  dSE pPSIS weight
m5.1u 126.0 12.83   0.0   NA   3.7   0.67
m5.3u 127.4 12.75   1.4 0.75   4.7   0.33
m5.2u 139.5  9.95  13.6 9.33   3.0   0.00
```
=#
