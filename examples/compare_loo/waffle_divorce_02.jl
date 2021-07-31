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

    loo_compare(Array.(ll); model_names=mnames)
    #(lls=Array.(ll), model_names=mnames)
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
    se = Vector{PsisLoo}(undef, nmodels)
    loov = Vector{Float64}(undef, nmodels)
    loos = Vector{Vector{Float64}}(undef, nmodels)

    for i in 1:nmodels
        psis[i] = psis_loo(loglikelihoods[i])
        se[i] = psis[i].estimates(:loo_score, :se)
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

    psis_values = [-2loov[i] for i in 1:length(loov)]
    data = psis_values

    se = [sqrt(size(lls[i], 1)*var2(-2loos[i])) for i in 1:nmodels]
    data = hcat(data, se)
    
    dpsis = zeros(nmodels)
    for i in 2:nmodels
        dpsis[i] = psis_values[i] - psis_values[1]
    end
    data = hcat(data, dpsis)

    dse = zeros(nmodels)
    for i in 2:nmodels
        diff = 2(loos[1] - loos[i])
        dse[i] = √(length(loos[i]) * var2(diff))
    end
    data = hcat(data, dse)

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
    data = hcat(data, weights)
    
    table = KeyedArray(
        data,
        model = mnames,
        statistic = [:PSIS, :SE, :ΔPSIS, :ΔSE, :weights],
    )
    LooCompare(lls, psis, table)

end

import Base.show
function Base.show(io::IO, ::MIME"text/plain", loo_compare::LooCompare)
    table = loo_compare.table
    return pretty_table(
        table;
        compact_printing=false,
        header=table.statistic,
        row_names=table.model,
        formatters=ft_printf("%5.2f"),
        alignment=:r,
    )
end


m5_1s = SampleModel("m5.1s", stan5_1)
rc5_1s = stan_sample(m5_1s; data)
m5_2s = SampleModel("m5.2s", stan5_2)
rc5_2s = stan_sample(m5_2s; data)
m5_3s = SampleModel("m5.3s", stan5_3)
rc5_3s = stan_sample(m5_3s; data)

if success(rc5_1s) && success(rc5_2s) && success(rc5_3s)
    models = [m5_1s, m5_2s, m5_3s]
    loglikelihood_name = :log_lik
    #loglikelihoods, model_names = loo_compare(models)
    #psis = loo_compare(loglikelihoods; model_names)
    loo_comparison = loo_compare(models)
    println()
    loo_comparison |> display
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
