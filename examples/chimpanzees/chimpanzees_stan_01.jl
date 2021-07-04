#
# This example assumes below packages are available in your environment
#

using StanSample, StatsModelComparisons, ParetoSmooth
using StatsFuns

# Just to access the data

using DrWatson
using StatisticalRethinking: sr_datadir

df = CSV.read(sr_datadir("chimpanzees.csv"), DataFrame)

stan11_4 = "
data{
    int<lower=1> N;
    int y[N];
    int prosoc[N];
}
parameters{
    real a;
    real bP;
}
model{
    vector[N] p;
    bP ~ normal( 0 , 1 );
    a ~ normal( 0 , 10 );
    for ( i in 1:N ) {
        p[i] = a + bP * prosoc[i];
    }
    y ~ binomial_logit( 1 , p );
}
generated quantities{
    vector[N] p;
    vector[N] log_lik;
    for ( i in 1:N ) {
        p[i] = a + bP * prosoc[i];
        log_lik[i] = binomial_logit_lpmf( y[i] | 1 , p[i] );
    }
}
";

m11_4s = SampleModel("m11_4s", stan11_4)
data = (N = size(df, 1), y = df.pulled_left, prosoc = df.prosoc_left)
rc11_4s = stan_sample(m11_4s; data)

if success(rc11_4s)
    st11_4s = read_samples(m11_4s; output_format=:table);

    log_lik = matrix(st11_4s, "log_lik")
    n_sam, n_obs = size(log_lik)
    lppd = reshape(logsumexp(log_lik .- log(n_sam); dims=1), n_obs)

    pwaic = [var(log_lik[:, i]) for i in 1:n_obs]
    @show -2(sum(lppd) - sum(pwaic))
    println()

    @show waic(log_lik)
    println()

    loo, loos, pk = psisloo(log_lik)
    @show -2loo

    ll = Matrix(log_lik');
    llr = reshape(ll, 504, 1000, 4);
    psis_ll = psis(llr);

    lwp = deepcopy(llr);
    lwp += psis_ll.weights;
    lwpt = Matrix(reshape(lwp, 504, 4000)');
    loos = reshape(logsumexp(lwpt; dims=1), size(lwpt, 2));

    @show loo = sum(loos)
    @show 2loo

end
