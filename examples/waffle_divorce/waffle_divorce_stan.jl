#
# This example assumes below packages available in your environment
#

using StanSample, StatsModelComparisons, ParetoSmooth
using StatsFuns, Test

# Just to access the data

using DrWatson
using StatisticalRethinking: sr_datadir

df = CSV.read(sr_datadir("WaffleDivorce.csv"), DataFrame);

stan5_1_t = "
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
    vector[N] mu;
    mu = a + + bA * A;
}

model {
  a ~ normal( 0 , 0.2 );
  bA ~ normal( 0 , 0.5 );
  sigma ~ exponential( 1 );
  D ~ student_t( 2, mu , sigma );
}
generated quantities{
    vector[N] log_lik;
    for (i in 1:N)
        log_lik[i] = student_t_lpdf(D[i] | 2, mu[i], sigma);
}
";

begin
    data = (N=size(df, 1), D=df.Divorce, A=df.MedianAgeMarriage,
        M=df.Marriage)
    m5_1s_t = SampleModel("m5.1s_t", stan5_1_t)
    rc5_1s_t = stan_sample(m5_1s_t; data)
end

if success(rc5_1s_t)
    st5_1_t = read_samples(m5_1s_t; output_format=:table);

    @test names(st5_1_t)[end] == Symbol("log_lik.50")
    @test size(DataFrame(st5_1_t)) == (4000, 103)

    log_lik = matrix(st5_1_t, "log_lik")
    @test size(log_lik) == (4000, 50)

    n_sam, n_obs = size(log_lik)
    lppd = reshape(logsumexp(log_lik .- log(n_sam); dims=1), n_obs)

    pwaic = [var(log_lik[:, i]) for i in 1:n_obs]
    @show -2(sum(lppd) - sum(pwaic))
    println()

    @show waic(log_lik)
    println()

    loo, loos, pk = psisloo(log_lik)
    @show -2loo

    println("\nUsing new psis\n")

    ll = Matrix(log_lik');
    llr = reshape(ll, 50, 1000, 4);
    psis_ll = psis(llr);

    lwp = deepcopy(llr);
    lwp += psis_ll.weights;
    lwpt = Matrix(reshape(lwp, 50, 4000)');
    loos = reshape(logsumexp(lwpt; dims=1), size(lwpt, 2));

    @show loo = sum(loos)
    @show 2loo

    
end

