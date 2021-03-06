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