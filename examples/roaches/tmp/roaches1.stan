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