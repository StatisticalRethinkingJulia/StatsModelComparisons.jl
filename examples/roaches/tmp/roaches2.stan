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