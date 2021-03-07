        library("rstan")
        library("loo")
        
        setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
        options(mc.cores = parallel::detectCores())
        set.seed(5054)
        
        data(cars)
        data = list("N" = dim(cars)[1], speed = cars$speed, dist=cars$dist)
        
        #fit <- stan( file="cars_model.stan", data = data, sample_file="test_chains/R_Samples.csv", seed=987,
        #             iter=2000, warmup=1000)
        fit <- stan( file="cars_model.stan", data = data, seed=987,
                     iter=2000, warmup=1000)
        
        
        # Extract pointwise log-likelihood
        # using merge_chains=FALSE returns an array, which is easier to 
        # use with relative_eff()
        log_lik <- extract_log_lik(fit, merge_chains = FALSE)
        
        # as of loo v2.0.0 we can optionally provide relative effective sample sizes
        # when calling loo, which allows for better estimates of the PSIS effective
        # sample sizes and Monte Carlo error
        r_eff <- relative_eff(exp(log_lik), cores = 2)
        
        # preferably use more than 2 cores (as many cores as possible)
        # will use value of 'mc.cores' option if cores is not specified
        loo <- loo(log_lik, r_eff = r_eff, cores = 2)
        print(loo)
        
        waic = waic(log_lik)
        