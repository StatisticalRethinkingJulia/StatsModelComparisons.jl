# StatsModelComparisons.jl

| **Project Status**                                                               |  **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
|![][project-status-img] | ![][CI-build] |


### Purpose of this package

This package implements model comparison methods as used and explained in StatisticalRethinking (chapter 7). Thus, StatsModelComparisons.jl is part of the [StatisticalRethinking family of packages](https://github.com/StatisticalRethinkingJulia/StatisticalRethinking.jl).

The most important methods are *Pareto smoothed importance sampling* (PSIS) and
PSIS leave-one-out cross-validation based on the [Matlab package called `PSIS` by Aki Vehtari](https://github.com/avehtari/PSIS.git). The Julia translation has been done by @alvaro1101 (on Github) in a (unpublished) package called [PSIS.jl](https://github.com/alvaro1101/PSIS.jl).

Updates for Julia v1+, the new Pkg ecosystem and the addition of WAIC and pk utilities have been done by Rob J Goedman. DIC has been added by Chris Fisher. Major code improvements have been done by David Widmann. The status of the package remains experimental and is, as is StatisticalRethinking.jl, primarily intended for learning statistical modeling approaches and pitfalls.

### Installation

StatsModelComparisons.jl can be installed with:
```
Pkg.add("StatsModelComparisons")
```

Each example and notebook will expect additional packages to be installed in your environment. These are listed at the top of each example or notebook.

Usually I have only a few packages `permanently` installed, e.g.:
```
(@v1.6) pkg> st
      Status `~/.julia/environments/v1.6/Project.toml`
  [634d3b9d] DrWatson v1.16.6
  [44cfe95a] Pkg
```
To use the demonstration Pluto notebooks, you can add:
```
  [c3e4b0f8] Pluto v0.12.18
  [7f904dfe] PlutoUI v0.6.11
```

To run the notebooks, I typically use an `alias`:
```
alias pluto="clear; j -i -e 'using Pkg; import Pluto; Pluto.run()'"
```
and then do:
```
$ cd ~/.julia/dev/StatsModelComparisons
$ pluto
```
to start Pluto from within that directory. 

The cars WAIC example requires RDatasets.jl to be installed and functioning.

### Included functions

`psisloo()` -
    Pareto smoothed importance sampling leave-one-out log predictive densities.

`psislw()` -
    Pareto smoothed importance sampling.

`waic()` -
    Compute WAIC for a loglikelihood matrix.

`dic()` -
    Deviance Information Criterion.

`pk_qualify()` -
    Show location of pk values.

`pk_plot()` -
    Plot pk values.

Additional function:

`gpdfitnew()` -
    Estimate the paramaters for the Generalized Pareto Distribution (GPD).

`gpinv()` -
    Inverse Generalised Pareto distribution function.

`var2()` -
    Uncorrected variance.

### Corresponding R code

Corresponding R code for the PSIS methods can be found in [R package called
`loo`](https://github.com/stan-dev/loo) which is available in CRAN.
                 
### References

- Aki Vehtari, Andrew Gelman and Jonah Gabry (2016). Practical
  Bayesian model evaluation using leave-one-out cross-validation
  and WAIC. Statistics and Computing, [doi:10.1007/s11222-016-9696-4](http://dx.doi.org/10.1007/s11222-016-9696-4). [arXiv preprint arXiv:1507.04544](http://arxiv.org/abs/1507.04544)
- Aki Vehtari, Andrew Gelman and Jonah Gabry (2016). Pareto
  smoothed importance sampling. [arXiv preprint arXiv:1507.02646](http://arxiv.org/abs/1507.02646)
- Jin Zhang & Michael A. Stephens (2009) A New and Efficient
  Estimation Method for the Generalized Pareto Distribution,
  Technometrics, 51:3, 316-325, DOI: 10.1198/tech.2009.08017
- Richard McElreath [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/)

[CI-build]: https://github.com/StatisticalRethinkingJulia/StatsModelComparisons.jl/workflows/CI/badge.svg?branch=master

[issues-url]: https://github.com/StatisticalRethinkingJulia/StatsModelComparisons.jl/issues

[project-status-img]: https://img.shields.io/badge/lifecycle-wip-orange.svg
