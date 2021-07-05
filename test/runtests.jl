using StatsModelComparisons, StanSample
using Test

println()
include("model_comparison_tests.jl")
include("test_demo_wells.jl")
include("test_cars.jl")
include("../examples/cars_waic/cars_stan.jl")
#include("../examples/chimpanzees/chimpanzees_stan_01.jl")
#include("../examples/chimpanzees/chimpanzees_stan_02.jl")
#include("../examples/cars_waic/cars_stan.jl")
#include("../examples/waffle_divorce/waffle_divorce_stan.jl")
