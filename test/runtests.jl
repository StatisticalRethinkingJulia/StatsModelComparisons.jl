using ModelComparisons, StanSample
using Test

if haskey(ENV, "JULIA_CMDSTAN_HOME")

    ProjDir = @__DIR__
    include(joinpath(ProjDir, "test_demo_wells.jl"))

else
  println("\nJULIA_CMDSTAN_HOME not set. Skipping tests")
end
