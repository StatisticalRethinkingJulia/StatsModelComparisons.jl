using Test

@testset "BIC"  begin 
    using StatsModelComparisons

    v1 = bic(-200.0, 10, 100)
    v2 = bic(-200.0, 2, 100)
    v3 = bic(-200.0, 2, 10)
    v4 = bic(-100.0, 2, 10)
    @test v2 < v1
    @test v3 < v2
    @test v4 < v3
end

@testset "AIC"  begin 
    using StatsModelComparisons

    v1 = aic(-200.0, 10)
    v2 = aic(-200.0, 2)
    v3 = aic(-100.0, 2)
    @test v2 < v1
    @test v3 < v2
end

@testset "DIC"  begin 
    using StatsModelComparisons
    x = randn(100)
    LLs = x .- mean(x)
    v1 = dic(LLs)
    v2 = dic(LLs*2)
    v3 = dic(LLs .+ 1)
    @test v1 < v2
    @test v3 < v1
end