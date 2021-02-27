using Documenter, StatsModelComparisons

makedocs(
    modules = [StatsModelComparisons],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Rob J Goedman",
    sitename = "StatsModelComparisons.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/goedman/StatsModelComparisons.jl.git",
    push_preview = true
)
