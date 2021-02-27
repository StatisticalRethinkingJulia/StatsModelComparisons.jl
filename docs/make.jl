using Documenter, ModelComparisons

makedocs(
    modules = [ModelComparisons],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Rob J Goedman",
    sitename = "ModelComparisons.jl",
    pages = Any["index.md"]
    # strict = true,
    # clean = true,
    # checkdocs = :exports,
)

deploydocs(
    repo = "github.com/goedman/ModelComparisons.jl.git",
    push_preview = true
)
