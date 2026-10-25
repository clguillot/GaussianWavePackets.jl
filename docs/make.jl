using Documenter
using GaussianWavePackets

DocMeta.setdocmeta!(GaussianWavePackets, :DocTestSetup, :(using GaussianWavePackets, StaticArrays, LinearAlgebra); recursive=true)

makedocs(
    sitename = "GaussianWavePackets.jl",
    authors = "Clement Guillot",
    modules = [GaussianWavePackets],
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages = [
        "Home" => "index.md",
        "API reference" => "api.md",
    ],
    checkdocs = :exports,
)

# The documentation is not deployed for now (no call to deploydocs)
