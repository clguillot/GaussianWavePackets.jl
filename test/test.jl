using GaussianWavePackets

using Test
using Random
using LinearAlgebra
using FastGaussQuadrature
using SpecialFunctions
using StaticArrays
using NullNumbers
using Cubature
using Aqua
using JET

function hermite_poly(n::Integer, x::T) where{T<:Number}
    v = zero(T)
    u = one(T)
    for j in 1:n
        w = u
        u = 2*x*u - 2*(j-1)*v
        v = w
    end
    return u
end

#ψₙ(z, q, p, x) = (a/π)^(1/4) / sqrt(2ⁿn!) * Hₙ(√a*(x-q)) * exp(-z(x-q)²/2) * exp(ipx)
function ψ(z, q, p, n, x)
    a = real(z)
    return (a/π)^(1/4) / sqrt(2.0^n * gamma(n+1)) * hermite_poly(n, sqrt(a) * (x-q)) * exp(-z*(x-q)^2/2) * cis(p*x)
end

#=
    Integral of a complex function f on the box [a, b]
    The absolute tolerance is relative to ∫|f| by default: with abstol = 0, the cubature of a part
    which vanishes (e.g. the imaginary part of a real function) would never converge
=#
function complex_cubature(f, a, b; reltol=1e-12, abstol=nothing, maxevals=10^7)
    if abstol === nothing
        abstol = reltol * hcubature(x -> abs(f(x)), a, b; reltol=1e-6, maxevals=maxevals)[1]
    end
    Ir = hcubature(x -> real(f(x)), a, b; reltol=reltol, abstol=abstol, maxevals=maxevals)
    Ic = hcubature(x -> imag(f(x)), a, b; reltol=reltol, abstol=abstol, maxevals=maxevals)
    return complex(Ir[1], Ic[1])
end

#=
    Integration domain for a cubature: a box containing the centers c (vectors), with a margin
    of `nσ` standard deviations of the narrowest real covariance a (exp(-nσ²/2) is negligible)
=#
function integration_box(centers, amin; nσ=10)
    w = nσ / sqrt(amin)
    lo = [minimum(c[k] for c in centers) - w for k in eachindex(first(centers))]
    hi = [maximum(c[k] for c in centers) + w for k in eachindex(first(centers))]
    return lo, hi
end
# The smallest eigenvalue of the real part of a covariance (matrix or number)
min_eig(z::Number) = real(z)
min_eig(z::AbstractMatrix) = eigmin(Symmetric(real(Matrix(z))))
# The center of a wave packet (0 if q is not present)
center(q::SVector) = map(x -> x isa NullNumber ? 0.0 : x, q)
center(q::Number) = SVector(q isa NullNumber ? 0.0 : q)
# The entries of a vector, with the NullNumbers replaced by 0.0 (for the comparisons with ≈)
num(v) = map(y -> y isa NullNumber ? 0.0 : y, v)
center(G::GaussianWavePacket) = center(G.q)
center(H::HermiteWavePacket1D) = center(H.q)

# Relative error, with a floor for the quantities which can be small
err_rel(a, b; floor=0.0) = abs(a - b) / max(abs(b), floor)

# Allocations are only reported (in green if there are none, in red otherwise) and not tested:
# whether a call allocates depends on the Julia version and on how the calling code is compiled
function report_alloc(name, alloc)
    printstyled("Allocations ($name): $alloc bytes\n"; bold=true, color=(alloc == 0 ? :green : :red))
end
# Number of bytes allocated by f(args...), measured behind a function barrier (the loops over
# wave packets of different types are not type stable), after a first call for the compilation
# (the arity is fixed: with a splatted `args...`, the compilation is much slower)
alloc_of(f, a) = (f(a); @allocated f(a))
alloc_of(f, a, b) = (f(a, b); @allocated f(a, b))
alloc_of(f, a, b, c) = (f(a, b, c); @allocated f(a, b, c))
alloc_of(f, a, b, c, d) = (f(a, b, c, d); @allocated f(a, b, c, d))

# Whether the return type of f(args...) is inferred (prints the call otherwise)
function is_inferred(f, args...)
    try
        @inferred f(args...)
        return true
    catch e
        e isa ErrorException || rethrow()
        @info "Not inferred" f typeof.(args)
        return false
    end
end

#=
    RANDOM WAVE PACKETS
    covariance: :diag or :sym, real or complex; q and p: present or NullNumber
=#
function rand_cov(::Val{D}, kind::Symbol, cplx::Bool) where D
    if kind === :diag
        a = 4 .* rand(SVector{D}) .+ 0.5
        return cplx ? Diagonal(a .+ im .* (4 .* rand(SVector{D}) .- 2)) : Diagonal(a)
    else
        A = rand(SMatrix{D, D})
        B = rand(SMatrix{D, D})
        a = A' * A + 0.5I
        return cplx ? Symmetric(a + im * (B + B')) : Symmetric(a)
    end
end
rand_vec(::Val{D}, present::Bool) where D = present ? 4 .* (rand(SVector{D}) .- 0.5) : zeros(SVector{D, NullNumber})
rand_gwp(::Val{D}, kind, cplx, hq, hp) where D =
    GaussianWavePacket(rand() + 1im * rand(), rand_cov(Val(D), kind, cplx), rand_vec(Val(D), hq), rand_vec(Val(D), hp))
# One random wave packet for each combination of the parameters
gwp_combos(::Val{D}) where D =
    Any[rand_gwp(Val(D), k, c, q, p) for k in (:diag, :sym), c in (false, true), q in (false, true), p in (false, true)]

rand_scalar(present::Bool) = present ? 4 * (rand() - 0.5) : NullNumber()
rand_hermite(N, cplx, hq, hp) =
    HermiteWavePacket1D(SVector{N}(rand(N) + im * rand(N)), 4 * rand() + 0.5 + (cplx ? im * (4 * rand() - 2) : 0),
                        rand_scalar(hq), rand_scalar(hp))
hermite_combos(N) = Any[rand_hermite(N, c, q, p) for c in (false, true), q in (false, true), p in (false, true)]

include("test_gaussian.jl")
include("test_gaussian_wave_packet.jl")
include("test_hermite_wave_packet_1d.jl")
include("test_wave_packet_sum.jl")
include("test_type_stability.jl")
include("test_quality.jl")
