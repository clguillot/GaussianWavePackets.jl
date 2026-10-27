
function Base.zero(::T) where{T<:AbstractWavePacket}
    return zero(T)
end

"""
    core_type(G)
    core_type(T::Type)
    core_type(G1, G2, ...)

Returns the type of the numbers used by a wave packet (or a number, or a type of wave packet):
the promotion of the types of all its parameters. With several arguments, returns the
promotion of their core types.
"""
function core_type(::T) where{T<:AbstractWavePacket}
    return core_type(T)
end

function core_type(::Type{T}) where{T<:Number}
    return T
end
function core_type(::T) where{T<:Number}
    return core_type(T)
end

function core_type(S, T)
    return promote_type(core_type(S), core_type(T))
end
function core_type(S, T, U...)
    return promote_type(core_type(S), core_type(T), core_type(U...))
end

@generated function fitting_float(::Type{T}) where{T<:Number}
    prec = precision(real(T))
    if prec == precision(Float16)
        return :( Float16 )
    elseif prec == precision(Float32)
        return :( Float32 )
    elseif prec == precision(Float64)
        return :( Float64 )
    else
        throw(ArgumentError("The precision format of $T is unsupported"))
    end
end
function fitting_float(T...)
    return fitting_float(core_type(T...))
end

@inline function Base.:*(G::AbstractWavePacket, w::Number)
    return w * G
end

norm_L2(G::AbstractWavePacket) = sqrt(norm2_L2(G))
norm_∇(G::AbstractWavePacket) = sqrt(norm2_∇(G))
norm_∇(G::AbstractWavePacket, c::AbstractVector) = sqrt(norm2_∇(G, c))
norm_x(G::AbstractWavePacket) = sqrt(norm2_x(G))
norm_x(G::AbstractWavePacket, c::AbstractVector) = sqrt(norm2_x(G, c))
∫(G::AbstractWavePacket; kwargs...) = integral(G; kwargs...)
# Evaluation at a point given by its coordinates (which may be of different numeric types)
(G::AbstractWavePacket{D})(x::Number...) where D = G(SVector{D}(promote(x...)))

#=
    DOCUMENTATION OF THE GENERIC FUNCTIONS
    (the methods are defined in the files of each type of wave packet)
=#

"""
    AbstractWavePacket{D}

Supertype of the wave packets in dimension `D`: [`GaussianWavePacket`](@ref),
[`HermiteWavePacket1D`](@ref) and [`WavePacketSum`](@ref).

A wave packet `G` can be evaluated at a point with `G(x)` (where `x` is a vector of length `D`)
or `G(x₁, ..., x_D)`.
"""
AbstractWavePacket

"""
    evaluate(G, x)
    evaluate(G, x; dims)
    G(x)
    G(x₁, ..., x_D)

Evaluates the wave packet `G` at the point `x` (a vector of length `D`, or a number in
dimension 1).

For a [`GaussianWavePacket`](@ref) (or a sum of them), `dims` (a tuple of distinct integers)
selects the variables which are fixed: `x[k]` is then the value of the variable number
`dims[k]`, and the result is a wave packet in the other variables (in increasing order).
For instance `evaluate(G, SVector(0.7); dims=(2,))` freezes `x₂ = 0.7`.
`dims` should be a constant for the result to be type-stable. When it is not known at compile
time, `evaluate(G, x, Val(dims))` can be used instead.
"""
function evaluate end

"""
    integral(G)
    integral(G; dims)
    ∫(G)

Computes the integral of the wave packet `G` over ℝᴰ.

For a [`GaussianWavePacket`](@ref) (or a sum of them), `dims` (a tuple of distinct integers)
selects the variables which are integrated, and the result is a wave packet in the other
variables (in increasing order). For instance `integral(G; dims=(2,))` integrates out `x₂`.
`dims` should be a constant for the result to be type-stable. When it is not known at compile
time, `integral(G, Val(dims))` can be used instead.

Throws an `ArgumentError` for a plane wave (a `GaussianWavePacket` whose `z` is a `NullNumber`).
"""
function integral end

"""
    ∫(G; dims)

Same as [`integral`](@ref).
"""
∫

"""
    unitary_product(G, b, q=0, p=0)

Multiplies the wave packet `G` by the unitary factor

    exp(-i(x-q)ᵀb(x-q)/2) exp(ip⋅x)

where `b` is a real symmetric matrix (a real number for a [`HermiteWavePacket1D`](@ref)) and
`q`, `p` are real vectors (real numbers in the Hermite case).
"""
function unitary_product end

"""
    polynomial_product(H, P, q=0)

Multiplies the 1D wave packet `H` by the polynomial

    P(x) = ∑ₖ P[k](x-q)^(k-1)

where `P` is an `SVector` of coefficients. The result is a [`HermiteWavePacket1D`](@ref)
(computed exactly) with `N + length(P) - 1` coefficients.
"""
function polynomial_product end

"""
    fourier(G)

Computes the Fourier transform of the wave packet `G`, defined as

    TF(G)(ξ) = ∫dx exp(-ix⋅ξ) G(x)

The result is a wave packet of the same kind.
"""
function fourier end

"""
    inv_fourier(G)

Computes the inverse Fourier transform of the wave packet `G`, defined as

    ITF(G)(x) = (2π)⁻ᴰ ∫dξ exp(ix⋅ξ) G(ξ)

The result is a wave packet of the same kind.
"""
function inv_fourier end

"""
    convolution(G1, G2)

Computes the convolution product `(G1 ∗ G2)(x) = ∫dy G1(y) G2(x-y)` of two wave packets.
"""
function convolution end

"""
    dot_L2(G1, G2)

Computes the L² inner product `∫dx conj(G1(x)) G2(x)` (conjugate on the first argument).
"""
function dot_L2 end

"""
    norm2_L2(G)

Computes the squared L² norm `∫dx |G(x)|²`.
"""
function norm2_L2 end

"""
    norm_L2(G)

Computes the L² norm `sqrt(∫dx |G(x)|²)`.
"""
norm_L2

"""
    dot_∇(G1, G2)
    dot_∇(G1, G2, c)

Computes the homogeneous H¹ inner product `∫dx ⟨∇G1(x), ∇G2(x)⟩` (where ⟨,⟩ is the
hermitian product in ℂᴰ, conjugate on the first argument).
With `c` (a real vector), computes `∫dx ⟨(D-c)G1(x), (D-c)G2(x)⟩` where `D = -i∇`.

Defined for [`GaussianWavePacket`](@ref)s and for the sums of them.
"""
function dot_∇ end

"""
    norm2_∇(G)
    norm2_∇(G, c)

Computes `∫dx |∇G(x)|²` (or, with `c`, `∫dx |(D-c)G(x)|²` where `D = -i∇`).
"""
function norm2_∇ end

"""
    norm_∇(G)
    norm_∇(G, c)

Square root of [`norm2_∇`](@ref).
"""
norm_∇

"""
    dot_x(G1, G2)
    dot_x(G1, G2, c)

Computes `∫dx |x|² conj(G1(x)) G2(x)`, that is the L² inner product of `xG1` and `xG2`
(the hermitian product in ℂᴰ being used for the vectors).
With `c` (a real vector), computes `∫dx |x-c|² conj(G1(x)) G2(x)`.

Defined for [`GaussianWavePacket`](@ref)s and for the sums of them.
"""
function dot_x end

"""
    norm2_x(G)
    norm2_x(G, c)

Computes `∫dx |x|² |G(x)|²` (or, with `c`, `∫dx |x-c|² |G(x)|²`).
"""
function norm2_x end

"""
    norm_x(G)
    norm_x(G, c)

Square root of [`norm2_x`](@ref).
"""
norm_x

"""
    moment_x(G1, G2, c=0)
    moment_x(G, c=0)

Computes the vector `(∫dx (x-c)ⱼ conj(G1(x)) G2(x))ⱼ`, as an `SVector` (`c` is a real vector,
absent by default). With a single argument, computes the associated quadratic form
`(∫dx (x-c)ⱼ |G(x)|²)ⱼ`, which is real.

Defined for [`GaussianWavePacket`](@ref)s and for the sums of them.
"""
function moment_x end

"""
    moment_D(G1, G2, c=0)
    moment_D(G, c=0)

Computes the vector `(∫dx conj((D-c)ⱼG1(x)) G2(x))ⱼ`, as an `SVector`, where `Dⱼ = -i∂ⱼ` (`c` is
a real vector, absent by default). With a single argument, computes the associated quadratic
form `(∫dx conj((D-c)ⱼG(x)) G(x))ⱼ`, which is real.

Defined for [`GaussianWavePacket`](@ref)s and for the sums of them.
"""
function moment_D end

"""
    moment_xx(G1, G2, c=0)
    moment_xx(G, c=0)

Computes the matrix `(∫dx (x-c)ᵢ(x-c)ⱼ conj(G1(x)) G2(x))ᵢⱼ`, as an `SMatrix` (`c` is a real
vector, absent by default). With a single argument, computes the associated quadratic form
`(∫dx (x-c)ᵢ(x-c)ⱼ |G(x)|²)ᵢⱼ`, which is real.

Defined for [`GaussianWavePacket`](@ref)s and for the sums of them.
"""
function moment_xx end

"""
    moment_xD(G1, G2, cx=0, cD=0)
    moment_xD(G, cx=0, cD=0)

Computes the matrix `(∫dx conj(G1(x)) (AᵢⱼG2)(x))ᵢⱼ`, as an `SMatrix`, where
`Aᵢⱼ = (XᵢPⱼ + PⱼXᵢ)/2 = XᵢPⱼ - iδᵢⱼ/2` is the symmetrized (self-adjoint) product of
`Xᵢ = (x-cx)ᵢ` and `Pⱼ = (D-cD)ⱼ`, with `Dⱼ = -i∂ⱼ` (`cx` and `cD` are real vectors, absent by
default). With a single argument, computes the associated quadratic form, which is real.

Defined for [`GaussianWavePacket`](@ref)s and for the sums of them.
"""
function moment_xD end

"""
    coulomb_integral(G::GaussianWavePacket{3})

Computes `∫dx G(x)/|x|` for an isotropic 3D Gaussian wave packet (`z` must be a multiple of the
identity, either `Diagonal` or `Symmetric`). Throws an `ArgumentError` otherwise.
"""
function coulomb_integral end
