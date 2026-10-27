
#=
    The type of a covariance matrix z: either a dense D×D symmetric matrix,
    or (when there is no correlation between dimensions) a cheaper diagonal matrix.
    Used for both the general complex z of GaussianWavePacket and the real z of Gaussian.
=#
const CovMat{D, Tz} = Union{Symmetric{Tz, <:SMatrix{D, D, Tz}}, SDiagonal{D, Tz}}

"""
    GaussianWavePacket(λ, z, q=0, p=0)
    GaussianWavePacket{D}(λ, z, q, p)

Represents the complex Gaussian wave packet in dimension `D`

    G(x) = λ exp(-(x-q)ᵀz(x-q)/2) exp(ip⋅x)

where `λ` is a complex number, `z` is a complex symmetric matrix whose real part is positive
definite, and `q`, `p` are real vectors.

- In dimension 1, `z`, `q` and `p` can be given as plain numbers.
- In dimension `D`, `z` is a `Diagonal` of an `SVector` or a `Symmetric` of an `SMatrix`,
  and `q`, `p` are `SVector`s. With `GaussianWavePacket{D}`, plain arrays are also accepted.
- `q` and `p` are optional: when they are omitted, they are stored as `NullNumber`s, which
  are statically known to be zero (so that the corresponding computations are skipped).

A `Symmetric` z is always stored with its upper triangle (`:U`).

`z` may also be a `NullNumber` (in dimension 1, or as a `Diagonal` of `NullNumber`s), which
represents a plane wave `λ exp(ip⋅x)`. Plane waves can be evaluated and multiplied by
other wave packets, but the operations which require an integrable function (`integral`,
`fourier`, ...) throw an `ArgumentError`.

See also [`Gaussian`](@ref), [`HermiteWavePacket1D`](@ref), [`WavePacketSum`](@ref).

# Examples
```julia
G = GaussianWavePacket(1.0 + 0.0im, 2.0 + 0.5im, 0.3, 1.2)
G(0.5)

z = Symmetric(@SMatrix [2.0 0.3; 0.3 1.0])
G2 = GaussianWavePacket(1.0, z, SVector(0.0, 1.0), SVector(0.5, 0.0))
G2(0.1, 0.2)
```
"""
struct GaussianWavePacket{D, Tλ<:Number, Tz<:Number, Cz<:CovMat{D, Tz}, Tq<:RealOrNull, Tp<:RealOrNull} <: AbstractWavePacket{D}
    λ::Tλ
    z::Cz
    q::SVector{D, Tq}
    p::SVector{D, Tp}

    function GaussianWavePacket(λ::Tλ, z::SDiagonal{D, Tz},
                q::SVector{D, Tq}=zeros(SVector{D, NullNumber}),
                p::SVector{D, Tp}=zeros(SVector{D, NullNumber})) where{D, Tλ<:Number, Tz<:Number, Tq<:RealOrNull, Tp<:RealOrNull}
        return new{D, Tλ, Tz, typeof(z), Tq, Tp}(λ, z, q, p)
    end
    function GaussianWavePacket(λ::Tλ, z::Symmetric{Tz, <:SMatrix{D, D, Tz}},
                q::SVector{D, Tq}=zeros(SVector{D, NullNumber}),
                p::SVector{D, Tp}=zeros(SVector{D, NullNumber})) where{D, Tλ<:Number, Tz<:Number, Tq<:RealOrNull, Tp<:RealOrNull}
        # The upper triangle is always the one which is stored (it is the one read by special_cholesky)
        zs = Symmetric(SMatrix{D, D, Tz}(z))
        return new{D, Tλ, Tz, typeof(zs), Tq, Tp}(λ, zs, q, p)
    end
    function GaussianWavePacket{D}(λ::Tλ, z::Symmetric{Tz},
                q::AbstractVector{Tq}=zeros(SVector{D, NullNumber}),
                p::AbstractVector{Tp}=zeros(SVector{D, NullNumber})) where{D, Tλ<:Number, Tz<:Number, Tq<:RealOrNull, Tp<:RealOrNull}
        return GaussianWavePacket(λ, Symmetric(SMatrix{D, D}(z)), SVector{D}(q), SVector{D}(p))
    end
    function GaussianWavePacket{D}(λ::Tλ, z::Diagonal{Tz},
                q::AbstractVector{Tq}=zeros(SVector{D, NullNumber}),
                p::AbstractVector{Tp}=zeros(SVector{D, NullNumber})) where{D, Tλ<:Number, Tz<:Number, Tq<:RealOrNull, Tp<:RealOrNull}
        return GaussianWavePacket(λ, Diagonal(SVector{D}(z.diag)), SVector{D}(q), SVector{D}(p))
    end
    function GaussianWavePacket(λ::Tλ, z::Tz, q::Tq=NullNumber(), p::Tp=NullNumber()) where{Tλ<:Number, Tz<:Number, Tq<:RealOrNull, Tp<:RealOrNull}
        return GaussianWavePacket(λ, Diagonal(SVector(z)), SVector(q), SVector(p))
    end
end

"""
    Gaussian(λ, z, q=0)
    Gaussian{D}(λ, z, q)

Represents the Gaussian function

    G(x) = λ exp(-(x-q)ᵀz(x-q)/2)

with a real `z` and no momentum. It is an alias of a [`GaussianWavePacket`](@ref) whose
`p` is a `NullNumber`, so that all the operations on wave packets apply.
"""
const Gaussian{D, Tλ<:Number, Tz<:RealOrNull, Cz<:CovMat{D, Tz}, Tq<:RealOrNull} =
            GaussianWavePacket{D, Tλ, Tz, Cz, Tq, NullNumber}
function Gaussian(λ::Tλ, z::CovMat{D, Tz},
                q::SVector{D, Tq}=zeros(SVector{D, NullNumber})) where{D, Tλ<:Number, Tz<:RealOrNull, Tq<:RealOrNull}
    return GaussianWavePacket(λ, z, q)
end
function Gaussian{D}(λ::Tλ, z::Symmetric{Tz},
                q::AbstractVector{Tq}=zeros(SVector{D, NullNumber})) where{D, Tλ<:Number, Tz<:RealOrNull, Tq<:RealOrNull}
    return GaussianWavePacket{D}(λ, z, q)
end
function Gaussian{D}(λ::Number, z::Diagonal{Tz},
                q::AbstractVector{Tq}=zeros(SVector{D, NullNumber})) where{D, Tz<:RealOrNull, Tq<:RealOrNull}
    return GaussianWavePacket{D}(λ, z, q)
end
function Gaussian(λ::Number, z::RealOrNull,
                q::RealOrNull = NullNumber())
    return Gaussian(λ, Diagonal(SVector(z)), SVector(q))
end

#=
    CONVERSIONS
=#

# To a diagonal covariance: only possible if z is diagonal
function Base.convert(::Type{GaussianWavePacket{D, Tλ, Tz, Cz, Tq, Tp}}, G::GaussianWavePacket{D}) where {D, Tλ, Tz, Cz<:Diagonal, Tq, Tp}
    isdiag(G.z) || throw(InexactError(:convert, GaussianWavePacket{D, Tλ, Tz, Cz, Tq, Tp}, G))
    return GaussianWavePacket(convert(Tλ, G.λ), Diagonal(convert.(Tz, diag(G.z))), convert.(Tq, G.q), convert.(Tp, G.p))
end
# To a dense covariance (from a dense or a diagonal one)
function Base.convert(::Type{GaussianWavePacket{D, Tλ, Tz, Cz, Tq, Tp}}, G::GaussianWavePacket{D}) where {D, Tλ, Tz, Cz<:Symmetric, Tq, Tp}
    return GaussianWavePacket(convert(Tλ, G.λ), Symmetric(convert.(Tz, SMatrix{D, D}(G.z))), convert.(Tq, G.q), convert.(Tp, G.p))
end

"""
    truncate_to_gaussian(G)

Returns the Gaussian part of a wave packet, as a [`GaussianWavePacket`](@ref): `G` itself
for a `GaussianWavePacket`, and the component on ψ₀ for a [`HermiteWavePacket1D`](@ref).
"""
function truncate_to_gaussian(G::GaussianWavePacket)
    return G
end

#=
    PROMOTIONS
=#

#
function Base.promote_rule(::Type{<:GaussianWavePacket}, ::Type{GaussianWavePacket})
    return GaussianWavePacket
end
# Two diagonal covariances give a diagonal covariance, any other combination a dense one
_promote_cov(::Type{<:Diagonal}, ::Type{<:Diagonal}, ::Val{D}, ::Type{Tz}) where{D, Tz} = SDiagonal{D, Tz}
_promote_cov(::Type, ::Type, ::Val{D}, ::Type{Tz}) where{D, Tz} = Symmetric{Tz, SMatrix{D, D, Tz, D*D}}
function Base.promote_rule(::Type{GaussianWavePacket{D, Tλ1, Tz1, Cz1, Tq1, Tp1}}, ::Type{GaussianWavePacket{D, Tλ2, Tz2, Cz2, Tq2, Tp2}}) where{D, Tλ1, Tz1, Cz1, Tq1, Tp1, Tλ2, Tz2, Cz2, Tq2, Tp2}
    Tz = promote_type(Tz1, Tz2)
    return GaussianWavePacket{D, promote_type(Tλ1, Tλ2), Tz, _promote_cov(Cz1, Cz2, Val(D), Tz), promote_type(Tq1, Tq2), promote_type(Tp1, Tp2)}
end


#=
    DISPLAY
=#

# Prints "x" if there is no shift (q is a NullNumber), "(x - q)" otherwise
_var_str(::NullNumber, var::String) = var
_var_str(q::Real, var::String) = string("(", var, " - ", q, ")")

# 1D case: printed as an explicit formula in x
function Base.show(io::IO, G::GaussianWavePacket{1})
    y = _var_str(first(G.q), "x")
    print(io, "(", G.λ, ") * exp(-(", G.z[1, 1], ")/2*", y, "^2)")
    if !(eltype(G.p) <: NullNumber)
        print(io, " * cis((", first(G.p), ")*x)")
    end
end

# D>1 case: a symbolic formula, with z, q, p spelled out below
# (q and/or p are omitted when they are NullNumber, i.e. not present)
function Base.show(io::IO, G::GaussianWavePacket{D}) where D
    print(io, "GaussianWavePacket{", D, "}: (", G.λ, ") * exp(-(x-q)ᵀz(x-q)/2)")
    eltype(G.p) <: NullNumber || print(io, " * exp(i⋅pᵀx)")
    print(io, "\n  z = ", G.z)
    eltype(G.q) <: NullNumber || print(io, "\n  q = ", G.q)
    eltype(G.p) <: NullNumber || print(io, "\n  p = ", G.p)
end

#=
    HELPERS
=#

# The imaginary part of a covariance matrix (NullNumbers if it is real), with the same structure
_imagz(z::Diagonal) = Diagonal(imagz.(z.diag))
_imagz(z::Symmetric) = z.uplo == 'U' ? Symmetric(imagz.(parent(z)), :U) : Symmetric(imagz.(parent(z)), :L)

# Sum of two covariance matrices (dense as soon as one of them is dense)
# (the methods must be at least as specific as CovMat, otherwise the last one is always selected)
_cov_add(z1::SDiagonal{D}, z2::SDiagonal{D}) where D = Diagonal(z1.diag + z2.diag)
_cov_add(z1::CovMat{D}, z2::CovMat{D}) where D = Symmetric(SMatrix{D, D}(z1) + SMatrix{D, D}(z2))

# yᵀzy and xᵀy (without complex conjugation)
@inline _quad(y::SVector, z) = sum(y .* (z * y))
@inline _bdot(x::SVector, y::SVector) = sum(x .* y)

# Throws an error if G is not integrable (a plane wave, z being a NullNumber)
_check_integrable(::GaussianWavePacket{D, <:Number, NullNumber}) where D =
    throw(ArgumentError("This operation requires an integrable wave packet, but z is a NullNumber (plane wave)"))
_check_integrable(::GaussianWavePacket) = nothing
# Same for conj(G1)G2 (which is integrable as soon as one of them is)
_check_integrable(::GaussianWavePacket{D, <:Number, NullNumber}, ::GaussianWavePacket{D, <:Number, NullNumber}) where D =
    throw(ArgumentError("This operation requires an integrable product, but both z are NullNumbers (plane waves)"))
_check_integrable(::GaussianWavePacket, ::GaussianWavePacket) = nothing

# Checks a dimension selector (dims) for a wave packet of dimension D
function _check_dims(dims, D)
    dims isa Tuple{Vararg{Integer}} || return "dims must be a tuple of integers"
    all(n -> 1 ≤ n ≤ D, dims) || return "dims contains a dimension which does not exist (D = $D)"
    allunique(dims) || return "dims contains duplicate dimensions"
    return nothing
end
# The dimensions which are not in dims, in increasing order
_other_dims(dims, D) = Tuple(n for n in 1:D if n ∉ dims)

#=
    BASIC OPERATIONS
=#

# A non-degenerate covariance (z = identity), used by Base.zero below.
# zero(Cz) is not used instead because it is either undefined (Cz<:Symmetric) or,
# being a degenerate (singular) covariance, breaks fourier/integral with NaNs.
_id_cov(::Type{<:Diagonal{Tz}}, ::Val{D}) where{Tz, D} = Diagonal(ones(SVector{D, Tz}))
_id_cov(::Type{<:Symmetric{Tz}}, ::Val{D}) where{Tz, D} = Symmetric(SMatrix{D, D, Tz}(I))

# Returns a null gaussian
function Base.zero(::Type{GaussianWavePacket{D, Tλ, Tz, Cz, Tq, Tp}}) where{D, Tλ, Tz, Cz, Tq, Tp}
    return GaussianWavePacket(zero(Tλ), _id_cov(Cz, Val(D)), zeros(SVector{D, Tq}), zeros(SVector{D, Tp}))
end

# Creates a copy of a gaussian
function Base.copy(G::GaussianWavePacket)
    return GaussianWavePacket(G.λ, G.z, G.q, G.p)
end

#
function core_type(::Type{GaussianWavePacket{D, Tλ, Tz, Cz, Tq, Tp}}) where{D, Tλ, Tz, Cz, Tq, Tp}
    return promote_type(Tλ, Tz, Tq, Tp)
end

# Returns the complex conjugate of a gaussian
function Base.conj(G::GaussianWavePacket)
    return GaussianWavePacket(conj(G.λ), conj(G.z), G.q, -G.p)
end

#
function Base.:-(G::GaussianWavePacket)
    return GaussianWavePacket(-G.λ, G.z, G.q, G.p)
end

# Computes the product of a scalar and a gaussian
function Base.:*(w::Number, G::GaussianWavePacket)
    return GaussianWavePacket(w * G.λ, G.z, G.q, G.p)
end

# Computes the product of a gaussian by a scalar
function Base.:/(G::GaussianWavePacket, w::Number)
    return GaussianWavePacket(G.λ / w, G.z, G.q, G.p)
end

# Evaluates a gaussian at x (or, if dims is given, only along the dimensions in dims)
@inline Base.@constprop :aggressive function evaluate(G::GaussianWavePacket{D}, x::AbstractVector{<:Number}; dims=nothing) where D
    if dims === nothing
        xs = SVector{D}(x)
        ys = xs - G.q
        return G.λ * exp(-_quad(ys, G.z) / 2) * cis(_bdot(G.p, xs))
    else
        return evaluate(G, x, Val(dims))
    end
end
# Evaluates a 1D gaussian at x
function evaluate(G::GaussianWavePacket{1}, x::Number)
    y = x - first(G.q)
    return G.λ * exp(-G.z[1, 1] * y^2 / 2) * cis(first(G.p) * x)
end
# Evaluates a gaussian at x along the dimensions contained in dims (x[k] being the value of the
#   variable number dims[k]), and returns a gaussian in the other variables (in increasing order)
@generated function evaluate(G::GaussianWavePacket{D, Tλ, Tz, Cz}, x::AbstractVector{<:Number}, ::Val{dims}) where{D, Tλ, Tz, Cz, dims}
    msg = _check_dims(dims, D)
    msg === nothing || return :( throw(ArgumentError($msg)) )

    D2 = length(dims)
    I1 = SVector(_other_dims(dims, D)...)
    I2 = SVector(dims...)

    if D2 == 0
        return :( G )
    elseif D2 == D
        # Reorders x so that x[k] is the value of the variable number k
        J = SVector(invperm(collect(dims))...)
        return :( evaluate(G, SVector{$D}(x)[$J]) )
    elseif Cz <: Diagonal
        code = quote
            x2 = SVector{$D2}(x)
            z1 = Diagonal(diag(G.z)[$I1])
            z2 = Diagonal(diag(G.z)[$I2])
            G2 = GaussianWavePacket(G.λ, z2, G.q[$I2], G.p[$I2])
            return GaussianWavePacket(evaluate(G2, x2), z1, G.q[$I1], G.p[$I1])
        end
        return code
    else
        code = quote
            x2 = SVector{$D2}(x)
            z1 = Symmetric(G.z[$I1, $I1])
            z2 = Symmetric(G.z[$I2, $I2])
            q1 = G.q[$I1]
            q2 = G.q[$I2]
            p2 = G.p[$I2]
            w = G.z[$I1, $I2]
            U = special_cholesky(real(z1))
            y2 = x2 - q2
            y1 = real(w) * y2
            y1_ = U \ (transpose(U) \ y1)
            p0 = imagz.(w) * y2 - _imagz(z1) * y1_
            # All the exponents are combined in a single exp and a single cis
            E = (_quad(y1_, z1) - _quad(y2, z2)) / 2
            λ = G.λ * exp(real(E)) * cis(imagz(E) + _bdot(p2, x2) + _bdot(p0, q1))
            return GaussianWavePacket(λ, z1, q1 - y1_, G.p[$I1] - p0)
        end
        return code
    end
end
# Evaluates a gaussian at x
(G::GaussianWavePacket{D})(x::AbstractVector{<:Number}) where D = evaluate(G, x)
(G::GaussianWavePacket{1})(x::Number) = evaluate(G, x)

#=
    TRANSFORMATIONS
=#

# Multiplies a gaussian wave packet by exp(-ib/2(x-q)⋅(x-q)) * exp(ipx)
function unitary_product(G::GaussianWavePacket{D}, b::CovMat{D, <:RealOrNull},
            q::SVector{D, <:RealOrNull} = zeros(SVector{D, NullNumber}),
            p::SVector{D, <:RealOrNull} = zeros(SVector{D, NullNumber})) where D
    z_ = _cov_add(G.z, im * b)
    q_ = G.q
    p_ = G.p + p + b * (q - G.q)
    λ_ = G.λ * cis(dot(G.q - q, b, G.q + q) / 2)
    return GaussianWavePacket(λ_, z_, q_, p_)
end

# Computes the product of two gaussians
function Base.:*(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D
    z = _cov_add(G1.z, G2.z)
    U = special_cholesky(real(z))
    q = U \ (transpose(U) \ (real(G1.z) * G1.q + real(G2.z) * G2.q))
    y1 = q - G1.q
    y2 = q - G2.q
    p0 = -(_imagz(G1.z) * y1 + _imagz(G2.z) * y2)
    p = G1.p + G2.p + p0
    # λ = G1(q) * G2(q) * cis(-p⋅q), with all the exponents combined in a single exp and a single cis
    E = (_quad(y1, G1.z) + _quad(y2, G2.z)) / 2
    λ = G1.λ * G2.λ * exp(-real(E)) * cis(-imagz(E) - _bdot(p0, q))
    return GaussianWavePacket(λ, z, q, p)
end
# Product of two plane waves
function Base.:*(G1::GaussianWavePacket{D, <:Number, NullNumber}, G2::GaussianWavePacket{D, <:Number, NullNumber}) where D
    return GaussianWavePacket(G1.λ * G2.λ, G1.z, zeros(SVector{D, NullNumber}), G1.p + G2.p)
end

# Computes |G(x)| and |G(x)|^2
Base.abs(G::GaussianWavePacket) = Gaussian(abs(G.λ), real(G.z), G.q)
Base.abs2(G::GaussianWavePacket) = Gaussian(abs2(G.λ), 2*real(G.z), G.q)

# Computes the integral of a gaussian (or, if dims is given, only over the dimensions in dims)
@inline Base.@constprop :aggressive function integral(G::GaussianWavePacket{D}; dims=nothing) where D
    if dims === nothing
        _check_integrable(G)
        U = special_cholesky(G.z)
        r = transpose(U) \ G.p
        return G.λ / prod(invsqrt2π * diag(U)) * cis(_bdot(G.p, G.q)) * exp(-sum(r.^2) / 2)
    else
        return integral(G, Val(dims))
    end
end
#=
    Integrates over all the variables with indices contained in dims, and returns a gaussian in
    the other variables (in increasing order).
    With y = x - q split into (y₁, y₂) (y₂ being integrated), and z = [A B; Bᵀ C], one has
        ∫dy₂ exp(-yᵀzy/2 + ip₂⋅y₂) = c exp(-y₁ᵀSy₁/2 + wᵀy₁)
    where S = A - BC⁻¹Bᵀ is the Schur complement, w = -iBC⁻¹p₂, and c is the integral of the
    gaussian in y₂ alone. The complex linear term wᵀy₁ is then absorbed in a shift of q and p.
=#
@generated function integral(G::GaussianWavePacket{D, Tλ, Tz, Cz}, ::Val{dims}) where{D, Tλ, Tz, Cz, dims}
    msg = _check_dims(dims, D)
    msg === nothing || return :( throw(ArgumentError($msg)) )

    D2 = length(dims)
    I1 = SVector(_other_dims(dims, D)...)
    I2 = SVector(dims...)

    if D2 == 0
        return :( G )
    elseif D2 == D
        return :( integral(G) )
    elseif Cz <: Diagonal
        code = quote
            _check_integrable(G)
            zd = diag(G.z)
            c = integral(GaussianWavePacket(G.λ, Diagonal(zd[$I2]), G.q[$I2], G.p[$I2]))
            return GaussianWavePacket(c, Diagonal(zd[$I1]), G.q[$I1], G.p[$I1])
        end
        return code
    else
        code = quote
            _check_integrable(G)
            A = G.z[$I1, $I1]
            B = G.z[$I1, $I2]
            C = Symmetric(G.z[$I2, $I2])
            q1 = G.q[$I1]
            p1 = G.p[$I1]
            p2 = G.p[$I2]
            c = integral(GaussianWavePacket(G.λ, C, G.q[$I2], p2))
            UC = special_cholesky(C)
            V = transpose(UC) \ transpose(B)
            S = Symmetric(A - transpose(V) * V)
            w = -im * (transpose(V) * (transpose(UC) \ p2))
            # Absorbs the linear term: Re(S)δ = Re(w)
            US = special_cholesky(real(S))
            δ = US \ (transpose(US) \ real(w))
            q_ = q1 + δ
            p_ = imagz.(w) - _imagz(S) * δ
            E = _bdot(w, δ) - _quad(δ, S) / 2
            λ = c * exp(real(E)) * cis(imagz(E) - _bdot(p_, q_))
            return GaussianWavePacket(λ, S, q_, p1 + p_)
        end
        return code
    end
end

#=
    Computes the Fourier transform of a gaussian
    The Fourier transform is defined as
        TF(ψ)(ξ) = ∫dx e^(-ixξ) ψ(x)
=#
function fourier(G::GaussianWavePacket{D}) where D
    _check_integrable(G)
    U = special_cholesky(G.z)
    U_inv = _upper_tri(U \ Diagonal(ones(SVector{D, Bool})))
    z_tf = _symmetric(U_inv * transpose(U_inv))
    q_tf = G.p
    p_tf = -G.q
    λ_tf = G.λ / prod(invsqrt2π * diag(U)) * cis(_bdot(G.p, G.q))
    return GaussianWavePacket(λ_tf, z_tf, q_tf, p_tf)
end

#=
    Computes the inverse Fourier transform of a gaussian
    The inverse Fourier transform is defined as
        ITF(ψ)(x) = (2π)⁻ᴰ∫dξ e^(ixξ) ψ(ξ)
=#
function inv_fourier(G::GaussianWavePacket{D}) where D
    _check_integrable(G)
    U = special_cholesky(G.z)
    U_inv = _upper_tri(U \ Diagonal(ones(SVector{D, Bool})))
    z_tf = _symmetric(U_inv * transpose(U_inv))
    q_tf = -G.p
    p_tf = G.q
    λ_tf = G.λ / prod(sqrt2π * diag(U)) * cis(_bdot(G.p, G.q))
    return GaussianWavePacket(λ_tf, z_tf, q_tf, p_tf)
end

# Computes the convolution product of two gaussians
function convolution(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D
    return inv_fourier(fourier(G1) * fourier(G2))
end

# The inverse of a covariance matrix Z (as a plain matrix), and the diagonal of its (analytic)
# Cholesky factor (whose product is the branch of sqrt(det(Z)) with a positive real part)
function _inv_diagchol(Z::Diagonal)
    U = special_cholesky(Z)
    return Diagonal(inv.(Z.diag)), U.diag
end
function _inv_diagchol(Z::Symmetric{<:Number, <:SMatrix{D, D}}) where D
    U = special_cholesky(Z)
    U_inv = U \ Diagonal(ones(SVector{D, Bool}))
    return SMatrix(U_inv * transpose(U_inv)), diag(U)
end

#=
    Computes the integral I₀ of
        conj(G1)G2 = λ̄₁λ₂ exp(-uᵀZu/2 + kᵀu + c₀)  (u = x - q₂, Z = z̄₁ + z₂)
    where k = i(p₂ - p₁) - z̄₁(q₂ - q₁) and c₀ = -(q₂ - q₁)ᵀz̄₁(q₂ - q₁)/2 + i(p₂ - p₁)⋅q₂, that is
        I₀ = λ̄₁λ₂ (2π)^(D/2) det(Z)^(-1/2) exp(kᵀZ⁻¹k/2 + c₀)
    Also returns the Cholesky factor U of Z (UᵀU = Z), r = U⁻ᵀk and q₂ - q₁
=#
function _overlap(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D
    _check_integrable(G1, G2)
    z1c = conj(G1.z)
    U = special_cholesky(_cov_add(z1c, G2.z))
    d = G2.q - G1.q
    Δp = G2.p - G1.p
    k = im * Δp - z1c * d
    r = transpose(U) \ k
    E = (sum(r.^2) - _quad(d, z1c)) / 2 + im * _bdot(Δp, G2.q)
    I0 = conj(G1.λ) * G2.λ / prod(invsqrt2π * diag(U)) * exp(E)
    return I0, U, r, d
end

# Computes the L² product of two gaussian wave packets
dot_L2(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D = first(_overlap(G1, G2))
# Computes the square L² norm of a gaussian wave packet
norm2_L2(G::GaussianWavePacket) = integral(abs2(G))

#=
    Computes ∫<∇G1,∇G2> (where <,> is the hermitian product in ℂᴰ)
    Since ∇Gₖ = (-zₖ(x-qₖ) + ipₖ)Gₖ, the integrand is a quadratic polynomial times conj(G1)G2,
    whose integral I₀, mean m = Z⁻¹k and covariance Z⁻¹ are known in closed form (see _overlap):
        ∫<∇G1,∇G2> = I₀ (tr(z̄₁z₂Z⁻¹) + A(m)ᵀB(m))
    where A(u) = -z̄₁(u + q₂ - q₁) - ip₁ and B(u) = -z₂u + ip₂.
=#
function dot_∇(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D
    _check_integrable(G1, G2)
    z1c = conj(G1.z)
    # Same computation of I₀ as in _overlap, but Z⁻¹ is computed explicitly (which is faster than
    # triangular solves here, since it is needed for the trace anyway)
    Zinv, dU = _inv_diagchol(_cov_add(z1c, G2.z))
    d = G2.q - G1.q
    Δp = G2.p - G1.p
    k = im * Δp - z1c * d
    m = Zinv * k
    E = (_bdot(k, m) - _quad(d, z1c)) / 2 + im * _bdot(Δp, G2.q)
    I0 = conj(G1.λ) * G2.λ / prod(invsqrt2π * dU) * exp(E)
    Am = -(z1c * (m + d)) - im * G1.p
    Bm = -(G2.z * m) + im * G2.p
    return I0 * (tr(Zinv * (z1c * G2.z)) + _bdot(Am, Bm))
end
# Computes ∫<(D-c)G1,(D-c)G2> (where <,> is the hermitian product in ℂᴰ, and D=-i∇)
function dot_∇(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}, c::AbstractVector{<:RealOrNull}) where D
    cs = SVector{D}(c)
    return dot_∇(GaussianWavePacket(G1.λ, G1.z, G1.q, G1.p - cs),
                 GaussianWavePacket(G2.λ, G2.z, G2.q, G2.p - cs))
end

# tr(b a⁻¹ b) where a = RᵀR (b real symmetric, possibly made of NullNumbers)
_tr_bab(R::Diagonal, ::Diagonal{NullNumber}) = false
_tr_bab(R, ::Symmetric{NullNumber}) = false
_tr_bab(R::Diagonal, b::Diagonal) = sum((b.diag ./ R.diag).^2)
_tr_bab(R, b::Symmetric{<:Real, <:SMatrix{D, D}}) where D = sum(abs2, transpose(R) \ SMatrix{D, D}(b))

#=
    Computes ∫|∇G|^2
    With z = a + ib, this is the case G1 = G2 of dot_∇, where Z = 2a is real and m = 0:
        ∫|∇G|^2 = ‖G‖² (tr(a)/2 + tr(ba⁻¹b)/2 + |p|²)
=#
function norm2_∇(G::GaussianWavePacket{D}) where D
    _check_integrable(G)
    a = real(G.z)
    R = special_cholesky(a)
    nrm2 = abs2(G.λ) / prod(invsqrtπ * diag(R))
    return nrm2 * ((tr(a) + _tr_bab(R, _imagz(G.z))) / 2 + sum(abs2, G.p))
end
# Computes ∫|(D-c)G|^2
function norm2_∇(G::GaussianWavePacket{D}, c::AbstractVector{<:RealOrNull}) where D
    cs = SVector{D}(c)
    return norm2_∇(GaussianWavePacket(G.λ, G.z, G.q, G.p - cs))
end

#=
    Same as _overlap, but returns I₀, Z⁻¹ and m = Z⁻¹k, so that conj(G1)G2 is I₀ times a normalized
    gaussian in u = x - q₂, of mean m and covariance Z⁻¹ (Z⁻¹ is computed explicitly as in dot_∇)
=#
function _overlap_inv(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D
    _check_integrable(G1, G2)
    z1c = conj(G1.z)
    Zinv, dU = _inv_diagchol(_cov_add(z1c, G2.z))
    d = G2.q - G1.q
    Δp = G2.p - G1.p
    k = im * Δp - z1c * d
    m = Zinv * k
    E = (_bdot(k, m) - _quad(d, z1c)) / 2 + im * _bdot(Δp, G2.q)
    I0 = conj(G1.λ) * G2.λ / prod(invsqrt2π * dU) * exp(E)
    return I0, Zinv, m
end

#=
    Computes ∫<(x-c)G1,(x-c)G2> = ∫|x-c|² conj(G1)G2
    With the notations of _overlap_inv,
        ∫|x-c|² conj(G1)G2 = I₀ (tr(Z⁻¹) + yᵀy)
    where y = m + q₂ - c.
=#
function dot_x(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D},
            c::AbstractVector{<:RealOrNull}=zeros(SVector{D, NullNumber})) where D
    I0, Zinv, m = _overlap_inv(G1, G2)
    y = m + (G2.q - SVector{D}(c))
    return I0 * (tr(Zinv) + _bdot(y, y))
end

#=
    Computes ∫|x-c|²|G|²
    With a = Re(z) = RᵀR, this is the case G1 = G2 of dot_x, where Z = 2a and m = 0:
        ∫|x-c|²|G|² = ‖G‖² (tr(a⁻¹)/2 + |q-c|²)
=#
function norm2_x(G::GaussianWavePacket{D}, c::AbstractVector{<:RealOrNull}=zeros(SVector{D, NullNumber})) where D
    _check_integrable(G)
    R = special_cholesky(real(G.z))
    nrm2 = abs2(G.λ) / prod(invsqrtπ * diag(R))
    R_inv = R \ Diagonal(ones(SVector{D, Bool}))
    return nrm2 * (sum(abs2, R_inv) / 2 + sum(abs2, G.q - SVector{D}(c)))
end

# The indices of the dimensions in dims (all the dimensions if dims is nothing), as an SVector
@generated function _dims_index(::Val{D}, ::Val{dims}) where{D, dims}
    dims === nothing && return :( SVector{$D, Int}($(1:D...)) )
    msg = _check_dims(dims, D)
    msg === nothing || return :( throw(ArgumentError($msg)) )
    return :( SVector{$(length(dims)), Int}($(dims...)) )
end

#=
    Computes the vector (∫xⱼ conj(G1)G2)ⱼ for j in dims
    With the notations of _overlap, conj(G1)G2 is I₀ times a normalized gaussian in u = x - q₂,
    of mean m = Z⁻¹k = U⁻¹r, so that ∫x conj(G1)G2 = I₀(m + q₂)
=#
@inline Base.@constprop :aggressive function moment_x(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}; dims=nothing) where D
    return moment_x(G1, G2, Val(dims))
end
function moment_x(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}, ::Val{dims}) where{D, dims}
    I = _dims_index(Val(D), Val(dims))
    I0, U, r, _ = _overlap(G1, G2)
    m = U \ r
    return I0 * (m + G2.q)[I]
end
# Computes the vector (∫xⱼ|G|²)ⱼ = ‖G‖² q[dims]
@inline Base.@constprop :aggressive function moment_x(G::GaussianWavePacket; dims=nothing)
    return moment_x(G, Val(dims))
end
function moment_x(G::GaussianWavePacket{D}, ::Val{dims}) where{D, dims}
    I = _dims_index(Val(D), Val(dims))
    return norm2_L2(G) * G.q[I]
end

#=
    Computes the vector (∫conj(DⱼG1)G2)ⱼ for j in dims (where D = -i∇)
    Since D is self-adjoint, this is ∫conj(G1)DG2, and DG2 = (p₂ + iz₂u)G2 (u = x - q₂), so that
    with the notations of moment_x, ∫conj(G1)DG2 = I₀(p₂ + iz₂m)
=#
@inline Base.@constprop :aggressive function moment_D(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}; dims=nothing) where D
    return moment_D(G1, G2, Val(dims))
end
function moment_D(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}, ::Val{dims}) where{D, dims}
    I = _dims_index(Val(D), Val(dims))
    I0, U, r, _ = _overlap(G1, G2)
    m = U \ r
    return I0 * (G2.p + im * (G2.z * m))[I]
end
# Computes the vector (∫conj(DⱼG)G)ⱼ = ‖G‖² p[dims]
@inline Base.@constprop :aggressive function moment_D(G::GaussianWavePacket; dims=nothing)
    return moment_D(G, Val(dims))
end
function moment_D(G::GaussianWavePacket{D}, ::Val{dims}) where{D, dims}
    I = _dims_index(Val(D), Val(dims))
    return norm2_L2(G) * G.p[I]
end

#=
    Computes the matrix (∫xᵢxⱼ conj(G1)G2)ᵢⱼ
    With the notations of _overlap_inv, and y = m + q₂ the mean of x, this is I₀(Z⁻¹ + yyᵀ)
=#
function moment_xx(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D
    I0, Zinv, m = _overlap_inv(G1, G2)
    y = m + G2.q
    return I0 * (SMatrix{D, D}(Zinv) + y * transpose(y))
end
#=
    Computes the matrix (∫xᵢxⱼ|G|²)ᵢⱼ
    With a = Re(z) = RᵀR, this is the case G1 = G2 of moment_xx, where Z = 2a and m = 0:
        ∫xxᵀ|G|² = ‖G‖² (a⁻¹/2 + qqᵀ)
=#
function moment_xx(G::GaussianWavePacket{D}) where D
    _check_integrable(G)
    R = special_cholesky(real(G.z))
    nrm2 = abs2(G.λ) / prod(invsqrtπ * diag(R))
    R_inv = SMatrix{D, D}(R \ Diagonal(ones(SVector{D, Bool})))
    return nrm2 * (R_inv * transpose(R_inv) / 2 + G.q * transpose(G.q))
end

#=
    Computes the matrix (∫conj(G1) Aᵢⱼ G2)ᵢⱼ where Aᵢⱼ = (xᵢDⱼ + Dⱼxᵢ)/2 = xᵢDⱼ - iδᵢⱼ/2
    (the symmetrized product, which is self-adjoint)
    Since DG2 = (p₂ + iz₂u)G2 (u = x - q₂), with the notations of _overlap_inv and y = m + q₂,
        ∫conj(G1) xDᵀ G2 = I₀(y(p₂ + iz₂m)ᵀ + iZ⁻¹z₂)
=#
function moment_xD(G1::GaussianWavePacket{D}, G2::GaussianWavePacket{D}) where D
    I0, Zinv, m = _overlap_inv(G1, G2)
    y = m + G2.q
    return I0 * (y * transpose(G2.p + im * (G2.z * m)) + im * (SMatrix{D, D}(Zinv * G2.z) - I / 2))
end
# Computes the matrix (∫conj(G) Aᵢⱼ G)ᵢⱼ, which is real (= ‖G‖²(qpᵀ - a⁻¹b/2) where z = a + ib)
moment_xD(G::GaussianWavePacket) = real(moment_xD(G, G))

"""
    gaussian_hull(G)

Returns a [`GaussianWavePacket`](@ref) which approximates the wave packet `G`: `G` itself for a
`GaussianWavePacket`. See the method for [`WavePacketSum`](@ref).
"""
gaussian_hull(G::GaussianWavePacket) = G

# The scalar z₀ such that z = z₀I (throws an error if z is not isotropic)
function _isotropic_scalar(z::CovMat{D}) where D
    z0 = z[1, 1]
    for j in 1:D, i in 1:D
        if z[i, j] != (i == j ? z0 : zero(z0))
            throw(ArgumentError("Coulomb integral can only be computed for isotropic wave packets (z must be a multiple of the identity)"))
        end
    end
    return z0
end

# Computes ∫dx 1/|x| G(x) where G is an isotropic 3d gaussian wave packet
function coulomb_integral(G::GaussianWavePacket{3})
    _check_integrable(G)
    z0 = _isotropic_scalar(G.z)
    r = sqrt(sum(x -> x^2, G.q + im * G.p / z0))
    if r == 0
        return fourπ * G.λ * cis(_bdot(G.q, G.p)) * exp(-sum(abs2, G.p) / (2*z0)) / z0
    else
        s = twoπ / z0
        return G.λ * (s * sqrt(s)) * cis(_bdot(G.q, G.p)) * exp(-sum(abs2, G.p) / (2*z0)) * erf(sqrt(z0 / 2) * r) / r
    end
end
