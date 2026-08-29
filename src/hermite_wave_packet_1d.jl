
#=
    Reminder :
    -gausshermite(p) computes the weights (wⱼ)ⱼ and nodes (xⱼ)ⱼ (j=1,...,p) such that the quadrature formula
        ∫dx f(x)exp(-x²) ≈ ∑ⱼ wⱼf(xⱼ)
     is exact for any polynomial f up to degree 2m-1

    We have the folowing properties (cf https://mathworld.wolfram.com/HermitePolynomial.html)
        ∫dx Hₘ(x)Hₙ(x)exp(-x²) = δₘₙ2ⁿn!√π
        Hₙ₊₁(x) = 2xHₙ(x) - 2nHₙ₋₁(x)
        Hₙ'(x) = 2n Hₙ₋₁(x)

    Now define for z = a + ib, a > 0, and p, q real numbers
        ψₙ(z, q, p, x) = (a/π)^(1/4) / sqrt(2ⁿn!) * Hₙ(√a*(x-q)) * exp(-z(x - q)²/2) * exp(ipx)
    (we shall only write ψₙ(x) when the context is clear enough)
    then it follows from the previous properties that
        ∫̄ψₘψₙ = δₘₙ
        ψₙ₊₁ = (√(2a)*(x-q)*ψₙ - √n*ψₙ₋₁) / √(n+1) (where a = real(z))
        ̂ψₙ = √(2π) * (-i)ⁿ * exp(ipq) * ψₙ(1/a, p, -q, ξ)
=#

#=
    Represents the complex hermite wave packet
        ∑ₙ Λ[n+1] ψₙ(z, q, p, x) (n=0,...,N-1)
=#
struct HermiteWavePacket1D{N, TΛ<:Number, Tz<:Number, Tq<:RealOrNull, Tp<:RealOrNull} <: AbstractWavePacket{1}
    Λ::SVector{N, TΛ}
    z::Tz
    q::Tq
    p::Tp

    function HermiteWavePacket1D(Λ::SVector{N, TΛ}, z::Tz, q::Tq=NullNumber(), p::Tp=NullNumber()) where{N, TΛ<:Number, Tz<:Number, Tq<:RealOrNull, Tp<:RealOrNull}
        return new{N, TΛ, Tz, Tq, Tp}(Λ, z, q, p)
    end
    function HermiteWavePacket1D(λ::TΛ, z::Tz, q::Tq=NullNumber(), p::Tp=NullNumber()) where{TΛ<:Number, Tz<:Number, Tq<:RealOrNull, Tp<:RealOrNull}
        return HermiteWavePacket1D(SVector(λ), z, q, p)
    end
end

#=
    Represents the real hermite function
        ∑ₙ Λ[n+1] ψₙ(a, q, x) (n=0,...,N-1)
=#
const HermiteFct1D{N, TΛ<:Number, Tz<:Real, Tq<:RealOrNull} = HermiteWavePacket1D{N, TΛ, Tz, Tq, NullNumber}
function HermiteFct1D(Λ::SVector{N, <:Number}, z::Real, q::RealOrNull=NullNumber()) where N
    return HermiteWavePacket1D(Λ, z, q, NullNumber())
end
function HermiteFct1D(λ::Number, z::Real, q::RealOrNull=NullNumber())
    return HermiteFct1D(SVector(λ), z, q)
end
function HermiteFct1D(G::Gaussian{1})
    return HermiteWavePacket1D(G)
end

#=
    CONVERSIONS
=#

function Base.convert(::Type{HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}}, H::HermiteWavePacket1D{N2}) where{N, TΛ, Tz, Tq, Tp, N2}
    if N < N2
        throw(ArgumentError("Cannot convert HermiteWavePacket1D{$N2} to HermiteWavePacket1D{$N}: N=$N is smaller than N2=$N2"))
    end
    Λ = SVector{N}(n <= N2 ? convert(TΛ, H.Λ[n]) : zero(TΛ) for n in 1:N)
    return HermiteWavePacket1D(Λ, convert(Tz, H.z), convert(Tq, H.q), convert(Tp, H.p))
end

function truncate_to_gaussian(H::HermiteWavePacket1D)
    a = real(H.z)
    return GaussianWavePacket((invπ*a)^Rational(1, 4) * first(H.Λ), H.z, H.q, H.p)
end

#=
    PROMOTIONS
=#

function Base.promote_rule(::Type{<:HermiteWavePacket1D}, ::Type{HermiteWavePacket1D})
    return HermiteWavePacket1D
end
function Base.promote_rule(::Type{HermiteWavePacket1D{N1, TΛ1, Tz1, Tq1, Tp1}}, ::Type{HermiteWavePacket1D{N2, TΛ2, Tz2, Tq2, Tp2}}) where{N1, TΛ1, Tz1, Tq1, Tp1, N2, TΛ2, Tz2, Tq2, Tp2}
    return HermiteWavePacket1D{max(N1, N2), promote_type(TΛ1, TΛ2), promote_type(Tz1, Tz2), promote_type(Tq1, Tq2), promote_type(Tp1, Tp2)}
end

#=
    DISPLAY
=#

# Represents ∑ₙ Λ[n]⋅ψₙ(z,q,p,x), with Λ, z, q, p spelled out below
# (q and/or p are omitted when they are NullNumber, i.e. not present)
function Base.show(io::IO, H::HermiteWavePacket1D{N}) where N
    print(io, "HermiteWavePacket1D{", N, "}: ∑ₙ Λ[n]⋅ψₙ(z,q,p,x)")
    print(io, "\n  Λ = ", H.Λ)
    print(io, "\n  z = ", H.z)
    H.q isa NullNumber || print(io, "\n  q = ", H.q)
    H.p isa NullNumber || print(io, "\n  p = ", H.p)
end

#=
    BASIC OPERATIONS
=#

# Returns a null hermite wave packet
function Base.zero(::Type{HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}}) where{N, TΛ, Tz, Tq, Tp}
    return HermiteWavePacket1D(zeros(SVector{N, TΛ}), one(Tz), zero(Tq), zero(Tp))
end

# Creates a copy of a hermite wave packet
function Base.copy(H::HermiteWavePacket1D)
    return HermiteWavePacket1D(H.Λ, H.z, H.q, H.p)
end

#
function core_type(::Type{HermiteWavePacket1D{N, TΛ, Tz, Tq, Tp}}) where{N, TΛ, Tz, Tq, Tp}
    return promote_type(TΛ, Tz, Tq, Tp)
end

# Returns the complex conjugate of a hermite wave packet
function Base.conj(H::HermiteWavePacket1D)
    return HermiteWavePacket1D(conj(H.Λ), conj(H.z), H.q, -H.p)
end

#
function Base.:-(H::HermiteWavePacket1D)
    return HermiteWavePacket1D(-H.Λ, H.z, H.q, H.p)
end

# Computes the product of a scalar and a hermite wave packet
function Base.:*(w::Number, H::HermiteWavePacket1D)
    return HermiteWavePacket1D(w * H.Λ, H.z, H.q, H.p)
end

# Computes the division of a hermite wave packet by a scalar
function Base.:/(H::HermiteWavePacket1D, w::Number)
    return HermiteWavePacket1D(H.Λ / w, H.z, H.q, H.p)
end

# Evaluates a hermite wave packet at x
function evaluate(H::HermiteWavePacket1D, x::Number)
    a = real(H.z)
    y = x - H.q
    val = clenshaw_hermite_eval(H.Λ, sqrt(2a), y, exp(-a/2*y^2))
    return (invπ*a)^Rational(1, 4) * val * cis(-imagz(H.z)/2*y^2) * cis(x*H.p)
end
# Evaluates a hermite wave packet at all the points in x
function evaluate_grid(H::HermiteWavePacket1D, x::SVector{M, <:Number}) where M
    a = real(H.z)
    y = x .- H.q
    val = clenshaw_hermite_eval(H.Λ, sqrt(2a), y, @. exp(-a/2*y^2))
    return @. (invπ*a)^Rational(1, 4) * val * cis(-imagz(H.z)/2*y^2) * cis(x*H.p)
end
# Evaluates a hermite wave packet at x
(H::HermiteWavePacket1D)(x::Number) = evaluate(H, x)

#=
    TRANSFORMATIONS
=#

# Multiplies a hermite wave packet by exp(-ib/2*(x-q)^2) * exp(ipx)
function unitary_product(H::HermiteWavePacket1D, b::RealOrNull, q::RealOrNull=NullNumber(), p::RealOrNull=NullNumber())
    α = cis(b/2 * (H.q + q) * (H.q - q))
    return HermiteWavePacket1D(α .* H.Λ, H.z + im*b, H.q, H.p + p - b*(H.q - q))
end

# Computes the product of two hermite wave packets (real variance)
function Base.:*(H1::HermiteWavePacket1D{N1, TΛ1, Tz1}, H2::HermiteWavePacket1D{N2, TΛ2, Tz2}) where{N1, TΛ1, Tz1<:Real, N2, TΛ2, Tz2<:Real}
    N = max(N1 + N2 - 1, 0)
    z, q, p = first.(gaussian_product_arg(SVector(H1.z), SVector(H1.q), SVector(H1.p), SVector(H2.z), SVector(H2.q), SVector(H2.p)))
    x = hermite_grid(z, q, Val(N))
    Φ1 = evaluate_grid(H1, x)
    Φ2 = evaluate_grid(H2, x)
    Λ = hermite_discrete_transform(Φ1 .* Φ2, z, q, Val(N))
    return HermiteWavePacket1D(Λ, z, q, p)
end
# Computes the product of two hermite wave packets (general complex variance)
function Base.:*(H1::HermiteWavePacket1D{N1, TΛ1, Tz1}, H2::HermiteWavePacket1D{N2, TΛ2, Tz2}) where{N1, TΛ1, Tz1, N2, TΛ2, Tz2}
    z, q, p = first.(gaussian_product_arg(SVector(H1.z), SVector(H1.q), SVector(H1.p), SVector(H2.z), SVector(H2.q), SVector(H2.p)))
    Hr = HermiteFct1D(H1.Λ, real(H1.z), H1.q) * HermiteFct1D(H2.Λ, real(H2.z), H2.q)
    μ = cis(-(imagz(H1.z)*(q - H1.q)^2 + imagz(H2.z)*(q - H2.q)^2)/2 + (H1.p + H2.p - p)*q)
    return HermiteWavePacket1D(μ .* Hr.Λ, z, q, p)
end

#=
    Computes the product of a hermite wave packet with a polynomial
        P(x) = ∑ₖ P[k](x-q)^(k-1)
=#
function polynomial_product(H::HermiteFct1D{N}, P::SVector{NP, <:Number}, q::RealOrNull=NullNumber()) where{N, NP}
    N1 = max(N + NP - 1, 0)
    x = hermite_grid(H.z, H.q, Val(N1))
    Φ = evaluate_grid(H, x)
    Φ_P = horner_eval(P, x .- q)
    Λ = hermite_discrete_transform(Φ .* Φ_P, H.z, H.q, Val(N1))
    return HermiteFct1D(Λ, H.z, H.q)
end
function polynomial_product(H::HermiteWavePacket1D, P::SVector{NP, <:Number}, q::RealOrNull=NullNumber()) where NP
    Hr = polynomial_product(HermiteFct1D(H.Λ, real(H.z), H.q), P, q)
    return HermiteWavePacket1D(Hr.Λ, H.z, H.q, H.p)
end

# Computes the integral of a hermite wave packet
function integral(H::HermiteWavePacket1D)
    return fourier(H)(NullNumber())
end

#=
    Computes the Fourier transform of a hermite wave packet
    The Fourier transform is defined as
        TF(ψ)(ξ) = ∫dx e^(-ixξ) ψ(x)
=#
# Real variance
function fourier(H::HermiteWavePacket1D{N, TΛ, Tz}) where{N, TΛ, Tz<:Real}
    zf, qf, pf = first.(gaussian_fourier_arg(SVector(H.z), SVector(H.q), SVector(H.p)))
    M = SVector{N}(Complex{Int8}((-im)^n) for n in 0:N-1)
    Λf = (sqrt2π * cis(H.p*H.q)) .* M .* H.Λ
    return HermiteWavePacket1D(Λf, zf, qf, pf)
end
# Complex variance
function fourier(H::HermiteWavePacket1D{N, TΛ, Tz}) where{N, TΛ, Tz}
    zf, qf, pf = first.(gaussian_fourier_arg(SVector(H.z), SVector(H.q), SVector(H.p)))
    μ = abs2(H.z)^Rational(1, 4) / (invsqrt2π * sqrt(H.z)) * cis(H.p*H.q)
    α = -im * conj(H.z) / abs(H.z)
    Λf = μ .* SVector{N}(α^n * H.Λ[n+1] for n in 0:N-1)
    return HermiteWavePacket1D(Λf, zf, qf, pf)
end

#=
    Computes the inverse Fourier transform of a hermite wave packet
    The inverse Fourier transform is defined as
        ITF(ψ)(x) = (2π)⁻¹ ∫dξ e^(ixξ) ψ(ξ)
=#
# Real variance
function inv_fourier(H::HermiteWavePacket1D{N, TΛ, Tz}) where{N, TΛ, Tz<:Real}
    zf, qf, pf = first.(gaussian_inv_fourier_arg(SVector(H.z), SVector(H.q), SVector(H.p)))
    M = SVector{N}(Complex{Int8}(im^n) for n in 0:N-1)
    Λf = (invsqrt2π * cis(H.p*H.q)) .* M .* H.Λ
    return HermiteWavePacket1D(Λf, zf, qf, pf)
end
# Complex variance
function inv_fourier(H::HermiteWavePacket1D{N, TΛ, Tz}) where{N, TΛ, Tz}
    zf, qf, pf = first.(gaussian_inv_fourier_arg(SVector(H.z), SVector(H.q), SVector(H.p)))
    μ = abs2(H.z)^Rational(1, 4) / (sqrt2π * sqrt(H.z)) * cis(H.p*H.q)
    α = im * conj(H.z) / abs(H.z)
    Λf = μ .* SVector{N}(α^n * H.Λ[n+1] for n in 0:N-1)
    return HermiteWavePacket1D(Λf, zf, qf, pf)
end

# Computes the convolution product of two hermite wave packets
function convolution(H1::HermiteWavePacket1D, H2::HermiteWavePacket1D)
    return inv_fourier(fourier(H1) * fourier(H2))
end

#=
    Computes the L² product of two hermite wave packets
        ∫dx conj(H1(x)) H2(x)
=#
function dot_L2(H1::HermiteFct1D{N1}, H2::HermiteFct1D{N2}) where{N1, N2}
    N = cld(N1 + N2 - 1, 2)
    z, q, _ = first.(gaussian_product_arg(SVector(H1.z), SVector(H1.q), SVector(H1.p), SVector(H2.z), SVector(H2.q), SVector(H2.p)))
    x, w = hermite_quadrature(z/2, q, Val(N))
    Φ1 = evaluate_grid(H1, x)
    Φ2 = evaluate_grid(H2, x)
    return dot(w, conj.(Φ1) .* Φ2)
end
function dot_L2(H1::HermiteWavePacket1D, H2::HermiteWavePacket1D)
    return integral(conj(H1) * H2)
end

# Computes the square L² norm of a hermite wave packet
function norm2_L2(H::HermiteWavePacket1D)
    return sum(abs2, H.Λ; init=zero(real(eltype(H.Λ))))
end

#=

    Interface with GaussianWavePacket

=#

function HermiteWavePacket1D(G::GaussianWavePacket{1})
    λ = G.λ * (invπ*real(G.z[1, 1]))^Rational(-1, 4)
    return HermiteWavePacket1D(SVector(λ), G.z[1, 1], G.q[1], G.p[1])
end

function Base.promote_rule(::Type{<:GaussianWavePacket{1, Tλ1, Tz1, Cz1, Tq1, Tp1}},
                      ::Type{HermiteWavePacket1D{N, TΛ2, Tz2, Tq2, Tp2}}) where{Tλ1, Tz1, Cz1, Tq1, Tp1, N, TΛ2, Tz2, Tq2, Tp2}
    return HermiteWavePacket1D{N, promote_type(Tλ1, TΛ2), promote_type(Tz1, Tz2), promote_type(Tq1, Tq2), promote_type(Tp1, Tp2)}
end

Base.:*(G1::GaussianWavePacket{1}, H2::HermiteWavePacket1D) = HermiteWavePacket1D(G1) * H2
Base.:*(H1::HermiteWavePacket1D, G2::GaussianWavePacket{1}) = H1 * HermiteWavePacket1D(G2)
function polynomial_product(G::GaussianWavePacket{1}, P::SVector{NP, <:Number}, q::RealOrNull=NullNumber()) where NP
    return polynomial_product(HermiteWavePacket1D(G), P, q)
end
convolution(G1::GaussianWavePacket{1}, H2::HermiteWavePacket1D) = convolution(HermiteWavePacket1D(G1), H2)
convolution(H1::HermiteWavePacket1D, G2::GaussianWavePacket{1}) = convolution(H1, HermiteWavePacket1D(G2))
dot_L2(G1::GaussianWavePacket{1}, H2::HermiteWavePacket1D) = dot_L2(HermiteWavePacket1D(G1), H2)
dot_L2(H1::HermiteWavePacket1D, G2::GaussianWavePacket{1}) = dot_L2(H1, HermiteWavePacket1D(G2))
