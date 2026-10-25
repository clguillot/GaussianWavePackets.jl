
#=
    Reminder :
    - gausshermite(p) computes the weights (wⱼ)ⱼ and nodes (xⱼ)ⱼ (j=1,...,m) such that the quadrature formula
     ∫dx f(x)exp(-x²) ≈ ∑ⱼ wⱼf(xⱼ)
     is exact for any polynomial f up to degree 2m-1
=#


#=

    TRANSFORM

=#

#=
    Computes, in BigFloat precision, the Hermite functions
        hₙ(x) = π^(-1/4) / sqrt(2ⁿn!) * Hₙ(x) * exp(-x²/2)    (n=0,...,N-1)
    (i.e. hₙ = ψₙ(1, 0, 0, ⋅)), with the stable recurrence
        hₙ₊₁(x) = (√2*x*hₙ(x) - √n*hₙ₋₁(x)) / √(n+1)
    Returns the vector (hₙ(x))ₙ
=#
function _big_hermite_functions(x::BigFloat, N::Integer)
    h = zeros(BigFloat, N)
    if N > 0
        b = sqrt(BigFloat(2))
        h[1] = BigFloat(π)^(-BigFloat(1)/4) * exp(-x^2 / 2)
        if N > 1
            h[2] = b * x * h[1]
            for k=3:N
                h[k] = (b * x * h[k-1] - sqrt(BigFloat(k-2)) * h[k-2]) / sqrt(BigFloat(k-1))
            end
        end
    end
    return h
end

#=
    Computes, in BigFloat precision, the nodes (xⱼ)ⱼ and the weights (ωⱼ)ⱼ = (wⱼexp(xⱼ²))ⱼ
    of the Gauss-Hermite quadrature rule with N points (wⱼ being the usual Gauss-Hermite weights)
    The nodes computed by FastGaussQuadrature (in Float64) are refined with Newton's method
    on h_N (whose roots are the nodes), and the weights are given by
        ωⱼ = 1 / (N h_{N-1}(xⱼ)²)
=#
function _big_gausshermite(N::Integer)
    x0, _ = gausshermite(N)
    x = BigFloat.(x0)
    ω = zeros(BigFloat, N)
    for j in 1:N
        for _ in 1:20
            h = _big_hermite_functions(x[j], N+1)
            # h_N'(x) = √(2N) h_{N-1}(x) - x h_N(x)
            dh = sqrt(BigFloat(2N)) * h[N] - x[j] * h[N+1]
            δ = h[N+1] / dh
            x[j] -= δ
            abs(δ) <= 4 * eps(BigFloat) * max(1, abs(x[j])) && break
        end
        hN1 = _big_hermite_functions(x[j], N)[N]
        ω[j] = 1 / (N * hN1^2)
    end
    return x, ω
end

#=
    Returns x, ω, M where
    - x, ω are such that the quadrature rule
            ∫dx g(x) ≈ ∑ⱼ ωⱼg(xⱼ) (j=1,...,N)
        is exact for any function of the form g(x) = P(x)exp(-x²) where P is a polynomial
        of degree at most 2N-1 (that is, ωⱼ = wⱼexp(xⱼ²), (wⱼ)ⱼ being the usual Gauss-Hermite weights)
    - M is a matrix of size N×N that transforms the values of a function
        at the quadrature points x into its discrete transform on the Fourier base
     In other words, if φ(x) = ∑ₙ λₙψₙ(1, 0, x) and Φ = (φ(x[n+1]))ₙ (n=0,...,N-1), then
        MΦ = Λ, with Λ=(λₙ)ₙ
    Everything is computed in BigFloat precision, then rounded to T. Since the quadrature rule is exact
    for the products ψₘψₙ (m,n < N), the matrix L = (ψₙ(xⱼ))ⱼₙ satisfies LᵀΩL = I with Ω = diag(ω),
    hence M = L⁻¹ = LᵀΩ.
=#
function hermite_primitive_discrete_transform(::Type{T}, N::Integer) where{T<:Union{Float16, Float32, Float64}}
    x, ω, M = setprecision(BigFloat, 256) do
        x, ω = _big_gausshermite(N)
        L = zeros(BigFloat, N, N)
        for j in 1:N
            L[j, :] = _big_hermite_functions(x[j], N)
        end
        M = transpose(L) * Diagonal(ω)
        return x, ω, M
    end
    return T.(x), T.(ω), T.(M)
end

@generated function hermite_primitive_discrete_transform(::Type{T}, ::Val{N}) where{N, T<:Union{Float16, Float32, Float64}}
    x, w, M = hermite_primitive_discrete_transform(T, N)

    xs = SVector{N}(x)
    ws = SVector{N}(w)
    Ms = SMatrix{N, N}(M)
    return :( $xs, $ws, $Ms)
end

"""
    hermite_discrete_transform(Φ, a, q, Val(N))
    hermite_discrete_transform!(Λ, Φ, a, q, Val(N))

Hermite transform: if `x = hermite_grid(a, q, Val(N))` and `Φ` contains the values `φ(xⱼ)`
of a function of the form

    φ(x) = ∑ₙ λₙψₙ(a, q, x)    (n=0,...,N-1)

(see [`HermiteFct1D`](@ref)), returns the coefficients `Λ = (λₙ)ₙ`. The in-place version stores
the result in `Λ`. `q` may be a `NullNumber` (no shift).
"""
function hermite_discrete_transform end

"""
    hermite_discrete_transform!(Λ, Φ, a, q, Val(N))

In-place version of [`hermite_discrete_transform`](@ref): stores the coefficients in `Λ`.
"""
function hermite_discrete_transform!(Λ::AbstractVector{TΛ}, U::AbstractVector{TU}, a::Ta, q::Tq, ::Val{N}) where{TΛ<:Number, TU<:Number, Ta<:Real, Tq<:RealOrNull, N}
    T = fitting_float(promote_type(TΛ, TU, Ta, Tq))
    _, _, M0 = hermite_primitive_discrete_transform(T, Val(N))

    mul!(Λ, M0, U, inv(sqrt(sqrt(a))), zero(TΛ))

    return Λ
end

function hermite_discrete_transform(U::AbstractVector{TU}, a::Ta, q::Tq, ::Val{N}) where{TU<:Number, Ta<:Real, Tq<:RealOrNull, N}
    TΛ = promote_type(TU, Ta, Tq)
    T = fitting_float(TΛ)
    _, _, M0 = hermite_primitive_discrete_transform(T, Val(N))

    return inv(sqrt(sqrt(a))) .* (M0 * U)
end

"""
    hermite_grid(a, q, Val(N))

Returns the `N` points (as an `SVector`) on which a function `∑ₙ λₙψₙ(a, q, x)` (n=0,...,N-1)
must be evaluated to compute its coefficients with [`hermite_discrete_transform`](@ref).
These are the nodes of the Gauss-Hermite quadrature rule, scaled by `1/√a` and shifted by `q`.
"""
function hermite_grid(a::Real, q::Union{Real, NullNumber}, ::Val{N}) where N
    T = fitting_float(typeof(a), typeof(q))
    x0, _, _ = hermite_primitive_discrete_transform(T, Val(N))
    q_broad = @SVector fill(q, N)
    return x0 .* inv(sqrt(a)) .+ q_broad
end

#=

    QUADRATURE RULE

=#

"""
    x, w = hermite_quadrature(a, q, Val(N))

Computes the `N` nodes `x` and weights `w` (as `SVector`s) of a Gauss-Hermite quadrature rule
such that

    ∫dx g(x) ≈ ∑ⱼ wⱼg(xⱼ)    (j=1,...,N)

is exact for any function of the form `g(x) = f(x)exp(-a(x-q)²)` where `f` is a polynomial of
degree at most `2N-1`. Note that the weights already include the factor `exp(a(xⱼ-q)²)`:
the Gaussian is part of the integrand `g` and must be included in the values `g(xⱼ)`.
`q` may be a `NullNumber` (no shift).
"""
function hermite_quadrature(a::Real, q::Union{Real, NullNumber}, ::Val{N}) where N
    T = fitting_float(typeof(a), typeof(q))    
    x0, w0, _ = hermite_primitive_discrete_transform(T, Val(N))

    c = inv(sqrt(a))
    q_broad = SVector{N}(ntuple(_ -> q, N)...)
    return c.*x0.+q_broad, c.*w0
end