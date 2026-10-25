#
_upper_tri(A::Diagonal) = A
_upper_tri(A::AbstractMatrix) = UpperTriangular(A)
_symmetric(A::Diagonal) = A
_symmetric(A::AbstractMatrix) = Symmetric(A)

#=

    CLENSHAW ALGORITHM FOR HERMITE FUNCTIONS

=#

# A cheap norm, used to detect when the Clenshaw sums become too large
@inline _norm1(x::Real) = abs(x)
@inline _norm1(x::Complex) = abs(real(x)) + abs(imag(x))

#=
    Computes
        ∑ₙ Λ[n+1] αₙ(z, x)
    where
        α₀ = exp(e0)
        αₙ₊₁ = (z*x*αₙ(z, x) - √n*αₙ₋₁(z, x)) / √(n+1)
    The factor exp(e0) is only applied at the end. For large N and |x|, the Clenshaw sums can
    overflow while exp(e0) underflows (giving Inf*0 = NaN): the sums are therefore rescaled
    by 1/√floatmax whenever they become larger than √floatmax, and the scaling is put back
    in the final exponential.
=#
function clenshaw_hermite_eval(Λ::AbstractVector{TΛ}, z::Tz, x::Tx, e0::Te) where{TΛ<:Number, Tx<:Number, Tz<:Number, Te<:Number}
    T = promote_type(fitting_float(TΛ), fitting_float(Tz), fitting_float(Tx), fitting_float(Te))
    N = length(Λ)
    big = sqrt(floatmax(T))
    ibig = inv(big)
    logbig = log(big)
    u = zero(promote_type(TΛ, Tz, Tx, T))
    s = one(T)      # the actual sums are (u, v) / s
    k = 0           # s = ibig^k
    if N > 0
        u += Λ[N]
        if N > 1
            zx = z * x
            v = u
            u = Λ[N-1] + zx * u / sqrt(T(N-1))
            for j=N-3:-1:0
                (u, v) = (s * Λ[j+1] + zx * u / sqrt(T(j+1)) - sqrt(T(j+1) / T(j+2)) * v, u)
                if _norm1(u) > big
                    u *= ibig
                    v *= ibig
                    s *= ibig
                    k += 1
                end
            end
        end
    end
    return u * exp(e0 + k * logbig)
end
#=
    Computes
        ∑ₙ Λ[n+1] αₙ(z, xₘ) (m=1,...,M)
    where
        α₀ = exp(e0ₘ)
        αₙ₊₁ = (z*x*αₙ(z, x) - √n*αₙ₋₁(z, x)) / √(n+1)
    (with the same rescaling as above, done independently for each point)
=#
function clenshaw_hermite_eval(Λ::AbstractVector{TΛ}, z::Tz, x::SVector{M, Tx}, e0::SVector{M, Te}) where{TΛ<:Number, M, Tx<:Number, Tz<:Number, Te<:Number}
    T = promote_type(fitting_float(TΛ), fitting_float(Tz), fitting_float(Tx), fitting_float(Te))
    N = length(Λ)
    big = sqrt(floatmax(T))
    ibig = inv(big)
    logbig = log(big)
    u = zero(SVector{M, promote_type(TΛ, Tz, Tx, T)})
    s = ones(SVector{M, T})
    k = zeros(SVector{M, Int})
    if N > 0
        u = u .+ Λ[N]
        if N > 1
            zx = z * x
            v = u
            u = @. Λ[N-1] + zx * u / sqrt(T(N-1))
            for j=N-3:-1:0
                (u, v) = @. (s * Λ[j+1] + zx * u / sqrt(T(j+1)) - sqrt(T(j+1) / T(j+2)) * v, u)
                if any(y -> _norm1(y) > big, u)
                    rescale = @. _norm1(u) > big
                    u = @. ifelse(rescale, u * ibig, u)
                    v = @. ifelse(rescale, v * ibig, v)
                    s = @. ifelse(rescale, s * ibig, s)
                    k = k .+ rescale
                end
            end
        end
    end
    return @. u * exp(e0 + k * logbig)
end

# The powers (αⁿ)ₙ (n=0,...,N-1), computed as a running product
@inline function _powers(α::Number, ::Val{N}) where N
    return cumprod(SVector{N}(ntuple(n -> n == 1 ? one(α) : α, Val(N))))
end

#=

    HORNER ALGORITHM

=#

#=
    Computes
        ∑ₙ Λ[n+1] xⁿ
=#
function horner_eval(Λ::AbstractVector{TΛ}, x::Tx) where{TΛ<:Number, Tx<:Number}
    N = length(Λ)
    u = zero(promote_type(TΛ, Tx))
    if N > 0
        u += Λ[N]
        for k=N-1:-1:1
            u = Λ[k] + x * u
        end
    end
    return u
end
#=
    Computes
        ∑ₙ Λ[n+1] xₘⁿ (m=1,...,M)
=#
function horner_eval(Λ::AbstractVector{TΛ}, x::SVector{M, Tx}) where{TΛ<:Number, M, Tx<:Number}
    N = length(Λ)
    u = zero(SVector{M, promote_type(TΛ, Tx)})
    if N > 0
        u = u .+ Λ[N]
        for k=N-1:-1:1
            u = @. Λ[k] + x * u
        end
    end
    return u
end
