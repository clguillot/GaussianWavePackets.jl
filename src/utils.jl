#
_upper_tri(A::Diagonal) = A
_upper_tri(A::AbstractMatrix) = UpperTriangular(A)
_symmetric(A::Diagonal) = A
_symmetric(A::AbstractMatrix) = Symmetric(A)

#=

    CLENSHAW ALGORITHM FOR HERMITE FUNCTIONS

=#

#=
    Computes
        ∑ₙ Λ[n+1] αₙ(z, x)
    where
        α₀ = α0
        αₙ₊₁ = (z*x*αₙ(z, x) - √n*αₙ₋₁(z, x)) / √(n+1)
=#
function clenshaw_hermite_eval(Λ::AbstractVector{TΛ}, z::Tz, x::Tx, α0::Tα) where{TΛ<:Number, Tx<:Number, Tz<:Number, Tα<:Number}
    T = promote_type(fitting_float(TΛ), fitting_float(Tz), fitting_float(Tx), fitting_float(Tα))
    N = length(Λ)
    u = zero(promote_type(TΛ, Tz, Tx, T))
    if N > 0
        u += Λ[N]
        if N > 1
            zx = z * x
            v = u
            u = Λ[N-1] + zx * u / sqrt(T(N-1))
            for k=N-3:-1:0
                (u, v) = (Λ[k+1] + zx * u / sqrt(T(k+1)) - sqrt(T(k+1) / T(k+2)) * v, u)
            end
        end
    end
    return u * α0
end
#=
    Computes
        ∑ₙ Λ[n+1] αₙ(z, xₘ) (m=1,...,M)
    where
        αₙ₊₁ = (z*x*αₙ(z, x) - √n*αₙ₋₁(z, x)) / √(n+1)
=#
function clenshaw_hermite_eval(Λ::AbstractVector{TΛ}, z::Tz, x::SVector{M, Tx}, α0::SVector{M, Tα}) where{TΛ<:Number, M, Tx<:Number, Tz<:Number, Tα<:Number}
    T = promote_type(fitting_float(TΛ), fitting_float(Tz), fitting_float(Tx), fitting_float(Tα))
    N = length(Λ)
    u = zero(SVector{M, promote_type(TΛ, Tz, Tx, T)})
    if N > 0
        u = u .+ Λ[N]
        if N > 1
            zx = z * x
            v = u
            u = @. Λ[N-1] + zx * u / sqrt(T(N-1))
            for k=N-3:-1:0
                (u, v) = @. (Λ[k+1] + zx * u / sqrt(T(k+1)) - sqrt(T(k+1) / T(k+2)) * v, u)
            end
        end
    end
    return u .* α0
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
