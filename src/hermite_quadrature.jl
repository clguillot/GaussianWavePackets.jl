
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
    Returns x, w, M where
    - x, w, M are such that the quadrature rule
            ∫dx f(x) ≈ ∑ⱼ wⱼf(xⱼ) (j=1,...,N)
        is exact for any function of the form f(x)exp(-x²)
    - M is a matrix of size N×N that transforms the values of a function
        at the quadrature points x into its discrete transform on the Fourier base
     In other words, if φ(x) = ∑ₙ λₙψₙ(1, 0, x) and Φ = (φ(x[n+1]))ₙ (n=0,...,N-1), then
        MΦ = Λ, with Λ=(λₙ)ₙ
=#
function hermite_primitive_discrete_transform(::Type{T}, N::Integer) where{T<:Union{Float16, Float32, Float64}}
    x, _ = gausshermite(N)

    x = BigFloat.(x)
    L = zeros(BigFloat, N, N)

    if N > 0
        b = sqrt(BigFloat("2.0"))
        @views @. L[:, 1] = BigFloat(π)^(-1/BigFloat("4.0")) * exp(-x^2 / 2)
        if N > 1
            @views @. L[:, 2] = b * x * L[:, 1]

            for k=3:N
                @views @. L[:, k] = (b * x * L[:, k-1] - sqrt(BigFloat(k-2)) * L[:, k-2]) / sqrt(BigFloat(k-1))
            end
        end
    end

    M = L^(-1)
    w = @views [dot(M[:, k], M[:, k]) for k in 1:N]
    
    return T.(x), T.(w), T.(M)
end

@generated function hermite_primitive_discrete_transform(::Type{T}, ::Val{N}) where{N, T<:Union{Float16, Float32, Float64}}
    x, w, M = hermite_primitive_discrete_transform(T, N)

    xs = SVector{N}(x)
    ws = SVector{N}(w)
    Ms = SMatrix{N, N}(M)
    return :( $xs, $ws, $Ms)
end

#=
    Let x, w = hermite_integral_quadrature(a, q, Val(N))
    This functions performs a Hermite transform and returns the result in Λ
    - U is an AbstractVector containing (φ(x[k]))ₖ where φ is the function to be transformed
    In other words, if φ(x) = ∑ₙ λₙψₙ(a, q, x) and Φ = (φ(x[n+1]))ₙ (n=0,...,N-1), then
        Λ = (λₙ)ₙ
    Returns Λ
=#
function hermite_discrete_transform!(Λ::AbstractVector{TΛ}, U::AbstractVector{TU}, a::Ta, q::Tq, ::Val{N}) where{TΛ<:Number, TU<:Number, Ta<:Real, Tq<:Real, N}
    T = fitting_float(promote_type(TΛ, TU, Ta, Tq))
    _, _, M0 = hermite_primitive_discrete_transform(T, Val(N))

    mul!(Λ, M0, U, a^Rational(-1,4), zero(TΛ))

    return Λ
end

function hermite_discrete_transform(U::AbstractVector{TU}, a::Ta, q::Tq, ::Val{N}) where{TU<:Number, Ta<:Real, Tq<:Real, N}
    TΛ = promote_type(TU, Ta, Tq)
    T = fitting_float(TΛ)
    _, _, M0 = hermite_primitive_discrete_transform(T, Val(N))

    return a^Rational(-1,4) .* (M0 * U)
end

#
function hermite_grid(a::Real, q::Union{Real, NullNumber}, ::Val{N}) where N
    T = fitting_float(typeof(a), typeof(q))
    x0, _, _ = hermite_primitive_discrete_transform(T, Val(N))
    q_broad = @SVector fill(q, N)
    return x0 .* a^Rational(-1,2) .+ q_broad
end

#=

    QUADRATURE RULE

=#

#=
    Computes two lists x, w of N weights and nodes such that the quadrature formula
        ∫dx f(x)exp(-a(x-q)²) ≈ ∑ⱼ wⱼf(xⱼ) (j=1,...,N)
    is exact for any polynomial f up to degree 2N-1
=#
function hermite_quadrature(a::Real, q::Union{Real, NullNumber}, ::Val{N}) where N
    T = fitting_float(typeof(a), typeof(q))    
    x0, w0, _ = hermite_primitive_discrete_transform(T, Val(N))

    c = a^Rational(-1,2)
    q_broad = SVector{N}(ntuple(_ -> q, N)...)
    return c.*x0.+q_broad, c.*w0
end