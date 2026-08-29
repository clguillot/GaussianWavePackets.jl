using HermiteWavePackets

using LinearAlgebra
using FastGaussQuadrature
using SpecialFunctions
using StaticArrays
using Cubature

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

# Approximates the integral on F on [-X, X]
function legendre_quadrature(X, M, F)
    h = 2 * X / M
    x_legendre, w_legendre = gausslegendre(16)
    I = zero(Complex{BigFloat})
    for k=1:M
        x = -X + (k - 0.5) * h
        I += h/2 * dot(w_legendre, F.(x .+ h/2 * x_legendre))
    end
    return ComplexF64(I)
end

function complex_cubature(f, a, b; reltol=1e-13, abstol=0)
    Ir = hcubature(x -> real(f(x)), a, b; reltol=reltol, abstol=abstol)
    Ic = hcubature(x -> imag(f(x)), a, b; reltol=reltol, abstol=abstol)
    return complex(Ir[1], Ic[1])
end

include("test_gaussian.jl")
include("test_gaussian_wave_packet.jl");
include("test_hermite_wave_packet_1d.jl");