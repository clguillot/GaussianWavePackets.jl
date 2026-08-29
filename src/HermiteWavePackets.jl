module HermiteWavePackets

using LinearAlgebra
using StaticArrays
using IrrationalConstants
using FastGaussQuadrature
using SpecialFunctions
using NullNumbers

# Abtract types
abstract type AbstractWavePacket{D} end
export AbstractWavePacket

# Concrete types
export Gaussian
export GaussianWavePacket
export HermiteFct1D
export HermiteWavePacket1D
export WavePacketSum

# Functions
export hermite_grid
export evaluate
export evaluate_grid
export hermite_discrete_transform
export hermite_quadrature
export truncate_to_gaussian
export integral
export ∫
export evaluate
export unitary_product
export polynomial_product
export fourier
export inv_fourier
export convolution
export dot_L2
export norm2_L2
export norm_L2
export dot_∇
export norm2_∇
export norm_∇
export core_type
export coulomb_integral

#=
    A real number, or the statically-known absence of one (see NullNumbers.jl).
    Used throughout this file for q, p, and (for `Gaussian`) z, so that "this
    parameter is not present" is resolved at compile time instead of at runtime.
=#
const RealOrNull = Union{Real, NullNumber}

include("utils.jl")
include("special_cholesky.jl")

include("gaussian_arg.jl")
include("hermite_quadrature.jl")
include("generics.jl")

include("gaussian_wave_packet.jl")
include("hermite_wave_packet_1d.jl")

include("wave_packet_sum.jl")

end
