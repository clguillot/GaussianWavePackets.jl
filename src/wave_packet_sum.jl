#=

    SUM OF WAVE PACKETS

    A WavePacketSum{D, Ctype} represents the sum of the wave packets contained in g::Ctype.
    Ctype can be any container (a Tuple, a Vector, ...) of AbstractWavePacket{D}:
    - a Tuple is well suited to short sums, possibly mixing different types of wave packets
    - a Vector (with a concrete element type) is well suited to long, homogeneous sums
    The terms may themselves be sums, and are never merged, so that the types are left under
    the control of the user.
    Operations are applied term by term, using linearity (or bilinearity), so that
    the arguments of operations like unitary_product are simply forwarded to each term.

=#

# Checks, at the type level whenever possible, that all the terms are wave packets of dimension D
_terms_type_ok(::Type{Ctype}, ::Val{D}) where{Ctype<:Tuple, D} = all(T -> T <: AbstractWavePacket{D}, fieldtypes(Ctype))
_terms_type_ok(::Type{Ctype}, ::Val{D}) where{Ctype<:AbstractArray, D} = eltype(Ctype) <: AbstractWavePacket{D}
_terms_type_ok(::Type, ::Val) = false

struct WavePacketSum{D, Ctype} <: AbstractWavePacket{D}
    g::Ctype

    # Constrain Ctype using an inner constructor
    function WavePacketSum{D}(g::Ctype) where{D, Ctype}
        if !(_terms_type_ok(Ctype, Val(D)) || all(G -> G isa AbstractWavePacket{D}, g))
            throw(ArgumentError("All elements of g must be subtypes of AbstractWavePacket{$D}"))
        end
        return new{D, Ctype}(g)
    end
    function WavePacketSum{D}(g::AbstractWavePacket{D}...) where D
        return WavePacketSum{D}(g)
    end
    function WavePacketSum(g::AbstractWavePacket{D}...) where D
        return WavePacketSum{D}(g)
    end
end
function WavePacketSum(g::AbstractVector{<:AbstractWavePacket{D}}) where D
    return WavePacketSum{D}(g)
end
function WavePacketSum(g::Tuple{Vararg{AbstractWavePacket{D}}}) where D
    return WavePacketSum{D}(g)
end

# The terms of a wave packet: the container of the terms for a sum, the wave packet itself otherwise
_terms(G::AbstractWavePacket) = (G,)
_terms(G::WavePacketSum) = G.g


#=
    CREATION
=#

function Base.:+(G::AbstractWavePacket{D}...) where D
    return WavePacketSum{D}(G)
end

function Base.:-(G1::AbstractWavePacket{D}, G2::AbstractWavePacket{D}) where D
    return WavePacketSum{D}(G1, -G2)
end

#=
    DISPLAY
=#

# Shows the first terms of the sum (at most 4)
function Base.show(io::IO, G::WavePacketSum{D}) where D
    n = length(G.g)
    print(io, "WavePacketSum{", D, "} with ", n, n == 1 ? " term:" : " terms:")
    n_shown = min(n, 4)
    for (k, g) in enumerate(G.g)
        k > n_shown && break
        print(io, "\n  [", k, "] ", replace(sprint(show, g), "\n" => "\n      "))
    end
    n > n_shown && print(io, "\n  ⋮ (", n - n_shown, " more)")
end

#=
    BASIC OPERATIONS
=#

# Returns a null sum (each term is replaced by a null wave packet of the same type)
function Base.zero(::Type{WavePacketSum{D, Ctype}}) where{D, Ctype<:Tuple}
    return WavePacketSum{D}(map(zero, fieldtypes(Ctype)))
end
function Base.zero(::Type{WavePacketSum{D, Ctype}}) where{D, Ctype<:Vector}
    return WavePacketSum{D}(Ctype())
end
function Base.zero(G::WavePacketSum{D}) where D
    return WavePacketSum{D}(map(zero, G.g))
end

# Creates a copy of a sum
function Base.copy(G::WavePacketSum{D}) where D
    return WavePacketSum{D}(map(copy, G.g))
end

# Type of the numbers used by the terms (when it can be deduced from the type of the sum)
function core_type(::Type{WavePacketSum{D, Ctype}}) where{D, Ctype<:AbstractArray}
    return core_type(eltype(Ctype))
end
function core_type(::Type{WavePacketSum{D, Ctype}}) where{D, Ctype<:Tuple}
    return promote_type(core_type.(fieldtypes(Ctype))...)
end
# Checks that the types of all the terms (including those of nested sums) are fully known
_concrete_terms(::Type{T}) where T = true
_concrete_terms(::Type{WavePacketSum{D, Ctype}}) where{D, Ctype<:AbstractArray} = isconcretetype(eltype(Ctype)) && _concrete_terms(eltype(Ctype))
_concrete_terms(::Type{WavePacketSum{D, Ctype}}) where{D, Ctype<:Tuple} = all(_concrete_terms, fieldtypes(Ctype))
# If the type of the terms is not known (abstract element type), it is computed from the terms themselves
function core_type(G::WavePacketSum)
    if _concrete_terms(typeof(G))
        return core_type(typeof(G))
    else
        return mapreduce(core_type, promote_type, G.g; init=Union{})
    end
end

# Returns the complex conjugate of a sum
function Base.conj(G::WavePacketSum{D}) where D
    return WavePacketSum{D}(map(conj, G.g))
end

#=
    Computes ∑ₖ G[k](x)
=#
function evaluate(G::WavePacketSum, x::AbstractVector{<:Number})
    s = zero(promote_type(core_type(G), eltype(x)))
    return sum(g -> g(x), G.g; init=s)
end
(G::WavePacketSum)(x::AbstractVector{<:Number}) = evaluate(G, x)

#=
    TRANSFORMATIONS
=#

# Computes the opposite of a sum
function Base.:-(G1::WavePacketSum{D}) where D
    return WavePacketSum{D}(map(-, G1.g))
end

# Computes the product (or division) of a sum by a scalar
function Base.:*(w::Number, G::WavePacketSum{D}) where D
    return WavePacketSum{D}(map(g -> w * g, G.g))
end
function Base.:/(G::WavePacketSum{D}, w::Number) where D
    return WavePacketSum{D}(map(g -> g / w, G.g))
end

# Computes the product of two wave packets, at least one of which is a sum (the result is a sum of products)
function Base.:*(G1::WavePacketSum{D}, G2::WavePacketSum{D}) where D
    return WavePacketSum{D}(map(g1 -> g1 * G2, G1.g))
end
function Base.:*(G1::AbstractWavePacket{D}, G2::WavePacketSum{D}) where D
    return WavePacketSum{D}(map(g2 -> G1 * g2, G2.g))
end
function Base.:*(G1::WavePacketSum{D}, G2::AbstractWavePacket{D}) where D
    return G2 * G1
end

# Computes the product of a sum with a polynomial
# (the arguments are forwarded to each term, which is responsible for checking them)
function polynomial_product(G::WavePacketSum{D}, args...) where D
    return WavePacketSum{D}(map(g -> polynomial_product(g, args...), G.g))
end
# Multiplies a sum by a gaussian phase (see unitary_product for the wave packets)
# (the arguments are forwarded to each term, which is responsible for checking them)
function unitary_product(G::WavePacketSum{D}, args...) where D
    return WavePacketSum{D}(map(g -> unitary_product(g, args...), G.g))
end

# Computes the convolution product of two wave packets, at least one of which is a sum
function convolution(G1::WavePacketSum{D}, G2::WavePacketSum{D}) where D
    return WavePacketSum{D}(map(g1 -> convolution(g1, G2), G1.g))
end
function convolution(G1::AbstractWavePacket{D}, G2::WavePacketSum{D}) where D
    return WavePacketSum{D}(map(g2 -> convolution(G1, g2), G2.g))
end
function convolution(G1::WavePacketSum{D}, G2::AbstractWavePacket{D}) where D
    return convolution(G2, G1)
end

#=
    Computes the integral
        ∫ ∑ₖG[k]
=#
function integral(G::WavePacketSum)
    s = zero(core_type(G))
    return sum(g -> integral(g), G.g; init=s)
end

# Fourier transform
function fourier(G::WavePacketSum{D}) where D
    return WavePacketSum{D}(map(fourier, G.g))
end
# Inverse Fourier transform
function inv_fourier(G::WavePacketSum{D}) where D
    return WavePacketSum{D}(map(inv_fourier, G.g))
end

#=
    Helpers for the inner products and the norms
    G[k] denotes the k-th term of G (G itself if G is not a sum). If a term is a sum,
    it is handled by the recursive calls of f.
=#

# Computes ∑ₖₗ f(G1[k], G2[l])
function _all_terms(f, G1::AbstractWavePacket, G2::AbstractWavePacket)
    s = zero(promote_type(core_type(G1), core_type(G2)))
    return sum(f(g1, g2) for g1 in _terms(G1) for g2 in _terms(G2); init=s)
end
# Computes ∑ₖ real(f(G[k]))
function _diagonal_terms(f, G::AbstractWavePacket)
    s = zero(real(core_type(G)))
    return sum(g -> real(f(g)), _terms(G); init=s)
end
# Computes ∑ₖ<ₗ real(f(G[k], G[l]))
function _cross_terms(f, G::AbstractWavePacket)
    z = zero(real(core_type(G)))
    return _cross_terms(f, _terms(G), z, z)
end
# (a tuple is handled recursively, which is type stable even if the terms are of different types)
_cross_terms(f, ::Tuple{}, acc, z) = acc
function _cross_terms(f, g_list::Tuple, acc, z)
    g, rest = first(g_list), Base.tail(g_list)
    return _cross_terms(f, rest, acc + sum(h -> real(f(g, h)), rest; init=z), z)
end
# (any other container)
function _cross_terms(f, g_list, acc, z)
    return acc + sum(real(f(g1, g2)) for (k, g1) in enumerate(g_list) for g2 in Iterators.drop(g_list, k); init=z)
end
# Computes ∑ₖ f_norm(G[k]) + 2 ∑ₖ<ₗ real(f_dot(G[k], G[l]))
#   which is the squared norm of ∑ₖG[k] if f_norm(g) and f_dot(g1, g2) are the squared norm of g and the inner product of g1 and g2
function _norm2_terms(f_norm, f_dot, G::AbstractWavePacket)
    return _diagonal_terms(f_norm, G) + 2 * _cross_terms(f_dot, G)
end

#=
    Computes the L2 inner product ∫ conj(G1) G2, where at least one of the arguments is a sum
=#
dot_L2(G1::WavePacketSum{D}, G2::WavePacketSum{D}) where D = _all_terms(dot_L2, G1, G2)
dot_L2(G1::AbstractWavePacket{D}, G2::WavePacketSum{D}) where D = _all_terms(dot_L2, G1, G2)
dot_L2(G1::WavePacketSum{D}, G2::AbstractWavePacket{D}) where D = _all_terms(dot_L2, G1, G2)

#=
    Computes the squared L2 norm of a sum
=#
norm2_L2(G::WavePacketSum) = _norm2_terms(norm2_L2, dot_L2, G)

#=
    Computes ∫<(D-c)G1,(D-c)G2> where D = -i∇ and at least one of the arguments is a sum
    (the optional argument c is forwarded to each term, see dot_∇ for GaussianWavePacket)
=#
function dot_∇(G1::WavePacketSum{D}, G2::WavePacketSum{D}, c...) where D
    return _all_terms((g1, g2) -> dot_∇(g1, g2, c...), G1, G2)
end
function dot_∇(G1::AbstractWavePacket{D}, G2::WavePacketSum{D}, c...) where D
    return _all_terms((g1, g2) -> dot_∇(g1, g2, c...), G1, G2)
end
function dot_∇(G1::WavePacketSum{D}, G2::AbstractWavePacket{D}, c...) where D
    return _all_terms((g1, g2) -> dot_∇(g1, g2, c...), G1, G2)
end

#=
    Computes the squared homogeneous H1 norm ∫|(D-c)G|² of a sum
=#
function norm2_∇(G::WavePacketSum, c...)
    return _norm2_terms(g -> norm2_∇(g, c...), (g1, g2) -> dot_∇(g1, g2, c...), G)
end
