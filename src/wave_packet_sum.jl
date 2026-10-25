# Checks, at the type level whenever possible, that all the terms are wave packets of dimension D
_terms_type_ok(::Type{Ctype}, ::Val{D}) where{Ctype<:Tuple, D} = all(T -> T <: AbstractWavePacket{D}, fieldtypes(Ctype))
_terms_type_ok(::Type{Ctype}, ::Val{D}) where{Ctype<:AbstractArray, D} = eltype(Ctype) <: AbstractWavePacket{D}
_terms_type_ok(::Type, ::Val) = false

"""
    WavePacketSum(g1, g2, ...)
    WavePacketSum(container)
    WavePacketSum{D}(container)

Represents the sum of the wave packets contained in `g` (a field of type `Ctype`).
`Ctype` can be any container (a `Tuple`, a `Vector`, ...) of `AbstractWavePacket{D}`:
- a `Tuple` is well suited to short sums, possibly mixing different types of wave packets
  (this is what `+` builds),
- a `Vector` (with a concrete element type) is well suited to long, homogeneous sums.

The terms may themselves be sums, and are never merged, so that the types are left under
the control of the user. Operations are applied term by term, using linearity (or
bilinearity), so that the arguments of operations like `unitary_product` are simply
forwarded to each term.

# Examples
```julia
S = G1 + G2 + H                 # a WavePacketSum stored in a Tuple
V = WavePacketSum([GaussianWavePacket(1.0, 1.0, 0.1k) for k in 1:50])
S(0.3), integral(V), norm_L2(S)
```
"""
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
    # (at least one term, so that D is always determined)
    function WavePacketSum(g1::AbstractWavePacket{D}, g::AbstractWavePacket{D}...) where D
        return WavePacketSum{D}((g1, g...))
    end
end
function WavePacketSum(g::AbstractVector{<:AbstractWavePacket{D}}) where D
    return WavePacketSum{D}(g)
end
function WavePacketSum(g::Tuple{AbstractWavePacket{D}, Vararg{AbstractWavePacket{D}}}) where D
    return WavePacketSum{D}(g)
end

# The terms of a wave packet: the container of the terms for a sum, the wave packet itself otherwise
_terms(G::AbstractWavePacket) = (G,)
_terms(G::WavePacketSum) = G.g

#=
    Computes acc + ∑ₖ f(g[k])
    Tuples are handled recursively, which is type stable even if the terms are of different types,
    and does not rely on Base.sum (whose inference gives up when sums of closures are nested,
    as happens for sums of sums)
=#
# (f::F forces the specialization on f, which Julia does not do for a function which is only passed through)
@inline _sum_terms(f::F, ::Tuple{}, acc) where F = acc
@inline _sum_terms(f::F, g::Tuple, acc) where F = _sum_terms(f, Base.tail(g), acc + f(first(g)))
_sum_terms(f::F, g, acc) where F = sum(f, g; init=acc)


#=
    CREATION
=#

function Base.:+(G1::AbstractWavePacket{D}, G::AbstractWavePacket{D}...) where D
    return WavePacketSum{D}((G1, G...))
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

# Returns a null sum (for the static arrays, each term is replaced by a null wave packet)
function Base.zero(::Type{WavePacketSum{D, Ctype}}) where{D, Ctype<:StaticArray}
    return WavePacketSum{D}(Ctype(ntuple(_ -> zero(eltype(Ctype)), Val(length(Ctype)))))
end

# Type of the numbers used by the terms (when it can be deduced from the type of the sum)
# (Bool is neutral for promote_type among the number types, and makes the empty sums work)
# These functions of types are declared :foldable, so that the compiler evaluates them at compile time:
# otherwise, the inference of the recursive calls for nested sums is given up (and the result is Any)
Base.@assume_effects :foldable function core_type(::Type{WavePacketSum{D, Ctype}}) where{D, Ctype<:AbstractArray}
    return promote_type(Bool, core_type(eltype(Ctype)))
end
Base.@assume_effects :foldable function core_type(::Type{WavePacketSum{D, Ctype}}) where{D, Ctype<:Tuple}
    return promote_type(Bool, map(core_type, fieldtypes(Ctype))...)
end
# Checks that the types of all the terms (including those of nested sums) are fully known
_concrete_terms(::Type{T}) where T = true
Base.@assume_effects :foldable _concrete_terms(::Type{WavePacketSum{D, Ctype}}) where{D, Ctype<:AbstractArray} =
    isconcretetype(eltype(Ctype)) && _concrete_terms(eltype(Ctype))
Base.@assume_effects :foldable _concrete_terms(::Type{WavePacketSum{D, Ctype}}) where{D, Ctype<:Tuple} =
    all(_concrete_terms, fieldtypes(Ctype))
# If the type of the terms is not known (abstract element type), it is computed from the terms themselves
function core_type(G::WavePacketSum)
    if _concrete_terms(typeof(G))
        return core_type(typeof(G))
    else
        return mapreduce(core_type, promote_type, G.g; init=Bool)
    end
end

# Returns the complex conjugate of a sum
function Base.conj(G::WavePacketSum{D}) where D
    return WavePacketSum{D}(map(conj, G.g))
end

#=
    Computes ∑ₖ G[k](x)
    If dims is given, evaluates each term along the dimensions in dims only (see evaluate for
    GaussianWavePacket), and returns the sum of the results
=#
@inline Base.@constprop :aggressive function evaluate(G::WavePacketSum{D}, x::AbstractVector{<:Number}; dims=nothing) where D
    if dims === nothing
        s = zero(promote_type(core_type(G), eltype(x)))
        return _sum_terms(g -> evaluate(g, x), G.g, s)
    else
        return evaluate(G, x, Val(dims))
    end
end
function evaluate(G::WavePacketSum{D}, x::AbstractVector{<:Number}, ::Val{dims}) where{D, dims}
    if length(dims) == D
        s = zero(promote_type(core_type(G), eltype(x)))
        return _sum_terms(g -> evaluate(g, x, Val(dims)), G.g, s)
    else
        return WavePacketSum(map(g -> evaluate(g, x, Val(dims)), G.g))
    end
end
function evaluate(G::WavePacketSum{1}, x::Number)
    s = zero(promote_type(core_type(G), typeof(x)))
    return _sum_terms(g -> evaluate(g, x), G.g, s)
end
(G::WavePacketSum)(x::AbstractVector{<:Number}) = evaluate(G, x)
(G::WavePacketSum{1})(x::Number) = evaluate(G, x)

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
    If dims is given, only the variables in dims are integrated (see integral for
    GaussianWavePacket), and the result is the sum of the partial integrals of the terms
=#
@inline Base.@constprop :aggressive function integral(G::WavePacketSum{D}; dims=nothing) where D
    if dims === nothing
        s = zero(core_type(G))
        return _sum_terms(integral, G.g, s)
    else
        return integral(G, Val(dims))
    end
end
function integral(G::WavePacketSum{D}, ::Val{dims}) where{D, dims}
    if length(dims) == D
        s = zero(core_type(G))
        return _sum_terms(g -> integral(g, Val(dims)), G.g, s)
    else
        return WavePacketSum(map(g -> integral(g, Val(dims)), G.g))
    end
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
function _all_terms(f::F, G1::AbstractWavePacket, G2::AbstractWavePacket) where F
    s = zero(promote_type(core_type(G1), core_type(G2)))
    g2_list = _terms(G2)
    return _sum_terms(g1 -> _sum_terms(g2 -> f(g1, g2), g2_list, s), _terms(G1), s)
end
# Computes ∑ₖ real(f(G[k]))
function _diagonal_terms(f::F, G::AbstractWavePacket) where F
    s = zero(real(core_type(G)))
    return _sum_terms(g -> real(f(g)), _terms(G), s)
end
# Computes ∑ₖ<ₗ real(f(G[k], G[l]))
function _cross_terms(f::F, G::AbstractWavePacket) where F
    z = zero(real(core_type(G)))
    return _cross_terms(f, _terms(G), z, z)
end
# (a tuple is handled recursively, which is type stable even if the terms are of different types)
_cross_terms(f::F, ::Tuple{}, acc, z) where F = acc
function _cross_terms(f::F, g_list::Tuple, acc, z) where F
    g, rest = first(g_list), Base.tail(g_list)
    return _cross_terms(f, rest, _sum_terms(h -> real(f(g, h)), rest, acc), z)
end
# (arrays: explicit indexing over the pairs k < l)
function _cross_terms(f::F, g_list::AbstractArray, acc, z) where F
    idx = LinearIndices(g_list)
    n = length(idx)
    for a in 1:n, b in a+1:n
        acc += real(f(g_list[idx[a]], g_list[idx[b]]))
    end
    return acc
end
# (any other container)
function _cross_terms(f::F, g_list, acc, z) where F
    return acc + sum(real(f(g1, g2)) for (k, g1) in enumerate(g_list) for g2 in Iterators.drop(g_list, k); init=z)
end
# Computes ∑ₖ f_norm(G[k]) + 2 ∑ₖ<ₗ real(f_dot(G[k], G[l]))
#   which is the squared norm of ∑ₖG[k] if f_norm(g) and f_dot(g1, g2) are the squared norm of g and the inner product of g1 and g2
function _norm2_terms(f_norm::F1, f_dot::F2, G::AbstractWavePacket) where{F1, F2}
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
    (D-c)G = exp(ic⋅x) D(exp(-ic⋅x)G), so that c is handled by shifting the momentum of each term.
    The functions are passed directly (and not through closures), so that the recursive calls
    for nested sums are inferred
=#
_shift_p(G::GaussianWavePacket{D}, c::AbstractVector) where D = GaussianWavePacket(G.λ, G.z, G.q, G.p - SVector{D}(c))
_shift_p(G::WavePacketSum{D}, c::AbstractVector) where D = WavePacketSum{D}(map(g -> _shift_p(g, c), G.g))

dot_∇(G1::WavePacketSum{D}, G2::WavePacketSum{D}) where D = _all_terms(dot_∇, G1, G2)
dot_∇(G1::AbstractWavePacket{D}, G2::WavePacketSum{D}) where D = _all_terms(dot_∇, G1, G2)
dot_∇(G1::WavePacketSum{D}, G2::AbstractWavePacket{D}) where D = _all_terms(dot_∇, G1, G2)
dot_∇(G1::WavePacketSum{D}, G2::WavePacketSum{D}, c::AbstractVector) where D = dot_∇(_shift_p(G1, c), _shift_p(G2, c))
dot_∇(G1::AbstractWavePacket{D}, G2::WavePacketSum{D}, c::AbstractVector) where D = dot_∇(_shift_p(G1, c), _shift_p(G2, c))
dot_∇(G1::WavePacketSum{D}, G2::AbstractWavePacket{D}, c::AbstractVector) where D = dot_∇(_shift_p(G1, c), _shift_p(G2, c))

#=
    Computes the squared homogeneous H1 norm ∫|(D-c)G|² of a sum
=#
norm2_∇(G::WavePacketSum) = _norm2_terms(norm2_∇, dot_∇, G)
norm2_∇(G::WavePacketSum, c::AbstractVector) = norm2_∇(_shift_p(G, c))
