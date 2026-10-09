#=

    SUM OF WAVE PACKETS

    A WavePacketSum{D, Ctype} represents the sum of the wave packets contained in g::Ctype.
    Ctype can be any container (a Tuple, a Vector, ...) of AbstractWavePacket{D}:
    - a Tuple is well suited to short sums, possibly mixing different types of wave packets
    - a Vector (with a concrete element type) is well suited to long, homogeneous sums
    The terms are never merged, so that the types are left under the control of the user.
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

#
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

#
function Base.conj(G::WavePacketSum{D}) where D
    return WavePacketSum{D}(conj.(G.g))
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

#
function Base.:-(G1::WavePacketSum{D}) where D
    return WavePacketSum{D}(.- G1.g)
end

#
function Base.:*(w::Number, G::WavePacketSum{D}) where D
    return WavePacketSum{D}(w .* G.g)
end
function Base.:/(G::WavePacketSum{D}, w::Number) where D
    return WavePacketSum{D}(G.g ./ w)
end

#
function Base.:*(G1::WavePacketSum{D}, G2::WavePacketSum{D}) where D
    return WavePacketSum{D}(broadcast(g1 -> g1 * G2, G1.g))
end
function Base.:*(G1::AbstractWavePacket{D}, G2::WavePacketSum{D}) where D
    return WavePacketSum{D}(broadcast(g2 -> G1 * g2, G2.g))
end
function Base.:*(G1::WavePacketSum{D}, G2::AbstractWavePacket{D}) where D
    return G2 * G1
end

# The arguments are forwarded to each term, which is responsible for checking them
function polynomial_product(G::WavePacketSum{D}, args...) where D
    return WavePacketSum{D}(broadcast(g -> polynomial_product(g, args...), G.g))
end
function unitary_product(G::WavePacketSum{D}, args...) where D
    return WavePacketSum{D}(broadcast(g -> unitary_product(g, args...), G.g))
end

#
function convolution(G1::WavePacketSum{D}, G2::WavePacketSum{D}) where D
    return WavePacketSum{D}(broadcast(g1 -> convolution(g1, G2), G1.g))
end
function convolution(G1::AbstractWavePacket{D}, G2::WavePacketSum{D}) where D
    return WavePacketSum{D}(broadcast(g2 -> convolution(G1, g2), G2.g))
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

# Computes ∑ₖ<ₗ real(f(G[k], G[l]))
function _cross_terms(f, G::WavePacketSum)
    s = zero(real(core_type(G)))
    S = s
    for k in firstindex(G.g):lastindex(G.g)-1, l in k+1:lastindex(G.g)
        S += real(f(G.g[k], G.g[l]))
    end
    return S
end

#
function dot_L2(G1::WavePacketSum{D}, G2::WavePacketSum{D}) where D
    s = zero(promote_type(core_type(G1), core_type(G2)))
    return sum(g1 -> dot_L2(g1, G2), G1.g; init=s)
end
function dot_L2(G1::AbstractWavePacket{D}, G2::WavePacketSum{D}) where D
    s = zero(promote_type(core_type(G1), core_type(G2)))
    return sum(g2 -> dot_L2(G1, g2), G2.g; init=s)
end
function dot_L2(G1::WavePacketSum{D}, G2::AbstractWavePacket{D}) where D
    s = zero(promote_type(core_type(G1), core_type(G2)))
    return sum(g1 -> dot_L2(g1, G2), G1.g; init=s)
end

#=
    Computes the squared L2 norm of ∑ₖ G[k]
=#
function norm2_L2(G::WavePacketSum)
    s = zero(real(core_type(G)))
    return sum(g -> norm2_L2(g), G.g; init=s) + 2 * _cross_terms(dot_L2, G)
end



# The optional argument c is forwarded to each term (see dot_∇ for GaussianWavePacket)
function dot_∇(G1::WavePacketSum{D}, G2::WavePacketSum{D}, c...) where D
    s = zero(promote_type(core_type(G1), core_type(G2)))
    return sum(g1 -> dot_∇(g1, G2, c...), G1.g; init=s)
end
function dot_∇(G1::AbstractWavePacket{D}, G2::WavePacketSum{D}, c...) where D
    s = zero(promote_type(core_type(G1), core_type(G2)))
    return sum(g2 -> dot_∇(G1, g2, c...), G2.g; init=s)
end
function dot_∇(G1::WavePacketSum{D}, G2::AbstractWavePacket{D}, c...) where D
    s = zero(promote_type(core_type(G1), core_type(G2)))
    return sum(g1 -> dot_∇(g1, G2, c...), G1.g; init=s)
end

#=
    Computes the squared homogeneous H1 norm of ∑ₖ G[k]
=#
function norm2_∇(G::WavePacketSum, c...)
    s = zero(real(core_type(G)))
    return sum(g -> norm2_∇(g, c...), G.g; init=s) + 2 * _cross_terms((g1, g2) -> dot_∇(g1, g2, c...), G)
end

# Fourier transform
function fourier(G::WavePacketSum{D}) where D
    return WavePacketSum{D}(fourier.(G.g))
end
# Inverse Fourier transform
function inv_fourier(G::WavePacketSum{D}) where D
    return WavePacketSum{D}(inv_fourier.(G.g))
end

#=
    ITERATOR
    It is allowed to iterate over a WavePacketSum
    The iteration will be performed over all elements of the sum,
    by breaking down internal WavePacketSum
=#
function Base.iterate(G::WavePacketSum)
    sup_state = iterate(G.g)
    while !isnothing(sup_state)
        sub_state = iterate(sup_state[1])
        if !isnothing(sub_state)
            return (sub_state[1], (sup_state..., sub_state[2]))
        end
        sup_state = iterate(G.g, sup_state[2])
    end
    return nothing
end
function Base.iterate(G::WavePacketSum, state)
    sup_state = (state[1], state[2])
    sub_state = iterate(sup_state[1], state[3])

    # If an iterated element is found on the current element
    if !isnothing(sub_state)
        return (sub_state[1], (sup_state..., sub_state[2]))
    end

    # If no iterated element is found on the current element
    sup_state = iterate(G.g, sup_state[2])
    while !isnothing(sup_state)
        sub_state = iterate(sup_state[1])
        if !isnothing(sub_state)
            return (sub_state[1], (sup_state..., sub_state[2]))
        end
        sup_state = iterate(G.g, sup_state[2])
    end

    return nothing
end
