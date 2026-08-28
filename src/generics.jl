
function Base.zero(::T) where{T<:AbstractWavePacket}
    return zero(T)
end

function core_type(::T) where{T<:AbstractWavePacket}
    return core_type(T)
end

function core_type(::Type{T}) where{T<:Number}
    return T
end
function core_type(::T) where{T<:Number}
    return core_type(T)
end

function core_type(S, T)
    return promote_type(core_type(S), core_type(T))
end
function core_type(S, T, U...)
    return promote_type(core_type(S), core_type(T), core_type(U...))
end

@generated function fitting_float(::Type{T}) where{T<:Number}
    prec = precision(real(T))
    if prec == precision(Float16)
        return :( Float16 )
    elseif prec == precision(Float32)
        return :( Float32 )
    elseif prec == precision(Float64)
        return :( Float64 )
    else
        throw(ArgumentError("The precision format of $T is unsupported"))
    end
end
function fitting_float(T...)
    return fitting_float(core_type(T...))
end

@inline function Base.:*(G::AbstractWavePacket, w::Number)
    return w * G
end

function Base.iterate(G::AbstractWavePacket)
    return (G, nothing)
end
function Base.iterate(::AbstractWavePacket, ::Nothing)
    return nothing
end

# Base.:*(G1::AbstractWavePacket{D}, G2::AbstractWavePacket{D}) where D = *(promote(G1, G2)...)
# convolution(G1::AbstractWavePacket{D}, G2::AbstractWavePacket{D}) where D = convolution(promote(G1, G2)...)
# dot_L2(G1::AbstractWavePacket{D}, G2::AbstractWavePacket{D}) where D = dot_L2(promote(G1, G2)...)
norm_L2(G::AbstractWavePacket) = sqrt(norm2_L2(G))
norm_∇(G::AbstractWavePacket) = sqrt(norm2_∇(G))
norm_∇(G::AbstractWavePacket, c::AbstractVector) = sqrt(norm2_∇(G, c))
∫(G::AbstractWavePacket) = integral(G)
(G::AbstractWavePacket{D})(x::T...) where{D, T<:Number} = G(SVector{D}(x...))