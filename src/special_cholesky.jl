#=
    Computes the analytic extension of the Cholesky decomposition (UᵀU=A, without conjugation)
    of a complex symmetric matrix A whose real part is positive definite.
    Returns the upper triangular factor U (whose diagonal has a positive real part).
    Throws a PosDefException if a pivot does not have a (strictly) positive real part.
=#

# Diagonal
function special_cholesky(D::SDiagonal)
    any(x -> !(real(x) > zero(real(x))), D.diag) && throw(LinearAlgebra.PosDefException(1))
    C = sqrt.(D.diag)
    return SDiagonal(C)
end

# Dense
function special_cholesky(A::Symmetric{<:Number, <:StaticMatrix})
    # The algorithm reads the upper triangle
    data = A.uplo == 'U' ? SMatrix(parent(A)) : SMatrix(transpose(parent(A)))
    _special_cholesky(Size(A), data)
end

@inline function _sp_chol_failure(info)
    throw(LinearAlgebra.PosDefException(info))
end
# The pivots must have a positive real part (NaN fails the check)
@inline _sp_nonpdcheck(x::Number) = real(x) > zero(real(x))
@inline _sp_nonpdcheck(x) = x == x

#=
    Adapted from github.com/JuliaArrays/StaticArrays.jl, which is distributed
    under the following MIT license:

    Copyright (c) 2016: Andy Ferris.

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files (the
    "Software"), to deal in the Software without restriction, including
    without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to
    the following conditions:

    The above copyright notice and this permission notice shall be
    included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
    SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
=#
@generated function _special_cholesky(::Size{S}, A::StaticMatrix{M, M}) where{S, M}
    @assert (M,M) == S
    # M > 24 && return :(_cholesky_large(Size{$S}(), A, check))
    q = Expr(:block)
    for n ∈ 1:M
        for m ∈ n:M
            L_m_n = Symbol(:L_,m,:_,n)
            push!(q.args, :($L_m_n = @inbounds A[$n, $m]))
        end
        for k ∈ 1:n-1, m ∈ n:M
            L_m_n = Symbol(:L_,m,:_,n)
            L_m_k = Symbol(:L_,m,:_,k)
            L_n_k = Symbol(:L_,n,:_,k)
            push!(q.args, :($L_m_n = muladd(-$L_m_k, $L_n_k, $L_m_n)))
        end
        L_n_n = Symbol(:L_,n,:_,n)
        push!(q.args, :(_sp_nonpdcheck($L_n_n) || return _sp_chol_failure($n)))
        push!(q.args, :($L_n_n = sqrt($L_n_n)))
        Linv_n_n = Symbol(:Linv_,n,:_,n)
        push!(q.args, :($Linv_n_n = inv($L_n_n)))
        for m ∈ n+1:M
            L_m_n = Symbol(:L_,m,:_,n)
            push!(q.args, :($L_m_n *= $Linv_n_n))
        end
    end
    # Element type of the result: the promoted type of the entries of the factor
    # (and not eltype(A), which may be an integer type)
    push!(q.args, :(T = promote_type($((:(typeof($(Symbol(:L_,m,:_,n)))) for n ∈ 1:M for m ∈ n:M)...))))
    ret = Expr(:tuple)
    for n ∈ 1:M
        for m ∈ 1:n
            push!(ret.args, Symbol(:L_,n,:_,m))
        end
        for m ∈ n+1:M
            push!(ret.args, :(zero(T)))
        end
    end
    push!(q.args, :(UpperTriangular(SMatrix{M, M, T}($ret))))
    return Expr(:block, Expr(:meta, :inline), q)
end