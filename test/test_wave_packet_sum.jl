
function test_wave_packet_sum()
    @testset "WavePacketSum" begin
        tol = 1e-6

        # Mixed absolute/relative error (some quantities can be small)
        err_rel(a, b) = abs(a - b) / (1 + abs(b))

        # Allocations are measured with alloc_of (see test.jl). Only the sums stored in a Tuple
        # are measured (the ones stored in a Vector with an abstract element type necessarily allocate)

        rand_gauss() = GaussianWavePacket(rand() + 1im * rand(), 4 * rand() + 0.5 + im * (4 * rand() - 2),
                                          4 * (rand() - 0.5), 4 * (rand() - 0.5))
        rand_hermite(N) = HermiteWavePacket1D(SVector{N}(rand(N) + im * rand(N)), 4 * rand() + 0.5 + im * (4 * rand() - 2),
                                              4 * (rand() - 0.5), 4 * (rand() - 0.5))
        rand_x() = SVector(4.0 * (rand() - 0.5))

        # Terms (1D: gaussians and hermite wave packets can be mixed)
        G = [rand_gauss() for _ in 1:3]
        H = [rand_hermite(n) for n in (3, 2, 4)]
        w = rand() + 1im * rand()

        # Different containers
        SG = G[1] + G[2] + G[3]                                                 # Tuple
        SH = H[1] + H[2] + H[3]                                                 # Tuple, different types
        SM = G[1] + H[1] + G[2] + H[2]                                          # Tuple, mixed
        SV = WavePacketSum([rand_gauss() for _ in 1:20])                        # Vector, concrete type
        SA = WavePacketSum(AbstractWavePacket{1}[G[1], H[1], G[2], H[3]])       # Vector, abstract type

        terms(S) = collect(S.g)
        sum_terms(f, S) = sum(f(g) for g in terms(S))

        @testset "evaluate" begin
            err = 0.0
            alloc = 0
            for S in (SG, SH, SM, SV, SA), _=1:10
                x = rand_x()
                S.g isa Tuple && (alloc += alloc_of(S, x))
                err = max(err, err_rel(S(x), sum_terms(g -> g(x), S)))
                err = max(err, err_rel(S(x[1]), sum_terms(g -> g(x), S)))
                err = max(err, err_rel(evaluate(S, x), sum_terms(g -> g(x), S)))
            end

            @test err <= tol
            report_alloc("evaluate", alloc)
        end

        @testset "linear operations" begin
            err = 0.0
            alloc = 0
            for S in (SG, SH, SM, SV, SA)
                x = rand_x()
                S.g isa Tuple && (alloc += alloc_of(*, w, S) + alloc_of(/, S, w) + alloc_of(-, S) + alloc_of(conj, S))
                err = max(err, err_rel((w * S)(x), w * S(x)))
                err = max(err, err_rel((S * w)(x), w * S(x)))
                err = max(err, err_rel((S / w)(x), S(x) / w))
                err = max(err, err_rel((-S)(x), -S(x)))
                err = max(err, err_rel(conj(S)(x), conj(S(x))))
                err = max(err, err_rel(zero(S)(x), 0))
                err = max(err, err_rel(zero(typeof(S))(x), 0))
                err = max(err, err_rel(copy(S)(x), S(x)))
            end

            @test err <= tol
            report_alloc("linear operations", alloc)
        end

        @testset "product" begin
            err = 0.0
            alloc = 0
            for (S1, S2) in ((SG, SH), (SM, SG), (SV, SA), (SH, SH))
                x = rand_x()
                S1.g isa Tuple && S2.g isa Tuple && (alloc += alloc_of(*, S1, S2))
                err = max(err, err_rel((S1 * S2)(x), S1(x) * S2(x)))
                err = max(err, err_rel((S1 - S2)(x), S1(x) - S2(x)))
                err = max(err, err_rel((S1 + S2)(x), S1(x) + S2(x)))
            end
            for S in (SG, SH, SM)
                x = rand_x()
                err = max(err, err_rel((G[1] * S)(x), G[1](x) * S(x)))
                err = max(err, err_rel((S * H[1])(x), H[1](x) * S(x)))
                alloc += alloc_of(*, G[1], S) + alloc_of(*, S, H[1])
            end

            @test err <= tol
            report_alloc("product", alloc)
        end

        @testset "integral and Fourier" begin
            err = 0.0
            alloc = 0
            for S in (SG, SH, SM, SA)
                ξ = rand_x()
                S.g isa Tuple && (alloc += alloc_of(integral, S) + alloc_of(fourier, S) + alloc_of(inv_fourier, S))
                err = max(err, err_rel(integral(S), sum_terms(integral, S)))
                err = max(err, err_rel(fourier(S)(ξ), sum_terms(g -> fourier(g)(ξ), S)))
                err = max(err, err_rel(inv_fourier(fourier(S))(ξ), S(ξ)))
            end

            @test err <= tol
            report_alloc("integral and Fourier", alloc)
        end

        @testset "convolution" begin
            err = 0.0
            alloc = 0
            x0 = rand_x()
            for (S1, S2) in ((SG, SG), (SH, SH), (SG, SH))
                alloc += alloc_of(convolution, S1, S2)
                ref = sum(convolution(a, b)(x0) for a in terms(S1), b in terms(S2))
                err = max(err, err_rel(convolution(S1, S2)(x0), ref))
            end
            for S in (SG, SH)
                for g in (G[3], H[3])
                    ref1 = sum(convolution(a, g)(x0) for a in terms(S))
                    ref2 = sum(convolution(g, a)(x0) for a in terms(S))
                    err = max(err, err_rel(convolution(S, g)(x0), ref1))
                    err = max(err, err_rel(convolution(g, S)(x0), ref2))
                    alloc += alloc_of(convolution, S, g) + alloc_of(convolution, g, S)
                end
            end

            @test err <= tol
            report_alloc("convolution", alloc)
        end

        @testset "unitary product" begin
            err = 0.0
            b, q, p = 4 * (rand() - 0.5), 4 * (rand() - 0.5), 4 * (rand() - 0.5)
            alloc = alloc_of(unitary_product, SG, Diagonal(SVector(b)), SVector(q), SVector(p)) + alloc_of(unitary_product, SH, b, q, p)
            # Gaussians and hermite wave packets do not have the same convention for b, q, p
            # (matrix and vectors for the former, scalars for the latter)
            S1 = unitary_product(SG, Diagonal(SVector(b)), SVector(q), SVector(p))
            S2 = unitary_product(SH, b, q, p)
            S3 = unitary_product(SH, b)
            for _=1:10
                x = rand_x()
                err = max(err, err_rel(S1(x), sum_terms(g -> unitary_product(g, Diagonal(SVector(b)), SVector(q), SVector(p))(x), SG)))
                err = max(err, err_rel(S2(x), SH(x) * cis(-b/2 * (x[1] - q)^2) * cis(p * x[1])))
                err = max(err, err_rel(S3(x), SH(x) * cis(-b/2 * x[1]^2)))
            end

            @test err <= tol
            report_alloc("unitary product", alloc)
        end

        @testset "polynomial product" begin
            err = 0.0
            alloc = 0
            NP = 4
            P = SVector{NP}(rand(NP) + im * rand(NP))
            q0 = 4 * (rand() - 0.5)
            for S in (SG, SH)
                alloc += alloc_of(polynomial_product, S, P, q0)
                PS = polynomial_product(S, P, q0)
                for _=1:10
                    x = rand_x()
                    ref = S(x) * sum(P[k] * (x[1] - q0)^(k-1) for k in 1:NP)
                    err = max(err, err_rel(PS(x), ref))
                end
            end

            @test err <= tol
            report_alloc("polynomial product", alloc)
        end

        @testset "dot and norm L²" begin
            err = 0.0
            alloc = 0
            for S in (SG, SH, SM, SV, SA)
                S.g isa Tuple && (alloc += alloc_of(dot_L2, S, S) + alloc_of(norm2_L2, S))
                ref = sum(dot_L2(a, b) for a in terms(S), b in terms(S))
                err = max(err, err_rel(dot_L2(S, S), ref))
                err = max(err, err_rel(norm2_L2(S), real(ref)))
                err = max(err, err_rel(norm_L2(S)^2, real(ref)))
            end
            for S in (SG, SH, SM)
                for g in (G[3], H[3])
                    err = max(err, err_rel(dot_L2(S, g), sum_terms(a -> dot_L2(a, g), S)))
                    err = max(err, err_rel(dot_L2(g, S), sum_terms(a -> dot_L2(g, a), S)))
                    alloc += alloc_of(dot_L2, S, g) + alloc_of(dot_L2, g, S)
                end
            end

            @test err <= tol
            report_alloc("dot and norm L²", alloc)
        end

        @testset "dot and norm ∇" begin
            err = 0.0
            alloc = 0
            c = SVector(4 * (rand() - 0.5))
            for S in (SG, SV)
                S.g isa Tuple && (alloc += alloc_of(dot_∇, S, S) + alloc_of(norm2_∇, S) + alloc_of(dot_∇, S, S, c) + alloc_of(norm2_∇, S, c)
                                          + alloc_of(dot_∇, S, G[3]) + alloc_of(dot_∇, G[3], S))
                ref = sum(dot_∇(a, b) for a in terms(S), b in terms(S))
                err = max(err, err_rel(dot_∇(S, S), ref))
                err = max(err, err_rel(norm2_∇(S), real(ref)))
                err = max(err, err_rel(dot_∇(S, G[3]), sum_terms(a -> dot_∇(a, G[3]), S)))
                err = max(err, err_rel(dot_∇(G[3], S), sum_terms(a -> dot_∇(G[3], a), S)))

                ref_c = sum(dot_∇(a, b, c) for a in terms(S), b in terms(S))
                err = max(err, err_rel(dot_∇(S, S, c), ref_c))
                err = max(err, err_rel(norm2_∇(S, c), real(ref_c)))
                err = max(err, err_rel(norm_∇(S, c)^2, real(ref_c)))
            end

            @test err <= tol
            report_alloc("dot and norm ∇", alloc)
        end

        @testset "dot and norm x, moments" begin
            err = 0.0
            alloc = 0
            c = SVector(4 * (rand() - 0.5))
            for S in (SG, SV)
                S.g isa Tuple && (alloc += alloc_of(dot_x, S, S) + alloc_of(norm2_x, S) + alloc_of(dot_x, S, S, c) + alloc_of(norm2_x, S, c)
                                          + alloc_of(dot_x, S, G[3]) + alloc_of(dot_x, G[3], S)
                                          + alloc_of(moment_x, S, S) + alloc_of(moment_D, S, S) + alloc_of(moment_x, S) + alloc_of(moment_D, S)
                                          + alloc_of(moment_x, S, G[3]) + alloc_of(moment_D, G[3], S))
                ref = sum(dot_x(a, b) for a in terms(S), b in terms(S))
                err = max(err, err_rel(dot_x(S, S), ref))
                err = max(err, err_rel(norm2_x(S), real(ref)))
                err = max(err, err_rel(dot_x(S, G[3]), sum_terms(a -> dot_x(a, G[3]), S)))
                err = max(err, err_rel(dot_x(G[3], S), sum_terms(a -> dot_x(G[3], a), S)))
                ref_c = sum(dot_x(a, b, c) for a in terms(S), b in terms(S))
                err = max(err, err_rel(dot_x(S, S, c), ref_c))
                err = max(err, err_rel(norm2_x(S, c), real(ref_c)))
                err = max(err, err_rel(norm_x(S, c)^2, real(ref_c)))

                for f in (moment_x, moment_D)
                    ref = sum(f(a, b) for a in terms(S), b in terms(S))
                    err = max(err, err_rel(f(S, S)[1], ref[1]))
                    err = max(err, err_rel(f(S)[1], real(ref[1])))
                    err = max(err, err_rel(f(S, G[3])[1], sum_terms(a -> f(a, G[3])[1], S)))
                    err = max(err, err_rel(f(G[3], S)[1], sum_terms(a -> f(G[3], a)[1], S)))
                end
                @test moment_x(S) isa SVector{1, Float64} && moment_D(S) isa SVector{1, Float64}
                for f in (moment_xx, moment_xD)
                    ref = sum(f(a, b) for a in terms(S), b in terms(S))
                    err = max(err, err_rel(f(S, S)[1], ref[1]))
                    err = max(err, err_rel(f(S)[1], real(ref[1])))
                    err = max(err, err_rel(f(S, G[3])[1], sum_terms(a -> f(a, G[3])[1], S)))
                end
                # Shifts
                c2 = SVector(4 * (rand() - 0.5))
                for (f, args) in ((moment_x, (c,)), (moment_D, (c,)), (moment_xx, (c,)), (moment_xD, (c, c2)))
                    ref = sum(f(a, b, args...) for a in terms(S), b in terms(S))
                    err = max(err, err_rel(f(S, S, args...)[1], ref[1]))
                    err = max(err, err_rel(f(S, args...)[1], real(ref[1])))
                    err = max(err, err_rel(f(G[3], S, args...)[1], sum_terms(a -> f(G[3], a, args...)[1], S)))
                end
                S.g isa Tuple && (alloc += alloc_of(moment_xx, S, S) + alloc_of(moment_xD, S) + alloc_of(gaussian_hull, S)
                                  + alloc_of(moment_x, S, c) + alloc_of(moment_xD, S, S, c, c2))
            end

            @test err <= tol
            report_alloc("dot and norm x, moments", alloc)
        end

        @testset "nested sums" begin
            err = 0.0
            leaves = AbstractWavePacket{1}[G[1], G[2], H[1], H[2], H[3]]
            S = (G[1] + G[2]) + (H[1] + H[2] + H[3])        # sum of two sums
            T = S + G[3]                                      # sum of a sum and a wave packet
            leaves_T = [leaves; G[3]]
            ref_ST = sum(dot_L2(a, b) for a in leaves, b in leaves_T)
            for _=1:10
                x = rand_x()
                err = max(err, err_rel(S(x), sum(g(x) for g in leaves)))
                err = max(err, err_rel(T(x), S(x) + G[3](x)))
            end
            ref_SS = sum(dot_L2(a, b) for a in leaves, b in leaves)
            err = max(err, err_rel(dot_L2(S, S), ref_SS))
            err = max(err, err_rel(norm2_L2(S), real(ref_SS)))
            err = max(err, err_rel(dot_L2(S, T), ref_ST))
            err = max(err, err_rel(norm2_L2(T), real(sum(dot_L2(a, b) for a in leaves_T, b in leaves_T))))

            SGN = (G[1] + G[2]) + G[3]                        # gaussians only (the gradient is not defined for the others)
            ref_∇ = sum(dot_∇(a, b) for a in G, b in G)
            err = max(err, err_rel(dot_∇(SGN, SGN), ref_∇))
            err = max(err, err_rel(norm2_∇(SGN), real(ref_∇)))
            ref_x = sum(dot_x(a, b) for a in G, b in G)
            err = max(err, err_rel(dot_x(SGN, SGN), ref_x))
            err = max(err, err_rel(norm2_x(SGN), real(ref_x)))
            ref_m = sum(moment_D(a, b) for a in G, b in G)
            err = max(err, err_rel(moment_D(SGN, SGN)[1], ref_m[1]))
            err = max(err, err_rel(moment_D(SGN)[1], real(ref_m[1])))
            err = max(err, err_rel(integral(S), sum(integral(g) for g in leaves)))

            @test err <= tol
        end

        @testset "unsupported operations" begin
            # The operations not defined for the terms give a clean error (no infinite recursion)
            @test_throws MethodError dot_∇(H[1], H[2])
            @test_throws MethodError dot_∇(SH, SH)
            @test_throws MethodError norm2_∇(SH)
            @test_throws MethodError dot_x(SH, SH)
            @test_throws MethodError moment_x(SH)
        end

        @testset "display" begin
            shown = sprint(show, SV)
            @test occursin("20 terms", shown)
            @test occursin("more", shown)
            @test occursin("2 terms", sprint(show, G[1] + H[1]))
            @test occursin("1 term:", sprint(show, WavePacketSum(G[1])))
        end

        @testset "Constructor checks" begin
            @test_throws MethodError WavePacketSum{2}(G[1], G[2])
            @test_throws ArgumentError WavePacketSum{1}([1.0, 2.0])
            @test WavePacketSum{1}(G[1]) isa WavePacketSum{1}
        end

        @testset "scalar evaluation" begin
            err = 0.0
            alloc = 0
            for S in (SG, SH, SM, SV, SA), _=1:10
                x = rand_x()
                err = max(err, err_rel(evaluate(S, x[1]), S(x)), err_rel(S(x[1]), S(x)))
                S.g isa Tuple && (alloc += alloc_of(evaluate, S, x[1]))
            end

            @test err <= tol
            report_alloc("scalar evaluation", alloc)
        end

        @testset "dimension selectors" begin
            err = 0.0
            alloc = 0
            S2 = rand_gwp(Val(3), :sym, true, true, true) + rand_gwp(Val(3), :diag, true, true, false) + rand_gwp(Val(3), :sym, false, false, true)
            for _=1:10
                x = SVector{3}(4 .* (rand(3) .- 0.5))
                Sx = evaluate(S2, x[SVector(3, 1)]; dims=(3, 1))
                err = max(err, err_rel(Sx(x[SVector(2)]), S2(x)))
                err = max(err, err_rel(evaluate(S2, x[SVector(2, 3, 1)]; dims=(2, 3, 1)), S2(x)))
                Si = integral(S2; dims=(2,))
                err = max(err, err_rel(Si(x[SVector(1, 3)]), sum(integral(g; dims=(2,))(x[SVector(1, 3)]) for g in S2.g)))
            end
            err = max(err, err_rel(integral(S2; dims=(1, 2, 3)), integral(S2)))
            alloc += alloc_of(S -> integral(S; dims=(2,)), S2) + alloc_of((S, x) -> evaluate(S, x; dims=(3, 1)), S2, SVector(0.1, 0.2))

            @test err <= tol
            report_alloc("dimension selectors", alloc)
        end

        @testset "containers" begin
            # Static vector
            SS = WavePacketSum(SVector(G[1], G[2]))
            x = rand_x()
            @test SS(x) ≈ G[1](x) + G[2](x)
            @test zero(typeof(SS))(x) == 0
            @test norm2_L2(SS) ≈ norm2_L2(G[1] + G[2])
            # Empty sums
            E = WavePacketSum{1}(typeof(G[1])[])
            @test E(x) == 0 && integral(E) == 0 && norm2_L2(E) == 0
            @test zero(typeof(SV))(x) == 0
            EA = WavePacketSum{1}(AbstractWavePacket{1}[])
            @test EA(x) == 0 && integral(EA) == 0
        end

        @testset "inference" begin
            # Sums of sums (stored in tuples) must be inferred, including the nested reductions
            S = (G[1] + H[1]) + (G[2] + (H[2] + G[3]))
            x = rand_x()
            @test S(x) ≈ sum(g(x) for g in (G[1], H[1], G[2], H[2], G[3]))
            @test is_inferred(S, x)
            @test is_inferred(S, x[1])
            @test is_inferred(integral, S)
            @test is_inferred(dot_L2, S, S)
            @test is_inferred(dot_L2, S, G[1])
            @test is_inferred(dot_L2, H[1], S)
            @test is_inferred(norm2_L2, S)
            @test is_inferred(fourier, S)
            @test is_inferred(convolution, S, G[1])
            # (S * S itself has a huge type: 25 products of nested terms)
            @test is_inferred(*, G[1] + H[1], G[2] + (H[2] + G[3]))
            SGN = (G[1] + G[2]) + G[3]
            @test is_inferred(dot_∇, SGN, SGN)
            @test is_inferred(norm2_∇, SGN)
            @test is_inferred(norm2_∇, SGN, SVector(0.3))
            @test is_inferred(dot_∇, SGN, SGN, SVector(0.3))
            @test is_inferred(dot_∇, G[1], SGN, SVector(0.3))
            @test is_inferred(dot_x, SGN, SGN)
            @test is_inferred(norm2_x, SGN)
            @test is_inferred(dot_x, G[1], SGN, SVector(0.3))
            @test is_inferred(norm2_x, SGN, SVector(0.3))
            @test is_inferred(moment_x, SGN, SGN)
            @test is_inferred(moment_x, SGN)
            @test is_inferred(moment_D, G[1], SGN)
            @test is_inferred(moment_D, SGN)
            @test is_inferred(moment_xx, SGN)
            @test is_inferred(moment_xD, G[1], SGN)
            @test is_inferred(gaussian_hull, SGN)
            @test is_inferred(moment_D, SGN, SGN, SVector(0.3))
            @test is_inferred(moment_xD, SGN, SVector(0.3), SVector(0.1))
            report_alloc("nested sums (evaluate, dot_L2, norm2_L2, norm2_∇)",
                         alloc_of(evaluate, S, x) + alloc_of(dot_L2, S, S) + alloc_of(norm2_L2, S) + alloc_of(norm2_∇, SGN))
            report_alloc("nested sums (dot_x, norm2_x, moment_x, moment_D)",
                         alloc_of(dot_x, SGN, SGN) + alloc_of(norm2_x, SGN) + alloc_of(moment_x, SGN) + alloc_of(moment_D, SGN, SGN)
                         + alloc_of(gaussian_hull, SGN))
        end
    end
end
