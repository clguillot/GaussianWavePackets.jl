
function test_wave_packet_sum()

    printstyled("Testing wave packet sum:\n"; bold=true, color=:blue)

    tol = 1e-6

    # Mixed absolute/relative error (some quantities can be small)
    err_rel(a, b) = abs(a - b) / (1 + abs(b))
    function report(name, err)
        color = err > tol ? :red : :green
        printstyled("Error $name = $err\n"; bold=true, color=color)
    end

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

    let
        err = 0.0
        for S in (SG, SH, SM, SV, SA), _=1:10
            x = rand_x()
            err = max(err, err_rel(S(x), sum_terms(g -> g(x), S)))
            err = max(err, err_rel(S(x[1]), sum_terms(g -> g(x), S)))
            err = max(err, err_rel(evaluate(S, x), sum_terms(g -> g(x), S)))
        end
        report("evaluate", err)
    end

    let
        err = 0.0
        for S in (SG, SH, SM, SV, SA)
            x = rand_x()
            err = max(err, err_rel((w * S)(x), w * S(x)))
            err = max(err, err_rel((S * w)(x), w * S(x)))
            err = max(err, err_rel((S / w)(x), S(x) / w))
            err = max(err, err_rel((-S)(x), -S(x)))
            err = max(err, err_rel(conj(S)(x), conj(S(x))))
            err = max(err, err_rel(zero(S)(x), 0))
            err = max(err, err_rel(zero(typeof(S))(x), 0))
            err = max(err, err_rel(copy(S)(x), S(x)))
        end
        report("linear operations", err)
    end

    let
        err = 0.0
        for (S1, S2) in ((SG, SH), (SM, SG), (SV, SA), (SH, SH))
            x = rand_x()
            err = max(err, err_rel((S1 * S2)(x), S1(x) * S2(x)))
            err = max(err, err_rel((S1 - S2)(x), S1(x) - S2(x)))
            err = max(err, err_rel((S1 + S2)(x), S1(x) + S2(x)))
        end
        for S in (SG, SH, SM)
            x = rand_x()
            err = max(err, err_rel((G[1] * S)(x), G[1](x) * S(x)))
            err = max(err, err_rel((S * H[1])(x), H[1](x) * S(x)))
        end
        report("product", err)
    end

    let
        err = 0.0
        for S in (SG, SH, SM, SA)
            ξ = rand_x()
            err = max(err, err_rel(integral(S), sum_terms(integral, S)))
            err = max(err, err_rel(fourier(S)(ξ), sum_terms(g -> fourier(g)(ξ), S)))
            err = max(err, err_rel(inv_fourier(fourier(S))(ξ), S(ξ)))
        end
        report("integral and Fourier", err)
    end

    let
        err = 0.0
        x0 = rand_x()
        for (S1, S2) in ((SG, SG), (SH, SH), (SG, SH))
            ref = sum(convolution(a, b)(x0) for a in terms(S1), b in terms(S2))
            err = max(err, err_rel(convolution(S1, S2)(x0), ref))
        end
        for S in (SG, SH)
            for g in (G[3], H[3])
                ref1 = sum(convolution(a, g)(x0) for a in terms(S))
                ref2 = sum(convolution(g, a)(x0) for a in terms(S))
                err = max(err, err_rel(convolution(S, g)(x0), ref1))
                err = max(err, err_rel(convolution(g, S)(x0), ref2))
            end
        end
        report("convolution", err)
    end

    let
        err = 0.0
        b, q, p = 4 * (rand() - 0.5), 4 * (rand() - 0.5), 4 * (rand() - 0.5)
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
        report("unitary product", err)
    end

    let
        err = 0.0
        NP = 4
        P = SVector{NP}(rand(NP) + im * rand(NP))
        q0 = 4 * (rand() - 0.5)
        for S in (SG, SH)
            PS = polynomial_product(S, P, q0)
            for _=1:10
                x = rand_x()
                ref = S(x) * sum(P[k] * (x[1] - q0)^(k-1) for k in 1:NP)
                err = max(err, err_rel(PS(x), ref))
            end
        end
        report("polynomial product", err)
    end

    let
        err = 0.0
        for S in (SG, SH, SM, SV, SA)
            ref = sum(dot_L2(a, b) for a in terms(S), b in terms(S))
            err = max(err, err_rel(dot_L2(S, S), ref))
            err = max(err, err_rel(norm2_L2(S), real(ref)))
            err = max(err, err_rel(norm_L2(S)^2, real(ref)))
        end
        for S in (SG, SH, SM)
            for g in (G[3], H[3])
                err = max(err, err_rel(dot_L2(S, g), sum_terms(a -> dot_L2(a, g), S)))
                err = max(err, err_rel(dot_L2(g, S), sum_terms(a -> dot_L2(g, a), S)))
            end
        end
        report("dot and norm L²", err)
    end

    let
        err = 0.0
        c = SVector(4 * (rand() - 0.5))
        for S in (SG, SV)
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
        report("dot and norm ∇", err)
    end

    let
        # Nested sums are flattened when iterating, and can be displayed
        S = (G[1] + G[2]) + (H[1] + H[2] + H[3])
        n_terms = 0
        for _ in S
            n_terms += 1
        end
        shown = sprint(show, SV)
        ok = (n_terms == 5) && occursin("20 terms", shown) && occursin("more", shown)
        printstyled("Iteration and display: $(ok ? "ok" : "problem")\n"; bold=true, color=ok ? :green : :red)
    end

    let
        # Constructor checks
        ok = true
        ok &= (try WavePacketSum{2}(G[1], G[2]); false catch e; e isa Exception end)
        ok &= (try WavePacketSum{1}([1.0, 2.0]); false catch e; e isa ArgumentError end)
        ok &= (WavePacketSum{1}(G[1]) isa WavePacketSum{1})
        printstyled("Constructor checks: $(ok ? "ok" : "problem")\n"; bold=true, color=ok ? :green : :red)
    end
end
