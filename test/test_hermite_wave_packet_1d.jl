
# Explicit formula ∑ₙ Λ[n+1] ψₙ(z, q, p, x)
function hermite_formula(H::HermiteWavePacket1D, x)
    q = H.q isa NullNumber ? 0.0 : H.q
    p = H.p isa NullNumber ? 0.0 : H.p
    return sum(H.Λ[n+1] * ψ(H.z, q, p, n, x) for n in 0:length(H.Λ)-1)
end
# A random point close to the center of H
near(H::HermiteWavePacket1D) = center(H)[1] + randn() / sqrt(real(H.z))
# The integral of |H| (scale for the comparisons with a cubature)
function abs_integral(H::HermiteWavePacket1D)
    lo, hi = integration_box((center(H),), real(H.z); nσ=12)
    return hcubature(y -> abs(H(y[1])), lo, hi; reltol=1e-8)[1]
end
function _test_hwp_evaluate(nb_reps, tol, tol_cub)
    @testset "evaluate" begin
        err = 0.0
        alloc = 0
        for N in (1, 2, 6), _ in 1:nb_reps, H in hermite_combos(N)
            s = norm_L2(H) * real(H.z)^(1/4)
            for _ in 1:5
                x = near(H)
                err = max(err, abs(H(x) - hermite_formula(H, x)) / s)
                err = max(err, abs(H(SVector(x)) - hermite_formula(H, x)) / s)
                err = max(err, abs(evaluate(H, x) - hermite_formula(H, x)) / s)
            end
            alloc += alloc_of(evaluate, H, near(H))
        end

        @test err <= tol
        report_alloc("evaluate", alloc)
    end
end

function _test_hwp_evaluate_grid(nb_reps, tol, tol_cub)
    @testset "evaluate grid" begin
        err = 0.0
        alloc = 0
        for N in (1, 5), H in hermite_combos(N)
            xgrid = hermite_grid(4 * rand() + 0.5, rand_scalar(rand(Bool)), Val(8))
            A = evaluate_grid(H, xgrid)
            for j in eachindex(xgrid)
                err = max(err, err_rel(A[j], H(xgrid[j]); floor=norm_L2(H) * 1e-6))
            end
            alloc += alloc_of(evaluate_grid, H, xgrid)
        end

        @test err <= tol
        report_alloc("evaluate grid", alloc)
    end
end

function _test_hwp_conversions_and_truncation(nb_reps, tol, tol_cub)
    @testset "conversions and truncation" begin
        err = 0.0
        alloc = 0
        for G in gwp_combos(Val(1))
            H = HermiteWavePacket1D(G)
            G_ = truncate_to_gaussian(H)
            # Conversion to a larger type, and promotion
            H2 = HermiteWavePacket1D(SVector{3}(rand(3) + im * rand(3)), complex(G.z[1, 1]), 0.1, 0.2)
            Hc = convert(typeof(H2), G)
            Hp = [G, H2]
            @test eltype(Hp) === typeof(H2)
            for _ in 1:10
                x = near(H)
                err = max(err, err_rel(H(x), G(x)), err_rel(G_(x), G(x)), err_rel(Hc(x), G(x)))
                err = max(err, err_rel(Hp[1](x), G(x)), err_rel(Hp[2](x), H2(x)))
            end
            alloc += alloc_of(HermiteWavePacket1D, G) + alloc_of(truncate_to_gaussian, H)
        end
        # z cannot be a NullNumber
        @test_throws ArgumentError HermiteWavePacket1D(SVector(1.0, 2.0), NullNumber())
        @test_throws ArgumentError HermiteWavePacket1D(1.0, NullNumber(), 0.1, 0.2)
        Hg = HermiteFct1D(Gaussian(2.0, 1.5, 0.3))
        @test Hg isa HermiteFct1D && Hg(0.2) ≈ Gaussian(2.0, 1.5, 0.3)(0.2)
        # truncate
        for H in hermite_combos(5)
            H3 = truncate_hermite(H, Val(3))
            @test H3 isa HermiteWavePacket1D{3} && H3.Λ == H.Λ[SVector(1, 2, 3)]
            @test truncate_hermite(H, Val(5)).Λ == H.Λ
            H7 = truncate_hermite(H, Val(7))
            @test H7.Λ == [H.Λ; 0; 0] && (H7.z, H7.q, H7.p) === (H.z, H.q, H.p)
            @test norm2_L2(H) - norm2_L2(H3) ≈ sum(abs2, H.Λ[SVector(4, 5)])
            @test truncate_hermite(H, Val(1))(0.3) ≈ truncate_to_gaussian(H)(0.3)
            # truncate_hermite is not a method of Base.truncate
            @test !hasmethod(Base.truncate, Tuple{typeof(H), Val{3}})
            alloc += alloc_of(truncate_hermite, H, Val(3))
        end

        @test err <= tol
        report_alloc("conversions and truncation", alloc)
    end
end

function _test_hwp_integral(nb_reps, tol, tol_cub)
    @testset "integral" begin
        err = 0.0
        err_id = 0.0
        alloc = 0
        for N in (1, 4, 7), H in hermite_combos(N)
            lo, hi = integration_box((center(H),), real(H.z))
            I1 = complex_cubature(y -> H(y[1]), lo, hi)
            err = max(err, abs(integral(H) - I1) / abs_integral(H))
            alloc += alloc_of(integral, H)
        end
        for N in (1, 4, 7, 20), _ in 1:nb_reps, H in hermite_combos(N)
            # ∫H is the Fourier transform at 0 (evaluated at the Float64 0.0, not at a NullNumber)
            err_id = max(err_id, abs(integral(H) - fourier(H)(0.0)) / norm_L2(H))
            err_id = max(err_id, abs(∫(H) - integral(H)) / norm_L2(H))
        end

        @test err <= tol_cub
        @test err_id <= tol
        report_alloc("integral", alloc)
    end
end

function _test_hwp_product(nb_reps, tol, tol_cub)
    @testset "product" begin
        err = 0.0
        alloc = 0
        for _ in 1:nb_reps
            combos1 = hermite_combos(4)
            combos2 = shuffle(hermite_combos(3))
            for (H1, H2) in zip(combos1, combos2)
                H = H1 * H2
                @test H isa HermiteWavePacket1D{6}
                s = sqrt(norm_L2(H1) * norm_L2(H2)) * (real(H1.z) * real(H2.z))^(1/8)
                for _ in 1:10
                    x = near(H)
                    err = max(err, abs(H(x) - H1(x) * H2(x)) / s^2)
                end
                alloc += alloc_of(*, H1, H2)
            end
            # With gaussians
            for (H, G) in zip(hermite_combos(3), gwp_combos(Val(1)))
                x = near(H)
                s = norm_L2(H) * real(H.z)^(1/4) * abs(G.λ)
                err = max(err, abs((G * H)(x) - G(x) * H(x)) / s, abs((H * G)(x) - G(x) * H(x)) / s)
            end
        end
        # Products of hermite functions (no q), which used to fail
        H1 = HermiteFct1D(SVector(1.0, 2.0), 1.0)
        H2 = HermiteFct1D(SVector(0.5, 0.2, 0.1), 2.0)
        @test (H1 * H2) isa HermiteFct1D{4}
        @test (H1 * H2)(0.4) ≈ H1(0.4) * H2(0.4)

        @test err <= tol
        report_alloc("product", alloc)
    end
end

function _test_hwp_unitary_product(nb_reps, tol, tol_cub)
    @testset "unitary product" begin
        err = 0.0
        alloc = 0
        for N in (1, 6), H in hermite_combos(N)
            b = 4 * (rand() - 0.5)
            q2 = rand_scalar(rand(Bool))
            p2 = rand_scalar(rand(Bool))
            c2 = q2 isa NullNumber ? 0.0 : q2
            d2 = p2 isa NullNumber ? 0.0 : p2
            H2 = unitary_product(H, b, q2, p2)
            H3 = unitary_product(H, b)
            s = norm_L2(H) * real(H.z)^(1/4)
            for _ in 1:10
                x = near(H)
                err = max(err, abs(H2(x) - H(x) * cis(-b/2 * (x - c2)^2) * cis(d2 * x)) / s)
                err = max(err, abs(H3(x) - H(x) * cis(-b/2 * x^2)) / s)
            end
            alloc += alloc_of(unitary_product, H, b, q2, p2)
        end

        @test err <= tol
        report_alloc("unitary product", alloc)
    end
end

function _test_hwp_polynomial_product(nb_reps, tol, tol_cub)
    @testset "polynomial product" begin
        err = 0.0
        alloc = 0
        NP = 4
        for N in (1, 6), _ in 1:nb_reps, H in hermite_combos(N)
            P = SVector{NP}(rand(NP) + im * rand(NP))
            q2 = rand_scalar(rand(Bool))
            c2 = q2 isa NullNumber ? 0.0 : q2
            PH = polynomial_product(H, P, q2)
            @test PH isa HermiteWavePacket1D{N + NP - 1}
            f(x) = H(x) * sum(P[k] * (x - c2)^(k-1) for k in 1:NP)
            for _ in 1:10
                x = near(H)
                err = max(err, abs(f(x) - PH(x)) / (norm_L2(PH) * real(H.z)^(1/4)))
            end
            # Without q
            PH0 = polynomial_product(H, P)
            x = near(H)
            err = max(err, abs(H(x) * sum(P[k] * x^(k-1) for k in 1:NP) - PH0(x)) / (norm_L2(PH0) * real(H.z)^(1/4)))
            alloc += alloc_of(polynomial_product, H, P, q2)
        end
        # With a gaussian
        G = rand_gwp(Val(1), :diag, true, true, true)
        P = SVector(1.0, 2.0, 0.5)
        @test polynomial_product(G, P, 0.3)(0.2) ≈ G(0.2) * (1 + 2 * (0.2 - 0.3) + 0.5 * (0.2 - 0.3)^2)

        @test err <= tol
        report_alloc("polynomial product", alloc)
    end
end

function _test_hwp_fourier(nb_reps, tol, tol_cub)
    @testset "Fourier" begin
        err = 0.0
        err_id = 0.0
        alloc = 0
        for N in (1, 6), H in hermite_combos(N)
            HF = fourier(H)
            ξ = near(HF)
            lo, hi = integration_box((center(H),), real(H.z))
            I1 = complex_cubature(y -> H(y[1]) * cis(-y[1] * ξ), lo, hi)
            err = max(err, abs(HF(ξ) - I1) / abs_integral(H))
        end
        for N in (1, 6, 15), _ in 1:nb_reps, H in hermite_combos(N)
            H1 = inv_fourier(fourier(H))
            H2 = fourier(inv_fourier(H))
            s = norm_L2(H) * real(H.z)^(1/4)
            # Real variance: same result as with the complex variance formulas
            Hc = HermiteWavePacket1D(H.Λ, complex(H.z), H.q, H.p)
            for _ in 1:5
                x = near(H)
                err_id = max(err_id, abs(H1(x) - H(x)) / s, abs(H2(x) - H(x)) / s)
                ξ = near(fourier(H))
                sF = norm_L2(H) * sqrt(2π) * real(inv(H.z))^(1/4)
                err_id = max(err_id, abs(fourier(H)(ξ) - fourier(Hc)(ξ)) / sF)
                err_id = max(err_id, abs(inv_fourier(H)(ξ) - inv_fourier(Hc)(ξ)) / sF)
            end
            alloc += alloc_of(fourier, H) + alloc_of(inv_fourier, H)
        end

        @test err <= tol_cub
        @test err_id <= tol
        report_alloc("Fourier", alloc)
    end
end

function _test_hwp_convolution(nb_reps, tol, tol_cub)
    @testset "convolution" begin
        err = 0.0
        err_id = 0.0
        alloc = 0
        for (H1, H2) in zip(hermite_combos(3), shuffle(hermite_combos(4)))
            H = convolution(H1, H2)
            x0 = near(H)
            amin = min(real(H1.z), real(H2.z))
            lo, hi = integration_box((center(H1), SVector(x0) - center(H2)), amin)
            I1 = complex_cubature(y -> H1(y[1]) * H2(x0 - y[1]), lo, hi)
            err = max(err, abs(H(x0) - I1) / (abs_integral(H1) * abs_integral(H2)))
        end
        # Convolution theorem
        for _ in 1:nb_reps, (H1, H2) in zip(hermite_combos(3), shuffle(hermite_combos(4)))
            F1, F2 = fourier(H1), fourier(H2)
            FH = fourier(convolution(H1, H2))
            s = norm_L2(F1 * F2) * real((F1 * F2).z)^(1/4)
            for _ in 1:5
                ξ = near(F1 * F2)
                err_id = max(err_id, abs(FH(ξ) - F1(ξ) * F2(ξ)) / s)
            end
            alloc += alloc_of(convolution, H1, H2)
        end
        # With gaussians
        G = rand_gwp(Val(1), :diag, true, true, true)
        H = rand_hermite(3, true, true, true)
        @test convolution(G, H)(0.3) ≈ convolution(HermiteWavePacket1D(G), H)(0.3)
        @test convolution(H, G)(0.3) ≈ convolution(H, HermiteWavePacket1D(G))(0.3)

        @test err <= tol_cub
        @test err_id <= tol
        report_alloc("convolution", alloc)
    end
end

function _test_hwp_dot_L2(nb_reps, tol, tol_cub)
    @testset "dot L²" begin
        err = 0.0
        err_id = 0.0
        alloc = 0
        for (H1, H2) in zip(hermite_combos(5), shuffle(hermite_combos(4)))
            amin = min(real(H1.z), real(H2.z))
            lo, hi = integration_box((center(H1), center(H2)), amin)
            I1 = complex_cubature(y -> conj(H1(y[1])) * H2(y[1]), lo, hi)
            err = max(err, abs(dot_L2(H1, H2) - I1) / (norm_L2(H1) * norm_L2(H2)))
        end
        for _ in 1:nb_reps, (H1, H2) in zip(hermite_combos(5), shuffle(hermite_combos(4)))
            s = norm_L2(H1) * norm_L2(H2)
            I = dot_L2(H1, H2)
            err_id = max(err_id, abs(dot_L2(H2, H1) - conj(I)) / s)
            # Parseval
            err_id = max(err_id, abs(dot_L2(fourier(H1), fourier(H2)) / 2π - I) / s)
            err_id = max(err_id, err_rel(real(dot_L2(H1, H1)), norm2_L2(H1)), abs(imag(dot_L2(H1, H1))) / norm2_L2(H1))
            err_id = max(err_id, err_rel(norm_L2(H1)^2, norm2_L2(H1)))
            alloc += alloc_of(dot_L2, H1, H2) + alloc_of(norm2_L2, H1)
        end
        # With gaussians
        for (H, G) in zip(hermite_combos(3), gwp_combos(Val(1)))
            s = norm_L2(H) * norm_L2(G)
            err_id = max(err_id, abs(dot_L2(G, H) - dot_L2(HermiteWavePacket1D(G), H)) / s)
            err_id = max(err_id, abs(dot_L2(H, G) - conj(dot_L2(G, H))) / s)
        end

        @test err <= tol_cub
        @test err_id <= 1e-11
        report_alloc("dot L²", alloc)
    end
end

function _test_hwp_quadrature_and_transform(nb_reps, tol, tol_cub)
    @testset "quadrature and transform" begin
        err = 0.0
        alloc = 0
        for N in (1, 2, 5, 12, 40), q in (NullNumber(), 0.7)
            a = 4 * rand() + 0.5
            x, w = hermite_quadrature(a, q, Val(N))
            c = q isa NullNumber ? 0.0 : q
            # Exact for g(x) = (x-q)ᵏ exp(-a(x-q)²), k ≤ 2N-1
            for k in 0:2N-1
                ref = iseven(k) ? gamma((k + 1) / 2) / a^((k + 1) / 2) : 0.0
                val = sum(w .* (x .- c).^k .* exp.(-a .* (x .- c).^2))
                err = max(err, abs(val - ref) / (gamma((k + 1) / 2) / a^((k + 1) / 2)))
            end
            # Transform: the coefficients are recovered from the values on the grid
            Λ = SVector{N}(rand(N) + im * rand(N))
            H = HermiteFct1D(Λ, a, q)
            xg = hermite_grid(a, q, Val(N))
            Φ = evaluate_grid(H, xg)
            err = max(err, maximum(abs, hermite_discrete_transform(Φ, a, q, Val(N)) - Λ) / norm(Λ))
            Λ2 = zeros(ComplexF64, N)
            hermite_discrete_transform!(Λ2, Φ, a, q, Val(N))
            err = max(err, maximum(abs, Λ2 - Λ) / norm(Λ))
            alloc += alloc_of(hermite_quadrature, a, q, Val(N)) + alloc_of(hermite_discrete_transform, Φ, a, q, Val(N))
        end

        @test err <= tol
        report_alloc("quadrature and transform", alloc)
    end
end

function test_hermite_wave_packets_1d()
    @testset "HermiteWavePacket1D" begin
        nb_reps = 3
        tol = 1e-12         # identities
        tol_cub = 1e-6      # comparisons with a cubature (relative to ∫|f|)

        _test_hwp_evaluate(nb_reps, tol, tol_cub)
        _test_hwp_evaluate_grid(nb_reps, tol, tol_cub)
        _test_hwp_conversions_and_truncation(nb_reps, tol, tol_cub)
        _test_hwp_integral(nb_reps, tol, tol_cub)
        _test_hwp_product(nb_reps, tol, tol_cub)
        _test_hwp_unitary_product(nb_reps, tol, tol_cub)
        _test_hwp_polynomial_product(nb_reps, tol, tol_cub)
        _test_hwp_fourier(nb_reps, tol, tol_cub)
        _test_hwp_convolution(nb_reps, tol, tol_cub)
        _test_hwp_dot_L2(nb_reps, tol, tol_cub)
        _test_hwp_quadrature_and_transform(nb_reps, tol, tol_cub)
    end
end
