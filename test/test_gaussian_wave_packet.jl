
# Explicit formula λ exp(-(x-q)ᵀz(x-q)/2) exp(ip⋅x)
function gwp_formula(G::GaussianWavePacket{D}, x) where D
    q = center(G.q)
    p = map(y -> y isa NullNumber ? 0.0 : y, G.p)
    y = SVector{D}(x) - q
    return G.λ * exp(-transpose(y) * SMatrix{D, D}(G.z) * y / 2) * cis(dot(p, x))
end
# A random point close to the center of G (where its values are not too small)
near(G::GaussianWavePacket{D}) where D = center(G) + randn(SVector{D}) / sqrt(min_eig(G.z))
near_p(G::GaussianWavePacket{D}) where D = map(y -> y isa NullNumber ? 0.0 : y, G.p) + randn(SVector{D}) * sqrt(min_eig(G.z))

function _test_gwp_evaluate(nb_reps, tol, tol_cub)
    @testset "evaluate" begin
        err = 0.0
        alloc = 0
        for D in (1, 2, 3), _ in 1:nb_reps, G in gwp_combos(Val(D))
            for _ in 1:5
                x = near(G)
                err = max(err, err_rel(G(x), gwp_formula(G, x)))
                err = max(err, err_rel(evaluate(G, x), gwp_formula(G, x)))
                err = max(err, err_rel(G(x...), gwp_formula(G, x)))
                err = max(err, err_rel(G(Vector(x)), gwp_formula(G, x)))
                D == 1 && (err = max(err, err_rel(G(x[1]), gwp_formula(G, x))))
                D == 1 && (err = max(err, err_rel(evaluate(G, x[1]), gwp_formula(G, x))))
            end
            alloc += alloc_of(evaluate, G, near(G))
            D == 1 && (alloc += alloc_of(evaluate, G, 0.3))
        end
        # Mixed numeric types
        G = rand_gwp(Val(2), :sym, true, true, true)
        err = max(err, err_rel(G(1, 0.5), G(SVector(1.0, 0.5))))
        err = max(err, err_rel(G(1, 0.5f0), G(SVector(1.0, 0.5))))

        @test err <= tol
        report_alloc("evaluate", alloc)
    end
end

function _test_gwp_restriction(nb_reps, tol, tol_cub)
    @testset "restriction" begin
        err = 0.0
        alloc = 0
        for _ in 1:nb_reps, G in gwp_combos(Val(4))
            x = near(G)
            # x[k] is the value of the variable dims[k], the result is a function of the other variables
            for (dims, others) in (((2,), (1, 3, 4)), ((4, 1), (2, 3)), ((3, 2, 4), (1,)))
                xd = x[SVector(dims)]
                xo = x[SVector(others)]
                err = max(err, err_rel(evaluate(G, xd; dims=dims)(xo), G(x)))
                err = max(err, err_rel(evaluate(G, xd, Val(dims))(xo), G(x)))
            end
            err = max(err, err_rel(evaluate(G, x[SVector(4, 2, 1, 3)]; dims=(4, 2, 1, 3)), G(x)))
            @test evaluate(G, SVector{0, Float64}(); dims=()) === G
            alloc += alloc_of((G, y) -> evaluate(G, y; dims=(4, 1)), G, x[SVector(4, 1)])
        end

        @test err <= tol
        report_alloc("restriction", alloc)
    end
end

function _test_gwp_scalar_operations(nb_reps, tol, tol_cub)
    @testset "scalar operations" begin
        err = 0.0
        alloc = 0
        w = rand() + im * rand()
        for D in (1, 3), G in gwp_combos(Val(D))
            x = near(G)
            err = max(err, err_rel(conj(G)(x), conj(G(x))))
            err = max(err, err_rel((-G)(x), -G(x)))
            err = max(err, err_rel((w * G)(x), w * G(x)))
            err = max(err, err_rel((G * w)(x), w * G(x)))
            err = max(err, err_rel((G / w)(x), G(x) / w))
            err = max(err, err_rel(copy(G)(x), G(x)))
            err = max(err, err_rel(abs(G)(x), abs(G(x))))
            err = max(err, err_rel(abs2(G)(x), abs2(G(x))))
            @test zero(G)(x) == 0
            @test zero(typeof(G))(x) == 0
            alloc += alloc_of(conj, G) + alloc_of(*, w, G) + alloc_of(/, G, w) + alloc_of(abs2, G)
        end

        @test err <= tol
        report_alloc("scalar operations", alloc)
    end
end

function _test_gwp_product(nb_reps, tol, tol_cub)
    @testset "product" begin
        err = 0.0
        alloc = 0
        for D in (1, 2, 3), _ in 1:nb_reps
            combos = gwp_combos(Val(D))
            for (G1, G2) in zip(combos, shuffle(combos))
                G = G1 * G2
                for _ in 1:5
                    x = near(G)
                    err = max(err, err_rel(G(x), G1(x) * G2(x)))
                end
                # The structure of z is preserved (dense as soon as one of them is dense)
                @test (G.z isa Diagonal) == (G1.z isa Diagonal && G2.z isa Diagonal)
                alloc += alloc_of(*, G1, G2)
            end
        end

        @test err <= tol
        report_alloc("product", alloc)
    end
end

function _test_gwp_plane_waves(nb_reps, tol, tol_cub)
    @testset "plane waves" begin
        err = 0.0
        for D in (1, 2), G in gwp_combos(Val(D))
            λ = rand() + im * rand()
            p = 4 * (rand(SVector{D}) .- 0.5)
            W = GaussianWavePacket(λ, Diagonal(zeros(SVector{D, NullNumber})), zeros(SVector{D, NullNumber}), p)
            x = near(G)
            err = max(err, err_rel(W(x), λ * cis(dot(p, x))))
            err = max(err, err_rel((W * G)(x), W(x) * G(x)))
            err = max(err, err_rel((G * W)(x), W(x) * G(x)))
            err = max(err, err_rel((W * W)(x), W(x)^2))
            # ∫ conj(W)G and ∫<∇W,∇G> are given by the Fourier transform of G at p
            s = abs(λ) * integral(abs(G))
            err = max(err, abs(dot_L2(W, G) - conj(λ) * fourier(G)(p)) / s)
            err = max(err, abs(dot_∇(W, G) - conj(λ) * sum(abs2, p) * fourier(G)(p)) / (s * (1 + sum(abs2, p))))
            @test_throws ArgumentError integral(W)
            @test_throws ArgumentError fourier(W)
            @test_throws ArgumentError inv_fourier(W)
            @test_throws ArgumentError norm2_∇(W)
            @test_throws ArgumentError norm2_L2(W)
            @test_throws ArgumentError dot_L2(W, W)
            @test_throws ArgumentError dot_∇(W, W)
        end
        W1 = GaussianWavePacket(1.0, NullNumber(), NullNumber(), 0.5)
        @test_throws ArgumentError HermiteWavePacket1D(W1)

        @test err <= tol
    end
end

function _test_gwp_unitary_product(nb_reps, tol, tol_cub)
    @testset "unitary product" begin
        err = 0.0
        alloc = 0
        for D in (1, 3), _ in 1:nb_reps, G in gwp_combos(Val(D))
            b = rand_cov(Val(D), rand((:diag, :sym)), false)
            q2 = rand_vec(Val(D), rand(Bool))
            p2 = rand_vec(Val(D), rand(Bool))
            G2 = unitary_product(G, b, q2, p2)
            c2 = center(q2)
            for _ in 1:5
                x = near(G)
                ref = G(x) * cis(-dot(x - c2, b, x - c2) / 2) * cis(dot(map(y -> y isa NullNumber ? 0.0 : y, p2), x))
                err = max(err, err_rel(G2(x), ref))
            end
            x = near(G)
            err = max(err, err_rel(unitary_product(G, b)(x), G(x) * cis(-dot(x, b, x) / 2)))
            alloc += alloc_of(unitary_product, G, b, q2, p2)
        end

        @test err <= tol
        report_alloc("unitary product", alloc)
    end
end

function _test_gwp_integral(nb_reps, tol, tol_cub)
    @testset "integral" begin
        err = 0.0
        alloc = 0
        for D in (1, 2), G in gwp_combos(Val(D))
            lo, hi = integration_box((center(G),), min_eig(G.z))
            I1 = complex_cubature(y -> G(y), lo, hi)
            err = max(err, abs(integral(G) - I1) / integral(abs(G)))
            alloc += alloc_of(integral, G)
        end
        # Identity: ∫G is the Fourier transform at 0
        err_id = 0.0
        for D in (1, 2, 3), G in gwp_combos(Val(D))
            err_id = max(err_id, err_rel(integral(G), fourier(G)(zeros(SVector{D})); floor=integral(abs(G)) * 1e-6))
            err_id = max(err_id, err_rel(∫(G), integral(G)))
        end

        @test err <= tol_cub
        @test err_id <= tol
        report_alloc("integral", alloc)
    end
end

function _test_gwp_restricted_integral(nb_reps, tol, tol_cub)
    @testset "restricted integral" begin
        err = 0.0
        err_id = 0.0
        alloc = 0
        # Comparison with a cubature in the integrated variable
        for G in gwp_combos(Val(3))
            x = near(G)[SVector(1, 3)]
            Gx = evaluate(G, x; dims=(1, 3))
            lo, hi = integration_box((center(Gx),), min_eig(Gx.z))
            I1 = complex_cubature(y -> Gx(y), lo, hi)
            err = max(err, abs(integral(G; dims=(2,))(x) - I1) / integral(abs(Gx)))
        end
        # Identity: integrating over x₂ is evaluating the Fourier transform at ξ₂ = 0
        for _ in 1:nb_reps, G in gwp_combos(Val(4))
            F = fourier(G)
            for dims in ((2,), (4, 1), (1, 2, 4))
                d = length(dims)
                Gi = integral(G; dims=dims)
                ref = inv_fourier(evaluate(F, zeros(SVector{d}); dims=dims))
                for _ in 1:3
                    x = near(ref)
                    err_id = max(err_id, err_rel(Gi(x), ref(x)))
                end
                @test (Gi.z isa Diagonal) == (G.z isa Diagonal)
            end
            @test integral(G; dims=()) === G
            err_id = max(err_id, err_rel(integral(G; dims=(1, 2, 3, 4)), integral(G)))
            alloc += alloc_of(G -> integral(G; dims=(4, 1)), G)
        end

        @test err <= tol_cub
        @test err_id <= 1e-11
        report_alloc("restricted integral", alloc)
    end
end

function _test_gwp_fourier(nb_reps, tol, tol_cub)
    @testset "Fourier" begin
        err = 0.0
        err_id = 0.0
        alloc = 0
        for D in (1, 2), G in gwp_combos(Val(D))
            GF = fourier(G)
            ξ = near_p(G)
            lo, hi = integration_box((center(G),), min_eig(G.z))
            # (the integrand oscillates: a tighter cubature is needed to reach tol_cub)
            I1 = complex_cubature(y -> G(y) * cis(-dot(ξ, y)), lo, hi; reltol=1e-14, maxevals=10^9)
            err = max(err, abs(GF(ξ) - I1) / integral(abs(G)))
        end
        for D in (1, 2, 3), _ in 1:nb_reps, G in gwp_combos(Val(D))
            G1 = inv_fourier(fourier(G))
            G2 = fourier(inv_fourier(G))
            for _ in 1:5
                x = near(G)
                err_id = max(err_id, err_rel(G1(x), G(x)), err_rel(G2(x), G(x)))
            end
            alloc += alloc_of(fourier, G) + alloc_of(inv_fourier, G)
        end

        @test err <= tol_cub
        @test err_id <= tol
        report_alloc("Fourier", alloc)
    end
end

function _test_gwp_convolution(nb_reps, tol, tol_cub)
    @testset "convolution" begin
        err = 0.0
        err_id = 0.0
        alloc = 0
        for D in (1, 2)
            combos = gwp_combos(Val(D))
            for (G1, G2) in zip(combos, shuffle(combos))
                G = convolution(G1, G2)
                x0 = near(G)
                amin = min(min_eig(G1.z), min_eig(G2.z))
                lo, hi = integration_box((center(G1), x0 - center(G2)), amin)
                I1 = complex_cubature(y -> G1(y) * G2(x0 - y), lo, hi)
                err = max(err, abs(G(x0) - I1) / (integral(abs(G1)) * integral(abs(G2))))
            end
        end
        # Convolution theorem
        for D in (1, 2, 3), _ in 1:nb_reps
            combos = gwp_combos(Val(D))
            for (G1, G2) in zip(combos, shuffle(combos))
                F1, F2 = fourier(G1), fourier(G2)
                FG = fourier(convolution(G1, G2))
                for _ in 1:5
                    ξ = near(F1 * F2)
                    err_id = max(err_id, err_rel(FG(ξ), F1(ξ) * F2(ξ)))
                end
                alloc += alloc_of(convolution, G1, G2)
            end
        end

        @test err <= tol_cub
        @test err_id <= tol
        report_alloc("convolution", alloc)
    end
end

function _test_gwp_dot_L2(nb_reps, tol, tol_cub)
    @testset "dot L²" begin
        err = 0.0
        err_id = 0.0
        alloc = 0
        for D in (1, 2)
            combos = gwp_combos(Val(D))
            for (G1, G2) in zip(combos, shuffle(combos))
                amin = min(min_eig(G1.z), min_eig(G2.z))
                lo, hi = integration_box((center(G1), center(G2)), amin)
                I1 = complex_cubature(y -> conj(G1(y)) * G2(y), lo, hi)
                err = max(err, abs(dot_L2(G1, G2) - I1) / (norm_L2(G1) * norm_L2(G2)))
            end
        end
        for D in (1, 2, 3), _ in 1:nb_reps
            combos = gwp_combos(Val(D))
            for (G1, G2) in zip(combos, shuffle(combos))
                s = norm_L2(G1) * norm_L2(G2)
                I = dot_L2(G1, G2)
                err_id = max(err_id, abs(I - integral(conj(G1) * G2)) / s)
                # Parseval
                err_id = max(err_id, abs(I - dot_L2(fourier(G1), fourier(G2)) / (2π)^D) / s)
                err_id = max(err_id, abs(dot_L2(G2, G1) - conj(I)) / s)
                err_id = max(err_id, err_rel(norm2_L2(G1), real(dot_L2(G1, G1))), abs(imag(dot_L2(G1, G1))) / norm2_L2(G1))
                err_id = max(err_id, err_rel(norm_L2(G1)^2, norm2_L2(G1)))
                alloc += alloc_of(dot_L2, G1, G2) + alloc_of(norm2_L2, G1) + alloc_of(norm_L2, G1)
            end
        end

        @test err <= tol_cub
        @test err_id <= tol
        report_alloc("dot L²", alloc)
    end
end

function _test_gwp_dot_grad(nb_reps, tol, tol_cub)
    @testset "dot ∇" begin
        err = 0.0
        err_id = 0.0
        alloc = 0
        # ∫<(D-c)G1,(D-c)G2> = (2π)⁻ᴰ ∫|ξ-c|² conj(Ĝ1)Ĝ2
        for D in (1, 2)
            combos = gwp_combos(Val(D))
            for (G1, G2) in zip(combos, shuffle(combos))
                c = 2 * (rand(SVector{D}) .- 0.5)
                GF = conj(fourier(G1)) * fourier(G2)
                lo, hi = integration_box((center(GF), c), min_eig(GF.z))
                s = sqrt(norm2_∇(G1) * norm2_∇(G2)) + sqrt(norm2_∇(G1, c) * norm2_∇(G2, c))
                I1 = complex_cubature(y -> sum(abs2, y) * GF(y), lo, hi) / (2π)^D
                err = max(err, abs(dot_∇(G1, G2) - I1) / s)
                I1c = complex_cubature(y -> sum(abs2, y - c) * GF(y), lo, hi) / (2π)^D
                err = max(err, abs(dot_∇(G1, G2, c) - I1c) / s)
            end
        end
        for D in (1, 2, 3), _ in 1:nb_reps
            combos = gwp_combos(Val(D))
            for (G1, G2) in zip(combos, shuffle(combos))
                c = 2 * (rand(SVector{D}) .- 0.5)
                s = sqrt(norm2_∇(G1) * norm2_∇(G2))
                err_id = max(err_id, abs(dot_∇(G2, G1) - conj(dot_∇(G1, G2))) / s)
                err_id = max(err_id, err_rel(norm2_∇(G1), real(dot_∇(G1, G1))), abs(imag(dot_∇(G1, G1))) / norm2_∇(G1))
                err_id = max(err_id, err_rel(norm2_∇(G1, c), real(dot_∇(G1, G1, c))))
                err_id = max(err_id, err_rel(norm_∇(G1, c)^2, norm2_∇(G1, c)), err_rel(norm_∇(G1)^2, norm2_∇(G1)))
                alloc += alloc_of(dot_∇, G1, G2) + alloc_of(dot_∇, G1, G2, c) + alloc_of(norm2_∇, G1) + alloc_of(norm2_∇, G1, c)
            end
        end

        @test err <= tol_cub
        @test err_id <= tol
        report_alloc("dot ∇", alloc)
    end
end

function _test_gwp_coulomb(nb_reps, tol, tol_cub)
    @testset "Coulomb integral" begin
        D = 3
        M = 15.0
        err = 0.0
        alloc = 0
        for kind in (:diag, :sym), hq in (false, true), hp in (false, true)
            λ = rand()
            z0 = 1.0 + rand() + im * rand()
            z = kind === :diag ? Diagonal(SVector(z0, z0, z0)) : Symmetric(SMatrix{3, 3}(z0 * I))
            G = GaussianWavePacket(λ, z, rand_vec(Val(D), hq), rand_vec(Val(D), hp))

            alloc += alloc_of(coulomb_integral, G)
            I1 = coulomb_integral(G)
            F(y) = G(y) * min(100, 1 / norm(y))
            # (the box is not centered at the singularity: otherwise, when q = 0, all the points of the first
                # cubature rule but the center are in the tails, and the imaginary part is wrongly found to vanish)
                I2 = complex_cubature(y -> F(y), [-M + 0.123 for _ in 1:D], [M + 0.057 for _ in 1:D]; reltol=1e-7, abstol=1e-6, maxevals=typemax(Int))

            err = max(err, abs(I1 - I2))
        end
        @test_throws ArgumentError coulomb_integral(GaussianWavePacket(1.0, Diagonal(SVector(1.0, 2.0, 1.0))))
        @test_throws ArgumentError coulomb_integral(GaussianWavePacket(1.0, Symmetric(@SMatrix [1.0 0.1 0; 0.1 1.0 0; 0 0 1.0])))

        # The reference integral is a poor approximation because of the singularity of 1/|x|
        # (it is truncated at 100 and computed by cubature), hence a larger tolerance
        @test err <= 1e-3
        report_alloc("Coulomb integral", alloc)
    end
end

function _test_gwp_conversions(nb_reps, tol, tol_cub)
    @testset "conversions and promotions" begin
        Gd = GaussianWavePacket(1.0, Diagonal(SVector(1.0, 2.0)), SVector(0.1, 0.2))
        Gs = GaussianWavePacket(1.0 + 0im, Symmetric(@SMatrix [1.0 0.2; 0.2 2.0]), SVector(0.1, 0.2))
        x = SVector(0.3, -0.4)
        V = [Gd, Gs]
        @test eltype(V) <: GaussianWavePacket{2, ComplexF64, Float64, <:Symmetric}
        @test V[1](x) ≈ Gd(x) && V[2](x) ≈ Gs(x)
        # Diagonal -> Symmetric
        @test convert(typeof(Gs), Gd)(x) ≈ Gd(x)
        # Symmetric -> Diagonal: only for a diagonal z
        @test_throws InexactError convert(typeof(Gd), Gs)
        Gs_diag = GaussianWavePacket(1.0, Symmetric(@SMatrix [1.0 0.0; 0.0 2.0]), SVector(0.1, 0.2))
        @test convert(typeof(Gd), Gs_diag)(x) ≈ Gs_diag(x)
        # A Symmetric z is stored with its upper triangle
        zl = Symmetric(@SMatrix([2.0 99.0; 0.3 1.0]), :L)
        Gl = GaussianWavePacket(1.0, zl)
        @test Gl.z.uplo == 'U'
        @test integral(Gl) ≈ 2π / sqrt(2.0 - 0.09)
        @test GaussianWavePacket{2}(1.0, Symmetric([2.0 99.0; 0.3 1.0], :L)).z == zl
        # Integer z
        @test integral(GaussianWavePacket(1.0, Symmetric(@SMatrix [2 0; 0 1]))) ≈ 2π / sqrt(2)
        @test integral(GaussianWavePacket(1.0, Diagonal(SVector(2, 1)))) ≈ 2π / sqrt(2)
    end
end

function _test_gwp_errors(nb_reps, tol, tol_cub)
    @testset "errors" begin
        # The real part of z must be positive definite (strictly)
        @test_throws PosDefException integral(GaussianWavePacket(1.0, Diagonal(SVector(0.0, 1.0))))
        @test_throws PosDefException integral(GaussianWavePacket(1.0, Symmetric(@SMatrix [1.0 1.0; 1.0 1.0])))
        @test_throws PosDefException integral(GaussianWavePacket(1.0, Symmetric(@SMatrix [-1.0 0.0; 0.0 1.0])))
        # Dimension selectors
        G = GaussianWavePacket(1.0, Diagonal(SVector(1.0, 1.0, 1.0)))
        @test_throws ArgumentError evaluate(G, SVector(0.1, 0.2); dims=(2, 2))
        @test_throws ArgumentError evaluate(G, SVector(0.1); dims=(4,))
        @test_throws ArgumentError evaluate(G, SVector(0.1); dims=(0,))
        @test_throws ArgumentError integral(G; dims=(1, 1))
        @test_throws ArgumentError integral(G; dims=(5,))
    end
end

function test_gaussian_wave_packet()
    @testset "GaussianWavePacket" begin
        nb_reps = 3
        tol = 1e-12         # identities
        tol_cub = 1e-10     # comparisons with a cubature (relative to ∫|f|)

        _test_gwp_evaluate(nb_reps, tol, tol_cub)
        _test_gwp_restriction(nb_reps, tol, tol_cub)
        _test_gwp_scalar_operations(nb_reps, tol, tol_cub)
        _test_gwp_product(nb_reps, tol, tol_cub)
        _test_gwp_plane_waves(nb_reps, tol, tol_cub)
        _test_gwp_unitary_product(nb_reps, tol, tol_cub)
        _test_gwp_integral(nb_reps, tol, tol_cub)
        _test_gwp_restricted_integral(nb_reps, tol, tol_cub)
        _test_gwp_fourier(nb_reps, tol, tol_cub)
        _test_gwp_convolution(nb_reps, tol, tol_cub)
        _test_gwp_dot_L2(nb_reps, tol, tol_cub)
        _test_gwp_dot_grad(nb_reps, tol, tol_cub)
        _test_gwp_coulomb(nb_reps, tol, tol_cub)
        _test_gwp_conversions(nb_reps, tol, tol_cub)
        _test_gwp_errors(nb_reps, tol, tol_cub)
    end
end
