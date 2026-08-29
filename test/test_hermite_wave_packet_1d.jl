
function test_hermite_wave_packets_1d()

    printstyled("Testing hermite wave packets 1D:\n"; bold=true, color=:blue)

    nb_reps = 1
    M = 10.0
    tol = 1e-6
    int_tol = 1e-9

    #ψₙ(a, q, x) = (a/π)^(1/4) / sqrt(2ⁿn!) * Hₙ(√a*(x-q)) * exp(-a(x-q)²/2)
    function ψ(a, q, n, x)
        (a/π)^(1/4) / sqrt(2.0^n * gamma(n+1)) * hermite_poly(n, sqrt(a) * (x-q)) * exp(-a*(x-q)^2/2)
    end

    let
        err = 0.0
        N = 6
        alloc = 0
        for _=1:nb_reps
            x = 4.0 * (rand() - 0.5)
            Λ = SVector{N}(rand(N) + im * rand(N))
            z = 4 * rand() + 0.5 + im * (4 * rand() - 2)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            alloc += @allocated μ1 = H(x)
            μ2 = sum(Λ[n+1] * ψ(real(z), q, n, x) for n in 0:N-1) * exp(-im * imag(z) / 2 * (x-q)^2 + im * x * p)
            err = max(err, abs(μ1 - μ2) / abs(μ1))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error evaluate = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        N = 5
        Ngrid = 8
        alloc = 0
        for _=1:nb_reps
            Λ = SVector{N}(rand(N) + im * rand(N))
            z = 4 * rand() + 0.5 + im * (4 * rand() - 2)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            agrid = 4 * rand() + 0.5
            qgrid = 4 * (rand() - 0.5)
            xgrid = hermite_grid(agrid, qgrid, Val(Ngrid))

            alloc += @allocated A = evaluate_grid(H, xgrid)

            for j in eachindex(xgrid)
                μ1 = H(xgrid[j])
                μ2 = A[j]
                err = max(err, abs(μ1 - μ2) / abs(μ1))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error evaluate grid = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        alloc = 0
        for _=1:nb_reps
            λ = rand() + 1im * rand()
            z = (4 * rand() + 0.5) + 4im * (rand() - 0.5)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            G = GaussianWavePacket(λ, z, q, p)

            alloc += @allocated H = HermiteWavePacket1D(G)
            alloc += @allocated G_ = truncate_to_gaussian(H)

            for _ in 1:100
                x = 4.0 * (rand() - 0.5)
                err = max(err, abs(H(x) - G(x)))
                err = max(err, abs(G_(x) - G(x)))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error convert from gaussian = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        N = 8
        alloc = 0
        for _=1:nb_reps
            Λ = SVector{N}(rand(N) + im * rand(N))
            z = 4 * rand() + 0.5 + im * (4 * rand() - 2)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            alloc += @allocated μ1 = integral(H)
            μ2 = complex_cubature(y -> H(y[1]), [-M], [M]; abstol=int_tol)
            err = max(err, abs(μ1 - μ2))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error integral = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        N1 = 5
        N2 = 4
        alloc = 0
        for _=1:nb_reps
            Λ1 = SVector{N1}(rand(N1) + im * rand(N1))
            z1 = 4 * rand() + 0.5 + im * (4 * rand() - 2)
            q1 = 4 * (rand() - 0.5)
            p1 = 4 * (rand() - 0.5)
            H1 = HermiteWavePacket1D(Λ1, z1, q1, p1)

            Λ2 = SVector{N2}(rand(N2) + im * rand(N2))
            z2 = 4 * rand() + 0.5 + im * (4 * rand() - 2)
            q2 = 4 * (rand() - 0.5)
            p2 = 4 * (rand() - 0.5)
            H2 = HermiteWavePacket1D(Λ2, z2, q2, p2)

            alloc += @allocated H = H1 * H2

            for _=1:40
                x = 5 * (rand() - 0.5)
                err = max(err, abs(H(x) - H1(x) * H2(x)))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error product = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        N = 6
        alloc = 0
        for _=1:nb_reps
            Λ = SVector{N}(rand(N) + im * rand(N))
            z = 4 * rand() + 0.5 + im * (4 * rand() - 2)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            b = 4 * (rand() - 0.5)
            q2 = 4 * (rand() - 0.5)
            p2 = 4 * (rand() - 0.5)

            f(x) = H(x) * cis(-b/2 * (x-q2)^2) * cis(p2*x)
            alloc += @allocated H2 = unitary_product(H, b, q2, p2)

            for _=1:100
                x = 5 * (rand() - 0.5)
                err = max(err, abs(f(x) - H2(x)))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error unitary product = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        N = 6
        NP = 5
        alloc = 0
        for _=1:nb_reps
            Λ = SVector{N}(rand(N) + im * rand(N))
            z = 4 * rand() + 0.5 + im * (4 * rand() - 2)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            Λ_P = SVector{NP}(rand(NP) + im * rand(NP))
            q2 = 4 * (rand() - 0.5)

            alloc += @allocated PH = polynomial_product(H, Λ_P, q2)

            f(x) = H(x) * sum(Λ_P[k] * (x-q2)^(k-1) for k in 1:NP)

            for _=1:100
                x = 5 * (rand() - 0.5)
                err = max(err, abs(f(x) - PH(x)) / norm_L2(PH))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error polynomial product = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        N = 6
        alloc = 0
        for _=1:nb_reps
            Λ = SVector{N}(rand(N) + im * rand(N))
            z = 4 * rand() + 0.5 + im * (4 * rand() - 2)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            alloc += @allocated HF = fourier(H)

            ξ = 4.0 * (rand() - 0.5)
            μ1 = HF(ξ)
            μ2 = complex_cubature(y -> H(y[1]) * cis(-y[1]*ξ), [-M], [M]; abstol=int_tol)
            err = max(err, abs(μ1 - μ2))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error Fourier (complex variance) = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        N = 6
        alloc = 0
        for _=1:nb_reps
            Λ = SVector{N}(rand(N) + im * rand(N))
            z = 4 * rand() + 0.5
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)
            Hc = HermiteWavePacket1D(Λ, complex(z), q, p)

            alloc += @allocated HF = fourier(H)
            HFc = fourier(Hc)

            for _ in 1:100
                ξ = 4.0 * (rand() - 0.5)
                μ1 = HF(ξ)
                μ2 = HFc(ξ)
                err = max(err, abs(μ1 - μ2))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error Fourier (real variance) = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        N = 7
        alloc = 0
        for _=1:nb_reps
            Λ = SVector{N}(rand(N) + im * rand(N))
            z = 4 * rand() + 0.5 + im * (4 * rand() - 2)
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)
            HF = fourier(H)

            alloc += @allocated H_ = inv_fourier(HF)
            for _ in 1:100
                x = 4.0 * (rand() - 0.5)
                μ1 = H_(x)
                μ2 = H(x)
                err = max(err, abs(μ1 - μ2))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error inverse Fourier (complex variance) = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        N = 7
        alloc = 0
        for _=1:nb_reps
            Λ = SVector{N}(rand(N) + im * rand(N))
            z = 4 * rand() + 0.5
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)
            HF = fourier(H)

            alloc += @allocated H_ = inv_fourier(HF)
            for _ in 1:100
                x = 4.0 * (rand() - 0.5)
                μ1 = H_(x)
                μ2 = H(x)
                err = max(err, abs(μ1 - μ2))
            end
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error inverse Fourier (real variance) = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        err = 0.0
        N1 = 5
        N2 = 4
        alloc = 0
        for _=1:nb_reps
            Λ1 = SVector{N1}(rand(N1) + im * rand(N1))
            z1 = 4 * rand() + 0.5 + im * (4 * rand() - 2)
            q1 = 4 * (rand() - 0.5)
            p1 = 4 * (rand() - 0.5)
            H1 = HermiteWavePacket1D(Λ1, z1, q1, p1)

            Λ2 = SVector{N2}(rand(N2) + im * rand(N2))
            z2 = 4 * rand() + 0.5 + im * (4 * rand() - 2)
            q2 = 4 * (rand() - 0.5)
            p2 = 4 * (rand() - 0.5)
            H2 = HermiteWavePacket1D(Λ2, z2, q2, p2)

            alloc += @allocated μ1 = dot_L2(H1, H2)
            μ2 = complex_cubature(y -> conj(H1(y[1])) * H2(y[1]), [-M], [M]; abstol=int_tol)
            err = max(err, abs(μ1 - μ2))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error dot L2 = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        N = 9
        err = 0.0
        alloc = 0
        for _=1:nb_reps
            Λ = SVector{N}(rand(N) + im * rand(N))
            z = 4 * rand() + 0.5
            q = 4 * (rand() - 0.5)
            p = 4 * (rand() - 0.5)
            H = HermiteWavePacket1D(Λ, z, q, p)

            alloc += @allocated μ1 = norm_L2(H)
            μ2 = sqrt(dot_L2(H, H))
            err = max(err, abs(μ1 - μ2))
        end

        color = (err > tol || alloc > 0) ? :red : :green
        printstyled("Error norm L² = $err ($alloc bytes allocated)\n"; bold=true, color=color)
    end

    let
        N1 = 3
        N2 = 4
        N3 = 3

        Λ1 = SVector{N1}(rand(Float32, N1) + im * rand(Float32, N1))
        z1 = 4 * rand(Float32) + 0.5f0 + im * (4 * rand(Float32) - 2)
        q1 = 4 * (rand(Float32) - 0.5f0)
        p1 = 4 * (rand(Float32) - 0.5f0)
        H1 = HermiteWavePacket1D(Λ1, z1, q1, p1)

        Λ2 = SVector{N2}(rand(Float32, N2) + im * rand(Float32, N2))
        z2 = 4 * rand(Float32) + 0.5f0 + im * (4 * rand(Float32) - 2)
        q2 = 4 * (rand(Float32) - 0.5f0)
        p2 = 4 * (rand(Float32) - 0.5f0)
        H2 = HermiteWavePacket1D(Λ2, z2, q2, p2)

        Λ3 = SVector{N1}(rand(Float32, N1) + im * rand(Float32, N1))
        z3 = 4 * rand(Float32) + 0.5f0 + im * (4 * rand(Float32) - 2)
        q3 = 4 * (rand(Float32) - 0.5f0)
        p3 = 4 * (rand(Float32) - 0.5f0)
        H3 = HermiteWavePacket1D(Λ3, z3, q3, p3)

        H = convolution(H1 * H2, H3)
        res = H(rand(Float32)) + integral(H) + norm_L2(H)
        T_type = typeof(res)

        color = (T_type != ComplexF32) ? :red : :green
        printstyled("Expecting $ComplexF32 and got $T_type\n"; bold=true, color=color)
    end
end
