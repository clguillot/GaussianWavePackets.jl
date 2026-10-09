
function test_gaussian_wave_packet()
    @testset "GaussianWavePacket" begin
        nb_reps = 10
        M = 15.0
        tol = 1e-6
        int_tol = 1e-9
        idN(::Val{N}) where N = Diagonal(SVector{N}(ntuple(_ -> true, N)))

        @testset "evaluate" begin
            D = 3
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                x = SVector{D}(4.0 .* (rand(D) .- 0.5))
                λ = rand() + 1im * rand()
                A = SMatrix{D, D}(rand(D*D))
                B = SMatrix{D, D}(rand(D*D))
                z = Symmetric(A'*A + 0.5*idN(Val(D)) + im*(B'+B))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                p = SVector{D}(4 * (rand(D) .- 0.5))
                G = GaussianWavePacket(λ, z, q, p)

                μ = λ * exp(-dot(x-q, z, x-q) / 2) * cis(dot(p, x))
                alloc += @allocated err = max(err, abs(G(x) - μ) / abs(μ))
            end

            @test err <= tol
            report_alloc("evaluate", alloc)
        end

        @testset "restriction" begin
            D = 5
            err = 0.0
            alloc = 0
            # General
            for _=1:nb_reps
                I1 = SVector(1, 3, 4)
                I2 = SVector(2, 5)
                x = SVector{D}(4.0 .* (rand(D) .- 0.5))
                λ = rand() + 1im * rand()
                A = SMatrix{D, D}(rand(D*D))
                B = SMatrix{D, D}(rand(D*D))
                z = Symmetric(A'*A + 0.5*idN(Val(D)) + im*(B'+B))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                p = SVector{D}(4 * (rand(D) .- 0.5))
                G = GaussianWavePacket(λ, z, q, p)

                alloc += @allocated G_ = evaluate(G, x[I2], Tuple{I2...})

                μ = G(x)
                err = max(err, abs(G_(x[I1]) - μ) / abs(μ))
            end
            # Diagonal
            for _=1:nb_reps
                I1 = SVector(1, 3, 4)
                I2 = SVector(2, 5)
                x = SVector{D}(4.0 .* (rand(D) .- 0.5))
                λ = rand() + 1im * rand()
                z = Diagonal(SVector{D}((4 * rand(D) .+ 0.5) + 4im * (rand(D) .- 0.5)))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                p = SVector{D}(4 * (rand(D) .- 0.5))
                G = GaussianWavePacket(λ, z, q, p)

                alloc += @allocated G_ = evaluate(G, x[I2], Tuple{I2...})

                μ = G(x)
                err = max(err, abs(G_(x[I1]) - μ) / abs(μ))
            end

            @test err <= tol
            report_alloc("restriction", alloc)
        end

        @testset "conjugate" begin
            D = 3
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                x = SVector{D}(4.0 .* (rand(D) .- 0.5))
                λ = rand() + 1im * rand()
                A = SMatrix{D, D}(rand(D*D))
                B = SMatrix{D, D}(rand(D*D))
                z = Symmetric(A'*A + 0.5*idN(Val(D)) + im*(B'+B))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                p = SVector{D}(4 * (rand(D) .- 0.5))
                G = GaussianWavePacket(λ, z, q, p)

                alloc += @allocated Gc = conj(G)

                err = max(err, abs(Gc(x) - conj(G(x))) / abs(Gc(x)))
            end

            @test err <= tol
            report_alloc("conjugate", alloc)
        end

        @testset "product" begin
            D = 3
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                λ1 = rand() + 1im * rand()
                A1 = SMatrix{D, D}(rand(D*D))
                B1 = SMatrix{D, D}(rand(D*D))
                z1 = Symmetric(A1'*A1 + 0.5*idN(Val(D)) + im*(B1'+B1))
                q1 = SVector{D}(4 * (rand(D) .- 0.5))
                p1 = SVector{D}(4 * (rand(D) .- 0.5))
                G1 = GaussianWavePacket(λ1, z1, q1, p1)

                λ2 = rand() + 1im * rand()
                A2 = SMatrix{D, D}(rand(D*D))
                B2 = SMatrix{D, D}(rand(D*D))
                z2 = Symmetric(A2'*A2 + 0.5*idN(Val(D)) + im*(B2'+B2))
                q2 = SVector{D}(4 * (rand(D) .- 0.5))
                p2 = SVector{D}(4 * (rand(D) .- 0.5))
                G2 = GaussianWavePacket(λ2, z2, q2, p2)

                f(x) = G1(x) * G2(x)
                alloc += @allocated G = G1 * G2

                for _=1:100
                    x = SVector{D}(5 * (rand(D) .- 0.5))
                    err = max(err, abs(f(x) - G(x)) / abs(f(x)))
                end
            end

            @test err <= tol
            report_alloc("product", alloc)
        end

        @testset "unitary product" begin
            D = 3
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                λ = rand() + 1im * rand()
                A = SMatrix{D, D}(rand(D*D))
                B = SMatrix{D, D}(rand(D*D))
                z = Symmetric(A'*A + 0.5*idN(Val(D)) + im*(B'+B))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                p = SVector{D}(4 * (rand(D) .- 0.5))
                G = GaussianWavePacket(λ, z, q, p)

                b = Symmetric(SMatrix{D, D}(rand(D*D)))
                q2 = SVector{D}(4 * (rand(D) .- 0.5))
                p2 = SVector{D}(4 * (rand(D) .- 0.5))

                f(x) = G(x) * cis(-dot(x - q2, b, x - q2) / 2) * cis(dot(p2, x))
                alloc += @allocated G2 = unitary_product(G, b, q2, p2)

                for _=1:100
                    x = SVector{D}(5 * (rand(D) .- 0.5))
                    err = max(err, abs(f(x) - G2(x)))
                end
            end

            @test err <= tol
            report_alloc("unitary product", alloc)
        end

        @testset "integral" begin
            D = 2
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                λ = rand() + 1im * rand()
                z = SVector{D}(4 * rand(D) .+ 0.5 + im * (4 * rand(D) .- 2))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                p = SVector{D}(4 * (rand(D) .- 0.5))
                G = GaussianWavePacket(λ, Diagonal(z), q, p)

                I1 = complex_cubature(y -> G(y), [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)        
                alloc += @allocated I2 = integral(G)
                err = max(err, abs(I1 - I2))
            end

            for _=1:nb_reps
                λ = rand() + 1im * rand()
                A = SMatrix{D, D}(rand(D*D))
                B = SMatrix{D, D}(rand(D*D))
                z = Symmetric(A'*A + 0.5*idN(Val(D)) + im*(B'+B))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                p = SVector{D}(4 * (rand(D) .- 0.5))
                G = GaussianWavePacket(λ, z, q, p)

                I1 = complex_cubature(y -> G(y), [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)        
                alloc += @allocated I2 = integral(G)
                err = max(err, abs(I1 - I2))
            end

            @test err <= tol
            report_alloc("integral", alloc)
        end
        @testset "restricted integral" begin
            D = 5
            d = 2
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                I1 = SVector(1, 3, 4)
                I2 = SVector(2, 5)
                x = SVector{D-d}(rand(D-d))
                λ = rand() + 1im * rand()
                A = SMatrix{D, D}(rand(D*D))
                B = SMatrix{D, D}(rand(D*D))
                z = Symmetric(A'*A + 0.5*idN(Val(D)) + im*(B'+B))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                p = SVector{D}(4 * (rand(D) .- 0.5))
                G = GaussianWavePacket(λ, z, q, p)

                G_ = evaluate(G, x, Tuple{I1...})

                I1 = complex_cubature(y -> G_(y), [-M for _ in 1:d], [M for _ in 1:d]; abstol=int_tol)        
                alloc += @allocated I2 = integral(G, Tuple{I2...})(x)
                err = max(err, abs(I1 - I2))
            end

            @test err <= tol
            report_alloc("restricted integral", alloc)
        end

        @testset "convolution" begin
            D = 2
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                λ1 = rand() + 1im * rand()
                A1 = SMatrix{D, D}(rand(D*D))
                B1 = SMatrix{D, D}(rand(D*D))
                z1 = Symmetric(A1'*A1 + 0.5*idN(Val(D)) + im*(B1'+B1))
                q1 = SVector{D}(4 * (rand(D) .- 0.5))
                p1 = SVector{D}(4 * (rand(D) .- 0.5))
                G1 = GaussianWavePacket(λ1, z1, q1, p1)

                λ2 = rand() + 1im * rand()
                A2 = SMatrix{D, D}(rand(D*D))
                B2 = SMatrix{D, D}(rand(D*D))
                z2 = Symmetric(A2'*A2 + 0.5*idN(Val(D)) + im*(B2'+B2))
                q2 = SVector{D}(4 * (rand(D) .- 0.5))
                p2 = SVector{D}(4 * (rand(D) .- 0.5))
                G2 = GaussianWavePacket(λ2, z2, q2, p2)

                alloc += @allocated G = convolution(G1, G2)

                x0 = 5.0 * (rand(D) .- 0.5)
                I = complex_cubature(y -> G1(y) * G2(x0 - y), [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)
                err = max(err, abs(I - G(x0)))
            end

            @test err <= tol
            report_alloc("convolution", alloc)
        end

        @testset "dot L²" begin
            D = 2
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                λ1 = rand() + 1im * rand()
                z1 = SVector{D}(4 * rand(D) .+ 0.5 + im * (4 * rand(D) .- 2))
                q1 = SVector{D}(4 * (rand(D) .- 0.5))
                p1 = SVector{D}(4 * (rand(D) .- 0.5))
                G1 = GaussianWavePacket(λ1, Diagonal(z1), q1, p1)

                λ2 = rand() + 1im * rand()
                z2 = SVector{D}(4 * rand(D) .+ 0.5 + im * (4 * rand(D) .- 2))
                q2 = SVector{D}(4 * (rand(D) .- 0.5))
                p2 = SVector{D}(4 * (rand(D) .- 0.5))
                G2 = GaussianWavePacket(λ2, Diagonal(z2), q2, p2)

                I1 = complex_cubature(y -> conj(G1(y)) * G2(y), [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)
                alloc += @allocated I2 = dot_L2(G1, G2)
                err = max(err, abs(I1 - I2))
            end
            for _=1:nb_reps
                λ1 = rand() + 1im * rand()
                A1 = SMatrix{D, D}(rand(D*D))
                B1 = SMatrix{D, D}(rand(D*D))
                z1 = Symmetric(A1'*A1 + 0.5*idN(Val(D)) + im*(B1'+B1))
                q1 = SVector{D}(4 * (rand(D) .- 0.5))
                p1 = SVector{D}(4 * (rand(D) .- 0.5))
                G1 = GaussianWavePacket(λ1, z1, q1, p1)

                λ2 = rand() + 1im * rand()
                A2 = SMatrix{D, D}(rand(D*D))
                B2 = SMatrix{D, D}(rand(D*D))
                z2 = Symmetric(A2'*A2 + 0.5*idN(Val(D)) + im*(B2'+B2))
                q2 = SVector{D}(4 * (rand(D) .- 0.5))
                p2 = SVector{D}(4 * (rand(D) .- 0.5))
                G2 = GaussianWavePacket(λ2, z2, q2, p2)

                I1 = complex_cubature(y -> conj(G1(y)) * G2(y), [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)
                alloc += @allocated I2 = dot_L2(G1, G2)
                err = max(err, abs(I1 - I2))
            end

            @test err <= tol
            report_alloc("dot L²", alloc)
        end

        @testset "norm L²" begin
            D = 3
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                λ = rand() + 1im * rand()
                A = SMatrix{D, D}(rand(D*D))
                B = SMatrix{D, D}(rand(D*D))
                z = Symmetric(A'*A + 0.5*idN(Val(D)) + im*(B'+B))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                p = SVector{D}(4 * (rand(D) .- 0.5))
                G = GaussianWavePacket(λ, z, q, p)

                alloc += @allocated I = norm_L2(G)
                err = max(err, abs(I - sqrt(dot_L2(G, G))) / norm_L2(G))
            end

            @test err <= tol
            report_alloc("norm L²", alloc)
        end

        @testset "dot ∇" begin
            D = 2
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                λ1 = rand() + 1im * rand()
                A1 = SMatrix{D, D}(rand(D*D))
                B1 = SMatrix{D, D}(rand(D*D))
                z1 = Symmetric(A1'*A1 + 0.5*idN(Val(D)) + im*(B1'+B1))
                q1 = SVector{D}(4 * (rand(D) .- 0.5))
                p1 = SVector{D}(4 * (rand(D) .- 0.5))
                G1 = GaussianWavePacket(λ1, z1, q1, p1)

                λ2 = rand() + 1im * rand()
                A2 = SMatrix{D, D}(rand(D*D))
                B2 = SMatrix{D, D}(rand(D*D))
                z2 = Symmetric(A2'*A2 + 0.5*idN(Val(D)) + im*(B2'+B2))
                q2 = SVector{D}(4 * (rand(D) .- 0.5))
                p2 = SVector{D}(4 * (rand(D) .- 0.5))
                G2 = GaussianWavePacket(λ2, z2, q2, p2)

                c = SVector{D}(2 * (rand(D) .- 1))

                GF = conj(fourier(G1)) * fourier(G2)
                F(y) = (2π)^(-D) * sum(abs2, y - c) * GF(y)
                I1 = complex_cubature(F, [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)
                alloc += @allocated I2 = dot_∇(G1, G2, c)
                err = max(err, abs(I1 - I2))
            end

            @test err <= tol
            report_alloc("dot ∇", alloc)
        end

        @testset "norm ∇" begin
            D = 3
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                λ = rand() + 1im * rand()
                A = SMatrix{D, D}(rand(D*D))
                B = SMatrix{D, D}(rand(D*D))
                z = Symmetric(A'*A + 0.5*idN(Val(D)) + im*(B'+B))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                p = SVector{D}(4 * (rand(D) .- 0.5))
                G = GaussianWavePacket(λ, z, q, p)

                c = SVector{D}(2 * (rand(D) .- 1))

                alloc += @allocated I = norm_∇(G, c)
                err = max(err, abs(I - sqrt(dot_∇(G, G, c))))
            end

            @test err <= tol
            report_alloc("norm ∇", alloc)
        end

        @testset "Fourier" begin
            D = 1
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                ξ = 5 * (rand(D) .- 0.5)
                λ = rand() + 1im * rand()
                A = SMatrix{D, D}(rand(D*D))
                B = SMatrix{D, D}(rand(D*D))
                z = Symmetric(A'*A + 0.5*idN(Val(D)) + im*(B'+B))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                p = SVector{D}(4 * (rand(D) .- 0.5))
                G = GaussianWavePacket(λ, z, q, p)

                alloc += @allocated Gf = fourier(G)

                I = complex_cubature(y -> G(y) * cis(-dot(ξ, y)), [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)
                err = max(err, abs(I - Gf(ξ)))
            end

            @test err <= tol
            report_alloc("Fourier", alloc)
        end

        @testset "Inverse Fourier" begin
            D = 3
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                λ = rand() + 1im * rand()
                A = SMatrix{D, D}(rand(D*D))
                B = SMatrix{D, D}(rand(D*D))
                z = Symmetric(A'*A + 0.5*idN(Val(D)) + im*(B'+B))
                q = SVector{D}(4 .* (rand(D) .- 0.5))
                p = SVector{D}(4 .* (rand(D) .- 0.5))
                G = GaussianWavePacket(λ, z, q, p)
                GF = fourier(G)

                alloc += @allocated G_ = inv_fourier(GF)

                for _=1:100
                    ξ = 5 * (rand(D) .- 0.5)
                    err = max(err, abs(G(ξ) - G_(ξ)))
                end
            end

            @test err <= tol
            report_alloc("Inverse Fourier", alloc)
        end

        @testset "Coulomb integral" begin
            D = 3
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                λ = rand() #+ 1im * rand()
                z0 = 1.0 + rand() + im * rand()
                z = Diagonal(SVector(z0, z0, z0))
                q = SVector{D}(4 .* (rand(D) .- 0.5))
                p = SVector{D}(4 .* (rand(D) .- 0.5))
                G = GaussianWavePacket(λ, z, q, p)

                alloc += @allocated I1 = coulomb_integral(G)
                F(y) = G(y) * min(100, 1 / norm(y))
                I2 = complex_cubature(y -> F(y), [-M for _ in 1:D], [M for _ in 1:D]; abstol=1e-5)

                err = max(err, abs(I1 - I2))
            end

            # The reference integral is a poor approximation because of the singularity of 1/|x|
            # (it is truncated at 100 and computed by cubature), hence a larger tolerance
            @test err <= 1e-3
            report_alloc("Coulomb integral", alloc)
        end

        @testset "type stability (ComplexF32)" begin
            D = 3

            λ1 = rand(Float32) + 1im * rand(Float32)
            A1 = SMatrix{D, D}(rand(Float32, D*D))
            B1 = SMatrix{D, D}(rand(Float32, D*D))
            z1 = Symmetric(A1'*A1 + 0.5f0*idN(Val(D)) + im*(B1'+B1))
            q1 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
            p1 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
            G1 = GaussianWavePacket(λ1, z1, q1, p1)

            λ2 = rand(Float32) + 1im * rand(Float32)
            A2 = SMatrix{D, D}(rand(Float32, D*D))
            B2 = SMatrix{D, D}(rand(Float32, D*D))
            z2 = Symmetric(A2'*A2 + 0.5f0*idN(Val(D)) + im*(B2'+B2))
            q2 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
            p2 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
            G2 = GaussianWavePacket(λ2, z2, q2, p2)

            λ3 = rand(Float32) + 1im * rand(Float32)
            A3 = SMatrix{D, D}(rand(Float32, D*D))
            B3 = SMatrix{D, D}(rand(Float32, D*D))
            z3 = Symmetric(A3'*A3 + 0.5f0*idN(Val(D)) + im*(B3'+B3))
            q3 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
            p3 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
            G3 = GaussianWavePacket(λ3, z3, q3, p3)

            G = convolution(G1 * fourier(G2), inv_fourier(G3))
            res = G(rand(Float32, D)) + integral(G) + norm_L2(G)
            T_type = typeof(res)

            @test T_type == ComplexF32
        end
    end
end