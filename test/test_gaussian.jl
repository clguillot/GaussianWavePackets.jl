
function test_gaussian()
    @testset "Gaussian" begin
        nb_reps = 1
        M = 10.0
        tol = 1e-6
        int_tol = 1e-9

        @testset "evaluate" begin
            D = 3
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                x = SVector{D}(4.0 .* (rand(D) .- 0.5))
                λ = rand() + 1im * rand()
                z = Diagonal(SVector{D}((4 * rand(D) .+ 0.5)))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                G = Gaussian(λ, z, q)

                u = @. exp(-z/2 * (x - q)^2)
                μ = λ * prod(u)
                alloc += @allocated err = max(err, abs(G(x) - μ) / abs(μ))
            end

            @test err <= tol
            report_alloc("evaluate", alloc)
        end

        @testset "conjugate" begin
            D = 3
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                x = SVector{D}(4.0 .* (rand(D) .- 0.5))
                λ = rand() + 1im * rand()
                z = Diagonal(SVector{D}((4 * rand(D) .+ 0.5)))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                G = Gaussian(λ, z, q)

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
                z1 = Diagonal(SVector{D}((4 * rand(D) .+ 0.5)))
                q1 = SVector{D}(4 * (rand(D) .- 0.5))
                G1 = Gaussian(λ1, z1, q1)

                λ2 = rand() + 1im * rand()
                z2 = Diagonal(SVector{D}((4 * rand(D) .+ 0.5)))
                q2 = SVector{D}(4 * (rand(D) .- 0.5))
                G2 = Gaussian(λ2, z2, q2)

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

        @testset "integral" begin
            D = 2
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                λ = rand() + 1im * rand()
                z = Diagonal(SVector{D}((4 * rand(D) .+ 0.5)))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                G = Gaussian(λ, z, q)

                I = complex_cubature(y -> G(y), [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)
                alloc += @allocated err = max(err, abs(I - integral(G)))
            end

            @test err <= tol
            report_alloc("integral", alloc)
        end

        @testset "convolution" begin
            D = 2
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                λ1 = rand() + 1im * rand()
                z1 = Diagonal(SVector{D}((4 * rand(D) .+ 0.5)))
                q1 = SVector{D}(4 * (rand(D) .- 0.5))
                G1 = Gaussian(λ1, z1, q1)

                λ2 = rand() + 1im * rand()
                z2 = Diagonal(SVector{D}((4 * rand(D) .+ 0.5)))
                q2 = SVector{D}(4 * (rand(D) .- 0.5))
                G2 = Gaussian(λ2, z2, q2)

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
                z1 = Diagonal(SVector{D}((4 * rand(D) .+ 0.5)))
                q1 = SVector{D}(4 * (rand(D) .- 0.5))
                G1 = Gaussian(λ1, z1, q1)

                λ2 = rand() + 1im * rand()
                z2 = Diagonal(SVector{D}((4 * rand(D) .+ 0.5)))
                q2 = SVector{D}(4 * (rand(D) .- 0.5))
                G2 = Gaussian(λ2, z2, q2)

                I = complex_cubature(y -> conj(G1(y)) * G2(y), [-M for _ in 1:D], [M for _ in 1:D]; abstol=int_tol)
                alloc += @allocated err = max(err, abs(I - dot_L2(G1, G2)))
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
                z = Diagonal(SVector{D}((4 * rand(D) .+ 0.5)))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                G = Gaussian(λ, z, q)

                alloc += @allocated err = max(err, abs(norm_L2(G) - sqrt(dot_L2(G, G))) / norm_L2(G))
            end

            @test err <= tol
            report_alloc("norm L²", alloc)
        end

        @testset "Fourier" begin
            D = 2
            err = 0.0
            alloc = 0
            for _=1:nb_reps
                ξ = 5 * (rand(D) .- 0.5)
                λ = rand() + 1im * rand()
                z = Diagonal(SVector{D}((4 * rand(D) .+ 0.5)))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                G = Gaussian(λ, z, q)

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
                z = Diagonal(SVector{D}((4 * rand(D) .+ 0.5)))
                q = SVector{D}(4 * (rand(D) .- 0.5))
                G = Gaussian(λ, z, q)
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

        @testset "type stability (Float32)" begin
            D = 3

            λ1 = rand(Float32)
            z1 = Diagonal(SVector{D}((4 * rand(Float32, D) .+ 0.5f0)))
            q1 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
            G1 = Gaussian(λ1, z1, q1)

            λ2 = rand(Float32)
            z2 = Diagonal(SVector{D}((4 * rand(Float32, D) .+ 0.5f0)))
            q2 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
            G2 = Gaussian(λ2, z2, q2)

            λ3 = rand(Float32)
            z3 = Diagonal(SVector{D}((4 * rand(Float32, D) .+ 0.5f0)))
            q3 = SVector{D}(4 * (rand(Float32, D) .- 0.5f0))
            G3 = Gaussian(λ3, z3, q3)

            G = convolution(G1 * G2, G3)
            res = G(rand(Float32, D)) + integral(G) + norm_L2(G)
            T_type = typeof(res)

            @test T_type == Float32
        end
    end
end
