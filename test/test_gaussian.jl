
# Real gaussians (the Gaussian alias): the operations themselves are tested with all the other
# wave packets in test_gaussian_wave_packet.jl, this file checks the constructors and that the
# operations which should give a Gaussian do
function test_gaussian()
    @testset "Gaussian" begin
        nb_reps = 5
        tol = 1e-12

        @testset "constructors" begin
            err = 0.0
            for _ in 1:nb_reps
                λ = rand() + 1im * rand()
                a = 4 * rand() + 0.5
                q = 4 * (rand() - 0.5)
                x = 4 * (rand() - 0.5)
                G = Gaussian(λ, a, q)
                @test G isa Gaussian{1}
                err = max(err, err_rel(G(x), λ * exp(-a * (x - q)^2 / 2)))
                err = max(err, err_rel(Gaussian(λ, a)(x), λ * exp(-a * x^2 / 2)))

                D = 3
                z = Diagonal(4 .* rand(SVector{D}) .+ 0.5)
                qv = 4 .* (rand(SVector{D}) .- 0.5)
                xv = 4 .* (rand(SVector{D}) .- 0.5)
                G3 = Gaussian(λ, z, qv)
                ref = λ * prod(@. exp(-z.diag / 2 * (xv - qv)^2))
                err = max(err, err_rel(G3(xv), ref))
                err = max(err, err_rel(Gaussian{D}(λ, Diagonal(Vector(z.diag)), Vector(qv))(xv), ref))
                err = max(err, err_rel(Gaussian{D}(λ, Symmetric(Matrix(z)), Vector(qv))(xv), ref))
                @test Gaussian{D}(λ, Symmetric(Matrix(z)), Vector(qv)) isa Gaussian{D}
            end

            @test err <= tol
        end

        @testset "closed operations" begin
            for D in (1, 3), kind in (:diag, :sym)
                G1 = Gaussian(rand() + 1im * rand(), rand_cov(Val(D), kind, false), rand_vec(Val(D), true))
                G2 = Gaussian(rand(), rand_cov(Val(D), kind, false), rand_vec(Val(D), false))
                @test G1 * G2 isa Gaussian{D}
                @test (abs(G1) * G2).λ isa Float64                # real gaussians stay real
                @test convolution(abs(G1), G2).λ isa Float64
                @test convolution(G1, G2) isa Gaussian{D}
                @test conj(G1) isa Gaussian{D}
                @test abs(G1) isa Gaussian{D}
                @test abs2(G1) isa Gaussian{D}
                @test 2 * G1 isa Gaussian{D}
                @test zero(G1) isa Gaussian{D}
                @test !(fourier(G1) isa Gaussian)                 # the Fourier transform has a momentum
                @test fourier(G2) isa Gaussian{D}                 # but not without q
                x = rand(SVector{D})
                @test abs(G1)(x) ≈ abs(G1(x))
                @test abs2(G1)(x) ≈ abs2(G1(x))
            end
        end

        @testset "display" begin
            @test occursin("exp", sprint(show, Gaussian(1.0, 2.0, 0.5)))
            @test !occursin("cis", sprint(show, Gaussian(1.0, 2.0, 0.5)))
            @test occursin("cis", sprint(show, GaussianWavePacket(1.0, 2.0, 0.5, 1.0)))
            s = sprint(show, GaussianWavePacket(1.0, Diagonal(SVector(1.0, 2.0)), SVector(0.1, 0.2), SVector(0.3, 0.4)))
            @test occursin("q = ", s) && occursin("p = ", s)
            @test !occursin("p = ", sprint(show, Gaussian(1.0, Diagonal(SVector(1.0, 2.0)), SVector(0.1, 0.2))))
            @test occursin("Λ = ", sprint(show, HermiteFct1D(SVector(1.0, 2.0), 1.0)))
        end
    end
end
