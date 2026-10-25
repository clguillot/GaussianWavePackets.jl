
# Wrappers with constant dimension selectors (the result is type stable thanks to constant propagation)
integral_dims2(G) = integral(G; dims=(2,))
evaluate_dims31(G, x) = evaluate(G, x; dims=(3, 1))

# The operations of a gaussian wave packet, with their arguments
function gwp_operations(G::GaussianWavePacket{D}) where D
    x = SVector{D}(rand(D))
    c = SVector{D}(rand(D))
    b = Symmetric(SMatrix{D, D}(rand(D, D)))
    G2 = conj(G)
    ops = Any[(evaluate, (G, x)), (*, (G, G2)), (*, (2.0, G)), (conj, (G,)), (abs2, (G,)), (integral, (G,)),
              (fourier, (G,)), (inv_fourier, (G,)), (convolution, (G, G2)), (dot_L2, (G, G2)),
              (norm2_L2, (G,)), (norm_L2, (G,)), (dot_∇, (G, G2)), (dot_∇, (G, G2, c)), (norm2_∇, (G,)),
              (norm2_∇, (G, c)), (unitary_product, (G, b, x, c))]
    D == 1 && push!(ops, (evaluate, (G, 0.3)))
    D == 3 && push!(ops, (integral_dims2, (G,)), (evaluate_dims31, (G, SVector(0.1, 0.2))))
    return ops
end
# The operations of a hermite wave packet, with their arguments
function hermite_operations(H::HermiteWavePacket1D)
    H2 = conj(H)
    return Any[(evaluate, (H, 0.3)), (evaluate_grid, (H, SVector(0.1, 0.2))), (*, (H, H2)), (*, (2.0, H)),
               (integral, (H,)), (fourier, (H,)), (inv_fourier, (H,)), (convolution, (H, H2)), (dot_L2, (H, H2)),
               (norm2_L2, (H,)), (unitary_product, (H, 0.5, 0.1, 0.2)), (polynomial_product, (H, SVector(1.0, 2.0), 0.3)),
               (truncate, (H, Val(2))), (truncate_to_gaussian, (H,))]
end

function test_type_stability()
    @testset "type stability" begin
        @testset "@inferred" begin
            nb_fail = 0
            for D in (1, 3), G in gwp_combos(Val(D)), (f, args) in gwp_operations(G)
                ok = try
                    @inferred f(args...)
                    true
                catch e
                    @info "Not inferred" f typeof.(args)
                    false
                end
                nb_fail += !ok
            end
            for N in (1, 3), H in hermite_combos(N), (f, args) in hermite_operations(H)
                ok = try
                    @inferred f(args...)
                    true
                catch e
                    @info "Not inferred" f typeof.(args)
                    false
                end
                nb_fail += !ok
            end
            @test nb_fail == 0
        end

        @testset "JET (@test_opt)" begin
            for D in (1, 3), G in gwp_combos(Val(D))[1:4:end], (f, args) in gwp_operations(G)
                @test_opt target_modules=(GaussianWavePackets,) f(args...)
            end
            for H in hermite_combos(3)[1:3:end], (f, args) in hermite_operations(H)
                @test_opt target_modules=(GaussianWavePackets,) f(args...)
            end
            S = (rand_gwp(Val(1), :diag, true, true, true) + rand_hermite(2, true, true, true)) + rand_gwp(Val(1), :sym, true, false, true)
            for (f, args) in ((evaluate, (S, SVector(0.3))), (integral, (S,)), (dot_L2, (S, S)), (norm2_L2, (S,)), (fourier, (S,)))
                @test_opt target_modules=(GaussianWavePackets,) f(args...)
            end
        end

        @testset "Float32 preservation (Gaussian)" begin
            D = 3
            G1, G2, G3 = (Gaussian(rand(Float32), Diagonal(SVector{D}(4 * rand(Float32, D) .+ 0.5f0)),
                                   SVector{D}(4 * (rand(Float32, D) .- 0.5f0))) for _ in 1:3)
            G = convolution(G1 * G2, G3)
            @test (@inferred G(rand(Float32, D))) isa Float32
            @test (@inferred integral(G)) isa Float32
            @test (@inferred norm_L2(G)) isa Float32
            @test (@inferred norm2_∇(G)) isa Float32
            @test (@inferred fourier(G)).λ isa Float32
            # Without q
            G0 = Gaussian(1f0, Diagonal(SVector(1f0, 2f0)))
            @test (@inferred integral(G0)) isa Float32
            @test (@inferred fourier(G0)).λ isa Float32
            @test (@inferred inv_fourier(G0)).λ isa Float32
            @test (@inferred integral_dims2(Gaussian(1f0, Diagonal(SVector(1f0, 2f0, 3f0))))).λ isa Float32
        end

        @testset "Float32 preservation (GaussianWavePacket)" begin
            D = 3
            idN = Diagonal(SVector{D}(ntuple(_ -> true, D)))
            function rand_gwp32()
                A = SMatrix{D, D}(rand(Float32, D*D))
                B = SMatrix{D, D}(rand(Float32, D*D))
                z = Symmetric(A'*A + 0.5f0*idN + im*(B'+B))
                return GaussianWavePacket(rand(Float32) + 1im * rand(Float32), z, SVector{D}(4 * (rand(Float32, D) .- 0.5f0)),
                                          SVector{D}(4 * (rand(Float32, D) .- 0.5f0)))
            end
            G1, G2, G3 = rand_gwp32(), rand_gwp32(), rand_gwp32()
            G = convolution(G1 * fourier(G2), inv_fourier(G3))
            @test (@inferred G(rand(Float32, D))) isa ComplexF32
            @test (@inferred integral(G)) isa ComplexF32
            @test (@inferred norm_L2(G)) isa Float32
            @test (@inferred dot_L2(G1, G2)) isa ComplexF32
            @test (@inferred dot_∇(G1, G2)) isa ComplexF32
            @test (@inferred norm2_∇(G1)) isa Float32
            @test (@inferred integral_dims2(G)).λ isa ComplexF32
            @test (@inferred coulomb_integral(GaussianWavePacket(1f0, Diagonal(SVector(1f0, 1f0, 1f0)), SVector(0.1f0, 0f0, 0f0)))) isa Float32
        end

        @testset "Float32 preservation (Hermite)" begin
            N1, N2, N3 = 3, 4, 3
            rand_h32(N) = HermiteWavePacket1D(SVector{N}(rand(Float32, N) + im * rand(Float32, N)), 4 * rand(Float32) + 0.5f0 + im * (4 * rand(Float32) - 2),
                                              4 * (rand(Float32) - 0.5f0), 4 * (rand(Float32) - 0.5f0))
            H1, H2, H3 = rand_h32(N1), rand_h32(N2), rand_h32(N3)
            H = convolution(H1 * H2, H3)
            @test (@inferred H(rand(Float32))) isa ComplexF32
            @test (@inferred integral(H)) isa ComplexF32
            @test (@inferred norm_L2(H)) isa Float32
            @test (@inferred dot_L2(H1, H2)) isa ComplexF32
            # Real variance, no q and no p (the constants used to be promoted to Float64)
            Hr = HermiteFct1D(SVector(1f0, 2f0, 3f0), 1.5f0)
            @test (@inferred fourier(Hr)).Λ isa SVector{3, ComplexF32}
            @test (@inferred inv_fourier(Hr)).Λ isa SVector{3, ComplexF32}
            @test (@inferred integral(Hr)) isa ComplexF32
            @test (@inferred Hr(0.3f0)) isa Float32
            @test (@inferred (Hr * Hr)(0.3f0)) isa Float32
            @test (@inferred dot_L2(Hr, Hr)) isa Float32
            @test (@inferred polynomial_product(Hr, SVector(1f0, 2f0))).Λ isa SVector{4, Float32}
        end
    end
end
