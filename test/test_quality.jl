
function test_quality()
    @testset "Code quality" begin
        @testset "Aqua" begin
            Aqua.test_all(GaussianWavePackets)
        end
        @testset "JET" begin
            JET.test_package(GaussianWavePackets; target_modules=(GaussianWavePackets,))
        end
    end
end
