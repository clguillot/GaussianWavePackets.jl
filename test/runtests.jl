using Test
using Random

include("test.jl")

# The tests use random data. The seed is fixed to make the results reproducible: some tests
# compare with a numerical cubature, which is occasionally not accurate enough for the
# tolerances of the tests on some unlucky draws (this is not an error of the package).
const SEED = 1

@testset "HermiteWavePackets" begin
    for test in (test_gaussian, test_gaussian_wave_packet, test_hermite_wave_packets_1d, test_wave_packet_sum)
        Random.seed!(SEED)
        test()
    end
end
