# HermiteWavePackets.jl

**HermiteWavePackets.jl** is a Julia package for exact, allocation-free computations with **Gaussian wave packets** and **Hermite wave packets**: products, Fourier transforms, convolutions, integrals, $L^2$ and $\dot H^1$ inner products, etc. are computed exactly (up to rounding) from the parameters of the functions, with closed formulas for Gaussians and exact Gauss–Hermite quadrature for Hermite products, rather than by approximation on a grid.

Such computations appear in quantum chemistry and quantum dynamics (Gaussian basis sets, Gaussian wave packet propagation), signal processing and mathematical analysis.

Two families of functions are supported:

| Type | Function represented | Dimension |
|---|---|---|
| `GaussianWavePacket` (and `Gaussian`) | $\lambda\, e^{-\frac12 (x-q)^\top z\,(x-q)}\, e^{i p\cdot x}$ | any $D$ |
| `HermiteWavePacket1D` (and `HermiteFct1D`) | $\sum_{n=0}^{N-1} \Lambda_n\, \psi_n(z,q,p;x)$ | 1 |

where $\lambda\in\mathbb C$, $z$ is a complex symmetric matrix with positive definite real part (a plain complex number in the Hermite case), $q,p\in\mathbb R^D$ and the $\psi_n$ are the (complex-scaled) Hermite functions defined [below](#mathematical-conventions). Both families are stable under products, Fourier transforms and convolutions, and finite sums of them can be handled with `WavePacketSum`.

## Installation

The package is not registered. From the Julia package manager:

```julia
pkg> add https://github.com/clguillot/HermiteWavePackets.git
```

It requires Julia 1.11 or later.

## Quick start

```julia
using HermiteWavePackets, StaticArrays, LinearAlgebra
```

### Gaussian wave packets

In dimension 1, the parameters are plain numbers:

```julia
#                                   λ          z         q    p
G = GaussianWavePacket(1.0 + 0.0im, 2.0 + 0.5im, 0.3, 1.2)   # λ exp(-z/2 (x-q)²) exp(ipx)

G(0.5)                       # ≈ 0.798 + 0.535im
integral(G)                  # ∫ G(x) dx
norm_L2(G)                   # ‖G‖₂
inv_fourier(fourier(G))(0.5) # ≈ G(0.5)
```

`q` and `p` are optional (a missing `p` gives a real-phase function, a missing `q` means `q = 0`). The alias `Gaussian` is a `GaussianWavePacket` *without* momentum and with a real `z`:

```julia
R = Gaussian(2.0, 1.5, 0.7)   # 2 exp(-1.5/2 (x-0.7)²)
R isa Gaussian                # true
```

In dimension $D$, `z` is a `Diagonal` or `Symmetric` matrix (`SDiagonal`/`Symmetric` of an `SMatrix`), and `q`, `p` are `SVector`s:

```julia
z  = Symmetric(@SMatrix [2.0 0.3; 0.3 1.0])
G2 = GaussianWavePacket(1.0, z, SVector(0.0, 1.0), SVector(0.5, 0.0))

G2(SVector(0.1, 0.2))        # evaluation at a point
G2(0.1, 0.2)                 # same thing

# Plain arrays are also accepted, if the dimension is given explicitly
G2 = GaussianWavePacket{2}(1.0, Symmetric([2.0 0.3; 0.3 1.0]), [0.0, 1.0], [0.5, 0.0])

integral(G2, Tuple{2})       # integrates out the 2nd variable: a GaussianWavePacket in x₁
evaluate(G2, SVector(0.7), Tuple{2})  # freezes x₂ = 0.7: a GaussianWavePacket in x₁
```

Operations return new wave packets:

```julia
G3 = G * GaussianWavePacket(0.5, 1.0, -0.2, 0.0)    # pointwise product, still a GaussianWavePacket
C  = convolution(G, G3)                              # convolution, still a GaussianWavePacket

dot_L2(G, G3)                # ∫ conj(G) G3
dot_∇(G, G3)                 # ∫ ⟨∇G, ∇G3⟩
dot_∇(G, G3, [0.3])          # ∫ ⟨(D-c)G, (D-c)G3⟩ with D = -i∇ and c = 0.3
```

### Hermite wave packets (1D)

```julia
Λ = SVector(1.0, 0.5im, 0.25)                            # coefficients on ψ₀, ψ₁, ψ₂
H = HermiteWavePacket1D(Λ, 1.2 + 0.3im, 0.1, 0.8)        # Λ, z, q, p

H(0.3)                       # evaluation (Clenshaw algorithm)
norm_L2(H)                   # = sqrt(sum(abs2, Λ)), since the ψₙ are orthonormal
Hr = HermiteFct1D(SVector(1.0, 0.5), 1.2, 0.1)           # real variance, no momentum

# Hermite and Gaussian wave packets can be combined
HG = H * G                                  # HermiteWavePacket1D with N = 3
P  = polynomial_product(H, SVector(1.0, 2.0), 0.5)   # (1 + 2(x-0.5)) H(x)
dot_L2(H, G)
truncate_to_gaussian(H)                     # keeps the ψ₀ component, as a GaussianWavePacket
HermiteWavePacket1D(G)                      # exact conversion of a 1D Gaussian
```

The number of coefficients `N` is part of the type: products and convolutions grow it (`N₁ + N₂ - 1`), Fourier transforms preserve it.

### Sums

`WavePacketSum` represents a finite sum of wave packets. Operations are applied term by term (using linearity or bilinearity), and terms are **never merged**, so the types stay under your control.

```julia
S = G + H                              # a WavePacketSum (here a Tuple of two terms)
S(0.3) ≈ G(0.3) + H(0.3)
norm_L2(S)
convolution(S, G)                      # sum of convolutions

# long sums: use a Vector of a concrete type
V = WavePacketSum([GaussianWavePacket(1.0, 1.0, 0.1k) for k in 1:50])
V(0.3); integral(V)
```

Short, possibly heterogeneous sums are best stored in a `Tuple` (what `+` builds), long homogeneous sums in a `Vector`. Any container of wave packets works (`WavePacketSum{D}(container)`).

## Mathematical conventions

**Hermite functions.** For $z = a + ib$ with $a>0$ and $p,q\in\mathbb R$,

$$\psi_n(z,q,p;x) = \Big(\tfrac{a}{\pi}\Big)^{1/4}\frac{1}{\sqrt{2^n n!}}\, H_n\big(\sqrt a\,(x-q)\big)\, e^{-\frac z2 (x-q)^2}\, e^{ipx},$$

where $H_n$ is the (physicists') Hermite polynomial. For fixed $(z,q,p)$ they are orthonormal in $L^2(\mathbb R)$: $\int \overline{\psi_m}\,\psi_n = \delta_{mn}$. `HermiteWavePacket1D(Λ, z, q, p)` stores $\sum_n \Lambda[n+1]\,\psi_n$ (Julia indices start at 1).

**Other conventions.**

* **Fourier transform.** $\hat f(\xi) = \int e^{-ix\xi} f(x)\,dx$ and $\check f(x) = (2\pi)^{-D}\int e^{ix\xi} f(\xi)\,d\xi$.
* **Inner products.** $\langle f,g\rangle = \int \overline{f}\,g$ (conjugate on the first argument); $\langle\nabla f,\nabla g\rangle$ is the Hermitian product in $\mathbb C^D$ of the gradients.
* **Convolution.** $(f*g)(x) = \int f(y)\,g(x-y)\,dy$.
* `unitary_product(G, b, q, p)` multiplies $G$ by $e^{-\frac i2 (x-q)^\top b\,(x-q)}\,e^{ip\cdot x}$ (with $b$ real symmetric).
* `polynomial_product(H, P, q)` multiplies $H$ by $\sum_k P[k]\,(x-q)^{k-1}$.

## Operations

| Function | Description | Gaussian | Hermite 1D | Sum |
|---|---|:-:|:-:|:-:|
| `G(x)`, `evaluate(G, x)` | evaluation | ✓ | ✓ | ✓ |
| `evaluate_grid(H, x)` | evaluation at all the points of an `SVector` | | ✓ | |
| `evaluate(G, x, Tuple{i,…})` | freezes the listed variables, returns a wave packet in the others | ✓ | | |
| `conj`, `-G`, `w*G`, `G/w` | conjugate, opposite, scalar multiplication and division | ✓ | ✓ | ✓ |
| `G1 * G2` | pointwise product | ✓ | ✓ | ✓ |
| `unitary_product` | product with a Gaussian phase | ✓ | ✓ | ✓ |
| `polynomial_product` | product with a polynomial (1D) | ✓ | ✓ | ✓ |
| `integral`, `∫` | integral over $\mathbb R^D$ | ✓ | ✓ | ✓ |
| `integral(G, Tuple{i,…})` | integral over the listed variables | ✓ | | |
| `fourier`, `inv_fourier` | Fourier transforms | ✓ | ✓ | ✓ |
| `convolution` | convolution product | ✓ | ✓ | ✓ |
| `dot_L2`, `norm2_L2`, `norm_L2` | $L^2$ inner product and norm | ✓ | ✓ | ✓ |
| `dot_∇`, `norm2_∇`, `norm_∇` (+ optional `c`) | $\dot H^1$ inner product and norm (`c`: magnetic shift, see above) | ✓ | | ✓ ¹ |
| `abs`, `abs2` | modulus and squared modulus (a `Gaussian`) | ✓ | | |
| `coulomb_integral` | $\int G(x)/\lvert x\rvert\,dx$ for an isotropic 3D packet | ✓ | | |
| `truncate_to_gaussian` | Gaussian part of the function | ✓ | ✓ | |
| `zero`, `copy`, `show` | | ✓ | ✓ | ✓ |

¹ Provided that all the terms support it.

Mixed operations between a `GaussianWavePacket{1}` and a `HermiteWavePacket1D` are available for `*`, `polynomial_product`, `convolution` and `dot_L2`; the result of a product or a convolution is a `HermiteWavePacket1D`.

`hermite_grid`, `hermite_quadrature` and `hermite_discrete_transform` (Gauss–Hermite nodes and weights, and the transform from values at the nodes to Hermite coefficients) are also exported, as they are the building blocks of the Hermite products.

## Design notes

* **No allocations.** Parameters are stored in `StaticArrays` (`SVector`, `SMatrix`), and the operations on a single wave packet do not allocate (the tests check it). Evaluate with an `SVector` rather than a `Vector` to keep this property. Sums stored in a `Vector` necessarily allocate when they are transformed.
* **Absent parameters are free.** A missing shift or momentum is not stored as `0.0`: it is of type `NullNumber` (from [NullNumbers.jl](https://github.com/clguillot/NullNumbers.jl)), whose arithmetic is resolved at compile time. A `Gaussian` or a `HermiteFct1D` is simply the same type with a `NullNumber` momentum, so the same code handles the real and complex cases at no extra cost.
* **Complex variance.** The real part of `z` gives the Gaussian envelope and the Hermite scale; its imaginary part is treated as a (chirp) phase, so that the $\psi_n$ stay orthonormal.
* **Exactness.** Hermite products and $L^2$ products are computed exactly (up to rounding) with Gauss–Hermite quadrature rules sized for the polynomial degrees involved. The transform matrices are computed once, in extended precision, for each floating-point type and each size `N`, at compile time: the first use of a new `N` is therefore slower than the following ones.
* **Floating-point types.** `Float32` and `Float64` (and the complex counterparts) are supported and preserved by the operations.

## Limitations

* This is version 0.1: the API may still change.
* `HermiteWavePacket1D` only exists in dimension 1. It has no gradient inner products (`dot_∇`, `norm2_∇`), `abs`/`abs2`, or partial integrals yet.
* The real part of `z` must be positive definite (a `PosDefException` is thrown otherwise).
* `unitary_product` takes a matrix `b` and vectors `q`, `p` for Gaussians, but scalars for Hermite wave packets; a `WavePacketSum` mixing both kinds therefore cannot forward a single set of arguments to all its terms.
* `coulomb_integral` is only available for isotropic three-dimensional packets (`Diagonal` covariance with equal entries).

## Running the tests

```julia
pkg> test HermiteWavePackets
```

The tests (written with `Test.jl`) compare each operation with a direct computation (point values, or numerical cubature with [Cubature.jl](https://github.com/JuliaMath/Cubature.jl), which is only needed for testing), and check that the operations do not allocate. They use random data, with a fixed seed in `test/runtests.jl` for reproducibility; the tests can also be run with other draws by calling the functions of `test/test.jl` directly (e.g. `include("test/test.jl"); test_gaussian()`, in an environment with the test dependencies).

## Package layout

| File | Content |
|---|---|
| `src/gaussian_wave_packet.jl` | `GaussianWavePacket` and `Gaussian`, with all their operations |
| `src/hermite_wave_packet_1d.jl` | `HermiteWavePacket1D` and `HermiteFct1D`, and the interface with Gaussians |
| `src/wave_packet_sum.jl` | `WavePacketSum` |
| `src/generics.jl` | functions shared by all the wave packets |
| `src/hermite_quadrature.jl` | Gauss–Hermite nodes, weights and discrete Hermite transform |
| `src/gaussian_arg.jl` | parameter updates of Gaussians under product, Fourier transform and convolution |
| `src/utils.jl` | Clenshaw and Horner algorithms |
| `src/special_cholesky.jl` | Cholesky decomposition of complex symmetric matrices (adapted from StaticArrays.jl, MIT license) |
