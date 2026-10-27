# API reference

```@meta
CurrentModule = GaussianWavePackets
```

## Types

```@docs
AbstractWavePacket
GaussianWavePacket
Gaussian
HermiteWavePacket1D
HermiteFct1D
WavePacketSum
```

## Evaluation and integration

```@docs
evaluate
evaluate_grid
integral
∫
coulomb_integral
```

## Transformations

```@docs
unitary_product
polynomial_product
fourier
inv_fourier
convolution
truncate_to_gaussian
truncate_hermite
gaussian_hull
```

## Inner products and norms

```@docs
dot_L2
norm2_L2
norm_L2
dot_∇
norm2_∇
norm_∇
dot_x
norm2_x
norm_x
moment_x
moment_D
moment_xx
moment_xD
```

## Hermite quadrature

```@docs
hermite_quadrature
hermite_grid
hermite_discrete_transform
hermite_discrete_transform!
```

## Utilities

```@docs
core_type
GaussianWavePackets
```

## Index

```@index
```
