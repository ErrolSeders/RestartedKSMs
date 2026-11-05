module KrylovRestart

using ComplexQuadrature
using LinearAlgebra
using SparseArrays
using KrylovKit
using Integrals
using Logging

if Sys.ARCH == :aarch64
    using AppleAccelerate
end



include("./stopcode.jl")
include("./tracing.jl")
include("./utils.jl")
include("./lanczos.jl")
include("./arnoldi.jl")
include("./rationalapprox.jl")
include("./stieltjes.jl")
include("./krylov_approximation.jl")

export RationalApproximation, bestapprox_expm_data
export StieltjesFunction, inverse_pth_root, log1p_over_z
export arnoldi, lanczos
export StopCode, Trace, TraceType, krylov_approx
export krylov_approx_quad, krylov_approx_quad2, message
export krylov_approx_stieltjes
export krylov_approx_chen_explicit, krylov_approx_chen_implicit,
    krylov_approx_chen_implicit_2, quad_test
export log_metric!, log_value!, reset!

end # module KrylovRestart
