"""
This is a benchmark for the convection diffusion numerical experiment.
The idea is to give a clean readout of benchmarks without any overhead from the
Pluto runtime.
"""

begin
    # enable if using a Mac
    using AppleAccelerate # Use Apple accelerate BLAS & LAPACK

    push!(LOAD_PATH, joinpath(@__DIR__, "KrylovRestart"))
    push!(LOAD_PATH, joinpath(@__DIR__, "ComplexQuadrature"))
    using ComplexQuadrature
    using KrylovRestart

    using BenchmarkTools
    using DataFrames
    using LinearAlgebra
    using SparseArrays
    using Tullio
    using JLD2
    using CSV
end

begin

    ν = 100
    n1 = 200

    s = 2 * 10.0e-4

    v = [1, 1]

    T = SymTridiagonal(fill(2.0, n1), fill(-1.0, n1 - 1))
    Id = I(n1) |> sparse

    (⊗)(A, B) = kron(A, B)

    h = 1 / (n1 + 1)
    D2 = (-1 / h^2) * (T ⊗ Id + Id ⊗ T)

    C = Tridiagonal(fill(-1.0, n1 - 1), fill(0.0, n1), fill(1.0, n1 - 1))

    D1 = (1 / h) * (C ⊗ Id + Id ⊗ C)

    Lh = (D2 + ν * D1)
    sLh = s * Lh
end

begin
    b = ones(n1^2)
    b = b / norm(b)
end

m = 70

results = DataFrame(
    algorithm = String[],
    m = Int[],
    time_ns = Float64[],
    memory_B = Int[],
    allocs = Int[],
)

trial1 = @benchmark $krylov_approx($exp, $sLh, $b, $m)
est1 = median(trial1)
push!(results, ("RKSM-EE", m, est1.time, est1.memory, est1.allocs))

trial3 = @benchmark $krylov_approx_quad($exp, $sLh, $b, $m, contour_type = EllipseContour, quad_tol = 1.0e-9, contour_safety = 1.01)
est3 = median(trial3)
push!(results, ("RKSM-FGS", m, est3.time, est3.memory, est3.allocs))

trial4 = @benchmark $krylov_approx_quad2($exp, $sLh, $b, $m, contour_type = EllipseContour, quad_tol = 1.0e-12, contour_safety = 1.001)
est4 = median(trial4)
push!(results, ("RKSM-FGS-QuadGK", m, est4.time, est4.memory, est4.allocs))

CSV.write("benchmarks_convdiffu_nonherm.csv", results)
