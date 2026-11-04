"""
This is a benchmark for the inverse square root numerical experiment.
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
    n1 = 100

    T = SymTridiagonal(fill(2.0, n1), fill(-1.0, n1 - 1))
    Id = I(n1) |> sparse

    (⊗)(A, B) = kron(A, B)

    h = 1 / (n1 + 1)
    Lh = (1 / h^2) * (T ⊗ Id + Id ⊗ T)
end

begin
    b = ones(n1^2)
    b = b / norm(b)
end

invsqrt = inv ∘ sqrt
invsqrt_stjes = inverse_pth_root(2)
m = 50

results = DataFrame(
    algorithm = String[],
    m = Int[],
    time_ns = Float64[],
    memory_B = Int[],
    allocs = Int[],
)

trial1 = @benchmark $krylov_approx($invsqrt, $Lh, $b, $m, tol = 1.0e-13, max_restarts = 60)
est1 = median(trial1)
push!(results, ("RKSM-EE", m, est1.time, est1.memory, est1.allocs))

trial2 = @benchmark $krylov_approx_quad($invsqrt, $Lh, $b, $m, max_restarts = 60, contour_type = PacmanContour, quad_tol = 1.0e-13, stop_tol = 1.0e-13, contour_safety = 1.1)
est2 = median(trial2)
push!(results, ("RKSM-FGS", m, est2.time, est2.memory, est2.allocs))

trial3 = @benchmark $krylov_approx_quad2($invsqrt, $Lh, $b, $m, max_restarts = 60, contour_type = PacmanContour, quad_tol = 1.0e-13, stop_tol = 1.0e-13, contour_safety = 1.1)
est3 = median(trial3)
push!(results, ("RKSM-FGS-QuadGK", m, est3.time, est3.memory, est3.allocs))

trial4 = @benchmark krylov_approx_quad($invsqrt_stjes, $Lh, $b, $m, quad_tol = 1.0e-13, stop_tol = 1.0e-13, contour_type = SquaredStieltjesContour)
est4 = median(trial4)
push!(results, ("RKSM-FGS-Stieltjes", m, est4.time, est4.memory, est4.allocs))

trial5 = @benchmark krylov_approx_quad2($invsqrt_stjes, $Lh, $b, $m, stop_tol = 1.0e-13)
est5 = median(trial5)
push!(results, ("RKSM-FGS-Stieltjes-QuadGK", m, est5.time, est5.memory, est5.allocs))

trial6 = @benchmark krylov_approx_stieltjes($invsqrt_stjes, $Lh, $b, $m, order = 60)
est6 = median(trial6)
push!(results, ("RKSM-FS", m, est6.time, est6.memory, est6.allocs))

CSV.write("benchmarks_inverse_square_root.csv", results)
