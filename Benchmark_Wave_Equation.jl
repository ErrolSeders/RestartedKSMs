"""
This is a benchmark for the wave equation numerical experiment.
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
    c = 2

    T = SymTridiagonal(fill(2.0, n1), fill(-1.0, n1 - 1))
    Id = I(n1) |> sparse

    (⊗)(A, B) = kron(A, B)

    h = 1 / (n1 + 1)
    Lh = (c^2 / h^2) * (T ⊗ Id + Id ⊗ T)
end

function gaussian_b(n, h, σ)
    b = zeros(Float64, n, n)

    @tullio b[i, j] = exp(
        -(
            (i * h - 0.5)^2 +
                (j * h - 0.5)^2
        ) / (2σ^2)
    )

    b ./= norm(b)

    return vec(b)
end

b = gaussian_b(n1, h, 4 * h)
t = 0.5
func1 = z -> cos(sqrt(z) * t)
func2 = z -> (sin(sqrt(z) * t) * inv(sqrt(z)))
α = 1 / t
F(z) = func1(z) + α * func2(z)

m = 40

results = DataFrame(
    algorithm = String[],
    m = Int[],
    time_ns = Float64[],
    memory_B = Int[],
    allocs = Int[],
)

trial1 = @benchmark $krylov_approx($F, $Lh, $b, $m, tol = 1.0e-12)
est1 = median(trial1)
push!(results, ("RKSM-EE", m, est1.time, est1.memory, est1.allocs))

trial2 = @benchmark $krylov_approx_quad($F, $Lh, $b, $m, contour_safety = 1.0001, contour_type = EllipseContour, maxorder = 1.0e5, stop_tol = 1.0e-12, quad_tol = 1.0e-9)
est2 = median(trial2)
push!(results, ("RKSM-FGS", m, est2.time, est2.memory, est2.allocs))

trial3 = @benchmark $krylov_approx_quad2($F, $Lh, $b, $m, stop_tol = 1.0e-12, contour_type = EllipseContour, contour_safety = 1.001, quad_tol = 1.0e-12)
est3 = median(trial3)
push!(results, ("RKSM-FGS-QuadGK", m, est3.time, est3.memory, est3.allocs))

CSV.write("benchmarks_wave_equation.csv", results)
