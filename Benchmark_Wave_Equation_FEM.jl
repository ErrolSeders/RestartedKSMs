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
    using Ferrite
    using JLD2
    using CSV
end

function setup_fem(; L = 5.0, n = 64)
    grid = generate_grid(Triangle, (n, n), Tensors.Vec(-L, -L), Tensors.Vec(L, L))
    ip = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(2)
    cv = CellValues(qr, ip)

    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)

    ch = ConstraintHandler(dh)
    ∂Ω = union(
        getfacetset(grid, "left"), getfacetset(grid, "right"),
        getfacetset(grid, "top"), getfacetset(grid, "bottom")
    )
    add!(ch, Dirichlet(:u, ∂Ω, x -> 0.0))
    close!(ch)

    return grid, dh, cv, ch
end

function assemble_KM(cv, dh; c = 1.0)
    K = allocate_matrix(dh)
    M = allocate_matrix(dh)
    nb = getnbasefunctions(cv)
    Ke = zeros(nb, nb)
    Me = zeros(nb, nb)
    asmK = start_assemble(K)
    asmM = start_assemble(M)

    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0.0); fill!(Me, 0.0)
        for q in 1:getnquadpoints(cv)
            dΩ = getdetJdV(cv, q)
            for i in 1:nb
                Ni = shape_value(cv, q, i)
                ∇Ni = shape_gradient(cv, q, i)
                for j in 1:nb
                    Nj = shape_value(cv, q, j)
                    ∇Nj = shape_gradient(cv, q, j)
                    Ke[i, j] += c^2 * (∇Ni ⋅ ∇Nj) * dΩ
                    Me[i, j] += Ni * Nj * dΩ
                end
            end
        end
        assemble!(asmK, celldofs(cell), Ke)
        assemble!(asmM, celldofs(cell), Me)
    end
    return K, M
end

gaussian(x, x0, σ) = exp(-sum(abs2, x - x0) / (2σ^2))

function build_wave_problem(; L = 5.0, n = 64, c = 1.0, σ = 0.5, x0 = [0.0, 0.0], v_amp = 0.1, σ_v = 1.0)
    grid, dh, cv, ch = setup_fem(; L, n)
    K, M = assemble_KM(cv, dh; c)

    # nodal interpolation of the initial data (P1 dofs = nodes)
    u0 = zeros(ndofs(dh)); v0 = zeros(ndofs(dh))
    apply_analytical!(u0, dh, :u, x -> gaussian(x, x0, σ))
    apply_analytical!(v0, dh, :u, x -> v_amp * gaussian(x, x0, σ_v))

    # drop Dirichlet dofs
    free = setdiff(1:ndofs(dh), ch.prescribed_dofs)
    Kff = K[free, free]
    Mff = M[free, free]

    # row-sum lumping (positive diagonal for P1 on triangles)
    mL = vec(sum(Mff, dims = 2))
    Mhalf = Diagonal(sqrt.(mL))
    Mihalf = Diagonal(1 ./ sqrt.(mL))

    A = Symmetric(Mihalf * Kff * Mihalf)    # sparse SPD matrix of (2.7)
    u0 = Mhalf * u0[free]                    # initial "position"  for cos(√A t) y0
    v0 = Mhalf * v0[free]                    # initial "velocity"  for (√A)⁻¹sin(√A t) y0'


    to_full(z) = (u = zeros(ndofs(dh)); u[free] = Mihalf * z; u)

    return (; A, u0, v0, to_full, dh, grid, free, K = Kff, M = Mff, mL)
end

prob = build_wave_problem(n = 80, L = 6.0, c = 2.0)

begin
    A = prob.A |> sparse
    u0 = prob.u0
    v0 = prob.v0
end

begin
    t = 22.2
    cossqrt = (z) -> cos(sqrt(z) * t)
    sinsqrt = (z) -> inv(sqrt(z)) * sin(sqrt(z) * t)
end


m = 70

results = DataFrame(
    algorithm = String[],
    m = Int[],
    time_ns = Float64[],
    memory_B = Int[],
    allocs = Int[],
)

trial1cos = @benchmark $krylov_approx($cossqrt, $A, $u0, $m, tol = 1.0e-13)
est1cos = median(trial1cos)
push!(results, ("RKSM-EE-COS", m, est1cos.time, est1cos.memory, est1cos.allocs))

trial1sin = @benchmark $krylov_approx($sinsqrt, $A, $v0, $m, tol = 1.0e-13)
est1sin = median(trial1sin)
push!(results, ("RKSM-EE-SIN", m, est1sin.time, est1sin.memory, est1sin.allocs))

trial2cos = @benchmark $krylov_approx_quad($cossqrt, $A, $u0, $m, contour_safety = 1.001, contour_type = EllipseContour, stop_tol = 1.0e-13, quad_tol = 1.0e-10)
est2cos = median(trial2cos)
push!(results, ("RKSM-FGS-COS", m, est2cos.time, est2cos.memory, est2cos.allocs))

trial2sin = @benchmark $krylov_approx_quad($sinsqrt, $A, $v0, $m, contour_safety = 1.001, contour_type = EllipseContour, stop_tol = 1.0e-13, quad_tol = 1.0e-10)
est2sin = median(trial2sin)
push!(results, ("RKSM-FGS-SIN", m, est2sin.time, est2sin.memory, est2sin.allocs))

trial3cos = @benchmark $krylov_approx_quad2($cossqrt, $A, $u0, $m, stop_tol = 1.0e-13, contour_type = EllipseContour, contour_safety = 1.001, quad_tol = 1.0e-13)
est3cos = median(trial3cos)
push!(results, ("RKSM-FGS-QuadGK-COS", m, est3cos.time, est3cos.memory, est3cos.allocs))


trial3sin = @benchmark $krylov_approx_quad2($sinsqrt, $A, $v0, $m, stop_tol = 1.0e-13, contour_type = EllipseContour, contour_safety = 1.001, quad_tol = 1.0e-13)
est3sin = median(trial3sin)
push!(results, ("RKSM-FGS-QuadGK-SIN", m, est3sin.time, est3sin.memory, est3sin.allocs))

CSV.write("benchmarks_wave_equation_fem.csv", results)
