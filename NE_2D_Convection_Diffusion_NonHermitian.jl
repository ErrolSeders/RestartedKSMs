### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ d778a3b8-b2da-11f1-a327-974fecff1343
begin
    using Pkg
    Pkg.activate("KSMdev", shared = true)
    Pkg.precompile()
end

# ╔═╡ 75aeda5a-1cdd-4ab2-bf1b-a5a3fafe4e01
begin
    using KrylovRestart
    using ComplexQuadrature
    using LinearAlgebra
    using SparseArrays
    using AlgebraOfGraphics
    using CairoMakie
    using JLD2
end

# ╔═╡ 46730f03-c29a-41e5-be36-d2edd04f5b25
using KrylovKit

# ╔═╡ 8417774e-7507-4ef5-8b9d-a727b23fac61
using DataFrames

# ╔═╡ 539c2677-0703-4300-b057-a9413a039c4f
function plot_complex(z::AbstractVector{<:Complex})
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "Real",
        ylabel = "Imaginary",
        aspect = 1,
    )

    scatter!(ax, real.(z), imag.(z))

    xr = extrema(real.(z))
    yr = extrema(imag.(z))

    cx = sum(xr) / 2
    cy = sum(yr) / 2
    r = (max(xr[2] - xr[1], yr[2] - yr[1]) / 2) * 1.05

    xlims!(ax, cx - r, cx + r)
    ylims!(ax, cy - r, cy + r)

    return fig
end

# ╔═╡ 677ed6b4-b198-447d-9abd-799b1f7be65b
md"""

A 2D convection-diffusion problem:


```math
\Omega = (0,1)^2, \;
\frac{\partial u}{\partial t} = \Delta u + \nu \; v \cdot \nabla u
```

```math
	v(x,y) = (1,1), \;
	u(x,y,t) = 0 \text{ for } x,y \in \partial\Omega
```

"""

# ╔═╡ 0772c910-9639-4659-963d-fdfe18ccdb77
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

# ╔═╡ bb0390a0-10c9-45c6-b4b8-66c81b86c73b
begin
    b = ones(n1^2)
    b = b / norm(b)
end

# ╔═╡ b0a83837-17fc-4686-89ac-ca79875057f1
m = 70

# ╔═╡ 31576b4d-07fb-4277-ad48-a080830a8b8c
reference = exponentiate(Lh, s, b, maxiter = 20000, tol = 1.0e-16)[1]

# ╔═╡ 723e54c4-b247-4df8-9f54-016c883b9d24
begin
    (V, H, η, qm) = ishermitian(sLh) ? lanczos(sLh, b, m) : arnoldi(sLh, b, m)
    first_restart = V * exp(H)[:, 1]
    norm(reference - first_restart), norm(reference - first_restart) / norm(reference)
end

# ╔═╡ 6610ace8-39de-43a0-b177-64daa55894fd
ℓ = first(eigen(H))

# ╔═╡ 5c62e723-2179-4252-9613-ee7dd8889f66
E = build(EllipseContour, first(eigen(H)),1.001)

# ╔═╡ 1b1e3938-82cd-4e96-aab3-1228a264ea80
x,w = resolve(E, 200)

# ╔═╡ 7b0b5b2c-a6cf-41cb-a3fa-4816c5d57462
begin
fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "Real",
        ylabel = "Imaginary",
        aspect = 1,
    )

    scatter!(ax, real.(ℓ), imag.(ℓ), marker = :+, color=:cornflowerblue, alpha=0.6)
	scatter!(ax, real.(x), imag.(x), marker = :circle, color=:tomato)

    xr = extrema(real.(x))
    yr = extrema(imag.(x))

    cx = sum(xr) / 2
    cy = sum(yr) / 2
    rad = (max(xr[2] - xr[1], yr[2] - yr[1]) / 2) * 1.05

    xlims!(ax, cx - rad, cx + rad)
    ylims!(ax, cy - rad, cy + rad)
end

# ╔═╡ 160483b5-1de1-4ad8-aed5-42e959fee25a
fig

# ╔═╡ 9825ba46-01ce-4bad-9a65-9dc96db07177
first_restart

# ╔═╡ 0fe1b742-32a9-4ed1-bc61-77d8bdd04912
begin
    abs_err(res) = norm(reference - res)
    rel_err(res) = abs_err(res) / norm(reference)
end

# ╔═╡ b9542e6d-f17d-407a-96a7-9191fbb3a65b
r = bestapprox_expm_data(16)

# ╔═╡ d79fddcc-c697-4e71-8010-9b244f4d9bbd
trace_original = Trace()

# ╔═╡ 20bafe34-4bc8-471b-9049-1228fb8c568a
reset!(trace_original); res_orig = krylov_approx(exp, sLh, b, m, trace = trace_original, exact = reference)

# ╔═╡ 4e6e8c8a-888b-4aa7-85f8-13d1625c28b5
rel_err(res_orig)

# ╔═╡ 8ce088e6-fde0-4916-b3ee-89bb1ee423af
trace_original

# ╔═╡ 76e2373c-8be0-4a3c-88f4-84bc7f66443a
trace_quad1 = Trace()

# ╔═╡ 96518ac3-383a-4efe-86a4-ceca9e075a41
reset!(trace_quad1); res_quad1 = krylov_approx_quad(exp, sLh, b, m, order = 8, contour_type = EllipseContour,quad_tol = 2.0e-9, maxorder = 1.0e6, trace = trace_quad1, exact = reference,contour_safety=1.001)

# ╔═╡ 3a1a7c8f-f7d2-4fa8-b7e0-305df4d76748
trace_quad1

# ╔═╡ b54d537d-339f-4419-8f5a-a48004cd3dc4
trace_quad1.metrics[:quad_err]

# ╔═╡ 675832ff-9b17-4582-a293-57e9fe21e730
trace_quad1.metrics[:abs_err]

# ╔═╡ 2ab86fbe-3fe4-4bd4-9cf3-88fdd6ec94d5
rel_err(res_quad1)

# ╔═╡ ece9ae7b-9c0e-41e1-a8a2-b33eed97de45
trace_quad2 = Trace()

# ╔═╡ 766341a6-0b6c-489b-a1c8-7473ec15003a
reset!(trace_quad2); res_quad2 = krylov_approx_quad2(exp, sLh, b, m, contour_type = EllipseContour, quad_tol = 1.0e-12, trace = trace_quad2, contour_safety=1.001, exact = reference)

# ╔═╡ 9d4ab100-340f-401a-a0fd-1476d868d53b
rel_err(res_quad2)

# ╔═╡ fcf04aeb-5ef1-4f95-9881-f5ad44492dec
trace_quad2

# ╔═╡ 433f1a91-7c55-4638-ab32-21b3c3fb5f43
trace_quad2.metrics[:abs_err]

# ╔═╡ 8f1a5cad-7f01-462f-90e8-4dbe5d97c183
trace_quad2.metrics[:update_norm]

# ╔═╡ 81a9a11c-30f0-4d4e-8b35-edbcc168db40
begin
    function trace_to_dataframe(trace, algorithm)
        n = length(trace.metrics[:abs_err])

        return DataFrame(
            algorithm = fill(algorithm, n),
            restart = 1:n,
            abs_err = trace.metrics[:abs_err],
        )
    end

    function traces_to_dataframe(traces, algorithms)
        return vcat(
            [
                trace_to_dataframe(trace, algorithm)
                    for (trace, algorithm) in zip(traces, algorithms)
            ]...
        )
    end
end

# ╔═╡ 9351989b-0086-4cf2-9cb1-a52f747ed303
df = traces_to_dataframe([trace_original, trace_quad1, trace_quad2], ["Original","Quadrature-Paper", "Quadrature-Adaptive"])

# ╔═╡ b29c7f1e-e1fa-4131-b3fc-a868feaaa742
begin
    p = data(df) *
        mapping(:restart, :abs_err, color = :algorithm) *
        (visual(Lines) + visual(Scatter, marker = :x))

    draw(
        p,
        axis = (;
            xminorticksvisible = true,
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(5),
            yscale = log10,
            limits = ((0, 8), (10.0e-16, 10.0e0)),
        ),
        legend = (
            framevisible = false,
        ),
        figure = (;
            title = "Absolute Error for Restart Length $(m)",
            size = (850, 650),
        )
    )
end

# ╔═╡ Cell order:
# ╠═d778a3b8-b2da-11f1-a327-974fecff1343
# ╠═75aeda5a-1cdd-4ab2-bf1b-a5a3fafe4e01
# ╠═539c2677-0703-4300-b057-a9413a039c4f
# ╠═677ed6b4-b198-447d-9abd-799b1f7be65b
# ╠═0772c910-9639-4659-963d-fdfe18ccdb77
# ╠═bb0390a0-10c9-45c6-b4b8-66c81b86c73b
# ╠═b0a83837-17fc-4686-89ac-ca79875057f1
# ╠═46730f03-c29a-41e5-be36-d2edd04f5b25
# ╠═31576b4d-07fb-4277-ad48-a080830a8b8c
# ╠═723e54c4-b247-4df8-9f54-016c883b9d24
# ╠═6610ace8-39de-43a0-b177-64daa55894fd
# ╠═5c62e723-2179-4252-9613-ee7dd8889f66
# ╠═1b1e3938-82cd-4e96-aab3-1228a264ea80
# ╠═7b0b5b2c-a6cf-41cb-a3fa-4816c5d57462
# ╠═160483b5-1de1-4ad8-aed5-42e959fee25a
# ╠═9825ba46-01ce-4bad-9a65-9dc96db07177
# ╠═0fe1b742-32a9-4ed1-bc61-77d8bdd04912
# ╠═b9542e6d-f17d-407a-96a7-9191fbb3a65b
# ╠═d79fddcc-c697-4e71-8010-9b244f4d9bbd
# ╠═20bafe34-4bc8-471b-9049-1228fb8c568a
# ╠═4e6e8c8a-888b-4aa7-85f8-13d1625c28b5
# ╠═8ce088e6-fde0-4916-b3ee-89bb1ee423af
# ╠═76e2373c-8be0-4a3c-88f4-84bc7f66443a
# ╠═96518ac3-383a-4efe-86a4-ceca9e075a41
# ╠═3a1a7c8f-f7d2-4fa8-b7e0-305df4d76748
# ╠═b54d537d-339f-4419-8f5a-a48004cd3dc4
# ╠═675832ff-9b17-4582-a293-57e9fe21e730
# ╠═2ab86fbe-3fe4-4bd4-9cf3-88fdd6ec94d5
# ╠═ece9ae7b-9c0e-41e1-a8a2-b33eed97de45
# ╠═766341a6-0b6c-489b-a1c8-7473ec15003a
# ╠═9d4ab100-340f-401a-a0fd-1476d868d53b
# ╠═fcf04aeb-5ef1-4f95-9881-f5ad44492dec
# ╠═433f1a91-7c55-4638-ab32-21b3c3fb5f43
# ╠═8f1a5cad-7f01-462f-90e8-4dbe5d97c183
# ╠═8417774e-7507-4ef5-8b9d-a727b23fac61
# ╠═81a9a11c-30f0-4d4e-8b35-edbcc168db40
# ╠═9351989b-0086-4cf2-9cb1-a52f747ed303
# ╠═b29c7f1e-e1fa-4131-b3fc-a868feaaa742
