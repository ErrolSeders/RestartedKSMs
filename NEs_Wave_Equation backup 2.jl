### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ c0813416-bd04-11f1-809d-4f2d479061aa
begin
    # enable if using a Mac
    #using AppleAccelerate  Use Apple accelerate BLAS & LAPACK

    push!(LOAD_PATH, joinpath(@__DIR__, "KrylovRestart"))
    push!(LOAD_PATH, joinpath(@__DIR__, "ComplexQuadrature"))
    using ComplexQuadrature
    using KrylovRestart
end

# ╔═╡ a50ac98d-cf1f-4477-8b76-8c9daedae42b
begin
    using LinearAlgebra
    using SparseArrays
    using AlgebraOfGraphics
    using CairoMakie
    using Tullio
    using JLD2
end

# ╔═╡ 50719c76-7f59-4451-bc56-f71791fb8c82
using BenchmarkTools

# ╔═╡ 77650c2d-d0e9-4ad1-9564-629748d6f382
using DataFrames

# ╔═╡ ba98239f-39ad-4eb4-853d-190ed2243c6c
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

# ╔═╡ 17514572-15cd-4794-87b7-521e9d8755fc
begin
    n1 = 100
    c = 2

    T = SymTridiagonal(fill(2.0, n1), fill(-1.0, n1 - 1))
    Id = I(n1) |> sparse

    (⊗)(A, B) = kron(A, B)

    h = 1 / (n1 + 1)
    Lh = (c^2 / h^2) * (T ⊗ Id + Id ⊗ T)
end

# ╔═╡ 1341634e-2564-49c3-911d-84cabc602d26
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

# ╔═╡ c7d8e576-621e-4626-a58b-70e90c6b79db
b = gaussian_b(n1, h, 4 * h)

# ╔═╡ 7f9ecad6-9dfc-4dd1-ba4c-8bdc5890c397
B = reshape(b, n1, n1)

# ╔═╡ 6183574d-74d6-4443-bff7-1afa2ffe3896
heatmap(B)

# ╔═╡ 99be8925-4bad-4b0e-abaa-011d5083bebb
t = 0.5

# ╔═╡ b156003a-5b03-489c-81b1-3239b4f8e898
func1 = z -> cos(sqrt(z) * t)

# ╔═╡ 3cf3b31f-376c-4405-be00-a7b84d7add0c
func2 = z -> (sin(sqrt(z) * t) / inv(sqrt(z)))

# ╔═╡ 3ee79536-143c-45ef-910f-5fcc26f7ea67
α = 1 / t

# ╔═╡ 41040d91-a6e3-4327-abb5-d05cb8efc513
F(z) = func1(z) + α * func2(z)

# ╔═╡ 0389565c-1991-4cb4-a19a-b4e94415618f
begin
    λ, Q = eigen(T)

    D = Q' * B * Q

    # Eigenvalues of the 2D operator
    μ = (c^2 / h^2) .* (λ .+ λ')

    D .*= F.(μ)

    U = Q * D * Q'
end

# ╔═╡ 202f3e65-74b1-468c-9b60-c80af41e359d
heatmap(U)

# ╔═╡ 33508fab-3d00-4580-83c4-ab3d053fbfb6
# Reference answer is 1e-13 rel. error from an "exact" answer forming f(Lh), tested for a smaller value of n1 = 10.

reference = vec(U)

# ╔═╡ 1c2226bc-c5e1-41a2-832e-752493dd0644
ref_abs_err(res) = norm(reference - res)

# ╔═╡ 653a5b74-e001-436b-a198-8ddb019a559b
ref_rel_err(res) = ref_abs_err(res) / norm(reference)

# ╔═╡ f31bbe3f-3c19-4de9-9173-a5a1b9ebdad0


# ╔═╡ b41e04b6-eba8-46c7-b461-686216621404
m = 40

# ╔═╡ 6e9e40c1-8f70-4f22-afc0-34bba5dd778a
# ╠═╡ disabled = true
# ╠═╡ skip_as_script = true
#=╠═╡
begin

	using Logging
	contour_safe = 1.0001
	
	open("quadrature_values_wave_equation_$(contour_safe).log", "w") do io
		with_logger(SimpleLogger(io, Logging.Debug)) do
			quad1_trace = Trace(); res_quad1 = krylov_approx_quad(F, Lh, b, m, trace = quad1_trace, exact = reference, contour_type=EllipseContour,contour_safety=contour_safe, maxorder=1e5, stop_tol= 1.0e-12, quad_tol = 1.0e-9) .|> real
		end
	end
	
end
  ╠═╡ =#

# ╔═╡ 6399e558-4628-4528-94e3-5372429c98a7
begin
    (V, S, _, _) = lanczos(Lh, b, m)
    first_restart = V * F(S)[:, 1]
end

# ╔═╡ f082cf9c-1004-45ec-b46f-284077ce023e
heatmap(reshape(first_restart, n1, n1))

# ╔═╡ 4fbbf2c1-21b9-4a40-bb46-088abf794686
ref_abs_err(first_restart)

# ╔═╡ d426ea1f-aef1-4dd0-9e72-03b401b544d2
original_trace = Trace(); res_original = krylov_approx(F, Lh, b, m, exact = reference, tol=1.0e-12, trace = original_trace)

# ╔═╡ c89a22ae-7516-4312-82b5-c7afb843428c
@benchmark krylov_approx($F,$Lh,$b,$m, tol=1e-12)

# ╔═╡ d5ca0567-b018-4c9f-8c65-a4f12a91340c
heatmap(reshape(res_original, n1, n1))

# ╔═╡ 18f58be8-2f6b-45f2-b200-c1448072d8dd
ref_rel_err(res_original)

# ╔═╡ 4512b70e-7eba-4136-8522-593602b00f62
original_trace

# ╔═╡ 03b73f66-decb-42de-929a-982e185f7693
original_trace.metrics[:low_bnd]

# ╔═╡ 4637cefe-5aa7-4deb-b50d-97033201a8e0
quad1_trace = Trace(); res_quad1 = krylov_approx_quad(F, Lh, b, m, trace = quad1_trace, exact = reference, contour_type=EllipseContour,contour_safety=1.0001, maxorder=1e5, stop_tol= 1.0e-12, quad_tol = 1.0e-9) .|> real

# ╔═╡ f0e87fe1-607f-42d7-b388-f6f89b377d1d
@benchmark krylov_approx_quad($F,$Lh,$b,$m,contour_type=1.0001,maxorder=1e5,stop_tol=1e-12,quad_tol=1e-9)

# ╔═╡ 1e37f414-b5e3-4a79-9829-ea1ab94a848a
heatmap(reshape(res_quad1, n1, n1))

# ╔═╡ 88a01284-2b4f-4b3c-a63e-e12adb61a0d1
quad1_trace

# ╔═╡ f07f85c2-71bb-45b6-93c4-75b26551c08c
quad1_trace.metrics[:quad1_order]

# ╔═╡ 737f779f-f9f2-489e-a520-8fc0812d2588
ref_rel_err(res_quad1)

# ╔═╡ 66fef001-38f7-4207-a996-95b96a5bb98e
quad2_trace = Trace(); res_quad2 = krylov_approx_quad2(F, Lh, b, m, trace = quad2_trace, exact = reference, stop_tol = 1.0e-12, contour_type=EllipseContour, contour_safety=1.001, quad_tol = 1.0e-12) .|> real

# ╔═╡ 4d468049-1d4a-4a83-a302-5f5e6fa26501
@benchmark krylov_approx_quad2($F,$Lh,$b,$m,stop_tol=1e-12,contour_type=EllipseContour,contour_safety=1.001,quad_tol=1e-12)

# ╔═╡ 8e42632a-99ad-4c50-ba07-3afa19fc80f8
heatmap(reshape(res_quad2, n1, n1))

# ╔═╡ f1ca212b-9a4c-4b63-9a7e-3cafe5a2bb8f
quad2_trace

# ╔═╡ 64bcc25c-164c-4fa9-b029-82f3d348876f
ref_rel_err(res_quad2)

# ╔═╡ 2e8e0bb5-5be6-4cc8-b936-110a436edb6e
begin
	function trace_to_dataframe(trace, algorithm)
	    n = length(trace.metrics[:abs_err])
	
	    DataFrame(
	        algorithm = fill(algorithm, n),
	        restart = 1:n,
	        abs_err = trace.metrics[:abs_err],
	    )
	end
	
	function traces_to_dataframe(traces, algorithms)
	    vcat([
	        trace_to_dataframe(trace, algorithm)
	        for (trace, algorithm) in zip(traces, algorithms)
	    ]...)
	end
end

# ╔═╡ f6e505fa-a204-4d26-a923-97582dbb796d
df = traces_to_dataframe([original_trace, quad1_trace, quad2_trace],["RKSM-EE", "RKSM-FGS", "RKSM-FGS-QuadGK"])

# ╔═╡ 517593cf-f4d0-4772-955b-474af831b8cd
 begin
	p = data(df) *
    mapping(:restart => "Restart Cycle", :abs_err => "Absolute Error", color=:algorithm => "Algorithm") *
    (visual(Lines,alpha=0.6) + visual(Scatter, alpha=0.8,marker=:x))
	
	draw(p, 
		 axis = (; 
				 xminorticksvisible= true,
				 yminorticksvisible = true, 
				 yminorgridvisible = true,
        		 yminorticks = IntervalsBetween(5), 
				 yscale = log10, 
				 limits = ((0,20), (10e-12, 10e2))
				),
		 legend =(
			 framevisible=false,
			),
		 figure = (;
			 title = "Absolute Error for Restart Length $(m)",
			 size=(850,650)
		 )
		)	
end

# ╔═╡ Cell order:
# ╠═c0813416-bd04-11f1-809d-4f2d479061aa
# ╠═a50ac98d-cf1f-4477-8b76-8c9daedae42b
# ╠═50719c76-7f59-4451-bc56-f71791fb8c82
# ╠═ba98239f-39ad-4eb4-853d-190ed2243c6c
# ╠═17514572-15cd-4794-87b7-521e9d8755fc
# ╠═1341634e-2564-49c3-911d-84cabc602d26
# ╠═c7d8e576-621e-4626-a58b-70e90c6b79db
# ╠═7f9ecad6-9dfc-4dd1-ba4c-8bdc5890c397
# ╠═6183574d-74d6-4443-bff7-1afa2ffe3896
# ╠═99be8925-4bad-4b0e-abaa-011d5083bebb
# ╠═b156003a-5b03-489c-81b1-3239b4f8e898
# ╠═3cf3b31f-376c-4405-be00-a7b84d7add0c
# ╠═3ee79536-143c-45ef-910f-5fcc26f7ea67
# ╠═41040d91-a6e3-4327-abb5-d05cb8efc513
# ╠═0389565c-1991-4cb4-a19a-b4e94415618f
# ╠═202f3e65-74b1-468c-9b60-c80af41e359d
# ╠═33508fab-3d00-4580-83c4-ab3d053fbfb6
# ╠═1c2226bc-c5e1-41a2-832e-752493dd0644
# ╠═653a5b74-e001-436b-a198-8ddb019a559b
# ╠═f31bbe3f-3c19-4de9-9173-a5a1b9ebdad0
# ╠═b41e04b6-eba8-46c7-b461-686216621404
# ╠═6399e558-4628-4528-94e3-5372429c98a7
# ╠═f082cf9c-1004-45ec-b46f-284077ce023e
# ╠═4fbbf2c1-21b9-4a40-bb46-088abf794686
# ╠═d426ea1f-aef1-4dd0-9e72-03b401b544d2
# ╠═c89a22ae-7516-4312-82b5-c7afb843428c
# ╠═d5ca0567-b018-4c9f-8c65-a4f12a91340c
# ╠═18f58be8-2f6b-45f2-b200-c1448072d8dd
# ╠═4512b70e-7eba-4136-8522-593602b00f62
# ╠═03b73f66-decb-42de-929a-982e185f7693
# ╠═4637cefe-5aa7-4deb-b50d-97033201a8e0
# ╠═f0e87fe1-607f-42d7-b388-f6f89b377d1d
# ╠═6e9e40c1-8f70-4f22-afc0-34bba5dd778a
# ╠═1e37f414-b5e3-4a79-9829-ea1ab94a848a
# ╠═88a01284-2b4f-4b3c-a63e-e12adb61a0d1
# ╠═f07f85c2-71bb-45b6-93c4-75b26551c08c
# ╠═737f779f-f9f2-489e-a520-8fc0812d2588
# ╠═66fef001-38f7-4207-a996-95b96a5bb98e
# ╠═4d468049-1d4a-4a83-a302-5f5e6fa26501
# ╠═8e42632a-99ad-4c50-ba07-3afa19fc80f8
# ╠═f1ca212b-9a4c-4b63-9a7e-3cafe5a2bb8f
# ╠═64bcc25c-164c-4fa9-b029-82f3d348876f
# ╠═77650c2d-d0e9-4ad1-9564-629748d6f382
# ╠═2e8e0bb5-5be6-4cc8-b936-110a436edb6e
# ╠═f6e505fa-a204-4d26-a923-97582dbb796d
# ╠═517593cf-f4d0-4772-955b-474af831b8cd
