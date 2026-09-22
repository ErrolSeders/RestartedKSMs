### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 5db2b550-b1f7-11f1-8b63-ed978a5444b1
begin
	using Pkg
	Pkg.activate("KSMdev", shared = true)
	Pkg.precompile()
end

# ╔═╡ 7f8627a6-e723-4e59-b70a-3c1be6d3dec4
begin
	using KrylovRestart
	using ComplexQuadrature
	using LinearAlgebra
	using SparseArrays
	using AlgebraOfGraphics
	using CairoMakie
	using JLD2
end

# ╔═╡ 08e1d393-38f5-472d-85aa-e08198b49044
using DataFrames

# ╔═╡ 9a2fc887-5dce-485b-b3fc-3686a5e63fd0
function plot_complex(z::AbstractVector{<:Complex})
    fig = Figure()
    ax = Axis(
        fig[1, 1],
        xlabel = "Real",
        ylabel = "Imaginary",
        aspect = DataAspect(),
    )

    scatter!(ax, real.(z), imag.(z))

    return fig
end

# ╔═╡ 0ad6ec02-f225-48bf-b9a1-83a1eb065cf4
begin
	n1 = 100

	T = SymTridiagonal(fill(2.0, n1), fill(-1.0, n1 - 1))
	Id = I(n1) |> sparse

	(⊗)(A,B) = kron(A,B)

	h = 1 / (n1 + 1)
	Lh = (1/h^2) * (T ⊗ Id + Id ⊗ T)
end

# ╔═╡ 6fb1d88c-c952-451f-840d-70f58568bacd
invsqrt = inv ∘ sqrt

# ╔═╡ 44c37768-5e30-401e-a7f3-af093676387f
begin
	b = ones(n1^2)
	b = b / norm(b)
end

# ╔═╡ ed634488-2e06-4a59-8ec7-a2fce4d7da15
#Cond Lh with n1 = 100 ≈ 4080.4 :(

# ╔═╡ bf2b5f42-6316-490b-b909-ef9a333ba4ea
begin
	μ, Q = eigen(T)
	Μ = diagm(μ) |> sparse
	Qkron = (Q ⊗ Q)
	Qkronb = Qkron' * b
	U = spdiagm(invsqrt.(diag(Μ ⊗ Id + Id ⊗ Μ)))
	reference = h * Qkron * U * Qkronb
end

# ╔═╡ d0fcd886-9350-4904-928f-de9e82a1acff
m = 50

# ╔═╡ 03407561-3a72-4b4b-8dae-2fc90c943442
trace_original = Trace()

# ╔═╡ bf50905d-726d-43a6-95e6-150669953fb0
reset!(trace_original); res = krylov_approx(invsqrt,Lh,b,m,trace=trace_original,max_restarts=60,exact=reference)

# ╔═╡ b7c4f09a-db5f-4467-9447-9895eef48083
trace_original

# ╔═╡ 389353ac-a5db-4c66-b430-57ca5cc93ab0
norm(reference - res) / norm(reference)

# ╔═╡ 37f49c32-bc93-4793-9ede-0ced463fd054
trace_quad = Trace()

# ╔═╡ 32219cbe-c3d8-43ba-b679-6c0a69ce5a04
md"""
This is the absolute & relative error of the approximation after a single restart. If our method is not significantly more accuracte than this after several restarts then we know that the initial restart is probably the majority of the contribution to the final result
"""

# ╔═╡ 436cd93d-c9a5-4ed6-a496-f0476cdb07ef
begin 
	(V,H,η,qm) = lanczos(Lh,b,m)
	first_restart = V * invsqrt(H)[1,:]
	norm(reference - first_restart), norm(reference - first_restart) / norm(reference)
end

# ╔═╡ c947cf07-f771-4967-9878-5dda0c1af841
eigen(H)

# ╔═╡ 19d65a45-56d1-49ef-839a-a7d1ee70a1bf
P = build(PacmanContour, first(eigen(H)), 1.001)

# ╔═╡ 0a5aeec8-43b9-49db-b07b-c69822241a4a
plot_complex(resolve(P,200)[1])

# ╔═╡ ba63609d-543d-429f-9562-c125c9d9a672
first_restart

# ╔═╡ f2568672-c417-45f6-b386-380289e706e6
reset!(trace_quad); res2 = krylov_approx_quad(invsqrt,Lh,b,m,trace=trace_quad,max_restarts=60,contour_type=PacmanContour,maxorder=1e6,quad_tol=1e-12,exact=reference,contour_safety=1.001)

# ╔═╡ 9f193c2e-48d2-4acd-a569-f1b37f7dbd13
trace_quad

# ╔═╡ 64436a74-4fdc-4829-9d6e-e7cf64d20836


# ╔═╡ 567073e3-c61f-47b5-ade8-e2b12afc23fd
trace_quad.metrics[:update_norm]

# ╔═╡ cbb5cba4-9429-480d-9972-49c11549eaf9
trace_quad.metrics[:abs_err]

# ╔═╡ 3596ff4c-9ec7-45c1-8b8d-5402ed599d88
norm(reference - (res2 .|> real)) / norm(reference)

# ╔═╡ ef51db80-7065-47ef-ab6d-a6adfb4937ec
trace_quad2 = Trace()

# ╔═╡ 442fffdc-0fe7-46a8-abeb-021ae9abc0e0
reset!(trace_quad2); res3 = krylov_approx_quad2(invsqrt, Lh, b, m, trace=trace_quad2,max_restarts=60,exact=reference,contour_type=PacmanContour,quad_tol=1e-16,contour_safety=1.001)

# ╔═╡ ac33e289-9d15-4bcc-88a8-d9621c394636
norm(reference - res3) / norm(reference)

# ╔═╡ 46ad7c94-ef59-4080-9f7d-9a6116e4a7be
trace_quad2

# ╔═╡ 06457149-bada-4ba7-97d3-e38b50763351
trace_quad2.metrics[:update_norm]

# ╔═╡ f0efc690-7974-40b4-89a0-e5b94f47f044
invsqrt_stjes = inverse_pth_root(2)

# ╔═╡ 394dd7dd-c83a-4fd0-b432-acd54b7a4ee6
trace_stieltjes_1 = Trace()

# ╔═╡ bf16c9e0-c528-4908-b8dd-3c06f10f7b3f
reset!(trace_stieltjes_1); res4 = krylov_approx_quad(invsqrt_stjes,Lh,b,m,quad_tol=3e-10,maxorder=1e7,trace=trace_stieltjes_1,exact=reference)

# ╔═╡ 70b959c3-e6e2-4d1f-aea5-94af92a1b4bb
md"""
This method does not converge to an accurate answer because of the quadrature rule. There is a singularity at -∞ and I'm using a global Möbius transform from (-∞,0] to [-1,1] with a fixed Guass-Legendre quadrature rule. More nodes are placed near the endpoints, but its not able to account for the integrable singularity properly. Increasing the order could fix the problem, but very very slowly, to the point where it is infeasible. 
"""

# ╔═╡ 0098f654-391a-4f90-90db-37e30bc94fe7
norm(reference - res4) / norm(reference)

# ╔═╡ 17ceb2d2-0192-4ecd-9adf-697d9ba698fa
trace_stieltjes_1

# ╔═╡ 51b0ffb4-5dd1-493a-a16a-0df3641ced46
trace_stieltjes_1.metrics[:quad_err]

# ╔═╡ 2a75085a-633f-463c-ba33-dc03d3803bdd
trace_stieltjes_2 = Trace()

# ╔═╡ 6865682e-3019-4528-ae84-a9f91aa1cafa
reset!(trace_stieltjes_2); res5 = krylov_approx_quad2(invsqrt_stjes, Lh, b, m, trace=trace_stieltjes_2,exact=reference)

# ╔═╡ e4760f5c-672c-439c-a898-5404bd5e673e
md"""
This method succeeds because QuadGK uses Gauss-Kronrod quadrature over the entire improper integral. The singularity at -∞ can be properly handled.
"""

# ╔═╡ 98802a63-a6cb-4e9c-8900-d608b31ffd02
norm(reference - res5) / norm(reference)

# ╔═╡ 86db8b9a-2670-4530-94a7-f1e8f0e2eaf8
trace_stieltjes_2

# ╔═╡ d2698ab2-768c-4e4a-bfb2-ce587e7266db
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

# ╔═╡ 27eac18c-cbc6-4414-82cb-50dbce71533c
df = traces_to_dataframe([trace_original, trace_quad, trace_quad2, trace_stieltjes_1, trace_stieltjes_2],["Original", "Quadrature-Paper", "Quadrature-Adaptive","Quadrature-Stieltjes-Paper","Quadrature-Stieltjes-Adaptive"])

# ╔═╡ ded25da0-34c6-4db6-a084-beaa93f1f942
begin
	p = data(df) *
    mapping(:restart, :abs_err, color=:algorithm) *
    (visual(Lines,linestyle=:dash,alpha=0.6) + visual(Scatter, marker=:x))
	
	draw(p, 
		 axis = (; 
				 xminorticksvisible= true,
				 yminorticksvisible = true, 
				 yminorgridvisible = true,
        		 yminorticks = IntervalsBetween(5), 
				 yscale = log10, 
				 limits = ((0,22), (10e-16, 10e-2))
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
# ╠═5db2b550-b1f7-11f1-8b63-ed978a5444b1
# ╠═7f8627a6-e723-4e59-b70a-3c1be6d3dec4
# ╠═9a2fc887-5dce-485b-b3fc-3686a5e63fd0
# ╠═0ad6ec02-f225-48bf-b9a1-83a1eb065cf4
# ╠═6fb1d88c-c952-451f-840d-70f58568bacd
# ╠═44c37768-5e30-401e-a7f3-af093676387f
# ╠═ed634488-2e06-4a59-8ec7-a2fce4d7da15
# ╠═bf2b5f42-6316-490b-b909-ef9a333ba4ea
# ╠═d0fcd886-9350-4904-928f-de9e82a1acff
# ╠═03407561-3a72-4b4b-8dae-2fc90c943442
# ╠═bf50905d-726d-43a6-95e6-150669953fb0
# ╠═b7c4f09a-db5f-4467-9447-9895eef48083
# ╠═389353ac-a5db-4c66-b430-57ca5cc93ab0
# ╠═37f49c32-bc93-4793-9ede-0ced463fd054
# ╟─32219cbe-c3d8-43ba-b679-6c0a69ce5a04
# ╠═436cd93d-c9a5-4ed6-a496-f0476cdb07ef
# ╠═c947cf07-f771-4967-9878-5dda0c1af841
# ╠═19d65a45-56d1-49ef-839a-a7d1ee70a1bf
# ╠═0a5aeec8-43b9-49db-b07b-c69822241a4a
# ╠═ba63609d-543d-429f-9562-c125c9d9a672
# ╠═f2568672-c417-45f6-b386-380289e706e6
# ╠═9f193c2e-48d2-4acd-a569-f1b37f7dbd13
# ╠═64436a74-4fdc-4829-9d6e-e7cf64d20836
# ╠═567073e3-c61f-47b5-ade8-e2b12afc23fd
# ╠═cbb5cba4-9429-480d-9972-49c11549eaf9
# ╠═3596ff4c-9ec7-45c1-8b8d-5402ed599d88
# ╠═ef51db80-7065-47ef-ab6d-a6adfb4937ec
# ╠═442fffdc-0fe7-46a8-abeb-021ae9abc0e0
# ╠═ac33e289-9d15-4bcc-88a8-d9621c394636
# ╠═46ad7c94-ef59-4080-9f7d-9a6116e4a7be
# ╠═06457149-bada-4ba7-97d3-e38b50763351
# ╠═f0efc690-7974-40b4-89a0-e5b94f47f044
# ╠═394dd7dd-c83a-4fd0-b432-acd54b7a4ee6
# ╠═bf16c9e0-c528-4908-b8dd-3c06f10f7b3f
# ╠═70b959c3-e6e2-4d1f-aea5-94af92a1b4bb
# ╠═0098f654-391a-4f90-90db-37e30bc94fe7
# ╠═17ceb2d2-0192-4ecd-9adf-697d9ba698fa
# ╠═51b0ffb4-5dd1-493a-a16a-0df3641ced46
# ╠═2a75085a-633f-463c-ba33-dc03d3803bdd
# ╠═6865682e-3019-4528-ae84-a9f91aa1cafa
# ╠═e4760f5c-672c-439c-a898-5404bd5e673e
# ╠═98802a63-a6cb-4e9c-8900-d608b31ffd02
# ╠═86db8b9a-2670-4530-94a7-f1e8f0e2eaf8
# ╠═08e1d393-38f5-472d-85aa-e08198b49044
# ╠═d2698ab2-768c-4e4a-bfb2-ce587e7266db
# ╠═27eac18c-cbc6-4414-82cb-50dbce71533c
# ╠═ded25da0-34c6-4db6-a084-beaa93f1f942
