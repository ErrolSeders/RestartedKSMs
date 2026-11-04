using LinearAlgebra
using SparseArrays

using KrylovRestart
using ComplexQuadrature

using MatrixDepot
using DataFrames

using CairoMakie
import AlgebraOfGraphics as AOG

get_Lh(n) = mdopen("poisson", n).A

get_norm_ones_b(Lh) = ones(eltype(Lh), size(Lh, 1)) / norm(ones(eltype(Lh), size(Lh, 1)))

function quad_method_trace_to_dataframe(trace, cs)
    n = length(trace.metrics[:quad_err])

    restarts = first.(trace.metrics[:quad_err])
    abs_errs = [trace.metrics[:abs_err][r - 1] for r in restarts]
    errors = last.(trace.metrics[:quad_err])

    return DataFrame(
        contour_safety = fill(cs, n),
        restart = restarts,
        absolute_errors = abs_errs,
        quad_error = errors,
        quad_order = last.(trace.metrics[:quad1_order])
    )
end

function quad_method_traces_to_dataframe(traces)
    return vcat(
        [
            quad_method_trace_to_dataframe(trace, cs)
                for (cs, trace) in sort(collect(traces), by = first)
        ]...
    )
end

function plot_quad_method_results(df; output_dir = "plots")
    mkpath(output_dir)

    restart_errors = combine(
        groupby(df, [:contour_safety, :restart]),
        :absolute_errors => last => :absolute_errors,
    )
    sort!(restart_errors, [:contour_safety, :restart])
    restart_errors.contour_safety_label = string.(restart_errors.contour_safety)
    error_plot = AOG.data(restart_errors) *
        AOG.mapping(
        :restart => "Restart",
        :absolute_errors => "Absolute error",
        color = :contour_safety_label => "Contour safety",
    ) *
        (AOG.visual(Lines, alpha = 0.6, linestyle = :dash) + AOG.visual(Scatter, alpha = 0.6, marker = :x))

    error_figure = AOG.draw(
        error_plot;
        axis = (
            xlabel = "Restart",
            ylabel = "Absolute error",
            yscale = log10,
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(5),
        ),
        legend = (position = :top, framevisible = false),
        figure = (
            title = "Absolute Error Across Restarts",
            size = (850, 650),
        ),
    )

    convergence_data = copy(df)
    convergence_data.contour_safety_label = string.(convergence_data.contour_safety)
    sort!(convergence_data, [:restart, :contour_safety, :quad_order])
    convergence_plot = AOG.data(convergence_data) *
        AOG.mapping(
        :quad_order => "Quadrature order",
        :quad_error => "Quadrature error",
        color = :contour_safety_label => "Contour safety",
        layout = :restart => "Restart",
    ) *
        AOG.visual(Lines)

    convergence_figure = AOG.draw(
        convergence_plot;
        axis = (
            xlabel = "Quadrature order",
            ylabel = "Quadrature error",
            yscale = log10,
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(5),
        ),
        legend = (position = :top, framevisible = false),
        figure = (
            title = "Quadrature Convergence Within Each Restart",
            size = (1100, 800),
        ),
    )

    heatmap_data = combine(
        groupby(convergence_data, [:restart, :contour_safety, :quad_order]),
        :quad_error => first => :quad_error,
    )
    heatmap_data.log_quad_error = log10.(max.(heatmap_data.quad_error, eps(Float64)))
    heatmap_data.contour_safety_label = string.(heatmap_data.contour_safety)
    heatmap_plot = AOG.data(heatmap_data) *
        AOG.mapping(
        :quad_order => "Quadrature order",
        :contour_safety => "Contour safety",
        :log_quad_error => "log10 quadrature error",
        layout = :restart => "Restart",
    ) *
        AOG.visual(Heatmap)

    heatmap_figure = AOG.draw(
        heatmap_plot;
        axis = (
            xlabel = "Quadrature order",
            ylabel = "Contour safety",
        ),
        colorbar = (label = "log10 quadrature error",),
        figure = (
            title = "Quadrature Error by Contour Safety",
            size = (1100, 800),
        ),
    )

    save(joinpath(output_dir, "contour_safety_absolute_errors.png"), error_figure)
    save(joinpath(output_dir, "contour_safety_quadrature_convergence.png"), convergence_figure)
    save(joinpath(output_dir, "contour_safety_quadrature_heatmap.png"), heatmap_figure)

    return (
        absolute_errors = error_figure,
        quadrature_convergence = convergence_figure,
        quadrature_heatmap = heatmap_figure,
    )
end

function main()

    m = 10

    Lh = -get_Lh(20)
    b = get_norm_ones_b(Lh)

    F(z) = (cos ∘ inv ∘ sqrt)(z) + ((sin ∘ inv ∘ sqrt)(z) / (sqrt(z)))

    reference = F(Lh |> Matrix) * b

    contour_safeties = [2.0, 1.5, 1.3, 1.2, 1.1, 1.05, 1.01, 1.001, 1.0]
    traces = Dict()

    for contour_safety in contour_safeties

        trace = Trace()

        krylov_approx_quad(F, Lh, b, m, quad_tol = 1.0e-10, contour_safety = contour_safety, trace = trace, exact = reference, contour_type = EllipseContour)

        push!(traces, contour_safety => trace)
    end

    df = quad_method_traces_to_dataframe(traces)
    plot_quad_method_results(df)

    return df
end
