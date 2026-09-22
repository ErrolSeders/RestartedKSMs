begin
    using Pkg
    Pkg.activate("KSMdev", shared = true)
end

begin
    using KrylovRestart
    using LinearAlgebra
    using DataFrames
    using AlgebraOfGraphics
    using CairoMakie
end

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

into_df(tr::Trace) = into_df(Val(tr.type), tr)


function into_df(::Val{General}, tr::Trace)
    df = DataFrame(matvecs = (1:tr.restarts) .* tr.values[:restart_length], up_bnd = tr.metrics[:up_bnd], low_bnd = tr.metrics[:low_bnd])
    if haskey(tr.metrics, :abs_err)
        df[!, :abs_err] = tr.metrics[:abs_err]
    end
    return df
end


function plot_krylov_approx_trace(trace_df::DataFrame)

    plts = []

    recipe(symbol, name, color, linestyle) = data(trace_df) * mapping(:matvecs => "MatVec Multiplications", symbol => name) * visual(Lines, color = color, linestyle = linestyle) * visual(label = name)

    abs_errs_included = "abs_err" in names(trace_df)

    push!(plts, recipe(:up_bnd, "Upper Bound", :cornflowerblue, :dash))
    push!(plts, recipe(:low_bnd, "Lower Bound", :green4, :dash))
    if abs_errs_included
        push!(plts, recipe(:abs_err, "Absolute Error", :tomato, :solid))
    end

    return sum(plts)
end


function draw_err_plot(plt, title::String)
    return draw(
        plt,
        axis = (;
            yminorticksvisible = true,
            yminorgridvisible = true,
            yminorticks = IntervalsBetween(5),
            yscale = log10,
            limits = (nothing, (10.0e-22, 10.0e0)),
        ),
        legend = (
            position = :top,
            framevisible = false,
        ),
        figure = (;
            title = title,
            size = (650, 650),
        )
    )
end
