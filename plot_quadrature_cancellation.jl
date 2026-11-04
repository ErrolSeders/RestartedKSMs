using CairoMakie, LinearAlgebra

# ───────────────────────────── data structures ─────────────────────────────
mutable struct QuadCall
    order::Int
    k::Vector{Int}
    z::Vector{ComplexF64}
    ω::Vector{ComplexF64}
    y::Vector{Vector{ComplexF64}}            # shifted solve at each node
    ϕ::Vector{ComplexF64}
    f::Vector{ComplexF64}
    term::Vector{Vector{ComplexF64}}         # integral_term at each node
    running_sum::Vector{Vector{ComplexF64}}
end
QuadCall(order) = QuadCall(
    order, Int[], ComplexF64[], ComplexF64[], Vector{ComplexF64}[],
    ComplexF64[], ComplexF64[], Vector{ComplexF64}[], Vector{ComplexF64}[]
)

struct RestartBlock
    restart::Int
    calls::Vector{QuadCall}
end

# ───────────────────────────── parsing ─────────────────────────────
const NUM = "(?:\\d+\\.?\\d*(?:[eE][-+]?\\d+)?|NaN|Inf)"
const RE_COMPLEX = Regex("^\\s*([-+]?$NUM)\\s*([-+])\\s*($NUM)\\s*\\*?\\s*im\\s*\$")

# node k: z, weight: ω, shifted solve: [..], ϕ: ϕ, f: f, integral_term: [..], running_sum: [..]
const RE_NODE = r"node (\d+):\s*(.+?),\s*weight:\s*(.+?),\s*shifted solve:\s*(?:ComplexF64)?\[(.*?)\],\s*ϕ:\s*(.+?),\s*f:\s*(.+?),\s*integral_term:\s*(?:ComplexF64)?\[(.*?)\],\s*running_sum:\s*(?:ComplexF64)?\[(.*)\]"

function parse_complex(s)
    m = match(RE_COMPLEX, s)
    m === nothing && error("could not parse complex number: \"$s\"")
    re = parse(Float64, m[1])
    imv = parse(Float64, m[3])
    return complex(re, m[2] == "-" ? -imv : imv)
end

parse_complex_vector(s) =
    isempty(strip(s)) ? ComplexF64[] : ComplexF64[parse_complex(strip(p)) for p in split(s, ',')]

"""
Read the log and return every restart block in file order. Each block holds all
quadrature calls (one per `Total Order:` line) made during that restart.
"""
function parse_quadrature_log(path)
    blocks = RestartBlock[]
    for line in eachline(path)
        if (m = match(r"Debug: Restart:\s*(\d+)", line)) !== nothing
            push!(blocks, RestartBlock(parse(Int, m[1]), QuadCall[]))
        elseif (m = match(r"Debug: Total Order:\s*(\d+)", line)) !== nothing
            isempty(blocks) && continue
            push!(blocks[end].calls, QuadCall(parse(Int, m[1])))
        elseif (m = match(RE_NODE, line)) !== nothing
            (isempty(blocks) || isempty(blocks[end].calls)) && continue
            c = blocks[end].calls[end]
            push!(c.k, parse(Int, m[1]))
            push!(c.z, parse_complex(m[2]))
            push!(c.ω, parse_complex(m[3]))
            push!(c.y, parse_complex_vector(m[4]))
            push!(c.ϕ, parse_complex(m[5]))
            push!(c.f, parse_complex(m[6]))
            push!(c.term, parse_complex_vector(m[7]))
            push!(c.running_sum, parse_complex_vector(m[8]))
        end
    end
    return blocks
end

"""
Final call of a restart: the last call whose node count equals its `Total Order`.
Falls back to the last call (with a warning) if none is complete.
"""
function final_call(b::RestartBlock)
    isempty(b.calls) && return nothing
    complete = filter(c -> length(c.z) == c.order, b.calls)
    if isempty(complete)
        @warn "Restart $(b.restart): no call has a node count equal to its order; using the last call"
        return b.calls[end]
    end
    return complete[end]
end

"""
`restarts`: `:all`, `:last` (the last restart in the file), or a vector of restart numbers.
If a restart number appears more than once in the log, its last occurrence is used.
"""
function select_final_calls(blocks; restarts = :all)
    latest = Dict{Int, RestartBlock}()
    for b in blocks
        latest[b.restart] = b
    end
    nums = restarts === :all ? sort(collect(keys(latest))) :
        restarts === :last ? [blocks[end].restart] :
        collect(restarts)
    out = Tuple{Int, QuadCall}[]
    for r in nums
        haskey(latest, r) || (@warn "restart $r not found in log"; continue)
        c = final_call(latest[r])
        c === nothing || push!(out, (r, c))
    end
    return out
end

# ───────────────────────────── derived quantities ─────────────────────────────
lg(v) = log10.(max.(abs.(v), floatmin(Float64)))
phase_cos(a) = (m = abs(a); iszero(m) ? 0.0 : real(a) / m)
alignment(S, v) = (d = norm(S) * norm(v); iszero(d) ? 0.0 : real(dot(S, v)) / d)

"""
Check that each logged term equals (1/2πi)·ω·f·ϕ·y, and that the logged running sum
equals the cumulative sum of the terms. Returns the largest relative deviations.
"""
function check_call(c::QuadCall)
    N = length(c.z)
    dev_term = maximum(1:N) do k
        expected = (1 / (2π * im)) * c.ω[k] * c.f[k] * c.ϕ[k] * c.y[k]
        norm(expected - c.term[k]) / max(norm(c.term[k]), floatmin(Float64))
    end
    ps = cumsum(c.term)
    dev_sum = maximum(1:N) do k
        norm(ps[k] - c.running_sum[k]) / max(norm(c.running_sum[k]), floatmin(Float64))
    end
    return dev_term, dev_sum
end

# ───────────────────────────── plotting ─────────────────────────────
"""
Draw y(k) as solid-colored half-segments: each segment k→k+1 is split at its midpoint
and each half takes the color value of the node it touches.
"""
function plot_quadrature_log(
        path; restarts = :all, link_y = false,
        check = true, tol = 1.0e-8,
        colwidth = 520, height = 520, maxcols = 2,
        legendsize = 18, legendheight = 130
    )
    blocks = parse_quadrature_log(path)
    isempty(blocks) && error("no 'Restart:' entries found in $path")
    sel = select_final_calls(blocks; restarts)
    isempty(sel) && error("no complete quadrature calls found for the requested restarts")

    nplots = length(sel)
    ncols = min(maxcols, nplots)
    nrows = cld(nplots, ncols)

    # extra height at the top for the legend
    fig = Figure(size = (colwidth * ncols, height * nrows + legendheight))
    axs = Axis[]

    for (idx, (r, c)) in enumerate(sel)
        row = fld(idx - 1, ncols) + 1
        col = mod(idx - 1, ncols) + 1
        N = length(c.z)

        if check
            dt, ds = check_call(c)
            dt > tol && @warn "Restart $r: logged term differs from (1/2πi)·ω·f·ϕ·y by up to $(round(dt, sigdigits = 2)) (relative). Check the logged factors / normalization."
            ds > tol && @warn "Restart $r: logged running_sum differs from the cumulative sum of terms by up to $(round(ds, sigdigits = 2)) (relative)."
        end

        S = c.running_sum[end]                      # final sum of this call
        tn = [norm(t) for t in c.term]               # ‖tₖ‖, taken directly from the log
        yn = [norm(y) for y in c.y]                  # ‖y(zₖ)‖
        κ = sum(tn) / norm(S)

        ax = Axis(
            fig[row, col];
            xlabel = row == nrows || idx + ncols > nplots ? "Node Index j" : "",
            ylabel = col == 1 ? "Magnitude (Log10)" : "",
            title = "Restart $r,  order $N\n" *
                "κ = $(round(κ, sigdigits = 3))"
        )
        push!(axs, ax)

        lines!(ax, 1:N, lg(c.ω); linewidth = 2, label = "|ωⱼ|")
        lines!(ax, 1:N, lg(c.f); linewidth = 2, label = "|f(zⱼ)|")
        lines!(ax, 1:N, lg(c.ϕ); linewidth = 2, label = "|ϕ(zⱼ)|")
        lines!(ax, 1:N, lg(yn); linewidth = 2, label = "‖y(zⱼ)‖")
        lines!(ax, 1:N, lg(tn); color = :black, linewidth = 3, label = "‖sⱼ‖  (full summand)")
        hlines!(
            ax, [log10(norm(S))]; color = :red, linestyle = :dash, linewidth = 2,
            label = "‖S‖  (final sum)"
        )
    end

    link_y && linkyaxes!(axs...)

    # Single legend in a row above the grid (row 0 is created automatically).
    # All axes share the same series/labels, so the first axis is representative.
    Legend(
        fig[0, 1:ncols], axs[1];
        orientation = :horizontal, nbanks = 2,   # 2 rows of 3 entries
        labelsize = legendsize,
        patchsize = (45, 20),                     # longer line samples
        colgap = 30, rowgap = 6,
        framevisible = false, tellwidth = false
    )

    return fig, sel
end
