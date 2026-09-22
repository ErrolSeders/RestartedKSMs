function unit_vector(T::Type, size::Int, pos::Int)
    unit = zeros(T, size)
    unit[pos] = one(T)
    return unit
end

geteps(z) = z |> real |> typeof |> eps
geteps(A::AbstractArray) = A |> eltype |> real |> eps

function Base.show(io::IO, tr::Trace)
    return print(io, "Trace(restarts=$(tr.restarts), stop=$(isnothing(tr.stop) ? "nothing" : tr.stop), metrics=$(length(tr.metrics)))")
end

function Base.show(io::IO, ::MIME"text/plain", tr::Trace)
    println(io, "Trace")
    println(io, "  restarts: ", tr.restarts)
    if isnothing(tr.stop)
        println(io, "  stop: not set")
    else
        println(io, "  stop: ", tr.stop, " — ", message(tr.stop))
    end

    if isempty(tr.metrics)
        print(io, "  metrics: (none)")
        return
    end

    println(io, "  metrics:")
    for key in sort!(collect(keys(tr.metrics)))
        values = tr.metrics[key]
        n = length(values)
        print(io, "    - ", key, ": ", n, " entr", n == 1 ? "y" : "ies")
        if n > 0
            print(io, " (last=", values[end], ")")
        end
        println(io)
    end
    return
end

"""
Checks whether the entries of v are of the form 1,...,1,0,...0,1,1 (i.e., last two entries are true/1, but not all entries are true/1)
"""
function check_stop_condition(v)
    if length(v) < 2
        return false
    end

    if all(v[(end - 1):end]) && !all(v)
        return true
    end

    return false
end

"""Tracks the 'last-two-true but not all true' decay pattern without storing full history."""
mutable struct DecayTracker
    prev_metric
    have_prev::Bool
    last_ratio_gt::Bool
    prev_ratio_gt::Bool
    seen_ratio_false::Bool
end

DecayTracker() = DecayTracker(NaN, false, false, false, false)

@inline function update_decay!(dt::DecayTracker, metric, min_decay)
    if !dt.have_prev
        dt.prev_metric = metric
        dt.have_prev = true
        return false
    end

    ratio_gt = (metric / dt.prev_metric) > min_decay
    dt.prev_metric = metric

    dt.prev_ratio_gt = dt.last_ratio_gt
    dt.last_ratio_gt = ratio_gt

    if !ratio_gt
        dt.seen_ratio_false = true
    end

    return dt.prev_ratio_gt && dt.last_ratio_gt && dt.seen_ratio_false
end

"""Internal stopping logic (and optional logging)."""
function stop_conditions!(
        A,
        α1,
        β,
        h,
        qm,
        fk,
        m::Int,
        bound::Bool,
        exact,
        tol,
        min_decay,
        trace::Union{Trace, Nothing},
        k::Int,
        up_decay::DecayTracker,
        err_decay::DecayTracker
    )::Union{StopCode, Nothing}

    using_exact = !(exact === nothing || isempty(exact))

    function bound_check!(stop)
        low = β * abs(h[end - 1])
        w = A * qm
        up = β * norm((h[end - 1] - α1 * h[end]) * qm + h[end] * w)

        log_metric!(trace, :low_bnd, low)
        log_metric!(trace, :up_bnd, up)

        if update_decay!(up_decay, up, min_decay)
            stop = UpBndLinConv
        elseif up < tol
            stop = UpBndAcc
        end

        return stop
    end

    function abs_err_check!(stop)
        err = norm(fk - exact)
        log_metric!(trace, :abs_err, err)

        if update_decay!(err_decay, err, min_decay)
            stop = AbsErrLinConv
        elseif err < tol
            stop = AbsErrAcc
        end

        return stop
    end

    function update_norm_check!(stop)
        update_norm = β * norm(h[1:m])
        log_metric!(trace, :update_norm, update_norm)

        if update_norm < tol
            stop = UpdateAcc
        end

        return stop
    end

    stop = nothing

    if using_exact && bound
        stop = bound_check!(stop)
        abs_err_check!(stop)
    elseif !using_exact && bound
        stop = bound_check!(stop)
    elseif using_exact
        stop = abs_err_check!(stop)
    else
        update_norm_check!(stop)

    end

    return stop
end
