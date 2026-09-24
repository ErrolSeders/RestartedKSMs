@enum TraceType begin
    General
    KrylovApproxFunction
    KrylovApproxRational
    Quadrature1
    Quadrature1Stieltjes
    QuadratureSolver
    QuadratureSolverSieltjes
    FrommerStieltjes
    ChenImplicit
    ChenExplicit
end

mutable struct Trace
    type::TraceType
    restarts::Int
    stop::Union{Nothing, StopCode}
    values::Dict{Symbol, Any} #Use this to record single values that do not change between restarts
    metrics::Dict{Symbol, AbstractVector}
end

Trace() = Trace(General, 0, nothing, Dict(), Dict())

set_type!(::Nothing, _type::TraceType) = nothing
set_type!(tr::Trace, type::TraceType) = (tr.type = type)

set_restart!(::Nothing, _k::Int) = nothing
set_restart!(tr::Trace, k::Int) = (tr.restarts = k)

set_stop!(::Nothing, _stop::StopCode) = nothing
set_stop!(tr::Trace, stop::StopCode) = (tr.stop = stop)

log_value!(::Nothing, _value_symb::Symbol, _value) = nothing
log_value!(tr::Trace, value_symb::Symbol, value) = push!(tr.values, value_symb => value)

log_metric!(::Nothing, _metric_symb::Symbol, _metric) = nothing
log_metric!(tr::Trace, metric_symb::Symbol, metric) = push!(get!(tr.metrics, metric_symb, typeof(metric)[]), metric)

function reset!(tr::Trace)
    tr.type = General
    tr.restarts = 0
    tr.stop = nothing
    empty!(tr.values)
    empty!(tr.metrics)
    return tr
end
