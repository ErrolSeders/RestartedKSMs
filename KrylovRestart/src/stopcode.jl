@enum StopCode begin
    MaxRestarts #0
    AbsErrAcc #1
    UpBndAcc #2
    AbsErrLinConv #3
    UpBndLinConv #4
    UpdateAcc #5
    IndicatorAcc #6
    QuadErrorDivergence #7
    QuadOrderDivergence #8
    UpdateNormDivergence #9
end

message(::Val{MaxRestarts}) = "Maximum number of restarts reached."
message(::Val{AbsErrAcc}) = "Absolute error below stopping accuracy."
message(::Val{UpBndAcc}) = "Upper bound below stopping accuracy."
message(::Val{AbsErrLinConv}) = "Linear convergence rate of absolute error greater than minimum decay."
message(::Val{UpBndLinConv}) = "Linear convergence rate upper bound greater than minimum decay."
message(::Val{UpdateAcc}) = "Norm of updates below stopping accuracy."
message(::Val{IndicatorAcc}) = "Error Indicator below stopping accuracy"
message(::Val{QuadErrorDivergence}) = "Quadrature error diverged. Accuracy is likely poor!"
message(::Val{QuadOrderDivergence}) = "Quadrature order has exceeded maximum allowence. Accuracy is likely poor!"
message(::Val{UpdateNormDivergence}) = "Update norm greater than previous update"
message(::Val{M}) where {M} = "Invalid stop code encountered! This should never happen!"
message(c::StopCode) = message(Val(c))
