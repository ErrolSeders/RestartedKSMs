function _update_alphas(α1, α2, H)

    if size(H) == (1, 1)
        μ = λ = H[1]
    else
        s, _ = eigen(H)
        μ = s .|> real |> minimum
        λ = s .|> real |> maximum
    end
    return min(μ, α1), max(λ, α2)

end

"""
Check whether the inputs conform to expected sizes.
`A` is square of dim `n x n`, `b` is of length `n`
and `exact` is of length `n` if provided.
"""
function _check_sizes(exact, A, b)
    !(size(A, 1) == size(A, 2)) && throw(ArgumentError(lazy"Matrix A must be square"))
    !(size(b, 1) == size(A, 1)) && throw(ArgumentError(lazy"b must have the same dim as Matrix first dim"))
    if !isnothing(exact)
        !(size(exact, 1) == size(A, 1)) && throw(ArgumentError(lazy"exact must be same length as b"))
    end
    return nothing
end

"""
Build the expanded Arnoldi decomposition matrix `Hhat`
The resulting matrix is block lower triangular and of size `km + m + l` x `km + m + l`
Note: `concat_block` is expected to be square
"""
function _build_Hhat(Hhat, H::AbstractMatrix, η_prev, k, m; η = nothing, concat_block = [])

    km = (k - 1) * m
    l = size(concat_block, 1)
    Hhat_expand = zeros(eltype(Hhat), km + m + l, km + m + l)

    @views Hhat_expand[1:km, 1:km] .= Hhat[1:km, 1:km]
    @views Hhat_expand[(km + 1):(km + m), (km + 1):(km + m)] .= H

    Hhat_expand[((k - 1) * m) + 1, (k - 1) * m] = η_prev
    if !isnothing(η)
        Hhat_expand[km + m + 1, km + m] = η
    end
    Hhat_expand[(km + m + 1):(km + m + l), (km + m + 1):(km + m + l)] = concat_block

    return Hhat_expand
end

"""
Build the expanded Lanczos decomposition matrix `That`
The resulting matrix is `Tridiagonal` and of size `km + m + l` x `km + m + l` where `l`
is the size of `concat_block`.
Note: `concat_block` is expected to be square and preserve the Tridiagonal structure!
"""
function _build_Hhat(That, T::SymTridiagonal, η_prev, k, m; η = 0.0, concat_block = [])

    km = (k - 1) * m

    That_sub_diag, That_diag, That_sup_diag = isempty(concat_block) ? (
            [diag(That, -1)[1:(km - 1)]; [η_prev]; T.ev],
            [diag(That)[1:km]; T.dv],
            [diag(That, 1)[1:(km - 1)]; [0.0]; T.ev],
        ) : (
            [diag(That, -1)[1:(km - 1)]; [η_prev]; T.ev; [η]; diag(concat_block, -1)],
            [diag(That)[1:km]; T.dv ; diag(concat_block)],
            [diag(That, 1)[1:(km - 1)]; [0.0]; T.ev; [0.0]; diag(concat_block, 1)],
        )

    return Tridiagonal(That_sub_diag, That_diag, That_sup_diag)
end

_build_extended_H_herm(H::SymTridiagonal, η, α1, α2) = Tridiagonal(
    [H.ev;[η, 1.0]],
    [H.dv;[α1, α2]],
    [H.ev;[0.0, 0.0]]
)

function _build_extended_H!(Hbar, H, η, α1, α2)
    m = size(H, 1)
    Hbar[1:m, 1:m] .= H
    Hbar[(m + 1):(m + 2), (m + 1):(m + 2)] = [α1 0.0; 1.0 α2]
    Hbar[m + 1, m] = η
    return nothing
end

krylov_approx(f, A::AbstractArray, b::AbstractVector, m::Integer; kwargs...) =
    krylov_approx(f, A, b; m = Int(m), kwargs...)

"""
Compute `f(A)b` in the manner corresponding to Alg. 1 in the paper.

If `trace::Trace` is provided, it is filled with per-restart statistics.
"""
function krylov_approx(
        f::Function,
        A::AbstractArray,
        b::AbstractVector;
        m::Int = 30,
        max_restarts::Int = 200,
        tol = nothing,
        bound::Bool = true,
        exact::Union{Nothing, AbstractVector} = nothing,
        min_decay = 0.95,
        trace::Union{Nothing, Trace} = nothing
    )

    if !isnothing(exact) && bound
        @warn "Both `exact` provided and `bound=true`; stopping will use bounds."
    end
    m < 1 && throw(ArgumentError(lazy"m must be ≥ 1"))

    _check_sizes(exact, A, b)

    set_type!(trace, KrylovApproxFunction)

    log_value!(trace, :restart_length, m)

    Tdefault = float(real(eltype(A)))
    tol = isnothing(tol) ? eps(Tdefault) : tol

    real_res = (eltype(A) <: AbstractFloat) && (eltype(b) <: AbstractFloat)

    # state for the linear-convergence-based stopping rules
    up_decay = DecayTracker()
    err_decay = DecayTracker()

    α1, α2 = if bound
        prevfloat(Tdefault(Inf)), nextfloat(Tdefault(-Inf))
    else
        zero(Tdefault), zero(Tdefault)
    end

    fk = zeros(promote_type(eltype(A), eltype(b)), length(b))

    β = norm(b)
    qm = (1 / β) * b

    η_prev = NaN

    is_A_hermitian = ishermitian(A)

    if !is_A_hermitian
        Hhat = Array{eltype(A)}(undef, 0, 0)
    end

    for k in 1:max_restarts

        set_restart!(trace, k)

        (Q, H, η, qm) = is_A_hermitian ? lanczos(A, qm, m) : arnoldi(A, qm, m)

        if bound
            α1, α2 = _update_alphas(α1, α2, H)
        end

        if k == 1
            # If I constantly have to do different things for Hermitian vs. Non-Hermitian then maybe I should dispatch a different algorithm for Hermitian.
            is_A_hermitian ? Hhat = _build_extended_H_herm(H, η, α1, α2) : (Hhat = zeros(eltype(H), m + 2, m + 2); _build_extended_H!(Hhat, H, η, α1, α2))
        else
            Hhat_expand = _build_Hhat(Hhat, H, η_prev, k, m, η = η, concat_block = [α1 0.0; 1.0 α2])
            Hhat = Hhat_expand
        end
        η_prev = η

        @views h = f(Hhat)[((k - 1) * m + 1):((k - 1) * m + m), 1]

        log_metric!(trace, :update_norm, norm(h))

        if real_res
            h = h .|> real
        end

        fk .+= β * (Q * h)

        stop = stop_conditions!(
            A, α1, β, h, qm, fk, m,
            bound, exact, tol, min_decay,
            trace, k,
            up_decay, err_decay
        )

        if stop !== nothing
            set_stop!(trace, stop)
            return fk
        end
    end

    set_stop!(trace, MaxRestarts)
    return fk
end

function _retrieve_poles_coeff(r::RationalApproximation)
    return (length(r.single_poles), [r.single_poles; r.conj_poles], [r.single_coeff; r.conj_coeff])
end

function _init_Bbar(poles, m)
    Bbar = zeros(ComplexF64, m + 2, length(poles))
    Bbar[end - 2, :] .= 1.0 + 0.0im
    return Bbar
end

function _update_Bbar!(Bbar, Hbar, poles, s, m)
    qbar = zeros(ComplexF64, m + 2)
    for p in eachindex(poles)
        qbar[1] = s * Bbar[m, p]
        Bbar[1:(m + 2), p] = (poles[p] * I - Hbar) \ qbar
    end
    return nothing
end

function _update_vector(Bbar, n_single, poles, coeff, m)
    # contribution from single poles
    @views h = n_single > 0 ? Bbar[1:(m + 2), 1:n_single] * coeff[1:n_single] : zeros(ComplexF64, m + 2)

    if n_single < length(poles) # add in the contrabution from the conjugate poles
        @views h .+= 2 * real(Bbar[1:(m + 2), (n_single + 1):end] * coeff[(n_single + 1):end])
    end

    return h
end

krylov_approx(r::RationalApproximation, A::AbstractArray, b::AbstractVector, m::Integer; kwargs...) =
    krylov_approx(r, A, b; m = Int(m), kwargs...)

"""
    krylov_approx(r::RationalApproximation, A, b; m=30, max_restarts=200, tol=nothing, bound=true,
                 exact=nothing, min_decay=0.95, trace=nothing, callback=nothing)

Compute `r(A)b ≈ f(A)b` in the manner corresponding to Alg. 2 in the paper.
"""
function krylov_approx(
        r::RationalApproximation,
        A::AbstractArray,
        b::AbstractVector;
        m::Int = 30,
        max_restarts::Int = 200,
        tol = nothing,
        bound::Bool = true,
        exact::Union{Nothing, AbstractVector} = nothing,
        min_decay = 0.95,
        trace::Union{Nothing, Trace} = nothing
    )

    if !isnothing(exact) && bound
        @warn "Both `exact` provided and `bound=true`; stopping will use bounds."
    end
    m < 1 && throw(ArgumentError("m must be ≥ 1"))

    _check_sizes(exact, A, b)

    set_type!(trace, KrylovApproxRational)

    log_value!(trace, :restart_length, m)

    Tdefault = float(real(eltype(A)))
    tol = isnothing(tol) ? eps(Tdefault) : tol

    # state for the linear-convergence-based stopping rules
    up_decay = DecayTracker()
    err_decay = DecayTracker()

    α1, α2 = if bound
        prevfloat(Tdefault(Inf)), nextfloat(Tdefault(-Inf))
    else
        zero(Tdefault), zero(Tdefault)
    end

    real_res = (eltype(A) <: AbstractFloat) && (eltype(b) <: AbstractFloat)

    fk = real_res ? (r.absterm * b) : complex.(r.absterm * b)

    β = norm(b)
    qm = (1 / β) * b

    n_single, poles, coeff = _retrieve_poles_coeff(r)

    Bbar = _init_Bbar(poles, m)
    s = one(eltype(Bbar))

    is_A_hermitian = ishermitian(A)

    Hbar = is_A_hermitian ? nothing : zeros(eltype(A), m + 2, m + 2)

    for k in 1:max_restarts

        set_restart!(trace, k)

        (Q, H, η, qm) = is_A_hermitian ? lanczos(A, qm, m) : arnoldi(A, qm, m)

        if bound
            α1, α2 = _update_alphas(α1, α2, H)
        end

        is_A_hermitian ? Hbar = _build_extended_H_herm(H, η, α1, α2) : _build_extended_H!(Hbar, H, η, α1, α2)

        _update_Bbar!(Bbar, Hbar, poles, s, m)

        h = _update_vector(Bbar, n_single, poles, coeff, m)

        log_metric!(trace, :update_norm, norm(h))

        s = η

        if real_res
            h = h .|> real
        end

        @views fk .+= β * (Q * h[1:m])

        stop = stop_conditions!(
            A, α1, β, h, qm, fk, m,
            bound, exact, tol, min_decay,
            trace, k,
            up_decay, err_decay
        )

        if stop !== nothing
            set_stop!(trace, stop)
            return fk
        end
    end

    set_stop!(trace, MaxRestarts)
    return fk
end

function _initialize_krylov_approx_quad(f, A, b, m, order)

    β = norm(b)

    qm = (1 / β) * b
    is_A_hermitian = ishermitian(A)

    (Q, H, η, qm) = is_A_hermitian ? lanczos(A, qm, m) : arnoldi(A, qm, m)

    Hs = Vector([H])
    ηs = [η]

    h = f(H)[:, 1]
    fk = β * Q * h

    e1 = unit_vector(eltype(H), m, 1)

    quad1::Int64 = order
    quad2::Int64 = round(sqrt(2) * quad1)

    h2 = h

    return β, qm, is_A_hermitian, Q, H, Hs, η, ηs, e1, quad1, quad2, h2, fk
end

function _check_quad_err(err, tol, quad1, quad2, accurate, refined)
    if err < tol
        accurate = true
    else
        quad1 = quad2
        quad2 = round(sqrt(2) * quad1) |> Int
        refined = true
    end

    return quad1, quad2, accurate, refined
end

function _check_exact_err(exact, fk, trace)
    if !(exact === nothing || isempty(exact))
        abs_err = norm(exact - fk)
        log_metric!(trace, :abs_err, abs_err)
    end

    return nothing
end

function _refine_quad_order(refined, quad1, quad2)
    if !refined
        quad2 = quad1
        quad1 = round(quad2 / sqrt(2)) |> Int
    end

    return quad1, quad2
end

function _check_quad_order_divergence(quad1, quad2, maxorder, trace)
    if quad2 >= maxorder || quad1 >= maxorder
        log_value!(trace, :diverged, quad2)
        return QuadOrderDivergence
    end
    return nothing
end

_NaNcheck(err) = isnan(err) ? QuadErrorDivergence : nothing

"""
    _integral_error_correction(
        f::Function,
        H::AbstractArray,
        Hs,
        ηs::AbstractVector,
        order::Int,
        R::Float64,
        c,
        e1
    )

    evaluate the integral formulation for the error term at the transformed quadrature points

"""
function _integral_error_correction(
        f::Function,
        H::AbstractArray,
        Hs,
        ηs::AbstractVector,
        x,
        w,
        e1
    )

    m = size(H, 1)

    S = zeros(ComplexF64, m)


    for (t, ω) in zip(x, w)
        y = (t * I - H) \ e1

        ϕ = prod(
            ηs[j] * ((t * I - Hs[j]) \ e1)[end] for j in eachindex(ηs)
        )


        S += (1 / (2π * 1im)) * ω * f(t) * ϕ * y
    end
    return S
end

"""
    krylov_approx_quad(
        f::Function,
        A::AbstractArray,
        b::AbstractVector,
        m::Int
        ;
        tol = 1.0e-16,
        max_restarts = 200,
        contour_safety = 2.0,
        order = 8,
        maxorder = 20000,
        trace::Union{Nothing, Trace} = nothing,
        exact::Union{Nothing, AbstractVector} = nothing
    )

Approximate ``f(A)b`` by Krylov subsapce method with a quadrature formulation of the error.
Here we assume that the function `f` is holomorphic in a neighborhood around the spectrum of `A`. In this case
we are guarenteed that the interal exists and the quadrature based error formula actually represents the error.

"""
function krylov_approx_quad(
        f::Function,
        A::AbstractArray,
        b::AbstractVector,
        m::Int
        ;
        quad_tol = 1.0e-12,
        stop_tol = 1.0e-16,
        max_restarts = 200,
        contour_safety = 1.1,
        order = 8,
        maxorder = 20000,
        contour_type::Type{<:AbstractContour} = CircleContour,
        rebuild_contour = true,
        trace::Union{Nothing, Trace} = nothing,
        contour_override::Union{Nothing, AbstractContour} = nothing,
        exact::Union{Nothing, AbstractVector} = nothing
    )

    real_res = (eltype(A) <: AbstractFloat) && (eltype(b) <: AbstractFloat)

    _check_sizes(exact, A, b)

    set_type!(trace, Quadrature1)
    log_value!(trace, :restart_length, m)


    β, qm, is_A_hermitian,
        Q, H, Hs, _, ηs, e1,
        quad1, quad2, h2, fk = _initialize_krylov_approx_quad(f, A, b, m, order)

    _check_exact_err(exact, fk, trace)

    contour = isnothing(contour_override) ?
        begin
            ritz, _ = eigen(H)
            build(contour_type, ritz, contour_safety)
        end : contour_override

    update_norm_prev = Inf

    for k in 2:max_restarts

        set_restart!(trace, k)

        #@info "Iteration $_k"
        (Q, H, η, qm) = is_A_hermitian ? lanczos(A, qm, m) : arnoldi(A, qm, m)

        if isnothing(contour_override) && rebuild_contour
            append!(ritz, first(eigen(H)))
            contour = build(contour_type, ritz, contour_safety)
        end

        accurate = false
        refined = false

        while !accurate

            if !isnothing(_check_quad_order_divergence(quad1, quad2, maxorder, trace))
                isnothing(trace) && throw(ErrorException(lazy"quadrature order exceeded allowence"))
                set_stop!(trace, QuadOrderDivergence)
                return fk
            end

            log_metric!(trace, :quad1_order, quad1)
            log_metric!(trace, :quad2_order, quad2)

            x1, w1 = resolve(contour, quad1)
            x2, w2 = resolve(contour, quad2)
            h1 = _integral_error_correction(f, H, Hs, ηs, x1, w1, e1)
            h2 = _integral_error_correction(f, H, Hs, ηs, x2, w2, e1)

            err = norm(h2 - h1)
            log_metric!(trace, :quad_err, err)
            if !isnothing(_NaNcheck(err))
                isnothing(trace) && throw(OverflowError(lazy"quadrature error diverged to NaN"))
                set_stop!(trace, QuadErrorDivergence)
                return real_res ? fk .|> real : fk
            end

            quad1, quad2, accurate, refined = _check_quad_err(err, quad_tol, quad1, quad2, accurate, refined)
        end

        update = β * Q * h2

        update_norm = norm(update)
        log_metric!(trace, :update_norm, update_norm)
        if (10 * update_norm_prev) < update_norm
            set_stop!(trace, UpdateNormDivergence)
            return real_res ? fk .|> real : fk
        end

        update_norm_prev = update_norm

        fk += update

        _check_exact_err(exact, fk, trace)

        if update_norm < stop_tol
            set_stop!(trace, UpdateAcc)
            return real_res ? fk .|> real : fk
        end

        push!(Hs, H)
        push!(ηs, η)

        quad1, quad2 = _refine_quad_order(refined, quad1, quad2)

    end
    set_stop!(trace, MaxRestarts)
    return real_res ? fk .|> real : fk
end

"""
    _integral_error_correction_stieltjes(
        f,
        H::AbstractArray,
        Hs,
        ηs::AbstractVector,
        order::Int,
        e1
    )

TBW
"""
function _integral_error_correction_stieltjes(
        f,
        H::AbstractArray,
        Hs,
        ηs::AbstractVector,
        x,
        w,
        e1
    )

    m = size(H, 1)

    S = zeros(ComplexF64, m)

    for (t, ω) in zip(x, w)
        y = (t * I - H) \ e1

        ϕ = prod(
            ηs[j] * ((t * I - Hs[j]) \ e1)[end] for j in eachindex(ηs)
        )

        S += f.constant * ω * f.inner(t) * ϕ * y
    end
    return S

end

"""
    krylov_approx_quad(
        f::StieltjesFunction,
        A::AbstractArray,
        b::AbstractVector,
        m::Int
        ;
        tol = 1.0e-12,
        max_restarts = 200,
        contour_safety = nothing,
        order = 8,
        maxorder = 20000,
        trace::Union{Nothing, Trace} = nothing,
        contour::Union{Nothing, AbstractContour} = nothing,
        exact::Union{Nothing, AbstractVector} = nothing
    )

Approximate `f(A)b` using an integral representation for the error where f is a Stieltjes. In this case we can approximate the error using an integral that is independent of the spectrum of `A`.
"""
function krylov_approx_quad(
        f::StieltjesFunction,
        A::AbstractArray,
        b::AbstractVector,
        m::Int
        ;
        quad_tol = 1.0e-12,
        stop_tol = 1.0e-16,
        max_restarts = 200,
        order = 8,
        maxorder = 20000,
        trace::Union{Nothing, Trace} = nothing,
        contour_override::Union{Nothing, AbstractContour} = nothing,
        exact::Union{Nothing, AbstractVector} = nothing
    )

    if !isnothing(contour_override)
        @warn "Stieltjes functions always use a spectrum independent contour. Provided Contour will be ignored."
    end

    real_res = (eltype(A) <: AbstractFloat) && (eltype(b) <: AbstractFloat)

    _check_sizes(exact, A, b)

    set_type!(trace, Quadrature1Stieltjes)
    log_value!(trace, :restart_length, m)

    β, qm, is_A_hermitian,
        Q, H, Hs, η, ηs, e1,
        quad1, quad2, h2, fk = _initialize_krylov_approx_quad(f, A, b, m, order)

    _check_exact_err(exact, fk, trace)

    contour = StieltjesContour()

    update_norm_prev = Inf

    for k in 2:max_restarts

        set_restart!(trace, k)
        (Q, H, η, qm) = is_A_hermitian ? lanczos(A, qm, m) : arnoldi(A, qm, m)

        accurate = false
        refined = false

        while !accurate

            if !isnothing(_check_quad_order_divergence(quad1, quad2, maxorder, trace))
                isnothing(trace) && throw(ErrorException(lazy"quadrature order exceeded allowence"))
                set_stop!(trace, QuadOrderDivergence)
                return real_res ? fk .|> real : fk
            end

            log_metric!(trace, :quad1_order, quad1)
            log_metric!(trace, :quad2_order, quad2)

            x1, w1 = resolve(contour, quad1)
            x2, w2 = resolve(contour, quad2)

            h1 = _integral_error_correction_stieltjes(f, H, Hs, ηs, x1, w1, e1)
            h2 = _integral_error_correction_stieltjes(f, H, Hs, ηs, x2, w2, e1)

            err = norm(h2 - h1)

            log_metric!(trace, :quad_err, err)

            if !isnothing(_NaNcheck(err))
                isnothing(trace) && throw(OverflowError(lazy"quadrature error diverged to NaN"))
                set_stop!(trace, QuadErrorDivergence)
                return real_res ? fk .|> real : fk
            end

            quad1, quad2, accurate, refined = _check_quad_err(err, quad_tol, quad1, quad2, accurate, refined)
        end

        update = β * Q * h2

        update_norm = norm(update)

        log_metric!(trace, :update_norm, update_norm)
        if (10 * update_norm_prev) < update_norm
            set_stop!(trace, UpdateNormDivergence)
            return real_res ? fk .|> real : fk
        end

        update_norm_prev = update_norm

        fk += update

        _check_exact_err(exact, fk, trace)

        if update_norm < stop_tol
            set_stop!(trace, UpdateAcc)
            return real_res ? fk .|> real : fk
        end

        push!(Hs, H)
        push!(ηs, η)

        quad1, quad2 = _refine_quad_order(refined, quad1, quad2)
    end
    set_stop!(trace, MaxRestarts)
    return real_res ? fk .|> real : fk
end

"""
    _make_integral_function(f, H, Hs, ηs, R, c, e1)

    Generate a function `F` to be integrated.
"""
function _make_integral_function(f, H, Hs, ηs, cov::ComplexQuadrature.ChangeOfVariables, e1)
    function F(x, p = nothing)

        t = cov.zmap(x)

        w = cov.dzmap(x)

        y = (t * I - H) \ e1

        ϕ = one(eltype(H))

        for j in eachindex(ηs)
            res = (t * I - Hs[j]) \ e1
            ϕ *= ηs[j] * res[end]
        end

        return ((1 / (2π * 1im)) * w * f(t) * ϕ) * y
    end

    return F

end

"""
    krylov_approx_quad2(f,A,b,m)

    Approximate ``f(A)b`` using Gauss-Kronrod quadrature
    directly assuming that we can attain an accurate quadrature based
    error correction in 1 shot using Julia's quadrature libraries

    !! Right now this appears to give incorrect answers for non-symmetric matrices !!
"""
function krylov_approx_quad2(
        f::Function,
        A::AbstractArray,
        b::AbstractVector,
        m::Int
        ;
        quad_tol = 1.0e-12,
        stop_tol = 1.0e-16,
        max_restarts = 200,
        contour_safety = 1.1,
        rebuild_contour = true,
        alg = QuadGKJL(),
        contour_type::Type{<:AbstractContour} = CircleContour,
        trace::Union{Nothing, Trace} = nothing,
        contour_override::Union{Nothing, AbstractContour} = nothing,
        exact::Union{Nothing, AbstractVector} = nothing,
    )

    real_res = (eltype(A) <: AbstractFloat) && (eltype(b) <: AbstractFloat)

    _check_sizes(exact, A, b)

    if (eltype(A) <: AbstractFloat) && (eltype(b) <: AbstractFloat)
        real_res = true
    end

    set_type!(trace, QuadratureSolver)

    β = norm(b)

    qm = (1 / β) * b

    is_A_hermitian = ishermitian(A)

    (Q, H, η, qm) = is_A_hermitian ? lanczos(A, qm, m) : arnoldi(A, qm, m)

    Hs = [H]
    ηs::Vector{eltype(H)} = [η]

    h = f(H)[:, 1]

    fk = β * Q * h

    _check_exact_err(exact, fk, trace)

    e1 = unit_vector(eltype(H), m, 1)

    contour::AbstractContour = isnothing(contour_override) ? begin
            ritz = first(eigen(H))
            build(contour_type, ritz, contour_safety)
        end : contour_override

    for k in 2:max_restarts

        set_restart!(trace, k)

        (Q, H, η, qm) = is_A_hermitian ? lanczos(A, qm, m) : arnoldi(A, qm, m)

        if isnothing(contour_override) && rebuild_contour
            append!(ritz, first(eigen(H)))
            contour = build(contour_type, ritz, contour_safety)
        end

        update = contour isa PacmanContour ? begin
                S = zeros(eltype(H), size(H, 1))
                for cov in contour.covs
                    F = _make_integral_function(f, H, Hs, ηs, cov, e1)
                    prob = IntegralProblem(F, (-1, 1))
                    sol = solve(prob, alg, reltol = quad_tol, abstol = quad_tol)
                    S += sol.u
            end
                β * Q * S
            end : begin
                F = _make_integral_function(f, H, Hs, ηs, contour.cov, e1)
                prob = IntegralProblem(F, (-1, 1))
                sol = solve(prob, alg, reltol = quad_tol, abstol = quad_tol)
                β * Q * sol.u
            end

        update_norm = update |> norm

        log_metric!(trace, :update_norm, update_norm)

        fk += update

        _check_exact_err(exact, fk, trace)

        if update_norm < stop_tol
            set_stop!(trace, UpdateAcc)
            return real_res ? fk .|> real : fk
        end

        push!(Hs, H)
        push!(ηs, η)
    end

    set_stop!(trace, MaxRestarts)
    return real_res ? fk .|> real : fk
end

"""
    _make_integral_function_stieltjes(f, H, Hs, ηs, e1)

    Generate a function `F` to be integrated when
    the matrix function to be approximanted is a Stieltjes function.

"""
function _make_integral_function_stieltjes(f, H, Hs, ηs, e1)

    function F(x, p = nothing)

        y = (x * I - H) \ e1

        ϕ = one(eltype(H))

        for j in eachindex(ηs)
            res = (x * I - Hs[j]) \ e1
            ϕ *= ηs[j] * res[end]
        end

        return f.constant * (f.inner(x) * ϕ) * y
    end

    return F
end


"""
    krylov_approx_quad2(
        f::StieltjesFunction,
        A::AbstractArray,
        b::AbstractVector,
        m::Int
        ;
        tol = 1.0e-16,
        max_restarts = 200,
        contour_safety = 2.0,
        alg = QuadGKJL(),
        trace::Union{Nothing, Trace} = nothing,
        exact::Union{Nothing, AbstractVector} = nothing
    )

Approximate ``f(A)b`` using an integral formulation for the error evaluated by Gauss-Kronrad quadrature. When `f` is a Stieltjes function we can evaluate the integral on the interval [-∞,0] independent of the spectrum of `A`.
`A` must have no negative eigenvalues!
"""
function krylov_approx_quad2(
        f::StieltjesFunction,
        A::AbstractArray,
        b::AbstractVector,
        m::Int
        ;
        quad_tol = 1.0e-16,
        stop_tol = 1.0e-16,
        max_restarts = 200,
        alg = QuadGKJL(),
        trace::Union{Nothing, Trace} = nothing,
        exact::Union{Nothing, AbstractVector} = nothing
    )

    _check_sizes(exact, A, b)

    real_res = (eltype(A) <: AbstractFloat) && (eltype(b) <: AbstractFloat)

    set_type!(trace, QuadratureSolverSieltjes)

    β = norm(b)

    qm = (1 / β) * b

    is_A_hermitian = ishermitian(A)

    (Q, H, η, qm) = is_A_hermitian ? lanczos(A, qm, m) : arnoldi(A, qm, m)

    Hs = [H]
    ηs::Vector{eltype(H)} = [η]

    h = f(H)[:, 1]
    fk = β * Q * h

    _check_exact_err(exact, fk, trace)

    e1 = unit_vector(eltype(H), m, 1)

    for k in 2:max_restarts

        set_restart!(trace, k)

        (Q, H, η, qm) = is_A_hermitian ? lanczos(A, qm, m) : arnoldi(A, qm, m)

        F = _make_integral_function_stieltjes(f, H, Hs, ηs, e1)
        prob = IntegralProblem(F, (-Inf, 0.0))
        sol = solve(prob, alg, reltol = quad_tol, abstol = quad_tol)

        update = β * Q * sol.u

        update_norm = update |> norm

        log_metric!(trace, :update_norm, update_norm)

        fk += update
        _check_exact_err(exact, fk, trace)
        if update_norm < stop_tol
            set_stop!(trace, UpdateAcc)
            return real_res ? fk .|> real : fk
        end

        push!(Hs, H)
        push!(ηs, η)
    end
    set_stop!(trace, MaxRestarts)
    return real_res ? fk .|> real : fk
end

"""
    krylov_approx_2norm(r::RationalApproximation, A, b, m)

Implementation of ``f(A)b`` for HPD matrices with an error indicator given
from "2-Norm Error bounds and estimates for Lanczos approximations to
linear systems and rational matrix functions".

For this algorithm we need `f` to be a rational function
(or able to be accurately approximated by a rational function)
and furthermore `A` to be HPD.

"""
function krylov_approx_2norm(r::RationalApproximation, A, b, m)

end

"""
    norm_of_hwz(z, w::Real, a, b)

Chen dissertation: Lemma 7.8
"""
function norm_of_hwz(z, w::Real, a, b)
    z_re = real(z)
    z_im = imag(z)
    left = abs((a - w) / (a - z))
    right = abs((b - w) / (b - z))
    x = (z_re^2 + z_im^2 - z_re * w) / (z_re - w)
    if x >= a && x <= b && z_im != 0
        middle = abs((1.0 / z_im) * (z - w))
    else
        middle = 0
    end

    @assert middle != NaN

    return max(left, middle, right)
end

"""
    chen_values(f, contour, order, w, λmax, λmin)

TBW
"""
function chen_values(f, contour, order, w, λmax, λmin)

    nodes, weights = resolve(contour, order)

    fval = nodes .|> f
    absfval = fval .|> abs

    norm_curry = z -> norm_of_hwz(z, w, λmin, λmax)
    norm_val = nodes .|> norm_curry

    return nodes, weights, fval, absfval, norm_val
end

"""
Implementation of an implicit restarted KSM using an error indicator adapted from Chen et al. and an implicit
restarting scheme derived from the cauchy integral form of the KSM approximation

For this algorithm we require that `A` be Hermitian as the error bound is only valid for the Lanczos decomposition.
The adaptation for the restarted scheme was derived assuming that we have a decomposition as given by Eiermann and Ernst (2006).

This algorithm is rather sensitive to its hyperparameters, we ideally want to choose `w` less than the minimum
eigenvalue of `A`. Furthermore, in this algorithm we try determine a countour the fully encloses the eigenvalues
of `A` and all subsequent `Tk` in one fell swoop based upon the initial ritz values extracted from `T1`. The ritz
values of `T1` should be an okay approximation to the extreme values of `A`. Therefore, we have the variable
`contour_safety` which will scale the radius of the circular contour beyond the initial maximum ritz value.
A sufficiently large number of quadrature nodes must be choosen as well with `order`.

Two key variables that need to be controlled for this algorithm to work are the value `w` which is a single
complex point that we need to choose
to be disjoint from both the spectrum of `A` and the ritz-values of each `T`.
We also will need to choose a contour for evaluating an integral such that every value of
the contour is far away from the spectrum of `A` and ritz-values of each `T`.
"""
function krylov_approx_chen_implicit(
        f, A, b, m, w::AbstractFloat = 0.0;
        tol = 1.0e-16,
        max_restarts = 200,
        contour_safety = 2.0,
        order = 80,
        trace::Union{Nothing, Trace} = nothing,
        contour::Union{Nothing, AbstractContour} = nothing,
        exact::Union{Nothing, AbstractVector} = nothing
    )

    @assert ishermitian(A)
    n = size(A, 1)

    _check_sizes(exact, A, b)

    err = 0.0

    set_type!(trace, ChenImplicit)

    log_value!(trace, :restart_length, m)
    log_value!(trace, :w_value, w)

    β = norm(b)
    qm = (1 / β) * b

    (Q, T, η, qm) = lanczos(A, qm, m)


    λs, _ = eigen(T)
    λmax = λs |> maximum
    λmin = λs |> minimum


    #The contour is set once based on the initial Ritz values
    if isnothing(contour)
        r = λmin / contour_safety
        R = λmax * contour_safety
        contour = PacmanContour(r, R, r, float(π), complex(w))
    end

    nodes, weights, fval, absfval, norm_val = chen_values(f, contour, order, w, λmax, λmin)

    integral_constant = (1 / (2 * π * im))

    h = f(T)[:, 1]
    fk = β * Q * h

    # Dets are computed in log-space for numerical stability
    w_det_accum = (T - w * I) |> logabsdet |> first
    dets_accum = [(T - node * I) |> logabsdet |> first for node in nodes]

    μs = Vector{ComplexF64}[]
    μ_coeff = iseven(m) ? 1.0 : -1.0 #(-1)^m

    Q_herm_b = Q' * b
    Q_herm_bs = [Q_herm_b]

    ηs = [η]

    e1 = unit_vector(eltype(A), m, 1)
    em = unit_vector(eltype(A), m, m)

    # eₘᵀ(T - zI)⁻¹ at each quad node
    right_vecs = [mapreduce(node -> (T - node * I)' \ em, hcat, nodes)]

    #Preallocation - Continually write over the same arrays
    left_vec = Array{ComplexF64}(undef, m, order)
    right_vec = Array{ComplexF64}(undef, m, order)
    cauchy_integral = Array{ComplexF64}(undef, n, order)
    dets = Array{Tuple{Float64, ComplexF64}}(undef, order)
    recurrence_terms = Array{ComplexF64}(undef, size(A, 1), order)

    for k in 2:max_restarts

        set_restart!(trace, k)

        (Q, T, η, qm) = lanczos(A, qm, m)

        Q_herm_b = Q' * b
        push!(Q_herm_bs, Q_herm_b)

        sub_diag = diag(T, -1)
        sub_diag_prod = prod(sub_diag)

        w_det_accum += (T - w * I) |> logabsdet |> first

        #Compute the quantities in the integral at each quadrature node
        for (j, node) in enumerate(nodes)

            T_shift = T - node * I
            F = factorize(T_shift)

            left_vec[:, j] = F \ e1
            right_vec[:, j] = F' \ em
            cauchy_solve = F \ Q_herm_b
            cauchy_integral[:, j] = weights[j] * Q * cauchy_solve

            dets[j] = logabsdet(F)
            dets_accum[j] += dets[j] |> first

        end

        push!(right_vecs, deepcopy(right_vec))

        det_sign = [det[1] for det in dets]
        inv_logabsdet_val = [-det[2] for det in dets]
        inv_det_val = (μ_coeff * det_sign) .* exp.(inv_logabsdet_val)

        push!(μs, sub_diag_prod .* inv_det_val)

        fill!(recurrence_terms, 0.0 + 0.0im)

        η_suffix = one(η)
        μ_suffix = ones(eltype(μs[1]), order)
        sgn = one(η)

        for i in (k - 1):-1:1
            coeff = sgn * η_suffix * η

            @inbounds for j in 1:order
                recurrence_terms[:, j] += weights[j] * fval[j] * coeff * Q * μ_suffix[j] * left_vec[:, j] * dot(right_vecs[i][:, j], Q_herm_bs[i])
            end

            η_suffix *= ηs[i]
            μ_suffix .*= μs[i]
            sgn = -sgn
        end

        h = (integral_constant * (sum(cauchy_integral, dims = 2) + sum(recurrence_terms, dims = 2))) .|> real
        fk .+= h

        push!(ηs, η)

        det_prod = exp.(-dets_accum .+ w_det_accum)
        error_indicator = integral_constant * dot(weights, absfval .* det_prod .* norm_val)

        log_metric!(trace, :error_indicator, error_indicator)

        if !isnothing(_NaNcheck(error_indicator))
            isnothing(trace) && throw(OverflowError(lazy"quadrature error diverged to NaN"))
            set_stop!(trace, QuadErrorDivergence)
            return fk
        end

        if !(exact === nothing || isempty(exact))
            abs_err = norm(exact - fk)
            log_metric!(trace, :abs_err, abs_err)
        end

        error_indicator_norm = error_indicator |> norm

        log_metric!(trace, :error_indicator_norm, error_indicator_norm)

        if error_indicator_norm < tol
            set_stop!(trace, IndicatorAcc)
            return fk
        end
    end
    set_stop!(trace, MaxRestarts)
    return fk
end

function krylov_approx_chen_implicit_2(
        f, A, b, m, w;
        tol = 1.0e-16,
        max_restarts = 200,
        contour_safety = 2.0,
        order = 80,
        trace::Union{Nothing, Trace} = nothing,
        exact::Union{Nothing, AbstractVector} = nothing
    )

    @assert ishermitian(A)
    n = size(A, 1)

    _check_sizes(exact, A, b)

    err = 0.0

    set_type!(trace, ChenImplicit)

    log_value!(trace, :restart_length, m)
    log_value!(trace, :w_value, w)

    β = norm(b)
    qm = (1 / β) * b

    (Q, T, η, qm) = lanczos(A, qm, m)

    T = SymTridiagonal(T)

    #The contour is set once based on the initial Ritz values
    #R, c, nodes, weights, fval, absfval, norm_val, error_integral_constant = build_contour_chen(f, T, contour_safety, order, w)

    R, c = fetch_circle(T, contour_safety)
    nodes, weights = circle_contour(order, R, c)

    fval = f.(nodes)

    update_integral_constant = (1 / (2 * π))


    # rv = mapreduce(node -> (T - node * I) \ em, hcat, nodes)
    # lv = mapreduce(node -> (T - node * I) \ e1, hcat, nodes)

    Qhb = Q' * b

    cauchy_sol = Array{ComplexF64}(undef, order, n)

    for (j, node) in enumerate(nodes)
        cauchy_sol[j, :] = Q * ((T - node * I) \ Qhb)
    end

    @info size(weights)
    @info size(fval)
    @info size(cauchy_sol)

    fk = sum(weights[j] * fval[j] * cauchy_sol[j, :] for j in eachindex(weights))
    fk *= update_integral_constant
    return fk
end

"""
Does it even work?
"""
function quad_test(f, a, R, order = 80)
    contour = CircleContour(R, a)
    x, w = resolve(contour, order)

    s = f.(x) ./ (x .- a)

    return (1 / (2 * π * im)) * sum(w .* s)
end

"""
Implementation of an explicit restarted KSM using an error indicator adapted from Chen et al.

Here we contruct the extended block diagonal matrix `That` like in `krylov_approx`. We evaluate the error using quadrature. Because we have access to the full matrix `That` we are able to rebuild our contour every restart based on the current Ritz values of `That`. This allows us to use a lower `order`.
For this algorithm we require that `A` be Hermitian as the error bound is only valid for the Lanczos decomposition.
The adaptation for the restarted scheme was derived assuming that we have a decomposition as given by Eiermann and Ernst (2006).
"""
function krylov_approx_chen_explicit(
        f, A, b, m, w::AbstractFloat = 0.0;
        tol = 1.0e-16,
        max_restarts = 200,
        contour_safety = 2.0,
        order = 200,
        rebuild_contour = true,
        trace::Union{Nothing, Trace} = nothing,
        contour::Union{Nothing, AbstractContour} = nothing,
        exact::Union{Nothing, AbstractVector} = nothing
    )

    @assert ishermitian(A)

    _check_sizes(exact, A, b)

    set_type!(trace, ChenExplicit)

    β = norm(b)
    qm = (1 / β) * b

    (Q, T, η_prev, qm) = lanczos(A, qm, m)

    That = T

    λs, _ = eigen(T)

    λmax = λs |> maximum
    λmin = λs |> minimum

    if isnothing(contour)
        r = λmin / contour_safety
        R = λmax * contour_safety
        contour = PacmanContour(r, R, r, float(π), complex(w))
        log_metric!(trace, :radii, (r, R))
    end

    nodes, weights, fval, absfval, norm_val = chen_values(f, contour, order, w, λmax, λmin)

    integral_constant = (1 / (2 * π * im))

    log_metric!(trace, :norm_val, norm_val)
    log_metric!(trace, :absfval, absfval)

    fk = β * Q * f(That)[:, 1]

    for k in 2:max_restarts

        set_restart!(trace, k)

        (Q, T, η, qm) = lanczos(A, qm, m)

        That = _build_Hhat(That, T, η_prev, k, m)

        η_prev = η

        @views h = f(That)[((k - 1) * m + 1):((k - 1) * m + m), 1]

        log_metric!(trace, :update_norm, norm(h))

        fk .+= β * (Q * h)

        if rebuild_contour && (contour isa PacmanContour)
            λs, _ = eigen(T)
            cur_λmax = λs |> maximum
            λmax = cur_λmax > λmax ? cur_λmax : λmax
            cur_λmin = λs |> minimum
            λmin = cur_λmin < λmin ? cur_λmin : λmin

            r = λmin / contour_safety
            R = λmax * contour_safety
            contour = PacmanContour(r, R, r, float(π), complex(w))

            nodes, weights, fval, absfval, norm_val = chen_values(f, contour, order, w, λmax, λmin)
            log_metric!(trace, :radii, (r, R))
            log_metric!(trace, :norm_val, norm_val)
            log_metric!(trace, :absfval, absfval)
        end

        norm_hwz_T = Array{ComplexF64}(undef, order)
        det_w = det(That - w * I)
        for (i, node) in enumerate(nodes)
            norm_hwz_T[i] = det_w / det(That - node * I)
        end

        log_metric!(trace, :norm_hwz_T, norm_hwz_T)

        error_indicator = (integral_constant * dot(weights, absfval .* norm_hwz_T .* norm_val))

        log_metric!(trace, :error_indicator, error_indicator)

        if !isnothing(_NaNcheck(error_indicator))
            isnothing(trace) && throw(OverflowError(lazy"quadrature error diverged to NaN"))
            set_stop!(trace, QuadErrorDivergence)
            return fk
        end

        error_indicator_norm = error_indicator |> real |> abs

        if !(exact === nothing || isempty(exact))
            abs_err = norm(exact - fk)
            log_metric!(trace, :abs_err, abs_err)
        end

        if error_indicator_norm < tol
            set_stop!(trace, IndicatorAcc)
            return fk
        end
    end
    set_stop!(trace, MaxRestarts)
    return fk
end
