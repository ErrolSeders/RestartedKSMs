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

    Hs = is_A_hermitian ? Vector([H]) : Vector([UpperHessenberg(H)])
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

function _shift_solve_e1!(y, H::UpperHessenberg, t, e1)

    copyto!(y, e1)
    y = rmul!(y, -one(eltype(y)))
    ldiv!(H, y; shift = -t)
    return y
end

function _shift_solve_e1!(y, H::SymTridiagonal, t, e1)
    copyto!(y, e1)
    rmul!(y, -one(eltype(y)))
    ldiv!(H, y; shift = -t)
    return y
end


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
        e1,
        trace::Union{Nothing, Trace}
    )

    T = promote_type(eltype(H), eltype(x), eltype(e1))

    S = zeros(ComplexF64, size(H, 1))
    y = Vector{ComplexF64}(undef, size(H, 1))

    sum_of_term_norms = 0

    isnothingtrace = isnothing(trace)

    for (i, (t, ω)) in enumerate(zip(x, w))

        ϕ = one(ComplexF64)


        for j in eachindex(Hs)
            _shift_solve_e1!(y, Hs[j], t, e1)
            ϕ *= ηs[j] * y[end]
        end

        _shift_solve_e1!(y, H, t, e1)

        coeff = (1 / (2π * 1im)) * ω * f(t) * ϕ

        if !isnothingtrace
            sum_of_term_norms += norm(coeff * y)
        end

        @. S += coeff * y
    end

    log_metric!(trace, :quad_conditioning, (length(x), sum_of_term_norms / norm(S)))

    return S
end

"""
    krylov_approx_quad(
        f::Function,
        A::AbstractArray,
        b::AbstractVector,
        m::Int
        ;
        quad_tol = 1.0e-12,
        stop_tol = 1.0e-16,
        max_restarts = 200,
        contour_safety = 1.1,
        ellipse_height = 0.15
        order = 24,
        maxorder = 20_000,
        contour_type,
        rebuild_contour,
        trace::Union{Nothing, Trace} = nothing,
        contour_override,
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
        ellipse_height = 0.15,
        order = 24,
        maxorder = 20_000,
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

    ritz = is_A_hermitian ? Float64[] : ComplexF64[]

    contour = isnothing(contour_override) ?
        begin
            # runic: off
            append!(ritz,first(eigen(H)))
            if contour_type isa EllipseContour
                build(contour_type, ritz, contour_safety, ellipse_height)
            else
                build(contour_type, ritz, contour_safety)
            end
            # runic: on
        end : contour_override

    log_metric!(trace, :ritz_contour, (deepcopy(ritz), contour))

    update_norm_prev = Inf

    for k in 2:max_restarts

        set_restart!(trace, k)

        @debug "Restart: $(k)"

        #@info "Iteration $_k"
        (Q, H, η, qm) = is_A_hermitian ? lanczos(A, qm, m) : arnoldi(A, qm, m)

        if isnothing(contour_override) && rebuild_contour
            append!(ritz, first(eigen(H)))
            contour = build(contour_type, ritz, contour_safety)
            log_metric!(trace, :ritz_contour, (deepcopy(ritz), contour))
        end

        accurate = false
        refined = false

        while !accurate

            if !isnothing(_check_quad_order_divergence(quad1, quad2, maxorder, trace))
                isnothing(trace) && throw(ErrorException(lazy"quadrature order exceeded allowence"))
                set_stop!(trace, QuadOrderDivergence)
                return fk
            end

            log_metric!(trace, :quad1_order, (k, quad1))
            log_metric!(trace, :quad2_order, (k, quad2))

            x1, w1 = resolve(contour, quad1)
            x2, w2 = resolve(contour, quad2)

            H = is_A_hermitian ? H : UpperHessenberg(H)

            h1 = _integral_error_correction(f, H, Hs, ηs, x1, w1, e1, trace)
            h2 = _integral_error_correction(f, H, Hs, ηs, x2, w2, e1, trace)

            err = norm(h2 - h1)
            log_metric!(trace, :quad_err, (k, err))
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
        e1,
        trace::Union{Nothing, Trace}
    )

    sum_of_term_norms = 0

    t1 = first(x)
    ω1 = first(w)

    T = promote_type(
        eltype(H),
        eltype(e1),
        typeof(t1),
        typeof(ω1),
        typeof(f.constant),
        typeof(f.inner(t1)),
    )

    isnothingtrace = isnothing(trace)

    S = zeros(T, size(H, 1))
    y = similar(S)

    for (t, ω) in zip(x, w)

        ϕ = one(T)

        for j in eachindex(Hs)
            _shift_solve_e1!(y, Hs[j], t, e1)
            ϕ *= ηs[j] * y[end]
        end

        _shift_solve_e1!(y, H, t, e1)

        coeff = f.constant * ω * f.inner(t) * ϕ

        if !isnothingtrace
            sum_of_term_norms += norm(coeff * y)
        end

        @. S += coeff * y
    end

    log_metric!(trace, :quad_conditioning, (length(x), sum_of_term_norms / norm(S)))

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
        contour_type::Type{<:AbstractStieltjesContour} = StieltjesContour,
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

    contour = build(contour_type)

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

            log_metric!(trace, :quad1_order, (k, quad1))
            log_metric!(trace, :quad2_order, (k, quad2))

            x1, w1 = resolve(contour, quad1)
            x2, w2 = resolve(contour, quad2)

            H = is_A_hermitian ? H : UpperHessenberg(H)

            h1 = _integral_error_correction_stieltjes(f, H, Hs, ηs, x1, w1, e1, trace)
            h2 = _integral_error_correction_stieltjes(f, H, Hs, ηs, x2, w2, e1, trace)

            err = norm(h2 - h1)

            log_metric!(trace, :quad_err, (k, err))

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

    y = Vector{ComplexF64}(undef, size(H, 1))

    function F(x, p = nothing)

        t = cov.zmap(x)

        w = cov.dzmap(x)

        ϕ = one(eltype(H))

        for j in eachindex(Hs)
            _shift_solve_e1!(y, Hs[j], t, e1)
            ϕ *= ηs[j] * y[end]
        end

        _shift_solve_e1!(y, H, t, e1)

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
        ellipse_height = 0.15,
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

    Hs = is_A_hermitian ? [H] : [UpperHessenberg(H)]
    ηs::Vector{eltype(H)} = [η]

    h = f(H)[:, 1]

    fk = β * Q * h

    _check_exact_err(exact, fk, trace)

    e1 = unit_vector(eltype(H), m, 1)

    ritz = is_A_hermitian ? Float64[] : ComplexF64[]

    contour::AbstractContour = isnothing(contour_override) ?
        begin
            # runic: off
            append!(ritz, first(eigen(H)))
            if contour_type isa EllipseContour
                build(contour_type, ritz, contour_safety, ellipse_height)
            else
                build(contour_type, ritz, contour_safety)
            end
            # runic: on
        end : contour_override

    log_metric!(trace, :ritz_contour, (deepcopy(ritz), contour))

    for k in 2:max_restarts

        set_restart!(trace, k)

        (Q, H, η, qm) = is_A_hermitian ? lanczos(A, qm, m) : arnoldi(A, qm, m)

        if isnothing(contour_override) && rebuild_contour
            append!(ritz, first(eigen(H)))
            contour = build(contour_type, ritz, contour_safety)
            log_metric!(trace, :ritz_contour, (deepcopy(ritz), contour))
        end

        H = is_A_hermitian ? H : UpperHessenberg(H)

        F = _make_integral_function(f, H, Hs, ηs, contour.cov, e1)
        prob = IntegralProblem(F, (-1, 1))
        sol = solve(prob, alg, reltol = quad_tol, abstol = quad_tol)
        log_metric!(trace, :quad_err, (k, sol.resid))
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
    _make_integral_function_stieltjes(f, H, Hs, ηs, e1)

    Generate a function `F` to be integrated when
    the matrix function to be approximanted is a Stieltjes function.

"""
function _make_integral_function_stieltjes(f, H, Hs, ηs, e1)

    y = Vector{eltype(H)}(undef, size(H, 1))

    function F(x, p = nothing)

        ϕ = one(eltype(H))

        for j in eachindex(Hs)
            _shift_solve_e1!(y, Hs[j], x, e1)
            ϕ *= ηs[j] * y[end]
        end

        _shift_solve_e1!(y, H, x, e1)

        return (f.constant * (f.inner(x) * ϕ)) * y
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

    Hs = is_A_hermitian ? [H] : [UpperHessenberg(H)]
    ηs::Vector{eltype(H)} = [η]

    h = f(H)[:, 1]
    fk = β * Q * h

    _check_exact_err(exact, fk, trace)

    e1 = unit_vector(eltype(H), m, 1)

    for k in 2:max_restarts

        set_restart!(trace, k)

        (Q, H, η, qm) = is_A_hermitian ? lanczos(A, qm, m) : arnoldi(A, qm, m)

        H = is_A_hermitian ? H : UpperHessenberg(H)

        F = _make_integral_function_stieltjes(f, H, Hs, ηs, e1)
        prob = IntegralProblem(F, f.limits)
        sol = solve(prob, alg, reltol = quad_tol, abstol = quad_tol)

        log_metric!(trace, :quad_err, (k, sol.resid))

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

function _lanczos_stieltjes_action(nodes, weights, ρ, T::SymTridiagonal)

    e1 = unit_vector(eltype(T), size(T, 1), 1)

    S = zeros(eltype(T), size(T, 1))
    y = Vector{eltype(T)}(undef, size(T, 1))

    for (i, t) in pairs(IndexLinear(), nodes)
        _shift_solve_e1!(y, T, t, -e1)
        S .+= weights[i] * ρ[i] * y
    end

    return S

end

function _lanczos_stieltjes_recurrence!(d, ρ, T, η, nodes)
    fill!(d, one(eltype(d)))
    m = size(T, 1)

    for r in 1:m
        α = T.dv[r]
        βprev = (r == 1) ? zero(η) : T.ev[r - 1]
        βout = (r == m) ? η : T.ev[r]

        @. d = α - nodes - βprev^2 / d
        @. ρ = ρ * βout / d
    end
    return d, ρ
end

function _radau_extension(T::SymTridiagonal, η, λ, em)
    shift_sol = SymTridiagonal(T.dv .- λ, T.ev) \ em
    δ = η^2 * shift_sol[end]

    return SymTridiagonal(
        vcat(T.dv, δ),
        vcat(T.ev, η),
    )
end

function _lanczos_stieltjes_restart(
        f::StieltjesFunction,
        A::AbstractArray,
        b::AbstractVector,
        m::Int,
        nodes,
        weights,
        tol,
        max_restarts::Int,
        safety_factor,
        trace::Union{Nothing, Trace} = nothing,
        exact::Union{Nothing, AbstractVector} = nothing
    )

    real_res = (eltype(A) <: AbstractFloat) && (eltype(b) <: AbstractFloat)

    normb = norm(b)

    qm = b / normb

    (Q, T, η, qm) = lanczos(A, qm, m)


    fk = normb * Q * f(T)[:, 1]

    _check_exact_err(exact, fk, trace)

    if iszero(η)
        set_stop!(trace, LanczosBreakdown)
        return fk
    end

    em = unit_vector(eltype(A), m, m)

    d = ones(eltype(weights), size(nodes))
    ρ = ones(eltype(weights), size(nodes))
    _lanczos_stieltjes_recurrence!(d, ρ, T, η, nodes)

    min_ritz = eigmin(T)

    for k in 2:max_restarts

        update_sign = isodd(m) && iseven(k) ? -1 : 1

        set_restart!(trace, k)

        (Q, T, η, qm) = lanczos(A, qm, m)

        gauss_update = _lanczos_stieltjes_action(nodes, weights, ρ, T)

        low_bnd = normb * norm(gauss_update)

        update = (normb * update_sign) * (Q * gauss_update)
        update_norm = update |> norm

        log_metric!(trace, :update_norm, update_norm)

        # In case of breakdown we apply the current update
        if iszero(η)
            set_stop!(trace, LanczosBreakdown)
            fk .+= update
            _check_exact_err(exact, fk, trace)
            return real_res ? fk .|> real : fk
        end

        θ = eigmin(T)
        min_ritz = min(min_ritz, θ)
        λ = safety_factor * min_ritz

        Tbar = _radau_extension(T, η, λ, em)

        radau_update = _lanczos_stieltjes_action(nodes, weights, ρ, Tbar)
        up_bnd = normb * norm(radau_update)

        log_metric!(trace, :low_bnd, low_bnd)
        log_metric!(trace, :up_bnd, up_bnd)
        log_metric!(trace, :delta_bnd, up_bnd - low_bnd)

        if up_bnd < tol
            set_stop!(trace, UpBndAcc)
            return real_res ? fk .|> real : fk
        end

        fk .+= update

        _check_exact_err(exact, fk, trace)

        if update_norm < tol
            set_stop!(trace, UpdateAcc)
            return real_res ? fk .|> real : fk
        end
        _lanczos_stieltjes_recurrence!(d, ρ, T, η, nodes)
    end
    set_stop!(trace, MaxRestarts)
    return real_res ? fk .|> real : fk
end

function _arnoldi_stieltjes_action(ρ, H::UpperHessenberg, η, nodes, weights, e1)

    h = zeros(eltype(H), size(H, 1))
    ρ_next = zeros(eltype(H), size(ρ))

    y = Vector{eltype(H)}(undef, size(H, 1))

    for (i, t) in pairs(IndexLinear(), nodes)
        _shift_solve_e1!(y, H, t, -e1)

        @. h += weights[i] * ρ[i] * y

        ρ_next[i] = -η * ρ[i] * y[end]
    end
    return h, ρ_next
end

function _arnoldi_stieltjes_restart(
        f::StieltjesFunction,
        A::AbstractArray,
        b::AbstractVector,
        m::Int,
        nodes,
        weights,
        tol,
        max_restarts::Int,
        safety_factor,
        trace::Union{Nothing, Trace} = nothing,
        exact::Union{Nothing, AbstractVector} = nothing
    )

    real_res = (eltype(A) <: AbstractFloat) && (eltype(b) <: AbstractFloat)

    normb = norm(b)

    qm = b / normb

    (Q, H, η, qm) = arnoldi(A, qm, m)

    H = UpperHessenberg(H)

    fk = normb * Q * f(H)[:, 1]

    _check_exact_err(exact, fk, trace)

    if iszero(η)
        set_stop!(trace, ArnoldiBreakdown)
        return fk
    end

    e1 = unit_vector(eltype(A), m, 1)

    ρ = ones(eltype(weights), size(nodes))


    _, ρ = _arnoldi_stieltjes_action(ρ, H, η, nodes, weights, e1)

    for k in 2:max_restarts

        set_restart!(trace, k)

        (Q, H, η, qm) = arnoldi(A, qm, m)
        H = UpperHessenberg(H)

        h, ρ_next = _arnoldi_stieltjes_action(ρ, H, η, nodes, weights, e1)

        error_indicator = normb * norm(h)

        update = normb * Q * h
        update_norm = update |> norm

        log_metric!(trace, :update_norm, update_norm)
        log_metric!(trace, :error_indicator, error_indicator)

        # In case of breakdown we apply the current update
        if iszero(η)
            set_stop!(trace, ArnoldiBreakdown)
            fk .+= update
            _check_exact_err(exact, fk, trace)
            return real_res ? fk .|> real : fk
        end

        if error_indicator < (safety_factor * tol)
            set_stop!(trace, IndicatorAcc)
            return real_res ? fk .|> real : fk
        end

        fk .+= update
        _check_exact_err(exact, fk, trace)

        if update_norm < tol
            set_stop!(trace, UpdateAcc)
            return real_res ? fk .|> real : fk
        end

        ρ = ρ_next

    end
    set_stop!(trace, MaxRestarts)
    return fk
end

"""
    Calculate `f(A)b` by the algorithm described by Frommer and Schweitzer for a Stieltjes function `f` and a positive definite matrix `A`.
"""
function krylov_approx_stieltjes(
        f::StieltjesFunction,
        A::AbstractArray,
        b::AbstractVector,
        m::Int,
        ;
        order = 40,
        tol = 1.0e-16,
        max_restarts = 200,
        safety_factor = 0.99,
        contour_type::Type{<:AbstractStieltjesContour} = SquaredStieltjesContour,
        trace::Union{Nothing, Trace} = nothing,
        exact::Union{Nothing, AbstractVector} = nothing
    )

    set_type!(trace, FrommerStieltjes)

    _check_sizes(exact, A, b)

    contour = build(contour_type, f.limits)
    nodes, ω = resolve(contour, order)
    weights = (-f.constant) .* ω .* f.inner.(nodes)

    A_is_hermitian = ishermitian(A)

    if A_is_hermitian
        fk = _lanczos_stieltjes_restart(f, A, b, m, nodes, weights, tol, max_restarts, safety_factor, trace, exact)
    else
        fk = _arnoldi_stieltjes_restart(f, A, b, m, nodes, weights, tol, max_restarts, safety_factor, trace, exact)
    end
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
