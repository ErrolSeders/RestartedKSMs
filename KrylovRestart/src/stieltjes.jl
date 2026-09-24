"""
A Stieltjes function is able to represented as a Riemann-Stieltjes integral with an `inner` function of constant sign on (-∞,0]
The value of the Riemann-Stieltjes integral is multiplied by `constant`.
The function `outer` is the standard representation of the function being represented and is expected to be able to be called on a matrix. Provide an `outer` function when Julia provides algorithms for calculating the matrix function already. Those will certainly be faster & more accurate than evaluating the integral by quadrature.

example: f(z) = z^(-1/2)
    outer = (sqrt ∘ inv)
    inner = (sqrt ∘ inv ∘ -) # equivalently: (t) -> inv(sqrt(-t))
    constant = sin((1/2 - 1)*π)/π
"""
struct StieltjesFunction
    outer::Union{Nothing, Function}
    inner::Function
    constant
end

"""
    (f::StieltjesFunction)(z; reltol = nothing, abstol = nothing)

Evaluate the Stieltjes function for a value `z` that is not on the negative real line.
"""
(f::StieltjesFunction)(z; reltol = nothing, abstol = nothing) = isnothing(
        f.outer
    ) ? begin
        Γ = [-Inf, 0]
        g = (t, p) -> (f.inner(t) / (t - z))
        prob = IntegralProblem(g, Γ)

        reltol = isnothing(reltol) ? geteps(z) : reltol
        abstol = isnothing(abstol) ? geteps(z) : abstol

        value = solve(prob, QuadGKJL(), reltol = reltol, abstol = abstol) |> only
        f.constant * value
    end : f.outer(z)

"""
    (f::StieltjesFunction)(A::AbstractArray; reltol = nothing, abstol = nothing)

Evaluate the Stieltjes function for a matrix `A` with no eigenvalues on the negative real line.
"""
(f::StieltjesFunction)(A::AbstractArray; reltol = nothing, abstol = nothing) = isnothing(
        f.outer
    ) ? begin
        Γ = [-Inf, 0]
        g = (t, p) -> f.inner(t) * inv(t * I - A)
        prob = IntegralProblem(g, Γ)

        reltol = isnothing(reltol) ? geteps(A) : reltol
        abstol = isnothing(abstol) ? geteps(A) : abstol

        value = solve(prob, QuadGKJL(), reltol = reltol, abstol = abstol)
        f.constant * value
    end : f.outer(A)

inverse_pth_root(p) = StieltjesFunction(
    (A) -> (inv(A^(1 / p))),
    (t) -> (inv((-t)^(1 / p))),
    sin((1 / p - 1) * π) / π
)
