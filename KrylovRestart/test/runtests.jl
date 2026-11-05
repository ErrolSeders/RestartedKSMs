using Test
using LinearAlgebra
using Random
using KrylovRestart
using ComplexQuadrature

relative_error(x, y) = norm(x - y) / max(norm(y), eps(real(eltype(y))))

@testset "KrylovRestart" begin
    Random.seed!(1234)

    @testset "Krylov decompositions" begin
        n, m = 12, 6
        A = 0.1 .* randn(n, n)
        b = randn(n)
        Q, H, η, qnext = arnoldi(A, b, m)

        @test size(Q) == (n, m)
        @test size(H) == (m, m)
        @test isapprox(Q' * Q, I(m); atol = 1.0e-11)
        @test norm(Q' * qnext) ≤ 1.0e-11

        residual = A * Q - Q * H -
            (η .* qnext) * [zeros(1, m - 1) 1]
        @test norm(residual) / norm(A * Q) ≤ 1.0e-11

        M = randn(n, n)
        S = 0.1 .* (M + M') / 2
        Q, T, η, qnext = lanczos(S, b, m)
        @test isapprox(Q' * Q, I(m); atol = 1.0e-11)
        @test isapprox(T, T'; atol = 1.0e-12)
        residual = S * Q - Q * T -
            (η .* qnext) * [zeros(1, m - 1) 1]
        @test norm(residual) / norm(S * Q) ≤ 1.0e-11
    end

    @testset "Stieltjes representations" begin
        f = log1p_over_z()
        @test f.outer === nothing
        @test f.limits == (-Inf, -1)
        @test isapprox(f(0.5; reltol = 1.0e-11, abstol = 1.0e-11),
            log(1.5) / 0.5; rtol = 1.0e-9)
        @test isapprox(f(2.0; reltol = 1.0e-11, abstol = 1.0e-11),
            log(3.0) / 2; rtol = 1.0e-9)

        A = Diagonal([0.25, 1.0, 4.0])
        expected = Diagonal(log.(1 .+ diag(A)) ./ diag(A))
        @test isapprox(f(Matrix(A); reltol = 1.0e-10, abstol = 1.0e-10),
            Matrix(expected); rtol = 1.0e-8, atol = 1.0e-10)

        finite = StieltjesFunction(nothing, _ -> 1.0, 1.0, (0.0, 1.0))
        @test isapprox(finite(2.0; reltol = 1.0e-11, abstol = 1.0e-11),
            -log(2); rtol = 1.0e-9)
        @test StieltjesFunction(nothing, _ -> 1.0, 1.0).limits == (-Inf, 0)
        @test inverse_pth_root(2).limits == (-Inf, 0)
    end

    @testset "Stieltjes contours" begin
        finite = StieltjesContour((0.0, 1.0))
        nodes, weights = resolve(finite, 32)
        @test finite.limits == (0.0, 1.0)
        @test all((0 .< nodes) .& (nodes .< 1))
        @test isapprox(sum(weights), 1.0; atol = 1.0e-13)

        semi_infinite = StieltjesContour((-Inf, -1.0))
        nodes, weights = resolve(semi_infinite, 32)
        @test all(nodes .< -1)
        @test all(isfinite, weights)

        squared = SquaredStieltjesContour((-Inf, -1.0))
        nodes, weights = resolve(squared, 32)
        @test squared.limits == (-Inf, -1.0)
        @test all(nodes .< -1)
        @test all(isfinite, weights)
        @test_throws ArgumentError SquaredStieltjesContour((0.0, 1.0))
    end

    @testset "Non-Stieltjes contours" begin
        circle = CircleContour(2.0, 0.5 + 0.25im)
        nodes, weights = resolve(circle, 32)
        @test all(isfinite, nodes)
        @test all(isfinite, weights)
        @test all(isapprox.(abs.(nodes .- circle.c), circle.R; rtol = 1.0e-13))

        ellipse = EllipseContour(-1.0 - 0.5im, 2.0 + 0.75im)
        nodes, weights = resolve(ellipse, 32)
        @test all(isfinite, nodes)
        @test all(isfinite, weights)
        @test isapprox(sum(weights), 0; atol = 1.0e-12)

        built = build(EllipseContour, [0.2, 0.8, 1.4], 1.2)
        nodes, _ = resolve(built, 32)
        @test all(isfinite, nodes)
        @test all(abs.(nodes .- built.c) .≤ abs(built.a) + abs(built.b))
        @test_throws ArgumentError build(EllipseContour, ComplexF64[], 1.2)
    end

    @testset "Non-Stieltjes quadrature methods" begin
        n, m = 6, 3
        λ = collect(range(-0.4, 0.8; length = n))
        Aherm = Matrix(Diagonal(λ))
        S = Matrix(I, n, n) + 0.05 .* triu(randn(n, n), 1)
        Anon = S * Aherm / S
        b = randn(n)

        for (name, f, exact_f) in (
            ("exp", exp, exp),
            ("cos", cos, cos),
        )
            for (matrix_name, A) in (("Hermitian", Aherm), ("non-Hermitian", Anon))
                @testset "$name on $matrix_name" begin
                    exact = exact_f(A) * b
                    quad1_circle = krylov_approx_quad(f, A, b, m;
                        order = 16, max_restarts = 8,
                        quad_tol = 1.0e-8, stop_tol = 1.0e-8,
                        contour_type = CircleContour)
                    quad1_ellipse = krylov_approx_quad(f, A, b, m;
                        order = 16, max_restarts = 8,
                        quad_tol = 1.0e-8, stop_tol = 1.0e-8,
                        contour_type = EllipseContour)
                    quad2_circle = krylov_approx_quad2(f, A, b, m;
                        max_restarts = 8, quad_tol = 1.0e-8,
                        stop_tol = 1.0e-8, contour_type = CircleContour)

                    @test relative_error(quad1_circle, exact) ≤ 1.0e-6
                    @test relative_error(quad1_ellipse, exact) ≤ 1.0e-6
                    @test relative_error(quad2_circle, exact) ≤ 1.0e-6
                end
            end
        end
    end

    @testset "Quadrature methods" begin
        n, m = 8, 4
        λ = collect(range(0.2, 2.0; length = n))
        Aherm = Matrix(Diagonal(λ))
        S = Matrix(I, n, n) + 0.08 .* triu(randn(n, n), 1)
        Anon = S * Aherm / S
        b = randn(n)

        fs = inverse_pth_root(2)
        fl = log1p_over_z()
        fs_exact(A) = inv(sqrt(A))
        fl_exact(A) = log(I + A) / A

        for (name, A) in (("Hermitian", Aherm), ("non-Hermitian", Anon))
            @testset "$name inverse square root" begin
                exact = fs_exact(A) * b
                quad1 = krylov_approx_quad(fs, A, b, m;
                    max_restarts = 10, quad_tol = 1.0e-8,
                    stop_tol = 1.0e-8, contour_type = SquaredStieltjesContour)
                quad2 = krylov_approx_quad2(fs, A, b, m;
                    max_restarts = 10, quad_tol = 1.0e-8, stop_tol = 1.0e-8)
                stieltjes = krylov_approx_stieltjes(fs, A, b, m;
                    order = 24, max_restarts = 10, tol = 1.0e-8,
                    contour_type = SquaredStieltjesContour)

                @test relative_error(quad1, exact) ≤ 1.0e-6
                @test relative_error(quad2, exact) ≤ 1.0e-6
                @test relative_error(stieltjes, exact) ≤ 1.0e-6
            end

            @testset "$name log(1+z)/z" begin
                exact = fl_exact(A) * b
                quad2 = krylov_approx_quad2(fl, A, b, m;
                    max_restarts = 10, quad_tol = 1.0e-8, stop_tol = 1.0e-8)
                stieltjes = krylov_approx_stieltjes(fl, A, b, m;
                    order = 24, max_restarts = 10, tol = 1.0e-8,
                    contour_type = StieltjesContour)

                @test relative_error(quad2, exact) ≤ 1.0e-6
                @test relative_error(stieltjes, exact) ≤ 1.0e-6
            end
        end
    end
end
