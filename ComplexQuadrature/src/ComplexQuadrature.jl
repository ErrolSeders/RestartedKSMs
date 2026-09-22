module ComplexQuadrature
"""
Structures for defining specfic integrals in the complex plane. Each struct provides an interface to build quadrature nodes and weights with a change of variables so the integral is transformed onto [-1,1].
"""

function transform_intervals(x, in, out = (-1, 1))
    a = first(in)
    b = last(in)
    c = first(out)
    d = last(out)

    return (d - c) / (b - a) * (x - a) + c
end


using FastGaussQuadrature
using LinearAlgebra

"""
    Contains two functions describing how to transform an integral from one coordinate system to another.
"""
struct ChangeOfVariables
    zmap::Function
    dzmap::Function
end

"""
    map_gausslegendre(zmap, dzmap, order)

    Perform a change of varaibles for Gauß-Legendre quadrature nodes and weights from [-1,1] to the contour.
    `order`: the number nodes & weights to generate
"""
function map_gausslegendre(cov::ChangeOfVariables, order::Int)
    x, w = gausslegendre(order)

    ξ = cov.zmap.(x)
    ω = w .* cov.dzmap.(x)

    return ξ, ω
end

"""
    piecewise_contour_quadrature(segments)

    For a piecewise defined contour, build the nodes and weights for each segment of the contour.
    Each segment of the contour has an independently defined order.

"""
function piecewise_contour_quadrature(segments)
    nodes = ComplexF64[]
    weights = ComplexF64[]

    for (cov, order) in segments
        zq, wq = map_gausslegendre(cov, order)

        append!(nodes, zq)
        append!(weights, wq)
    end

    return nodes, weights
end

abstract type AbstractContour end

"""
    resolve(C::AbtractContour, order)

    Resolve the quadrature nodes and weights for the contour.
    `order`: Number of quadrature nodes and weights to calculate
"""
function resolve(C::AbstractContour, order)
    return map_gausslegendre(C.cov, order)
end

"""
A circular contour defined by a radius `R` and a center `c`
"""
struct CircleContour{T <: Real} <: AbstractContour
    cov::ChangeOfVariables
    R::T
    c::Complex{T}
    CircleContour(R::T, c::Complex{T}) where {T <: Real} = new{T}(
        ChangeOfVariables(
            x -> c + R * cis(π * (x + 1)),
            x -> im * π * R * cis(π * (x + 1))
        )
        , R, c
    )
end

function build(CircleContour, λs, contour_buffer)::CircleContour
    c = sum(λs) / length(λs)
    dist = maximum(abs.(c .- λs))
    R = contour_buffer * dist
    R = max(contour_buffer, R)
    return CircleContour(R, complex(c))
end

"""
An elliptical contour described using semi-major axis `a`, semi-minor axis `b`, and center `c`
"""
struct EllipseContour{T <: Real} <: AbstractContour
    cov::ChangeOfVariables
    a::Complex{T}
    b::Complex{T}
    c::Complex{T}
    function EllipseContour(
            min::Complex{T}, max::Complex{T},
            real_safety = 1.1, imaginary_safety = 0.2
        ) where {T <: Real}

        c = (max + min) / 2
        d = (max - min) / 2

        a = real_safety * d
        b = imaginary_safety * d

        zmap = x -> begin
            θ = π * (x + 1)
            c + a * cos(θ) + 1im * b * sin(θ)
        end

        dzmap = x -> begin
            θ = π * (x + 1)
            π * (-a * sin(θ) + im * b * cos(θ))
        end

        return new{T}(
            ChangeOfVariables(zmap, dzmap),
            a, b, c
        )
    end

    EllipseContour{T}(
        cov::ChangeOfVariables, a::Complex{T}, b::Complex{T}, c::Complex{T}
    ) where {T <: Real} = new{T}(cov, a, b, c)
end

function EllipseContour(
        min::T, max::T,
        real_safety = 1.1, imaginary_safety = 0.2
    ) where {T <: Real}
    return EllipseContour(complex(min), complex(max), real_safety, imaginary_safety)
end

function build(::Type{EllipseContour}, λs, contour_buffer)
    isempty(λs) && throw(ArgumentError("cannot build an ellipse from an empty collection"))
    isfinite(contour_buffer) && contour_buffer > 0 ||
        throw(ArgumentError("contour_buffer must be finite and positive"))

    # Work in real coordinates so that the ellipse can be oriented along the
    # dominant direction of the eigenvalue cloud.
    λ = collect(λs)
    R = promote_type(Float64, typeof(real(first(λ))), typeof(imag(first(λ))))
    points = Matrix{R}(undef, length(λ), 2)
    for (i, z) in pairs(λ)
        points[i, 1] = real(z)
        points[i, 2] = imag(z)
    end

    center = vec(sum(points, dims = 1)) / length(λ)
    centered = points .- transpose(center)
    scatter = transpose(centered) * centered
    direction = if iszero(scatter)
        [one(R), zero(R)]
    else
        eigen(Symmetric(scatter)).vectors[:, 2]
    end
    # The second vector is the perpendicular direction for a symmetric
    # two-dimensional eigensystem.
    perpendicular = [-direction[2], direction[1]]

    major_coordinates = points * direction
    minor_coordinates = points * perpendicular
    major_center = (minimum(major_coordinates) + maximum(major_coordinates)) / 2
    minor_center = (minimum(minor_coordinates) + maximum(minor_coordinates)) / 2
    major_radius = (maximum(major_coordinates) - minimum(major_coordinates)) / 2
    minor_radius = (maximum(minor_coordinates) - minimum(minor_coordinates)) / 2
    ellipse_center = major_center * direction + minor_center * perpendicular

    # A zero-radius axis occurs for spectra on a line. Keep the contour
    # non-degenerate while making the added width negligible relative to the
    # scale of the spectrum.
    scale = max(one(R), major_radius, minor_radius)
    minor_radius = max(minor_radius, 0.15 * scale)
    major_radius = max(major_radius, 0.15 * scale)

    # Bounding-box radii alone do not always contain points in the corners of
    # the box. Scale by the largest normalized radius before applying the
    # requested safety factor.
    normalized_radius = maximum(
        sqrt.(
            (major_coordinates .- major_center) .^ 2 / major_radius^2 .+
                (minor_coordinates .- minor_center) .^ 2 / minor_radius^2
        )
    )
    factor = convert(R, contour_buffer) * max(one(R), normalized_radius)

    c = complex(ellipse_center[1], ellipse_center[2])
    axis = complex(direction[1], direction[2])
    a = complex(factor * major_radius * axis)
    b = complex(factor * minor_radius * axis)

    zmap = x -> begin
        θ = π * (x + 1)
        c + a * cos(θ) + 1im * b * sin(θ)
    end
    dzmap = x -> begin
        θ = π * (x + 1)
        π * (-a * sin(θ) + 1im * b * cos(θ))
    end

    return EllipseContour{R}(
        ChangeOfVariables(zmap, dzmap),
        a, b, c
    )
end
"""
Contour for use when integrating Stieljes functions. Always [-∞,0].
"""
struct StieltjesContour{T <: Real} <: AbstractContour
    cov::ChangeOfVariables
    StieltjesContour{T}() where {T <: Real} = new{T}(
        ChangeOfVariables(
            x -> -(1 + x) / (1 - x),
            x -> 2 / (1 - x)^2
        )
    )
end

StieltjesContour() = StieltjesContour{Float64}()

build(::Type{StieltjesContour}) = StieltjesContour()

build(::Type{StieltjesContour}, λs, contour_buffer) = StieltjesContour()

"""
    PacmanContour(r,R,h,θ,c)

A Pacman contour which consists of two concentric circles connected by a channel. This contour is useful for integrating functions with branch cuts along certain axis, for example the squre root function.
"""
struct PacmanContour{T <: Real} <: AbstractContour
    cov::ChangeOfVariables
    covs::Vector{ChangeOfVariables}
    r::T # Inner radius
    R::T # Outer radius
    h::T # Half-Diameter of channel
    θ::T # Cut angle
    c::Complex{T} # Center
end


function PacmanContour(r::T, R::T, h::T, θ::T, c::Complex{T}) where {T <: Real}

    # Angles where the channel interrupts the circles
    αr = asin(h / r)
    αR = asin(h / R)

    # Intersection points for channel
    sr = sqrt(r^2 - h^2)
    sR = sqrt(R^2 - h^2)

    # Outer circle

    z_out = x -> begin
        ρ = θ + αR + (π - αR) * (x + 1)
        c + R * cis(ρ)
    end

    dz_out = x -> begin
        ρ = θ + αR + (π - αR) * (x + 1)
        im * R * (π - αR) * cis(ρ)
    end


    # Lower channel

    z_low = x -> begin
        s = sR + (sr - sR) * (x + 1) / 2
        c + (s - im * h) * cis(θ)
    end

    dz_low = _ -> (sr - sR) / 2 * cis(θ)

    # Inner circle

    z_in = x -> begin
        ρ = θ + 2π - αr - (2π - 2αr) * (x + 1) / 2
        c + r * cis(ρ)
    end

    dz_in = x -> begin
        ρ = θ + 2π - αr - (2π - 2αr) * (x + 1) / 2
        -im * r * (π - αr) * cis(ρ)
    end

    # Upper channel

    z_up = x -> begin
        s = sr + (sR - sr) * (x + 1) / 2
        c + (s + im * h) * cis(θ)
    end

    dz_up = _ -> (sR - sr) / 2 * cis(θ)

    full_zmap = x -> x < -0.3 ? z_out(transform_intervals(x, (-1, -0.3), (0.0, 1.0))) : x <= -0.1 ? z_up(transform_intervals(x, (-0.3, -0.1))) : x <= 0.1 ? z_in(transform_intervals(x, (-0.1, 0.1))) : x <= 0.3 ? z_low(transform_intervals(x, (0.1, 0.3))) : x <= 1.0 ? z_out(transform_intervals(x, (0.3, 1.0), (-1.0, 0.0))) : z_out(0.0)

    full_dzmap = x -> x < -0.3 ? (1 / 0.7) * dz_out(transform_intervals(x, (-1, -0.3), (0.0, 1.0))) : x <= -0.1 ? 10 * dz_up(transform_intervals(x, (-0.3, -0.1))) : x <= 0.1 ? 10 * dz_in(transform_intervals(x, (-0.1, 0.1))) : x <= 0.3 ? 10 * dz_low(transform_intervals(x, (0.1, 0.3))) : x <= 1.0 ? (1 / 0.7) * dz_out(transform_intervals(x, (0.3, 1.0), (-1.0, 0.0))) : dz_out(0.0)

    return PacmanContour(
        ChangeOfVariables(full_zmap, full_dzmap),
        [
            ChangeOfVariables(z_out, dz_out),
            ChangeOfVariables(z_low, dz_low),
            ChangeOfVariables(z_in, dz_in),
            ChangeOfVariables(z_up, dz_up),
        ], r, R, h, θ, c
    )
end

"""
    build(PacmanContour, λs, contour_buffer; c=0, θ=0)

Build a Pacman contour around a cluster of eigenvalues. The inner radius is
chosen strictly inside the eigenvalue closest to `c`, while the outer radius
is expanded by `contour_buffer`. `θ` specifies the direction of the cut and
`c` its center. The channel half-width is chosen as half the inner radius.
"""
function build(::Type{PacmanContour}, λs, contour_buffer, θ = float(π), c = 0.0 + 0.0im)::PacmanContour

    distances = abs.(λs .- c)
    nearest = minimum(distances)
    farthest = maximum(distances)
    nearest > 0 ||
        throw(ArgumentError("the contour center must not coincide with an eigenvalue"))

    # Do not expand the inner radius: it must remain between c and the
    # closest eigenvalue so that the inner circle cannot cross the spectrum.
    T = promote_type(Float64, typeof(nearest), typeof(farthest), typeof(real(c)))
    r = convert(T, nearest / 2)
    R = convert(T, max(contour_buffer * farthest, r + eps(T) * max(one(T), r)))
    h = r / 2

    center = complex(convert(T, real(c)), convert(T, imag(c)))
    return PacmanContour(r, R, h, convert(T, θ), center)
end

"""
    resolve(P::PacmanContour, outer_order::Int, inner_order::Int, cut_order::Int)

    Return the nodes and weights for the given pacman contour. The final numbers of nodes will be
    ``outer_order + inner_order + (2 * cut_order)``
"""
function resolve(P::PacmanContour, outer_order::Int, inner_order::Int, cut_order::Int)
    orders = [outer_order, cut_order, inner_order, cut_order]
    segments = zip(P.covs, orders)
    return piecewise_contour_quadrature(segments)
end

"""
    resolve(P::PacmanContour, total_order::Int)

    Return the nodes and weights for the given Pacman contour. Attempts to divide the nodes across each segment with the majority going to the outer circle.
"""
function resolve(P::PacmanContour, total_order::Int)
    outer_order = max(8, floor(Int, total_order * 0.4))
    inner_order = max(4, floor(Int, total_order * 0.2))
    cut_order = max(4, floor(Int, total_order * 0.2))
    return resolve(P, outer_order, inner_order, cut_order)
end

export AbstractContour, ChangeOfVariables, CircleContour, EllipseContour, PacmanContour,
    StieltjesContour, build, resolve, transform_intervals

end
