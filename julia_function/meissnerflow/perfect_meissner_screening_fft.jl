# perfect_meissner_screening_fft.jl
#
# Vortex-free, ideal-Meissner screening current in a thin superconducting film.
#
# Model
# -----
# The film occupies a 2D region Omega in the z = 0 plane.  The sheet current
# K(x, y) [A/m] is represented by a stream function g(x, y) [A]:
#
#     Kx = d g / d y,
#     Ky = -d g / d x.
#
# The perpendicular self-field is computed with the thin-film Biot-Savart
# operator Q:
#
#     F[Q g](kx, ky) = |k|/2 * F[g].
#
# In the ideal, vortex-free, perfectly screening limit, the total normal field
# vanishes inside the film:
#
#     Delta B_ext + mu0 * Q[g] = 0       in Omega,
#     g = 0                              outside Omega.
#
# The code solves this restricted linear equation by conjugate gradients.  FFTs
# are used to apply Q.  The computational box must include vacuum padding around
# the film because FFTs impose periodic boundary conditions.
# This implementation assumes a simply connected film with no holes and no
# transport current.  Holes require additional stream-function constants or
# fluxoid constraints and are intentionally outside this first model.
#
# Important physical limitation
# -----------------------------
# In the strict ideal-screening limit the film thickness does NOT change the
# sheet-current distribution K.  Thickness is used only to convert K [A/m] to
# a volume current density J = K / thickness [A/m^2].  To make thickness affect
# the spatial distribution one must include a finite London penetration depth.
#
# Packages
# --------
#   import Pkg
#   Pkg.add(["FFTW", "CairoMakie"])
#
# References
# ----------
# L. Prigozhin and V. Sokolovsky, Supercond. Sci. Technol. 31, 055018 (2018).
# E. H. Brandt, Phys. Rev. B 72, 024529 (2005).

module PerfectMeissnerScreeningFFT

using FFTW
using LinearAlgebra
using Statistics
using Printf
using Serialization
using CairoMakie

export MU0,
       Grid2D,
       rectangle_mask,
       disk_mask,
       ellipse_mask,
       polygon_mask,
       make_shape_mask,
       solve_perfect_screening,
       simulate_perfect_screening,
       scale_screening_response,
       plot_screening_current,
       plot_field_check,
       save_screening_response,
       load_screening_response

const MU0 = 4pi * 1e-7

# -----------------------------------------------------------------------------
# Grid
# -----------------------------------------------------------------------------

"""
    Grid2D(nx, ny; pixel_size_um)
    Grid2D(nx, ny, Lx_m, Ly_m)

Uniform, cell-centred 2D grid.  Arrays use Julia image order `(ny, nx)`.
The physical x and y coordinates increase to the right and upward.
"""
struct Grid2D
    nx::Int
    ny::Int
    dx::Float64
    dy::Float64
    x::Vector{Float64}
    y::Vector{Float64}
    kx::Vector{Float64}
    ky::Vector{Float64}
    kmag::Matrix{Float64}
end

function _fft_wavenumbers(n::Int, d::Real)
    @assert n >= 2 "Grid size must be at least 2."
    @assert d > 0 "Grid spacing must be positive."

    modes = if iseven(n)
        vcat(0:(div(n, 2) - 1), -div(n, 2):-1)
    else
        m = div(n - 1, 2)
        vcat(0:m, -m:-1)
    end

    return (2pi / (n * Float64(d))) .* Float64.(modes)
end

function Grid2D(nx::Integer, ny::Integer, Lx_m::Real, Ly_m::Real)
    nx_i = Int(nx)
    ny_i = Int(ny)
    @assert nx_i >= 4 && ny_i >= 4 "Use at least a 4 x 4 grid."
    @assert Lx_m > 0 && Ly_m > 0 "Physical grid lengths must be positive."

    dx = Float64(Lx_m) / nx_i
    dy = Float64(Ly_m) / ny_i

    # Cell centres.  For an even number of points, x = 0 lies between two cells.
    x = ((collect(0:(nx_i - 1)) .- (nx_i - 1) / 2) .* dx)
    y = ((collect(0:(ny_i - 1)) .- (ny_i - 1) / 2) .* dy)

    kx = _fft_wavenumbers(nx_i, dx)
    ky = _fft_wavenumbers(ny_i, dy)

    KX = reshape(kx, 1, nx_i)
    KY = reshape(ky, ny_i, 1)
    kmag = sqrt.(KX .^ 2 .+ KY .^ 2)

    return Grid2D(nx_i, ny_i, dx, dy, x, y, kx, ky, kmag)
end

function Grid2D(nx::Integer, ny::Integer; pixel_size_um::Real)
    @assert pixel_size_um > 0 "pixel_size_um must be positive."
    d = Float64(pixel_size_um) * 1e-6
    return Grid2D(nx, ny, Int(nx) * d, Int(ny) * d)
end

# -----------------------------------------------------------------------------
# Film geometry
# -----------------------------------------------------------------------------

function _xy_mesh(grid::Grid2D)
    X = reshape(grid.x, 1, grid.nx)
    Y = reshape(grid.y, grid.ny, 1)
    return X, Y
end

"""
    rectangle_mask(grid; width_um, height_um, center_um=(0, 0), angle_deg=0)

Boolean mask for a rotated rectangle.
"""
function rectangle_mask(
    grid::Grid2D;
    width_um::Real,
    height_um::Real,
    center_um::Tuple{<:Real,<:Real} = (0.0, 0.0),
    angle_deg::Real = 0.0,
)
    @assert width_um > 0 && height_um > 0

    X, Y = _xy_mesh(grid)
    xc = Float64(center_um[1]) * 1e-6
    yc = Float64(center_um[2]) * 1e-6
    theta = deg2rad(Float64(angle_deg))

    c = cos(theta)
    s = sin(theta)
    xr = c .* (X .- xc) .+ s .* (Y .- yc)
    yr = -s .* (X .- xc) .+ c .* (Y .- yc)

    w = Float64(width_um) * 1e-6
    h = Float64(height_um) * 1e-6
    return BitMatrix((abs.(xr) .<= w / 2) .& (abs.(yr) .<= h / 2))
end

"""
    disk_mask(grid; radius_um, center_um=(0, 0))
"""
function disk_mask(
    grid::Grid2D;
    radius_um::Real,
    center_um::Tuple{<:Real,<:Real} = (0.0, 0.0),
)
    @assert radius_um > 0

    X, Y = _xy_mesh(grid)
    xc = Float64(center_um[1]) * 1e-6
    yc = Float64(center_um[2]) * 1e-6
    r = Float64(radius_um) * 1e-6

    return BitMatrix((X .- xc) .^ 2 .+ (Y .- yc) .^ 2 .<= r ^ 2)
end

"""
    ellipse_mask(grid; radius_x_um, radius_y_um,
                 center_um=(0, 0), angle_deg=0)
"""
function ellipse_mask(
    grid::Grid2D;
    radius_x_um::Real,
    radius_y_um::Real,
    center_um::Tuple{<:Real,<:Real} = (0.0, 0.0),
    angle_deg::Real = 0.0,
)
    @assert radius_x_um > 0 && radius_y_um > 0

    X, Y = _xy_mesh(grid)
    xc = Float64(center_um[1]) * 1e-6
    yc = Float64(center_um[2]) * 1e-6
    theta = deg2rad(Float64(angle_deg))

    c = cos(theta)
    s = sin(theta)
    xr = c .* (X .- xc) .+ s .* (Y .- yc)
    yr = -s .* (X .- xc) .+ c .* (Y .- yc)

    rx = Float64(radius_x_um) * 1e-6
    ry = Float64(radius_y_um) * 1e-6
    return BitMatrix((xr ./ rx) .^ 2 .+ (yr ./ ry) .^ 2 .<= 1.0)
end

function _point_in_polygon(x::Real, y::Real, vertices_m::Vector{Tuple{Float64,Float64}})
    inside = false
    n = length(vertices_m)
    j = n

    for i in 1:n
        xi, yi = vertices_m[i]
        xj, yj = vertices_m[j]

        crosses = ((yi > y) != (yj > y)) &&
                  (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        if crosses
            inside = !inside
        end
        j = i
    end

    return inside
end

"""
    polygon_mask(grid; vertices_um)

Create a mask from polygon vertices `[(x1, y1), (x2, y2), ...]` in micrometres.
The polygon should be simple and non-self-intersecting.
"""
function polygon_mask(grid::Grid2D; vertices_um)
    @assert length(vertices_um) >= 3 "A polygon needs at least three vertices."

    vertices_m = [
        (Float64(v[1]) * 1e-6, Float64(v[2]) * 1e-6)
        for v in vertices_um
    ]

    mask = falses(grid.ny, grid.nx)
    for iy in 1:grid.ny, ix in 1:grid.nx
        mask[iy, ix] = _point_in_polygon(grid.x[ix], grid.y[iy], vertices_m)
    end
    return BitMatrix(mask)
end

"""
    make_shape_mask(grid; shape, shape_params)

Supported shapes are `:rectangle`, `:disk`, `:ellipse`, and `:polygon`.
`shape_params` is a NamedTuple containing the keyword arguments for the
corresponding mask function.
"""
function make_shape_mask(
    grid::Grid2D;
    shape::Symbol,
    shape_params::NamedTuple,
)
    if shape == :rectangle
        return rectangle_mask(grid; shape_params...)
    elseif shape == :disk
        return disk_mask(grid; shape_params...)
    elseif shape == :ellipse
        return ellipse_mask(grid; shape_params...)
    elseif shape == :polygon
        return polygon_mask(grid; shape_params...)
    else
        error("Unsupported shape: $shape. Use :rectangle, :disk, :ellipse, or :polygon.")
    end
end

function _shape_extent_um(shape::Symbol, p::NamedTuple)
    if shape == :rectangle
        w = Float64(p.width_um)
        h = Float64(p.height_um)
        theta = deg2rad(Float64(hasproperty(p, :angle_deg) ? getproperty(p, :angle_deg) : 0.0))
        ex = abs(w * cos(theta)) + abs(h * sin(theta))
        ey = abs(w * sin(theta)) + abs(h * cos(theta))
        return ex, ey
    elseif shape == :disk
        d = 2 * Float64(p.radius_um)
        return d, d
    elseif shape == :ellipse
        rx = Float64(p.radius_x_um)
        ry = Float64(p.radius_y_um)
        theta = deg2rad(Float64(hasproperty(p, :angle_deg) ? getproperty(p, :angle_deg) : 0.0))
        ex = 2 * sqrt((rx * cos(theta)) ^ 2 + (ry * sin(theta)) ^ 2)
        ey = 2 * sqrt((rx * sin(theta)) ^ 2 + (ry * cos(theta)) ^ 2)
        return ex, ey
    elseif shape == :polygon
        verts = p.vertices_um
        xs = Float64[v[1] for v in verts]
        ys = Float64[v[2] for v in verts]
        return maximum(xs) - minimum(xs), maximum(ys) - minimum(ys)
    else
        error("Unsupported shape: $shape")
    end
end

# -----------------------------------------------------------------------------
# FFT Biot-Savart operator and conjugate-gradient solver
# -----------------------------------------------------------------------------

function _q_operator(g::AbstractMatrix, q_symbol::AbstractMatrix)
    return real.(ifft(q_symbol .* fft(g)))
end

function _restricted_cg(
    mask::BitMatrix,
    q_symbol::Matrix{Float64},
    rhs::Vector{Float64};
    reltol::Real = 1e-8,
    abstol::Real = 0.0,
    maxiter::Int = 1000,
    verbose::Bool = true,
    print_every::Int = 20,
)
    @assert length(rhs) == count(mask)
    @assert reltol > 0
    @assert abstol >= 0
    @assert maxiter >= 1

    inds = findall(mask)
    work = zeros(Float64, size(mask))

    function apply_A(v::Vector{Float64})
        fill!(work, 0.0)
        work[inds] .= v
        qwork = _q_operator(work, q_symbol)
        return qwork[inds]
    end

    x = zeros(Float64, length(rhs))
    r = rhs - apply_A(x)
    p = copy(r)

    rr = dot(r, r)
    rhs_norm = norm(rhs)
    target = max(Float64(abstol), Float64(reltol) * max(rhs_norm, eps(Float64)))

    history = Float64[sqrt(rr) / max(rhs_norm, eps(Float64))]
    converged = sqrt(rr) <= target
    iterations = 0

    if verbose
        @printf("CG: unknowns = %d, initial relative residual = %.3e\n", length(rhs), history[end])
    end

    for it in 1:maxiter
        converged && break

        Ap = apply_A(p)
        pAp = dot(p, Ap)
        if !(isfinite(pAp) && pAp > 0)
            error(
                "The restricted Biot-Savart operator was not positive definite " *
                "at iteration $it (p'Ap = $pAp). Increase vacuum padding or check the mask."
            )
        end

        alpha = rr / pAp
        x .+= alpha .* p
        r .-= alpha .* Ap

        rr_new = dot(r, r)
        relres = sqrt(rr_new) / max(rhs_norm, eps(Float64))
        push!(history, relres)
        iterations = it

        if verbose && (it == 1 || it % print_every == 0 || sqrt(rr_new) <= target)
            @printf("CG iteration %4d: relative residual = %.3e\n", it, relres)
        end

        if sqrt(rr_new) <= target
            converged = true
            rr = rr_new
            break
        end

        beta = rr_new / rr
        p .= r .+ beta .* p
        rr = rr_new
    end

    return (; x, inds, converged, iterations, residual_history = history)
end

# -----------------------------------------------------------------------------
# Current calculation
# -----------------------------------------------------------------------------

function _finite_difference_current(g::Matrix{Float64}, dx::Real, dy::Real)
    ny, nx = size(g)
    dgdx = zeros(Float64, ny, nx)
    dgdy = zeros(Float64, ny, nx)

    dgdx[:, 2:(nx - 1)] .= (g[:, 3:nx] .- g[:, 1:(nx - 2)]) ./ (2 * dx)
    dgdx[:, 1] .= (g[:, 2] .- g[:, 1]) ./ dx
    dgdx[:, nx] .= (g[:, nx] .- g[:, nx - 1]) ./ dx

    dgdy[2:(ny - 1), :] .= (g[3:ny, :] .- g[1:(ny - 2), :]) ./ (2 * dy)
    dgdy[1, :] .= (g[2, :] .- g[1, :]) ./ dy
    dgdy[ny, :] .= (g[ny, :] .- g[ny - 1, :]) ./ dy

    Kx = dgdy
    Ky = -dgdx
    return Kx, Ky
end

function _spectral_current(
    g::Matrix{Float64},
    grid::Grid2D;
    smoothing_sigma_pixels::Real = 1.0,
)
    sigma = Float64(smoothing_sigma_pixels) * sqrt(grid.dx ^ 2 + grid.dy ^ 2)
    filt = sigma > 0 ? exp.(-0.5 .* (grid.kmag .* sigma) .^ 2) : ones(size(g))

    KX = reshape(grid.kx, 1, grid.nx)
    KY = reshape(grid.ky, grid.ny, 1)
    ghat = fft(g)

    dgdx = real.(ifft((im .* KX) .* filt .* ghat))
    dgdy = real.(ifft((im .* KY) .* filt .* ghat))

    return dgdy, -dgdx
end

"""
    solve_perfect_screening(grid, mask;
        thickness_nm,
        delta_B_uT=1.0,
        derivative=:finite_difference,
        smoothing_sigma_pixels=1.0,
        reltol=1e-8,
        maxiter=1000,
        verbose=true)

Solve the vortex-free, ideal-screening thin-film problem.

Inputs
------
- `grid`: computational grid, including vacuum padding.
- `mask`: `true` inside the superconducting film.
- `thickness_nm`: film thickness.  It converts sheet current K [A/m] to
  volume current density J [A/m^2]; it does not alter K in this ideal model.
- `delta_B_uT`: uniform perpendicular external-field change in microtesla.

Returns a NamedTuple containing `g`, `Kx`, `Ky`, `Kmag`, `Jx`, `Jy`, `Jmag`,
`Bind_T`, and `Btotal_T`, as well as convergence diagnostics.

For ideal screening, the continuum sheet current has an edge singularity.
The finite grid regularises it, so the numerical maximum current is resolution-dependent.
The mask should represent a simply connected film without holes.
"""
function solve_perfect_screening(
    grid::Grid2D,
    mask_in::AbstractMatrix{Bool};
    thickness_nm::Real,
    delta_B_uT::Real = 1.0,
    derivative::Symbol = :finite_difference,
    smoothing_sigma_pixels::Real = 1.0,
    reltol::Real = 1e-8,
    abstol::Real = 0.0,
    maxiter::Int = 1000,
    verbose::Bool = true,
    print_every::Int = 20,
)
    @assert size(mask_in) == (grid.ny, grid.nx) "mask size must be (ny, nx)."
    @assert thickness_nm > 0 "thickness_nm must be positive."
    @assert count(mask_in) > 0 "The sample mask is empty."
    @assert count(mask_in) < length(mask_in) "Include vacuum padding; the mask cannot fill the FFT box."

    mask = BitMatrix(mask_in)

    # The sample must not touch the periodic boundary of the FFT box.
    if any(mask[1, :]) || any(mask[end, :]) || any(mask[:, 1]) || any(mask[:, end])
        error("The film touches the FFT-box boundary. Enlarge the computational box or add padding.")
    end

    delta_B_T = Float64(delta_B_uT) * 1e-6
    thickness_m = Float64(thickness_nm) * 1e-9
    Hext = delta_B_T / MU0

    # F[Q g] = |k|/2 F[g].  The k = 0 symbol is exactly zero.
    q_symbol = grid.kmag ./ 2

    rhs = fill(-Hext, count(mask))
    cg = _restricted_cg(
        mask,
        q_symbol,
        rhs;
        reltol = reltol,
        abstol = abstol,
        maxiter = maxiter,
        verbose = verbose,
        print_every = print_every,
    )

    g = zeros(Float64, grid.ny, grid.nx)
    g[cg.inds] .= cg.x

    Kx, Ky = if derivative == :finite_difference
        _finite_difference_current(g, grid.dx, grid.dy)
    elseif derivative == :spectral
        _spectral_current(
            g,
            grid;
            smoothing_sigma_pixels = smoothing_sigma_pixels,
        )
    else
        error("derivative must be :finite_difference or :spectral.")
    end

    # The physical sheet current is defined only in the film.
    Kx[.!mask] .= 0.0
    Ky[.!mask] .= 0.0
    Kmag = hypot.(Kx, Ky)

    Jx = Kx ./ thickness_m
    Jy = Ky ./ thickness_m
    Jmag = Kmag ./ thickness_m

    Hind = _q_operator(g, q_symbol)
    Bind_T = MU0 .* Hind
    Btotal_T = delta_B_T .+ Bind_T

    residual_inside = Btotal_T[mask]
    rms_screening_error_T = sqrt(mean(abs2.(residual_inside)))
    max_screening_error_T = maximum(abs.(residual_inside))
    relative_rms_screening_error = delta_B_T == 0 ? NaN : rms_screening_error_T / abs(delta_B_T)

    if verbose
        @printf("Converged: %s after %d CG iterations\n", string(cg.converged), cg.iterations)
        @printf("RMS residual field in film: %.3e T (relative %.3e)\n",
                rms_screening_error_T, relative_rms_screening_error)
        @printf("Maximum |K| on this grid: %.6g A/m\n", maximum(Kmag[mask]))
        @printf("Maximum |J| on this grid: %.6g A/m^2\n", maximum(Jmag[mask]))
    end

    return (;
        grid,
        mask,
        g,
        Kx,
        Ky,
        Kmag,
        Jx,
        Jy,
        Jmag,
        Bind_T,
        Btotal_T,
        delta_B_T,
        thickness_m,
        derivative,
        converged = cg.converged,
        iterations = cg.iterations,
        residual_history = cg.residual_history,
        rms_screening_error_T,
        max_screening_error_T,
        relative_rms_screening_error,
    )
end

"""
    simulate_perfect_screening(; shape, shape_params, thickness_nm,
        delta_B_uT=1.0, nx=512, ny=512, padding=2.5, ...)

Convenience wrapper that constructs a centred computational grid from a built-in
shape.  `padding` is the ratio of FFT-box size to sample bounding-box size.
Use at least about 2 to reduce periodic-image interactions, and check convergence
by increasing it.
"""
function simulate_perfect_screening(;
    shape::Symbol = :rectangle,
    shape_params::NamedTuple = (width_um = 200.0, height_um = 200.0),
    thickness_nm::Real,
    delta_B_uT::Real = 1.0,
    nx::Int = 512,
    ny::Int = 512,
    padding::Real = 2.5,
    derivative::Symbol = :finite_difference,
    smoothing_sigma_pixels::Real = 1.0,
    reltol::Real = 1e-8,
    abstol::Real = 0.0,
    maxiter::Int = 1000,
    verbose::Bool = true,
    print_every::Int = 20,
)
    @assert padding > 1.0 "padding must be greater than 1."

    extent_x_um, extent_y_um = _shape_extent_um(shape, shape_params)
    @assert extent_x_um > 0 && extent_y_um > 0

    grid = Grid2D(
        nx,
        ny,
        Float64(padding) * extent_x_um * 1e-6,
        Float64(padding) * extent_y_um * 1e-6,
    )

    mask = make_shape_mask(grid; shape = shape, shape_params = shape_params)

    return solve_perfect_screening(
        grid,
        mask;
        thickness_nm = thickness_nm,
        delta_B_uT = delta_B_uT,
        derivative = derivative,
        smoothing_sigma_pixels = smoothing_sigma_pixels,
        reltol = reltol,
        abstol = abstol,
        maxiter = maxiter,
        verbose = verbose,
        print_every = print_every,
    )
end

"""
    scale_screening_response(result, new_delta_B_uT)

Scale a previously computed ideal-Meissner solution to another external-field
change.  This requires no new FFT solve because the ideal model is linear.
"""
function scale_screening_response(result, new_delta_B_uT::Real)
    old_B = result.delta_B_T
    new_B = Float64(new_delta_B_uT) * 1e-6

    if old_B == 0
        error("Cannot scale a response computed for delta_B = 0.")
    end

    factor = new_B / old_B
    afactor = abs(factor)

    g = factor .* result.g
    Kx = factor .* result.Kx
    Ky = factor .* result.Ky
    Kmag = afactor .* result.Kmag
    Jx = factor .* result.Jx
    Jy = factor .* result.Jy
    Jmag = afactor .* result.Jmag
    Bind_T = factor .* result.Bind_T
    Btotal_T = new_B .+ Bind_T

    residual_inside = Btotal_T[result.mask]
    rms_screening_error_T = sqrt(mean(abs2.(residual_inside)))
    max_screening_error_T = maximum(abs.(residual_inside))
    relative_rms_screening_error = new_B == 0 ? NaN : rms_screening_error_T / abs(new_B)

    return merge(
        result,
        (;
            g,
            Kx,
            Ky,
            Kmag,
            Jx,
            Jy,
            Jmag,
            Bind_T,
            Btotal_T,
            delta_B_T = new_B,
            rms_screening_error_T,
            max_screening_error_T,
            relative_rms_screening_error,
        ),
    )
end

# -----------------------------------------------------------------------------
# Save and reload a unit-field response
# -----------------------------------------------------------------------------

"""
    save_screening_response(path, result)

Save a computed response with Julia's standard `Serialization` module.
Load the module file before deserializing in a new Julia session.
"""
function save_screening_response(path::AbstractString, result)
    open(path, "w") do io
        serialize(io, result)
    end
    return path
end

"""
    load_screening_response(path)

Reload a response saved by `save_screening_response`.
"""
function load_screening_response(path::AbstractString)
    open(path, "r") do io
        return deserialize(io)
    end
end

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

function _masked_values(a::AbstractMatrix, mask::BitMatrix)
    z = fill(NaN, size(a))
    z[mask] .= Float64.(a[mask])
    return z
end

function _safe_quantile(values, q::Real)
    vals = Float64[v for v in values if isfinite(v)]
    isempty(vals) && return 1.0
    v = quantile(vals, clamp(Float64(q), 0.0, 1.0))
    return v > 0 ? v : maximum(vals)
end

"""
    plot_screening_current(result;
        quantity=:sheet,
        outpath=nothing,
        stride=8,
        arrow_mode=:magnitude,
        arrow_length_um=nothing,
        color_quantile=0.99,
        arrow_quantile=0.95)

Plot current magnitude as a heatmap and current direction as arrows.

- `quantity=:sheet` plots K [A/m].
- `quantity=:volume` plots J [A/m^2].
- `arrow_mode=:magnitude` makes arrow length proportional to local current,
  with robust clipping at `arrow_quantile`.
- `arrow_mode=:direction` gives every arrow the same length.
"""
function plot_screening_current(
    result;
    quantity::Symbol = :sheet,
    outpath = nothing,
    title::AbstractString = "",
    stride::Int = 8,
    arrow_mode::Symbol = :magnitude,
    arrow_length_um = nothing,
    max_arrow_factor::Real = 1.5,
    color_quantile::Real = 0.99,
    arrow_quantile::Real = 0.95,
    colorrange = nothing,
    colormap = :viridis,
    figure_size = (950, 820),
)
    @assert stride >= 1

    grid = result.grid
    mask = result.mask

    if quantity == :sheet
        vx = result.Kx
        vy = result.Ky
        mag = result.Kmag
        cbar_label = "|K| (A/m)"
    elseif quantity == :volume
        vx = result.Jx
        vy = result.Jy
        mag = result.Jmag
        cbar_label = "|J| (A/m^2)"
    else
        error("quantity must be :sheet or :volume.")
    end

    x_um = grid.x .* 1e6
    y_um = grid.y .* 1e6
    heat = _masked_values(mag, mask)

    if colorrange === nothing
        vmax = _safe_quantile(mag[mask], color_quantile)
        colorrange = (0.0, vmax)
    end

    if arrow_length_um === nothing
        arrow_length_um = 0.75 * stride * min(grid.dx, grid.dy) * 1e6
    end

    xs = Float64[]
    ys = Float64[]
    ux = Float64[]
    uy = Float64[]
    local_mag = Float64[]

    for iy in 1:stride:grid.ny, ix in 1:stride:grid.nx
        if mask[iy, ix]
            m = mag[iy, ix]
            if isfinite(m) && m > 0
                push!(xs, x_um[ix])
                push!(ys, y_um[iy])
                push!(ux, vx[iy, ix])
                push!(uy, vy[iy, ix])
                push!(local_mag, m)
            end
        end
    end

    ref_mag = _safe_quantile(local_mag, arrow_quantile)
    uplot = Float64[]
    vplot = Float64[]

    for i in eachindex(local_mag)
        m = local_mag[i]
        length_factor = if arrow_mode == :magnitude
            min(m / ref_mag, Float64(max_arrow_factor))
        elseif arrow_mode == :direction
            1.0
        else
            error("arrow_mode must be :magnitude or :direction.")
        end

        push!(uplot, Float64(arrow_length_um) * length_factor * ux[i] / m)
        push!(vplot, Float64(arrow_length_um) * length_factor * uy[i] / m)
    end

    fig = Figure(size = figure_size)
    ax = Axis(
        fig[1, 1];
        xlabel = "x (um)",
        ylabel = "y (um)",
        aspect = DataAspect(),
    )

    hm = heatmap!(
        ax,
        x_um,
        y_um,
        permutedims(heat);
        colormap = colormap,
        colorrange = colorrange,
    )
    Colorbar(fig[1, 2], hm; label = cbar_label)

    if !isempty(xs)
        points = Point2f.(xs, ys)
        vectors = Vec2f.(uplot, vplot)

        # Black underlay plus white body remains visible across the heatmap.
        arrows2d!(
            ax,
            points,
            vectors;
            lengthscale = 1.0,
            align = :tail,
            tipwidth = 17,
            tiplength = 17,
            shaftwidth = 6,
            color = :black,
        )
        arrows2d!(
            ax,
            points,
            vectors;
            lengthscale = 1.0,
            align = :tail,
            tipwidth = 11,
            tiplength = 11,
            shaftwidth = 3,
            color = :white,
        )
    end

    delta_uT = result.delta_B_T * 1e6
    default_title = @sprintf(
        "Ideal Meissner screening, Delta Bext = %.6g uT | relative RMS residual = %.2e",
        delta_uT,
        result.relative_rms_screening_error,
    )
    ax.title = isempty(title) ? default_title : title

    if outpath !== nothing
        save(String(outpath), fig)
    end

    return fig
end

"""
    plot_field_check(result; outpath=nothing)

Diagnostic plot of the total perpendicular field.  It should be approximately
zero in the sample mask.
"""
function plot_field_check(
    result;
    outpath = nothing,
    colorrange = nothing,
    figure_size = (900, 780),
)
    grid = result.grid
    x_um = grid.x .* 1e6
    y_um = grid.y .* 1e6
    B_uT = result.Btotal_T .* 1e6

    if colorrange === nothing
        vmax = _safe_quantile(abs.(B_uT), 0.995)
        colorrange = (-vmax, vmax)
    end

    fig = Figure(size = figure_size)
    ax = Axis(
        fig[1, 1];
        xlabel = "x (um)",
        ylabel = "y (um)",
        aspect = DataAspect(),
        title = "Total perpendicular field: Delta Bext + Bind",
    )

    hm = heatmap!(
        ax,
        x_um,
        y_um,
        permutedims(B_uT);
        colormap = :balance,
        colorrange = colorrange,
    )
    Colorbar(fig[1, 2], hm; label = "Btotal,z (uT)")

    if outpath !== nothing
        save(String(outpath), fig)
    end

    return fig
end

end # module

# -----------------------------------------------------------------------------
# Example main block
# -----------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    using .PerfectMeissnerScreeningFFT

    # Example 1: 200 um x 200 um square film, 250 nm thick, Delta B = +1 uT.
    # The FFT box is 2.5 times larger than the sample bounding box.
    result = simulate_perfect_screening(
        shape = :rectangle,
        shape_params = (
            width_um = 200.0,
            height_um = 200.0,
            angle_deg = 0.0,
        ),
        thickness_nm = 250.0,
        delta_B_uT = 1.0,
        nx = 512,
        ny = 512,
        padding = 2.5,
        derivative = :finite_difference,
        reltol = 1e-8,
        maxiter = 1000,
        verbose = true,
    )

    plot_screening_current(
        result;
        quantity = :sheet,
        stride = 8,
        arrow_mode = :magnitude,
        outpath = "perfect_meissner_sheet_current.png",
    )

    plot_screening_current(
        result;
        quantity = :volume,
        stride = 8,
        arrow_mode = :magnitude,
        outpath = "perfect_meissner_volume_current.png",
    )

    plot_field_check(
        result;
        outpath = "perfect_meissner_field_check.png",
    )

    # No new solve is required for another field change in this ideal model.
    result_5uT = scale_screening_response(result, 5.0)
    plot_screening_current(
        result_5uT;
        quantity = :sheet,
        stride = 8,
        outpath = "perfect_meissner_sheet_current_5uT.png",
    )
end
