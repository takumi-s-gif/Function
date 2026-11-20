# whole_sample_meissner_workflow.jl
#
# Workflow for:
#   1. Solving the ideal vortex-free Meissner screening current over an
#      entire thin-film sample.
#   2. Plotting the whole-sample current magnitude and direction.
#   3. Later extracting a camera field of view by physical sample coordinates.
#
# Put this file in the same directory as:
#   perfect_meissner_screening_fft.jl

include(joinpath(@__DIR__, "perfect_meissner_screening_fft.jl"))
using .PerfectMeissnerScreeningFFT
using CairoMakie
using Statistics
using Printf

"""
    even_grid_size(n)

Return an even integer of at least 64. FFT calculations are usually easier to
handle with even grid sizes.
"""
even_grid_size(n::Integer) = max(64, iseven(n) ? Int(n) : Int(n) + 1)

"""
    make_whole_sample_rectangle(; sample_width_um, sample_height_um,
        thickness_nm, delta_B_uT=1.0, nx=768, padding=2.5, ...)

Solve the ideal Meissner screening problem for an entire rectangular sample.
The sample is centred at `(x, y) = (0, 0)` in micrometres.

`padding` is the ratio of the FFT computational-box size to the sample size.
The box includes vacuum around the sample, but plotting can be cropped to the
sample itself.
"""
function make_whole_sample_rectangle(;
    sample_width_um::Real,
    sample_height_um::Real,
    thickness_nm::Real,
    delta_B_uT::Real = 1.0,
    nx::Int = 768,
    padding::Real = 2.5,
    derivative::Symbol = :finite_difference,
    reltol::Real = 1e-8,
    maxiter::Int = 1500,
    verbose::Bool = true,
)
    @assert sample_width_um > 0
    @assert sample_height_um > 0
    @assert thickness_nm > 0
    @assert padding > 1

    # Preserve approximately square cells: dx ~= dy.
    ny_guess = round(Int, nx * sample_height_um / sample_width_um)
    nx_grid = even_grid_size(nx)
    ny_grid = even_grid_size(ny_guess)

    grid = Grid2D(
        nx_grid,
        ny_grid,
        padding * sample_width_um * 1e-6,
        padding * sample_height_um * 1e-6,
    )

    mask = rectangle_mask(
        grid;
        width_um = sample_width_um,
        height_um = sample_height_um,
        center_um = (0.0, 0.0),
        angle_deg = 0.0,
    )

    result = solve_perfect_screening(
        grid,
        mask;
        thickness_nm = thickness_nm,
        delta_B_uT = delta_B_uT,
        derivative = derivative,
        reltol = reltol,
        maxiter = maxiter,
        verbose = verbose,
    )

    return result
end

"""
    sample_bounds_um(result; margin_fraction=0.03)

Return plotting limits `(xlim, ylim)` around the complete sample mask in the
same physical coordinate system used by the solver.
"""
function sample_bounds_um(result; margin_fraction::Real = 0.03)
    @assert margin_fraction >= 0

    mask = result.mask
    grid = result.grid

    ix = findall(vec(any(mask; dims = 1)))
    iy = findall(vec(any(mask; dims = 2)))
    @assert !isempty(ix) && !isempty(iy) "The sample mask is empty."

    x_um = grid.x .* 1e6
    y_um = grid.y .* 1e6
    half_dx_um = 0.5 * grid.dx * 1e6
    half_dy_um = 0.5 * grid.dy * 1e6

    xmin = x_um[first(ix)] - half_dx_um
    xmax = x_um[last(ix)] + half_dx_um
    ymin = y_um[first(iy)] - half_dy_um
    ymax = y_um[last(iy)] + half_dy_um

    mx = margin_fraction * max(xmax - xmin, eps(Float64))
    my = margin_fraction * max(ymax - ymin, eps(Float64))

    return (xmin - mx, xmax + mx), (ymin - my, ymax + my)
end

function robust_positive_quantile(values, q::Real)
    vals = Float64[v for v in values if isfinite(v) && v >= 0]
    isempty(vals) && return 1.0
    vmax = quantile(vals, clamp(Float64(q), 0.0, 1.0))
    return vmax > 0 ? vmax : max(maximum(vals), eps(Float64))
end

"""
    plot_whole_sample_current(result; quantity=:sheet, ...)

Plot the current over the complete sample only.

- Heatmap: current magnitude, `|K|` or `|J|`.
- Arrows: local current direction; with `arrow_mode=:magnitude`, their length
  also represents the local current magnitude up to robust clipping.
- The FFT vacuum padding is not shown; axis limits are set from the sample mask.

Use `quantity=:sheet` for sheet current density K [A/m].
Use `quantity=:volume` for volume current density J [A/m^2].
"""
function plot_whole_sample_current(
    result;
    quantity::Symbol = :sheet,
    outpath = nothing,
    title::AbstractString = "",
    show_arrows::Bool = true,
    stride::Int = 8,
    arrow_mode::Symbol = :magnitude,
    arrow_length_um = nothing,
    arrow_quantile::Real = 0.95,
    max_arrow_factor::Real = 1.5,
    color_quantile::Real = 0.98,
    colorrange = nothing,
    colormap = :viridis,
    margin_fraction::Real = 0.03,
    figure_size = (980, 820),
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
        error("quantity must be :sheet or :volume")
    end

    x_um = grid.x .* 1e6
    y_um = grid.y .* 1e6

    # Hide the vacuum region while retaining the full FFT solution internally.
    heat = fill(NaN, size(mag))
    heat[mask] .= mag[mask]

    if colorrange === nothing
        vmax = robust_positive_quantile(mag[mask], color_quantile)
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

    if show_arrows
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
    end

    uplot = Float64[]
    vplot = Float64[]

    if !isempty(local_mag)
        ref_mag = robust_positive_quantile(local_mag, arrow_quantile)
        for i in eachindex(local_mag)
            m = local_mag[i]
            length_factor = if arrow_mode == :magnitude
                min(m / ref_mag, Float64(max_arrow_factor))
            elseif arrow_mode == :direction
                1.0
            else
                error("arrow_mode must be :magnitude or :direction")
            end

            push!(uplot, Float64(arrow_length_um) * length_factor * ux[i] / m)
            push!(vplot, Float64(arrow_length_um) * length_factor * uy[i] / m)
        end
    end

    fig = Figure(size = figure_size)
    ax = Axis(
        fig[1, 1];
        xlabel = "sample x (um)",
        ylabel = "sample y (um)",
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

        # Black outline and white body remain legible over the heatmap.
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

    xlim, ylim = sample_bounds_um(result; margin_fraction = margin_fraction)
    limits!(ax, xlim[1], xlim[2], ylim[1], ylim[2])

    delta_uT = result.delta_B_T * 1e6
    default_title = @sprintf(
        "Whole-sample ideal Meissner current, Delta Bext = %.6g uT",
        delta_uT,
    )
    ax.title = isempty(title) ? default_title : title

    if outpath !== nothing
        save(String(outpath), fig)
    end

    return fig
end

"""
    extract_current_fov(result; center_um, width_um, height_um)

Extract a rectangular field of view from an already-computed whole-sample
solution. No new electromagnetic solve is performed.

The coordinates use the same sample coordinate system as the whole-sample plot:
the sample centre is `(0, 0)` unless the shape mask was created elsewhere.
"""
function extract_current_fov(
    result;
    center_um::Tuple{<:Real,<:Real},
    width_um::Real,
    height_um::Real,
)
    @assert width_um > 0 && height_um > 0

    xc = Float64(center_um[1])
    yc = Float64(center_um[2])
    xmin = xc - Float64(width_um) / 2
    xmax = xc + Float64(width_um) / 2
    ymin = yc - Float64(height_um) / 2
    ymax = yc + Float64(height_um) / 2

    x_um_all = result.grid.x .* 1e6
    y_um_all = result.grid.y .* 1e6

    ix = findall(x -> xmin <= x <= xmax, x_um_all)
    iy = findall(y -> ymin <= y <= ymax, y_um_all)

    @assert !isempty(ix) "The requested FOV does not overlap the x grid."
    @assert !isempty(iy) "The requested FOV does not overlap the y grid."

    return (;
        center_um = (xc, yc),
        width_um = Float64(width_um),
        height_um = Float64(height_um),
        x_um = x_um_all[ix],
        y_um = y_um_all[iy],
        mask = result.mask[iy, ix],
        g = result.g[iy, ix],
        Kx = result.Kx[iy, ix],
        Ky = result.Ky[iy, ix],
        Kmag = result.Kmag[iy, ix],
        Jx = result.Jx[iy, ix],
        Jy = result.Jy[iy, ix],
        Jmag = result.Jmag[iy, ix],
        Bind_T = result.Bind_T[iy, ix],
        Btotal_T = result.Btotal_T[iy, ix],
        ix,
        iy,
    )
end

# -----------------------------------------------------------------------------
# Editable example
# -----------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    # Replace these dimensions with the actual whole-sample dimensions.
    sample_width_um = 1000.0
    sample_height_um = 1000.0
    thickness_nm = 250.0
    delta_B_uT = 1.0

    result = make_whole_sample_rectangle(
        sample_width_um = sample_width_um,
        sample_height_um = sample_height_um,
        thickness_nm = thickness_nm,
        delta_B_uT = delta_B_uT,
        nx = 768,
        padding = 2.5,
        derivative = :finite_difference,
        reltol = 1e-8,
        maxiter = 1500,
        verbose = true,
    )

    plot_whole_sample_current(
        result;
        quantity = :sheet,
        stride = 10,
        arrow_mode = :magnitude,
        color_quantile = 0.98,
        outpath = "whole_sample_sheet_current_1uT.png",
    )

    save_screening_response(
        "whole_sample_unit_response_1uT.jls",
        result,
    )

    # Later example, not required for the whole-sample plot:
    # fov = extract_current_fov(
    #     result;
    #     center_um = (150.0, -100.0),
    #     width_um = 200.0,
    #     height_um = 200.0,
    # )
end
