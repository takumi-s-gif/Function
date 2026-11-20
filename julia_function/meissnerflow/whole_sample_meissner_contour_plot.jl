# whole_sample_meissner_contour_plot.jl
#
# Adds a whole-sample current plot with:
#   - current-magnitude heatmap,
#   - current-magnitude contour lines,
#   - current-direction arrows placed on the contour lines,
#   - sample lower-left bounding-box corner used as (x, y) = (0, 0).
#
# Put this file in the same directory as:
#   whole_sample_meissner_workflow.jl
#   perfect_meissner_screening_fft.jl

if !isdefined(@__MODULE__, :sample_bounds_um)
    include(joinpath(@__DIR__, "whole_sample_meissner_workflow.jl"))
end
using CairoMakie
using Statistics
using Printf

"""
    sample_bbox_edges_um(result)

Return the sample bounding-box edges `(xmin, xmax, ymin, ymax)` in the
solver's original centred coordinate system, in micrometres.
"""
function sample_bbox_edges_um(result)
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

    return xmin, xmax, ymin, ymax
end

"""
    sample_lower_left_coordinates_um(result)

Return grid-centre coordinates shifted so that the lower-left corner of the
sample bounding box is `(0, 0)`.  For a rectangular sample this is the physical
lower-left sample corner.  For a disk, ellipse, or polygon it is the lower-left
corner of the sample's axis-aligned bounding box.
"""
function sample_lower_left_coordinates_um(result)
    xmin, xmax, ymin, ymax = sample_bbox_edges_um(result)

    x_um = result.grid.x .* 1e6 .- xmin
    y_um = result.grid.y .* 1e6 .- ymin

    return (;
        x_um,
        y_um,
        width_um = xmax - xmin,
        height_um = ymax - ymin,
        original_origin_um = (xmin, ymin),
    )
end

function _unique_positive_levels(raw_levels)
    raw = sort(Float64[x for x in raw_levels if isfinite(x) && x > 0])
    isempty(raw) && return Float64[]

    scale = max(maximum(abs.(raw)), 1.0)
    tol = 100 * eps(Float64) * scale
    levels = Float64[]

    for level in raw
        if isempty(levels) || abs(level - levels[end]) > tol
            push!(levels, level)
        end
    end

    return levels
end

function _make_current_contour_levels(
    values,
    contour_levels;
    mode::Symbol,
    min_quantile::Real,
    max_quantile::Real,
)
    vals = Float64[v for v in values if isfinite(v) && v > 0]
    @assert !isempty(vals) "No positive finite current values were found."

    if contour_levels isa Integer
        nlevels = Int(contour_levels)
        @assert nlevels >= 1 "contour_levels must be at least 1."
        qmin = clamp(Float64(min_quantile), 0.0, 1.0)
        qmax = clamp(Float64(max_quantile), 0.0, 1.0)
        @assert qmin < qmax "Require contour_min_quantile < contour_max_quantile."

        if mode == :quantile
            qs = range(qmin, qmax; length = nlevels)
            return _unique_positive_levels(quantile(vals, q) for q in qs)
        elseif mode == :linear
            lo = quantile(vals, qmin)
            hi = quantile(vals, qmax)
            return _unique_positive_levels(range(lo, hi; length = nlevels))
        else
            error("contour_level_mode must be :quantile or :linear")
        end
    elseif contour_levels isa AbstractVector
        return _unique_positive_levels(contour_levels)
    else
        error("contour_levels must be an integer or a vector of current values")
    end
end

function _level_crossing_candidates(mag, mask, level::Real)
    ny, nx = size(mag)
    candidates = Tuple{Int,Int}[]

    for iy in 2:(ny - 1), ix in 2:(nx - 1)
        mask[iy, ix] || continue
        z0 = mag[iy, ix]
        isfinite(z0) || continue

        crossed = false
        for (jy, jx) in ((iy, ix + 1), (iy, ix - 1), (iy + 1, ix), (iy - 1, ix))
            mask[jy, jx] || continue
            z1 = mag[jy, jx]
            isfinite(z1) || continue

            a = z0 - level
            b = z1 - level
            if a == 0 || b == 0 || signbit(a) != signbit(b)
                crossed = true
                break
            end
        end

        crossed && push!(candidates, (iy, ix))
    end

    return candidates
end

function _distance_to_points(x::Real, y::Real, points)
    isempty(points) && return Inf
    dmin = Inf
    for (xp, yp) in points
        dmin = min(dmin, hypot(Float64(x) - xp, Float64(y) - yp))
    end
    return dmin
end

function _select_spaced_contour_points(
    candidates,
    x_um,
    y_um;
    max_points::Int,
    min_spacing_um::Real,
    existing_points = Tuple{Float64,Float64}[],
)
    isempty(candidates) && return Tuple{Int,Int}[]
    max_points <= 0 && return Tuple{Int,Int}[]

    available = collect(eachindex(candidates))
    selected = Tuple{Int,Int}[]
    selected_xy = Tuple{Float64,Float64}[]

    while !isempty(available) && length(selected) < max_points
        best_available_position = 0
        best_score = -Inf

        for (available_position, candidate_index) in pairs(available)
            iy, ix = candidates[candidate_index]
            x = x_um[ix]
            y = y_um[iy]

            references = isempty(selected_xy) ? existing_points : vcat(existing_points, selected_xy)
            score = _distance_to_points(x, y, references)

            # Deterministic seed for the first point when no previous arrows exist.
            if isinf(score)
                score = x + 1e-6 * y
            end

            if score > best_score
                best_score = score
                best_available_position = available_position
            end
        end

        best_available_position == 0 && break

        candidate_index = available[best_available_position]
        iy, ix = candidates[candidate_index]
        x = Float64(x_um[ix])
        y = Float64(y_um[iy])

        references = isempty(selected_xy) ? existing_points : vcat(existing_points, selected_xy)
        distance = _distance_to_points(x, y, references)

        if !isempty(references) && distance < min_spacing_um
            break
        end

        push!(selected, (iy, ix))
        push!(selected_xy, (x, y))
        deleteat!(available, best_available_position)
    end

    return selected
end

"""
    plot_whole_sample_current_contours(result; ...)

Plot the whole-sample screening current using:

- a heatmap of `|K|` or `|J|`,
- contour lines of the same magnitude,
- fixed-length arrows placed near the contour lines and oriented along the
  actual current vector field,
- coordinates measured from the lower-left corner of the sample bounding box.

The arrows are not selected by a regular `stride`.  Instead, candidate grid
points are detected where each contour level crosses the current-magnitude
field, and spatially separated points are selected from those candidates.

Important: contours of current magnitude are not necessarily streamlines.
The arrow direction is therefore taken from the actual local vector
`(Kx, Ky)` or `(Jx, Jy)`, not from the tangent of the contour line.
"""
function plot_whole_sample_current_contours(
    result;
    quantity::Symbol = :sheet,
    outpath = nothing,
    title::AbstractString = "",
    show_heatmap::Bool = true,
    show_contours::Bool = true,
    show_arrows::Bool = true,
    show_boundary::Bool = true,
    contour_levels = 8,
    contour_level_mode::Symbol = :quantile,
    contour_min_quantile::Real = 0.15,
    contour_max_quantile::Real = 0.98,
    contour_labels::Bool = false,
    contour_linewidth::Real = 1.5,
    contour_color = :black,
    contour_halo::Bool = true,
    arrows_per_contour::Int = 10,
    avoid_cross_level_overlap::Bool = false,
    min_arrow_spacing_um = nothing,
    arrow_length_um = nothing,
    arrow_tipwidth::Real = 12,
    arrow_tiplength::Real = 12,
    arrow_shaftwidth::Real = 3,
    color_quantile::Real = 0.98,
    colorrange = nothing,
    colormap = :viridis,
    margin_fraction::Real = 0.0,
    figure_size = (980, 820),
)
    @assert arrows_per_contour >= 0
    @assert margin_fraction >= 0

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

    coordinates = sample_lower_left_coordinates_um(result)
    x_um = coordinates.x_um
    y_um = coordinates.y_um
    sample_width_um = coordinates.width_um
    sample_height_um = coordinates.height_um

    heat = fill(NaN, size(mag))
    heat[mask] .= mag[mask]

    if colorrange === nothing
        vmax = robust_positive_quantile(mag[mask], color_quantile)
        colorrange = (0.0, vmax)
    end

    levels = _make_current_contour_levels(
        mag[mask],
        contour_levels;
        mode = contour_level_mode,
        min_quantile = contour_min_quantile,
        max_quantile = contour_max_quantile,
    )
    @assert !isempty(levels) "No usable contour levels were generated."

    if min_arrow_spacing_um === nothing
        min_arrow_spacing_um = 0.07 * min(sample_width_um, sample_height_um)
    end
    if arrow_length_um === nothing
        arrow_length_um = 0.55 * Float64(min_arrow_spacing_um)
    end

    fig = Figure(size = figure_size)
    ax = Axis(
        fig[1, 1];
        xlabel = "x(um)",
        ylabel = "y(um)",
        aspect = DataAspect(),
    )

    if show_heatmap
        hm = heatmap!(
            ax,
            x_um,
            y_um,
            permutedims(heat);
            colormap = colormap,
            colorrange = colorrange,
        )
        Colorbar(fig[1, 2], hm; label = cbar_label)
    end

    if show_contours
        if contour_halo
            contour!(
                ax,
                x_um,
                y_um,
                permutedims(heat);
                levels = levels,
                color = :white,
                linewidth = Float64(contour_linewidth) + 2.0,
                labels = false,
            )
        end

        contour!(
            ax,
            x_um,
            y_um,
            permutedims(heat);
            levels = levels,
            color = contour_color,
            linewidth = contour_linewidth,
            labels = contour_labels,
        )
    end

    if show_boundary
        contour!(
            ax,
            x_um,
            y_um,
            permutedims(Float64.(mask));
            levels = [0.5],
            color = :black,
            linewidth = 2.0,
        )
    end

    arrow_points = Tuple{Float64,Float64}[]
    arrow_vectors = Tuple{Float64,Float64}[]

    if show_arrows
        occupied_points = Tuple{Float64,Float64}[]

        for level in levels
            candidates = _level_crossing_candidates(mag, mask, level)
            selected = _select_spaced_contour_points(
                candidates,
                x_um,
                y_um;
                max_points = arrows_per_contour,
                min_spacing_um = Float64(min_arrow_spacing_um),
                existing_points = avoid_cross_level_overlap ? occupied_points : Tuple{Float64,Float64}[],
            )

            for (iy, ix) in selected
                ux = vx[iy, ix]
                uy = vy[iy, ix]
                current_magnitude = hypot(ux, uy)

                if isfinite(current_magnitude) && current_magnitude > 0
                    x = Float64(x_um[ix])
                    y = Float64(y_um[iy])
                    u = Float64(arrow_length_um) * ux / current_magnitude
                    v = Float64(arrow_length_um) * uy / current_magnitude

                    push!(arrow_points, (x, y))
                    push!(arrow_vectors, (u, v))
                    push!(occupied_points, (x, y))
                end
            end
        end
    end

    if !isempty(arrow_points)
        points = Point2f.(first.(arrow_points), last.(arrow_points))
        vectors = Vec2f.(first.(arrow_vectors), last.(arrow_vectors))

        arrows2d!(
            ax,
            points,
            vectors;
            lengthscale = 1.0,
            align = :center,
            tipwidth = Float64(arrow_tipwidth) + 6.0,
            tiplength = Float64(arrow_tiplength) + 6.0,
            shaftwidth = Float64(arrow_shaftwidth) + 3.0,
            color = :black,
        )
        arrows2d!(
            ax,
            points,
            vectors;
            lengthscale = 1.0,
            align = :center,
            tipwidth = arrow_tipwidth,
            tiplength = arrow_tiplength,
            shaftwidth = arrow_shaftwidth,
            color = :white,
        )
    end

    mx = margin_fraction * sample_width_um
    my = margin_fraction * sample_height_um
    limits!(
        ax,
        -mx,
        sample_width_um + mx,
        -my,
        sample_height_um + my,
    )

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
    extract_current_fov_lower_left(result; lower_left_um, width_um, height_um)

Extract a rectangular FOV using the same lower-left-origin coordinates as
`plot_whole_sample_current_contours`.  No electromagnetic solve is repeated.
"""
function extract_current_fov_lower_left(
    result;
    lower_left_um::Tuple{<:Real,<:Real},
    width_um::Real,
    height_um::Real,
)
    @assert width_um > 0 && height_um > 0

    coordinates = sample_lower_left_coordinates_um(result)
    x_all = coordinates.x_um
    y_all = coordinates.y_um

    x0 = Float64(lower_left_um[1])
    y0 = Float64(lower_left_um[2])
    x1 = x0 + Float64(width_um)
    y1 = y0 + Float64(height_um)

    ix = findall(x -> x0 <= x <= x1, x_all)
    iy = findall(y -> y0 <= y <= y1, y_all)

    @assert !isempty(ix) "The requested FOV does not overlap the x grid."
    @assert !isempty(iy) "The requested FOV does not overlap the y grid."

    return (;
        lower_left_um = (x0, y0),
        width_um = Float64(width_um),
        height_um = Float64(height_um),
        x_um = x_all[ix],
        y_um = y_all[iy],
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

# Example:
#
# include("whole_sample_meissner_contour_plot.jl")
#
# fig = plot_whole_sample_current_contours(
#     result;
#     quantity = :sheet,
#     contour_levels = 8,
#     contour_level_mode = :quantile,
#     contour_min_quantile = 0.15,
#     contour_max_quantile = 0.98,
#     arrows_per_contour = 10,
#     min_arrow_spacing_um = 50.0,
#     arrow_length_um = 30.0,
#     color_quantile = 0.98,
#     outpath = "whole_sample_current_contours.png",
# )
#
# display(fig)
