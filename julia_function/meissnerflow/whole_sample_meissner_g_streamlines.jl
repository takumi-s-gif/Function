# whole_sample_meissner_g_streamlines.jl
#
# Plot closed shielding-current streamlines from contours of the stream
# function g.  For the convention used by perfect_meissner_screening_fft.jl,
#
#     Kx =  dg/dy,
#     Ky = -dg/dx,
#
# so K is tangent to every contour g = constant.
#
# Features
# --------
#   * optional |K| or |J| heatmap,
#   * closed contours of g,
#   * optional current-direction arrows placed on those contours,
#   * sample lower-left bounding-box corner displayed as (0, 0),
#   * no regular-grid stride is used for arrow placement.
#
# Put this file in the same directory as:
#   perfect_meissner_screening_fft.jl
#   whole_sample_meissner_workflow.jl

if !isdefined(@__MODULE__, :make_whole_sample_rectangle)
    include(joinpath(@__DIR__, "whole_sample_meissner_workflow.jl"))
end

using CairoMakie
using Statistics
using Printf

# -----------------------------------------------------------------------------
# Private helpers.  Names are prefixed to avoid collisions with other scripts.
# -----------------------------------------------------------------------------

function _gstream_sample_geometry_um(result)
    mask = result.mask
    grid = result.grid

    ix = findall(vec(any(mask; dims = 1)))
    iy = findall(vec(any(mask; dims = 2)))
    @assert !isempty(ix) && !isempty(iy) "The sample mask is empty."

    x_center_um = grid.x .* 1e6
    y_center_um = grid.y .* 1e6
    half_dx_um = 0.5 * grid.dx * 1e6
    half_dy_um = 0.5 * grid.dy * 1e6

    xmin_um = x_center_um[first(ix)] - half_dx_um
    xmax_um = x_center_um[last(ix)] + half_dx_um
    ymin_um = y_center_um[first(iy)] - half_dy_um
    ymax_um = y_center_um[last(iy)] + half_dy_um

    # Shift only the displayed coordinates.  The numerical solution is not
    # modified.
    x_um = x_center_um .- xmin_um
    y_um = y_center_um .- ymin_um

    return (;
        x_um,
        y_um,
        width_um = xmax_um - xmin_um,
        height_um = ymax_um - ymin_um,
        original_lower_left_um = (xmin_um, ymin_um),
    )
end

function _gstream_positive_quantile(values, q::Real)
    vals = Float64[v for v in values if isfinite(v) && v >= 0]
    isempty(vals) && return 1.0

    value = quantile(vals, clamp(Float64(q), 0.0, 1.0))
    return value > 0 ? value : max(maximum(vals), eps(Float64))
end

function _gstream_unique_levels(raw_levels)
    raw = sort(Float64[v for v in raw_levels if isfinite(v) && v != 0])
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

function _gstream_make_levels(
    g_inside,
    contour_levels;
    level_mode::Symbol,
    min_fraction::Real,
    max_fraction::Real,
    min_quantile::Real,
    max_quantile::Real,
)
    vals = Float64[v for v in g_inside if isfinite(v)]
    isempty(vals) && error("No finite stream-function values were found in the sample.")

    max_abs, ipeak = findmax(abs.(vals))
    max_abs > 0 || error("The stream function is zero everywhere; no contours can be drawn.")
    dominant_sign = sign(vals[ipeak])
    dominant_sign == 0 && error("Could not determine the sign of the stream function.")

    # In the vortex-free, simply connected problem under a uniform field, g is
    # normally one-signed.  Restrict automatic level selection to that branch
    # so that small numerical values of the opposite sign do not create noise.
    same_sign_vals = dominant_sign > 0 ?
        Float64[v for v in vals if v > 0] :
        Float64[v for v in vals if v < 0]

    if contour_levels isa Integer
        nlevels = Int(contour_levels)
        @assert nlevels >= 1 "contour_levels must be at least 1."

        if level_mode == :fraction
            fmin = Float64(min_fraction)
            fmax = Float64(max_fraction)
            @assert 0 < fmin < fmax < 1 "Require 0 < min_fraction < max_fraction < 1."

            fractions = range(fmin, fmax; length = nlevels)
            raw = dominant_sign .* max_abs .* fractions
            return _gstream_unique_levels(raw)

        elseif level_mode == :quantile
            qmin = clamp(Float64(min_quantile), 0.0, 1.0)
            qmax = clamp(Float64(max_quantile), 0.0, 1.0)
            @assert qmin < qmax "Require min_quantile < max_quantile."
            isempty(same_sign_vals) && error("No one-signed stream-function values were found.")

            qs = range(qmin, qmax; length = nlevels)
            raw = [quantile(same_sign_vals, q) for q in qs]
            return _gstream_unique_levels(raw)

        else
            error("level_mode must be :fraction or :quantile.")
        end

    elseif contour_levels isa AbstractVector
        levels = _gstream_unique_levels(contour_levels)
        isempty(levels) && error("The supplied contour_levels contain no finite nonzero values.")
        return levels

    else
        error("contour_levels must be an integer or a vector of g values in amperes.")
    end
end

@inline function _gstream_crosses(z1::Real, z2::Real, level::Real)
    a = Float64(z1) - Float64(level)
    b = Float64(z2) - Float64(level)
    return a == 0.0 || b == 0.0 || signbit(a) != signbit(b)
end

"""
Return interpolated candidate points where `g = level` crosses horizontal and
vertical grid edges.  K is interpolated at each candidate, so its direction is
tangent to the stream-function contour.
"""
function _gstream_crossing_candidates(
    g,
    Kx,
    Ky,
    mask,
    x_um,
    y_um,
    level::Real,
)
    ny, nx = size(g)
    candidates = NamedTuple[]

    # Edges parallel to x: (iy, ix) -> (iy, ix + 1)
    for iy in 1:ny, ix in 1:(nx - 1)
        mask[iy, ix] && mask[iy, ix + 1] || continue

        z1 = g[iy, ix]
        z2 = g[iy, ix + 1]
        isfinite(z1) && isfinite(z2) || continue
        _gstream_crosses(z1, z2, level) || continue

        dz = z2 - z1
        abs(dz) > eps(Float64) || continue
        t = clamp((Float64(level) - z1) / dz, 0.0, 1.0)

        x = (1 - t) * x_um[ix] + t * x_um[ix + 1]
        y = y_um[iy]
        kx = (1 - t) * Kx[iy, ix] + t * Kx[iy, ix + 1]
        ky = (1 - t) * Ky[iy, ix] + t * Ky[iy, ix + 1]

        push!(candidates, (; x = Float64(x), y = Float64(y), kx = Float64(kx), ky = Float64(ky)))
    end

    # Edges parallel to y: (iy, ix) -> (iy + 1, ix)
    for iy in 1:(ny - 1), ix in 1:nx
        mask[iy, ix] && mask[iy + 1, ix] || continue

        z1 = g[iy, ix]
        z2 = g[iy + 1, ix]
        isfinite(z1) && isfinite(z2) || continue
        _gstream_crosses(z1, z2, level) || continue

        dz = z2 - z1
        abs(dz) > eps(Float64) || continue
        t = clamp((Float64(level) - z1) / dz, 0.0, 1.0)

        x = x_um[ix]
        y = (1 - t) * y_um[iy] + t * y_um[iy + 1]
        kx = (1 - t) * Kx[iy, ix] + t * Kx[iy + 1, ix]
        ky = (1 - t) * Ky[iy, ix] + t * Ky[iy + 1, ix]

        push!(candidates, (; x = Float64(x), y = Float64(y), kx = Float64(kx), ky = Float64(ky)))
    end

    return candidates
end

@inline function _gstream_point_distance(a, b)
    return hypot(a.x - b.x, a.y - b.y)
end

"""
Greedy farthest-point selection.  This spreads a small number of arrows around
each closed contour without using a regular grid stride.
"""
function _gstream_select_spaced(
    candidates;
    max_points::Int,
    min_spacing_um::Real,
    seed_phase::Real = 0.0,
)
    isempty(candidates) && return NamedTuple[]
    max_points <= 0 && return NamedTuple[]

    # Remove candidates with unusable current vectors.
    valid = [c for c in candidates if isfinite(c.kx) && isfinite(c.ky) && hypot(c.kx, c.ky) > 0]
    isempty(valid) && return NamedTuple[]

    # Deterministic first point.  Varying seed_phase prevents arrows on nested
    # contours from all lining up along one radial direction.
    cx = mean(c.x for c in valid)
    cy = mean(c.y for c in valid)
    target_angle = mod(Float64(seed_phase), 2pi)
    first_index = 1
    first_score = Inf

    for (i, c) in pairs(valid)
        angle = atan(c.y - cy, c.x - cx)
        wrapped_difference = atan(
            sin(angle - target_angle),
            cos(angle - target_angle),
        )
        score = abs(wrapped_difference)

        if score < first_score
            first_score = score
            first_index = i
        end
    end

    selected = NamedTuple[valid[first_index]]
    remaining = collect(eachindex(valid))
    first_position = findfirst(==(first_index), remaining)
    first_position === nothing || deleteat!(remaining, first_position)

    while !isempty(remaining) && length(selected) < max_points
        best_position = 0
        best_distance = -Inf

        for (position, candidate_index) in pairs(remaining)
            candidate = valid[candidate_index]
            distance = minimum(_gstream_point_distance(candidate, s) for s in selected)
            if distance > best_distance
                best_distance = distance
                best_position = position
            end
        end

        best_position == 0 && break
        best_distance < Float64(min_spacing_um) && break

        candidate_index = remaining[best_position]
        push!(selected, valid[candidate_index])
        deleteat!(remaining, best_position)
    end

    return selected
end

# -----------------------------------------------------------------------------
# Public plotting function
# -----------------------------------------------------------------------------

"""
    plot_g_streamlines(result; kwargs...)

Plot closed shielding-current paths as contours of the stream function `g`.
With the solver convention

    Kx = dg/dy,   Ky = -dg/dx,

the current vector is tangent to every `g = constant` contour.  Thus these
contours are current streamlines, unlike contours of `|K|`.

Main keyword arguments
----------------------
- `quantity=:sheet` or `:volume`: quantity used only for the background
  magnitude heatmap.  The streamlines themselves always come from `g`.
- `contour_levels=10`: number of streamlines, or an explicit vector of `g`
  values in amperes.
- `level_mode=:fraction`: place levels between fractions of the peak `|g|`.
- `min_fraction=0.08`, `max_fraction=0.92`: avoid the zero-valued boundary and
  the single central extremum.
- `arrows_per_level=4`: maximum number of direction arrows on each streamline.
- `show_current_heatmap=true`: draw `|K|` or `|J|` behind the streamlines.
- Display coordinates use the lower-left corner of the sample bounding box as
  `(0, 0)` in micrometres.
"""
function plot_g_streamlines(
    result;
    quantity::Symbol = :sheet,
    outpath = nothing,
    title::AbstractString = "",
    show_current_heatmap::Bool = true,
    show_streamlines::Bool = true,
    show_direction_arrows::Bool = true,
    show_boundary::Bool = true,
    contour_levels = 10,
    level_mode::Symbol = :fraction,
    min_fraction::Real = 0.08,
    max_fraction::Real = 0.92,
    min_quantile::Real = 0.08,
    max_quantile::Real = 0.92,
    contour_color = :white,
    contour_linewidth::Real = 1.0,
    contour_halo::Bool = true,
    contour_labels::Bool = false,
    arrows_per_level::Int = 4,
    min_arrow_spacing_um = nothing,
    arrow_length_um = nothing,
    arrow_color = :white,
    arrow_tipwidth::Real = 11,
    arrow_tiplength::Real = 11,
    arrow_shaftwidth::Real = 3,
    color_quantile::Real = 0.98,
    colorrange = nothing,
    colormap = :viridis,
    margin_fraction::Real = 0.02,
    figure_size = (980, 820),
)
    @assert arrows_per_level >= 0
    @assert margin_fraction >= 0

    grid = result.grid
    mask = result.mask
    g = Float64.(result.g)

    if quantity == :sheet
        Kx = Float64.(result.Kx)
        Ky = Float64.(result.Ky)
        magnitude = Float64.(result.Kmag)
        cbar_label = "|K| (A/m)"
    elseif quantity == :volume
        # J has the same direction as K, so either vector field is suitable for
        # arrow orientation.  Use J consistently with the selected heatmap.
        Kx = Float64.(result.Jx)
        Ky = Float64.(result.Jy)
        magnitude = Float64.(result.Jmag)
        cbar_label = "|J| (A/m^2)"
    else
        error("quantity must be :sheet or :volume.")
    end

    coords = _gstream_sample_geometry_um(result)
    x_um = coords.x_um
    y_um = coords.y_um
    sample_width_um = coords.width_um
    sample_height_um = coords.height_um

    g_masked = fill(NaN, size(g))
    g_masked[mask] .= g[mask]

    magnitude_masked = fill(NaN, size(magnitude))
    magnitude_masked[mask] .= magnitude[mask]

    levels = _gstream_make_levels(
        g[mask],
        contour_levels;
        level_mode = level_mode,
        min_fraction = min_fraction,
        max_fraction = max_fraction,
        min_quantile = min_quantile,
        max_quantile = max_quantile,
    )
    isempty(levels) && error("No usable stream-function contour levels were generated.")

    if colorrange === nothing
        vmax = _gstream_positive_quantile(magnitude[mask], color_quantile)
        colorrange = (0.0, vmax)
    end

    if min_arrow_spacing_um === nothing
        min_arrow_spacing_um = 0.12 * min(sample_width_um, sample_height_um)
    end
    if arrow_length_um === nothing
        arrow_length_um = 0.045 * min(sample_width_um, sample_height_um)
    end

    if isempty(title)
        delta_B_uT = result.delta_B_T * 1e6
        title = @sprintf("Meissner-current streamlines, Delta B_ext = %.4g uT", delta_B_uT)
    end

    fig = Figure(size = figure_size)
    ax = Axis(
        fig[1, 1];
        xlabel = "x(um)",
        ylabel = "y(um)",
        aspect = DataAspect(),
        title = title,
    )

    if show_current_heatmap
        hm = heatmap!(
            ax,
            x_um,
            y_um,
            permutedims(magnitude_masked);
            colormap = colormap,
            colorrange = colorrange,
        )
        Colorbar(fig[1, 2], hm; label = cbar_label)
    end

    if show_streamlines
        if contour_halo
            contour!(
                ax,
                x_um,
                y_um,
                permutedims(g_masked);
                levels = levels,
                color = :black,
                linewidth = Float64(contour_linewidth) + 2.2,
                labels = false,
            )
        end

        contour!(
            ax,
            x_um,
            y_um,
            permutedims(g_masked);
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
            linewidth = 2.2,
        )
    end

    arrow_points = Point2f[]
    arrow_vectors = Vec2f[]

    if show_direction_arrows && arrows_per_level > 0
        for (ilevel, level) in enumerate(levels)
            candidates = _gstream_crossing_candidates(
                g,
                Kx,
                Ky,
                mask,
                x_um,
                y_um,
                level,
            )

            selected = _gstream_select_spaced(
                candidates;
                max_points = arrows_per_level,
                min_spacing_um = Float64(min_arrow_spacing_um),
                seed_phase = 2pi * (ilevel - 1) / max(length(levels), 1),
            )

            for p in selected
                current_magnitude = hypot(p.kx, p.ky)
                current_magnitude > 0 || continue

                u = Float64(arrow_length_um) * p.kx / current_magnitude
                v = Float64(arrow_length_um) * p.ky / current_magnitude

                push!(arrow_points, Point2f(p.x, p.y))
                push!(arrow_vectors, Vec2f(u, v))
            end
        end
    end

    if !isempty(arrow_points)
        # Dark outline.
        # arrows2d!(
        #     ax,
        #     arrow_points,
        #     arrow_vectors;
        #     lengthscale = 1.0,
        #     align = :center,
        #     tipwidth = Float64(arrow_tipwidth) + 5.0,
        #     tiplength = Float64(arrow_tiplength) + 5.0,
        #     shaftwidth = Float64(arrow_shaftwidth) + 3.0,
        #     color = :black,
        # )

        # Bright arrow body.
        arrows2d!(
            ax,
            arrow_points,
            arrow_vectors;
            lengthscale = 2.0,
            align = :center,
            tipwidth = arrow_tipwidth,
            tiplength = arrow_tiplength,
            shaftwidth = arrow_shaftwidth,
            color = arrow_color,
        )
    end

    margin_x = Float64(margin_fraction) * sample_width_um
    margin_y = Float64(margin_fraction) * sample_height_um
    limits!(
        ax,
        -margin_x,
        sample_width_um + margin_x,
        -margin_y,
        sample_height_um + margin_y,
    )

    if outpath !== nothing
        directory = dirname(String(outpath))
        directory == "." || mkpath(directory)
        save(outpath, fig)
    end

    return fig
end

# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------
#
# include("whole_sample_meissner_g_streamlines.jl")
#
# fig = plot_g_streamlines(
#     result;
#     quantity = :sheet,
#     contour_levels = 10,
#     level_mode = :fraction,
#     min_fraction = 0.08,
#     max_fraction = 0.92,
#     arrows_per_level = 4,
#     show_current_heatmap = true,
#     outpath = "whole_sample_g_streamlines.png",
# )
#
# display(fig)
