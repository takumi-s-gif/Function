# local_vortex_fraction_40um.jl
#
# This file is intended to be included AFTER track_peakmove_arrows2d(2).jl.
# It uses the existing peak_xy(...) function and analysis.per_frame[t].moving.

using CairoMakie

"""
    spatial_edges(range, binsize)

Create bin edges covering `range = (min, max)`.
The last bin is allowed to be shorter when the range is not exactly divisible by binsize.
"""
function spatial_edges(range::Tuple{<:Real,<:Real}, binsize::Real)
    lo = Float64(range[1])
    hi = Float64(range[2])
    bs = Float64(binsize)

    @assert isfinite(lo) && isfinite(hi) "range must be finite."
    @assert hi > lo "Require range[2] > range[1]."
    @assert bs > 0 "binsize must be positive."

    edges = collect(lo:bs:hi)

    if isempty(edges)
        edges = [lo]
    end

    # Add the exact upper boundary when the regular grid does not reach it.
    if edges[end] < hi
        push!(edges, hi)
    else
        edges[end] = hi
    end

    return edges
end

"""
    spatial_bin_index(value, edges)

Return the 1-based bin index containing `value`.
Bins are [edge[k], edge[k+1]), except that the final bin includes the upper boundary.
Return `nothing` when value is outside the range or is not finite.
"""
function spatial_bin_index(value::Real, edges::AbstractVector{<:Real})
    v = Float64(value)
    lo = Float64(first(edges))
    hi = Float64(last(edges))
    nbins = length(edges) - 1

    if !isfinite(v) || v < lo || v > hi
        return nothing
    elseif v == hi
        return nbins
    else
        i = searchsortedlast(edges, v)
        return 1 <= i <= nbins ? i : nothing
    end
end

"""
    local_vortex_statistics(p, result; xrange=(0, 200), yrange=(0, 200), binsize=40)

For one frame transition t -> t+1, calculate in each spatial bin:

- `total_count`: all vortices present in frame t
- `moving_count`: vortices judged to move, assigned by their starting position (d.x, d.y)
- `moving_fraction`: moving_count / total_count

A bin with no vortex has `NaN` moving_fraction.
"""
function local_vortex_statistics(
    p,
    result;
    xrange::Tuple{<:Real,<:Real}=(0.0, 200.0),
    yrange::Tuple{<:Real,<:Real}=(0.0, 200.0),
    binsize::Real=40.0,
)
    xedges = spatial_edges(xrange, binsize)
    yedges = spatial_edges(yrange, binsize)
    xcenters = (xedges[1:end-1] .+ xedges[2:end]) ./ 2
    ycenters = (yedges[1:end-1] .+ yedges[2:end]) ./ 2

    nx = length(xedges) - 1
    ny = length(yedges) - 1

    total_count = zeros(Int, ny, nx)
    moving_count = zeros(Int, ny, nx)

    # Denominator: all vortices in the initial frame t.
    xy = peak_xy(p)
    for k in axes(xy, 1)
        ix = spatial_bin_index(xy[k, 1], xedges)
        iy = spatial_bin_index(xy[k, 2], yedges)

        if ix !== nothing && iy !== nothing
            total_count[iy, ix] += 1
        end
    end

    # Numerator: moving vortices, classified by their starting coordinates.
    for d in result.moving
        ix = spatial_bin_index(d.x, xedges)
        iy = spatial_bin_index(d.y, yedges)

        if ix !== nothing && iy !== nothing
            moving_count[iy, ix] += 1
        end
    end

    moving_fraction = fill(NaN, ny, nx)
    valid = total_count .> 0
    moving_fraction[valid] .= moving_count[valid] ./ total_count[valid]

    return (;
        t=result.t,
        xrange=(Float64(xrange[1]), Float64(xrange[2])),
        yrange=(Float64(yrange[1]), Float64(yrange[2])),
        binsize=Float64(binsize),
        xedges,
        yedges,
        xcenters,
        ycenters,
        total_count,
        moving_count,
        moving_fraction,
    )
end

"""
    local_vortex_statistics_series(peaks, analysis; ...)

Calculate local statistics for every frame transition.
Also return pooled counts over all transitions.

`pooled.moving_fraction` is
    sum(moving_count over transitions) / sum(total_count over transitions)
and therefore treats each vortex observation in each initial frame as one sample.
"""
function local_vortex_statistics_series(
    peaks,
    analysis;
    xrange::Tuple{<:Real,<:Real}=(0.0, 200.0),
    yrange::Tuple{<:Real,<:Real}=(0.0, 200.0),
    binsize::Real=40.0,
)
    @assert length(analysis.per_frame) == length(peaks) - 1 "analysis.per_frame must correspond to consecutive peak frames."

    per_frame = [
        local_vortex_statistics(
            peaks[t],
            analysis.per_frame[t];
            xrange=xrange,
            yrange=yrange,
            binsize=binsize,
        )
        for t in eachindex(analysis.per_frame)
    ]

    first_stat = first(per_frame)
    pooled_total_count = zeros(Int, size(first_stat.total_count))
    pooled_moving_count = zeros(Int, size(first_stat.moving_count))

    for stat in per_frame
        pooled_total_count .+= stat.total_count
        pooled_moving_count .+= stat.moving_count
    end

    pooled_moving_fraction = fill(NaN, size(pooled_total_count))
    valid = pooled_total_count .> 0
    pooled_moving_fraction[valid] .=
        pooled_moving_count[valid] ./ pooled_total_count[valid]

    pooled = (;
        t=:all,
        xrange=first_stat.xrange,
        yrange=first_stat.yrange,
        binsize=first_stat.binsize,
        xedges=first_stat.xedges,
        yedges=first_stat.yedges,
        xcenters=first_stat.xcenters,
        ycenters=first_stat.ycenters,
        total_count=pooled_total_count,
        moving_count=pooled_moving_count,
        moving_fraction=pooled_moving_fraction,
    )

    return (; per_frame, pooled)
end

"""
    print_local_vortex_statistics(stat)

Print a compact table for one frame transition or pooled result.
Rows are ordered from low y to high y.
"""
function print_local_vortex_statistics(stat)
    ny, nx = size(stat.total_count)

    println("t = ", stat.t)
    println("ix iy   x-range       y-range       total moving fraction")

    for iy in 1:ny, ix in 1:nx
        frac = stat.moving_fraction[iy, ix]
        frac_text = isfinite(frac) ? string(round(100 * frac; digits=2), "%") : "NaN"

        println(
            lpad(ix, 2), " ", lpad(iy, 2), "   ",
            "[", stat.xedges[ix], ", ", stat.xedges[ix + 1], "]   ",
            "[", stat.yedges[iy], ", ", stat.yedges[iy + 1], "]   ",
            lpad(stat.total_count[iy, ix], 5), " ",
            lpad(stat.moving_count[iy, ix], 6), " ",
            lpad(frac_text, 8),
        )
    end

    return nothing
end

"""
    write_local_vortex_statistics_csv(path, series)

Write all frame-by-frame results and the pooled result to one CSV file.
"""
function write_local_vortex_statistics_csv(
    path::AbstractString,
    series,
)
    open(path, "w") do io
        println(
            io,
            "t,ix,iy,xmin_um,xmax_um,ymin_um,ymax_um,total_vortices,moving_vortices,moving_fraction,moving_percent",
        )

        function write_one(stat, tlabel)
            ny, nx = size(stat.total_count)
            for iy in 1:ny, ix in 1:nx
                frac = stat.moving_fraction[iy, ix]
                percent = isfinite(frac) ? 100 * frac : NaN

                println(io, join((
                    tlabel,
                    ix,
                    iy,
                    stat.xedges[ix],
                    stat.xedges[ix + 1],
                    stat.yedges[iy],
                    stat.yedges[iy + 1],
                    stat.total_count[iy, ix],
                    stat.moving_count[iy, ix],
                    frac,
                    percent,
                ), ','))
            end
        end

        for stat in series.per_frame
            write_one(stat, stat.t)
        end
        write_one(series.pooled, "all")
    end

    return path
end

"""
    plot_local_vortex_statistics(stat; outpath=nothing, title_prefix="")

Plot total count, moving count, and moving fraction in three panels.
Cell labels show the numerical values.
"""
function plot_local_vortex_statistics(
    stat;
    outpath::Union{Nothing,AbstractString}=nothing,
    title_prefix::AbstractString="",
    yreversed::Bool=false,
)
    fig = Figure(size=(1500, 470))

    ax_total = Axis(
        fig[1, 1];
        xlabel="x (μm)",
        ylabel="y (μm)",
        title="all vortices",
        aspect=DataAspect(),
        yreversed=yreversed,
    )
    ax_moving = Axis(
        fig[1, 2];
        xlabel="x (μm)",
        ylabel="y (μm)",
        title="moving vortices",
        aspect=DataAspect(),
        yreversed=yreversed,
    )
    ax_fraction = Axis(
        fig[1, 3];
        xlabel="x (μm)",
        ylabel="y (μm)",
        title="moving fraction (%)",
        aspect=DataAspect(),
        yreversed=yreversed,
    )

    hm_total = heatmap!(
        ax_total,
        stat.xedges,
        stat.yedges,
        permutedims(Float64.(stat.total_count));
        colormap=:viridis,
    )
    hm_moving = heatmap!(
        ax_moving,
        stat.xedges,
        stat.yedges,
        permutedims(Float64.(stat.moving_count));
        colormap=:viridis,
    )
    hm_fraction = heatmap!(
        ax_fraction,
        stat.xedges,
        stat.yedges,
        permutedims(100 .* stat.moving_fraction);
        colormap=:viridis,
        colorrange=(0, 100),
    )

    Colorbar(fig[2, 1], hm_total; vertical=false, label="count")
    Colorbar(fig[2, 2], hm_moving; vertical=false, label="count")
    Colorbar(fig[2, 3], hm_fraction; vertical=false, label="%")

    ny, nx = size(stat.total_count)
    for iy in 1:ny, ix in 1:nx
        x = stat.xcenters[ix]
        y = stat.ycenters[iy]

        text!(
            ax_total,
            x,
            y;
            text=string(stat.total_count[iy, ix]),
            align=(:center, :center),
            color=:white,
        )
        text!(
            ax_moving,
            x,
            y;
            text=string(stat.moving_count[iy, ix]),
            align=(:center, :center),
            color=:white,
        )

        frac = stat.moving_fraction[iy, ix]
        frac_text = isfinite(frac) ? string(round(100 * frac; digits=1)) : "-"
        text!(
            ax_fraction,
            x,
            y;
            text=frac_text,
            align=(:center, :center),
            color=:white,
        )
    end

    frame_text = stat.t === :all ? "all frame pairs" : "frame $(stat.t) -> $(stat.t + 1)"
    Label(
        fig[0, 1:3],
        isempty(title_prefix) ? frame_text : title_prefix * " | " * frame_text;
        fontsize=22,
    )

    if outpath !== nothing
        save(outpath, fig)
    end

    return fig
end

# -----------------------------------------------------------------------------
# Example
# -----------------------------------------------------------------------------
# include("track_peakmove_arrows2d(2).jl")
#
# analysis = analyze_peak_series(
#     peaks;
#     dt=1.0,
#     d_static=0.20,
#     d_move=0.50,
#     dmax=3.00,
# )
#
# local_stats = local_vortex_statistics_series(
#     peaks,
#     analysis;
#     xrange=(0.0, 200.0),
#     yrange=(0.0, 200.0),
#     binsize=40.0,
# )
#
# # One transition: frame 1 -> 2
# print_local_vortex_statistics(local_stats.per_frame[1])
# plot_local_vortex_statistics(
#     local_stats.per_frame[1];
#     outpath="local_vortex_fraction_frame_0001.png",
# )
#
# # Pooled over all frame transitions
# print_local_vortex_statistics(local_stats.pooled)
# plot_local_vortex_statistics(
#     local_stats.pooled;
#     outpath="local_vortex_fraction_all_frames.png",
# )
#
# write_local_vortex_statistics_csv(
#     "local_vortex_statistics_40um.csv",
#     local_stats,
# )
