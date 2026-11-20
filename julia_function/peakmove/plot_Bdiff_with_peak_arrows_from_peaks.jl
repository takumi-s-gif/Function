# plot_Bdiff_with_peak_arrows_from_peaks.jl
#
# Purpose:
#   Plot B_after - B_before and overlay vortex displacement arrows
#   directly from Bmat and peaks, without using analysis.
#
# Required packages:
#   using CairoMakie
#   using NearestNeighbors
#   using Statistics
#
# Typical usage:
#
#   include("plot_Bdiff_with_peak_arrows_from_peaks.jl")
#
#   t = 1
#   result = plot_Bdiff_with_peak_arrows(
#       Bmat[t],
#       Bmat[t+1],
#       peaks[t],
#       peaks[t+1];
#       xlim = (0.0, 130.0),
#       ylim = (0.0, 130.0),
#       dmax = 3.0,
#       d_move = 0.5,
#       colorrange = (-0.5, 0.5),
#       colormap = :balance,
#       colorbar_label = "B_after - B_before",
#       arrow_scale = 1.0,
#       outpath = "Bdiff_arrows_frame_0001.png",
#   )
#
#   # all frames
#   plot_Bdiff_with_peak_arrows_series(
#       Bmat,
#       peaks;
#       outdir = "Bdiff_arrow_output",
#       xlim = (0.0, 130.0),
#       ylim = (0.0, 130.0),
#       dmax = 3.0,
#       d_move = 0.5,
#       colorrange = (-0.5, 0.5),
#   )

using CairoMakie
using NearestNeighbors
using Statistics
using Printf

# ------------------------------------------------------------
# utilities
# ------------------------------------------------------------

function _getfield_or_key(obj, name::Symbol)
    if hasproperty(obj, name)
        return getproperty(obj, name)
    elseif obj isa AbstractDict && haskey(obj, name)
        return obj[name]
    elseif obj isa AbstractDict && haskey(obj, String(name))
        return obj[String(name)]
    else
        error("Cannot find $(name) in object of type $(typeof(obj))")
    end
end

function _peak_xy(peaks)
    x = collect(_getfield_or_key(peaks, :x))
    y = collect(_getfield_or_key(peaks, :y))
    @assert length(x) == length(y) "peaks.x and peaks.y must have the same length."
    return x, y
end

function _finite_values(A)
    vals = Float64[]
    for v in A
        if isfinite(v)
            push!(vals, Float64(v))
        end
    end
    return vals
end

function _auto_symmetric_colorrange(A; percentile::Real=99.0)
    vals = abs.(_finite_values(A))

    if isempty(vals)
        return (-1.0, 1.0)
    end

    vmax = quantile(vals, percentile / 100)

    if vmax == 0 || !isfinite(vmax)
        vmax = maximum(vals)
    end
    if vmax == 0 || !isfinite(vmax)
        vmax = 1.0
    end

    return (-vmax, vmax)
end

function _prepare_Bdiff(B_before::AbstractMatrix, B_after::AbstractMatrix; transpose_field::Bool=true)
    @assert size(B_before) == size(B_after) "B_before and B_after must have the same size."

    dB = B_after .- B_before
    Z = transpose_field ? collect(dB') : collect(dB)
    return Z
end

# ------------------------------------------------------------
# peak matching
# ------------------------------------------------------------

function match_peaks_mutual(peaks_before, peaks_after; dmax::Real=3.0)
    x1, y1 = _peak_xy(peaks_before)
    x2, y2 = _peak_xy(peaks_after)

    if isempty(x1) || isempty(x2)
        return NamedTuple[]
    end

    xy1 = hcat(x1, y1)
    xy2 = hcat(x2, y2)

    tree2 = KDTree(permutedims(xy2))   # 2 x N
    tree1 = KDTree(permutedims(xy1))   # 2 x N

    idx12, dist12 = knn(tree2, permutedims(xy1), 1, true)
    idx21, dist21 = knn(tree1, permutedims(xy2), 1, true)

    matches = NamedTuple[]

    for i in 1:size(xy1, 1)
        j = idx12[i][1]
        d = dist12[i][1]

        if d <= dmax && idx21[j][1] == i
            x_before = xy1[i, 1]
            y_before = xy1[i, 2]
            x_after  = xy2[j, 1]
            y_after  = xy2[j, 2]
            dx = x_after - x_before
            dy = y_after - y_before
            dr = sqrt(dx^2 + dy^2)

            push!(matches, (
                i_before = i,
                i_after = j,
                x = x_before,
                y = y_before,
                x_after = x_after,
                y_after = y_after,
                dx = dx,
                dy = dy,
                dr = dr,
            ))
        end
    end

    return matches
end

function _filter_and_downsample_matches(matches; d_move::Real=0.0, max_arrows::Int=2000)
    moving = filter(m -> m.dr >= d_move, matches)

    if length(moving) > max_arrows
        inds = unique(round.(Int, range(1, length(moving); length=max_arrows)))
        moving = moving[inds]
    end

    x  = Float64[m.x for m in moving]
    y  = Float64[m.y for m in moving]
    dx = Float64[m.dx for m in moving]
    dy = Float64[m.dy for m in moving]
    dr = Float64[m.dr for m in moving]

    return (; moving, x, y, dx, dy, dr)
end

# ------------------------------------------------------------
# main plot
# ------------------------------------------------------------

function plot_Bdiff_with_peak_arrows(
    B_before::AbstractMatrix,
    B_after::AbstractMatrix,
    peaks_before,
    peaks_after;
    xlim::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
    ylim::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
    dmax::Real=3.0,
    d_move::Real=0.5,
    colorrange=nothing,
    auto_colorrange_percentile::Real=99.0,
    colormap=:balance,
    colorbar_label::AbstractString="B_after - B_before",
    title::AbstractString="B_after - B_before with vortex displacement",
    xlabel::AbstractString="x",
    ylabel::AbstractString="y",
    arrow_scale::Real=1.0,
    max_arrows::Int=2000,
    transpose_field::Bool=true,
    reverse_y::Bool=false,
    show_start_points::Bool=true,
    show_before_after_points::Bool=false,
    marker_size::Real=4.0,
    arrow_color=:red,
    start_marker_color=(:black, 0.75),
    before_marker_color=(:cyan, 0.45),
    after_marker_color=(:yellow, 0.45),
    tipwidth::Real=10.0,
    tiplength::Real=12.0,
    shaftwidth::Real=3.0,
    figsize::Tuple{Int, Int}=(900, 760),
    outpath::Union{Nothing, AbstractString}="Bdiff_with_arrows.png",
)
    Z = _prepare_Bdiff(B_before, B_after; transpose_field=transpose_field)
    nx, ny = size(Z)

    if xlim === nothing || ylim === nothing
        xb, yb = _peak_xy(peaks_before)
        xa, ya = _peak_xy(peaks_after)
        xall = vcat(xb, xa)
        yall = vcat(yb, ya)

        if xlim === nothing
            xlim = isempty(xall) ? (1.0, Float64(nx)) : (minimum(xall), maximum(xall))
        end
        if ylim === nothing
            ylim = isempty(yall) ? (1.0, Float64(ny)) : (minimum(yall), maximum(yall))
        end
    end

    xgrid = range(xlim[1], xlim[2], length=nx)
    ygrid = range(ylim[1], ylim[2], length=ny)

    crange = colorrange === nothing ? _auto_symmetric_colorrange(Z; percentile=auto_colorrange_percentile) : colorrange

    matches = match_peaks_mutual(peaks_before, peaks_after; dmax=dmax)
    vecs = _filter_and_downsample_matches(matches; d_move=d_move, max_arrows=max_arrows)

    fig = Figure(size=figsize)
    ax = Axis(
        fig[1, 1],
        aspect=DataAspect(),
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
    )

    hm = heatmap!(
        ax,
        xgrid,
        ygrid,
        Z;
        colormap=colormap,
        colorrange=crange,
    )

    Colorbar(
        fig[1, 2],
        hm;
        label=colorbar_label,
    )

    if show_before_after_points
        xb, yb = _peak_xy(peaks_before)
        xa, ya = _peak_xy(peaks_after)

        scatter!(ax, xb, yb; markersize=3, color=before_marker_color, label="before peaks")
        scatter!(ax, xa, ya; markersize=3, marker=:xcross, color=after_marker_color, label="after peaks")
    end

    if !isempty(vecs.moving)
        if show_start_points
            scatter!(
                ax,
                vecs.x,
                vecs.y;
                markersize=marker_size,
                color=start_marker_color,
                label="moving start",
            )
        end

        arrows2d!(
            ax,
            Point2f.(vecs.x, vecs.y),
            Vec2f.(vecs.dx, vecs.dy);
            lengthscale=arrow_scale,
            align=:tail,
            tipwidth=tipwidth,
            tiplength=tiplength,
            shaftwidth=shaftwidth,
            color=arrow_color,
            label="displacement",
        )

        axislegend(ax; position=:rt)
    end

    xlims!(ax, xlim...)
    ylims!(ax, ylim...)

    if reverse_y
        ax.yreversed = true
    end

    Label(
        fig[2, 1],
        "matched = $(length(matches)), arrows = $(length(vecs.moving)), dmax = $(dmax), d_move = $(d_move), colorrange = $(crange)";
        tellwidth=false,
        fontsize=13,
    )

    if outpath !== nothing
        save(outpath, fig)
    end

    return (;
        fig,
        dB=Z,
        matches=matches,
        moving=vecs.moving,
        colorrange=crange,
        outpath,
    )
end

# ------------------------------------------------------------
# helper: all frames
# ------------------------------------------------------------

function plot_Bdiff_with_peak_arrows_series(
    Bmat,
    peaks;
    outdir::AbstractString="Bdiff_arrow_output",
    xlim::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
    ylim::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
    dmax::Real=3.0,
    d_move::Real=0.5,
    colorrange=nothing,
    auto_colorrange_percentile::Real=99.0,
    colormap=:balance,
    colorbar_label::AbstractString="B_after - B_before",
    arrow_scale::Real=1.0,
    max_arrows::Int=2000,
    transpose_field::Bool=true,
    reverse_y::Bool=false,
    show_start_points::Bool=true,
    show_before_after_points::Bool=false,
)
    @assert length(Bmat) == length(peaks) "Bmat and peaks must have the same length."
    @assert length(Bmat) >= 2 "Need at least 2 frames."

    mkpath(outdir)
    results = Vector{Any}(undef, length(Bmat) - 1)

    for t in 1:(length(Bmat) - 1)
        outpath = joinpath(outdir, "Bdiff_arrows_frame_$(lpad(t, 4, '0')).png")

        results[t] = plot_Bdiff_with_peak_arrows(
            Bmat[t],
            Bmat[t+1],
            peaks[t],
            peaks[t+1];
            xlim=xlim,
            ylim=ylim,
            dmax=dmax,
            d_move=d_move,
            colorrange=colorrange,
            auto_colorrange_percentile=auto_colorrange_percentile,
            colormap=colormap,
            colorbar_label=colorbar_label,
            title="B_after - B_before + arrows: frame $(t) -> $(t+1)",
            arrow_scale=arrow_scale,
            max_arrows=max_arrows,
            transpose_field=transpose_field,
            reverse_y=reverse_y,
            show_start_points=show_start_points,
            show_before_after_points=show_before_after_points,
            outpath=outpath,
        )
    end

    return results
end
