# plot_Bdiff_with_peak_arrows.jl
#
# Purpose:
#   Plot B_after - B_before and overlay vortex displacement arrows
#   using the result from your previous moving-vortex analysis.
#
# Typical usage:
#
#   include("plot_Bdiff_with_peak_arrows.jl")
#
#   # analysis was already made by run_moving_vortex_analysis(...)
#   t = 1
#   plot_Bdiff_with_arrows(
#       Bmat[t],
#       Bmat[t+1],
#       analysis.per_frame[t];
#       xlim = (0.0, 130.0),
#       ylim = (0.0, 130.0),
#       colorrange = (-0.5, 0.5),      # set manually
#       colormap = :balance,
#       arrow_scale = 1.0,
#       d_move = 0.5,
#       outpath = "Bdiff_arrows_frame_0001.png",
#   )
#
#   # make all frame plots
#   plot_Bdiff_with_arrows_series(
#       Bmat,
#       analysis;
#       xlim = (0.0, 130.0),
#       ylim = (0.0, 130.0),
#       colorrange = (-0.5, 0.5),
#       outdir = "Bdiff_arrow_output",
#   )
#
# Notes:
#   - Bmat[t] must be a Matrix.
#   - analysis.per_frame[t].moving must contain x, y, dx, dy, dr.
#   - colorrange controls the colorbar limits.
#   - Use transpose_field=true/false to align the image orientation.

using CairoMakie
using Statistics
using Printf

# ------------------------------------------------------------
# utilities
# ------------------------------------------------------------

function _get_field(v, name::Symbol)
    if hasproperty(v, name)
        return getproperty(v, name)
    elseif v isa AbstractDict && haskey(v, name)
        return v[name]
    elseif v isa AbstractDict && haskey(v, String(name))
        return v[String(name)]
    else
        error("Cannot get field/key $(name) from object of type $(typeof(v))")
    end
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

    # B' gives an Adjoint, so collect is used to convert it back to a normal Matrix.
    Z = transpose_field ? collect(dB') : collect(dB)

    return Z
end

function _moving_vectors(result; d_move::Real=0.0, max_arrows::Int=2000)
    moving = filter(m -> _get_field(m, :dr) >= d_move, result.moving)

    if length(moving) > max_arrows
        inds = unique(round.(Int, range(1, length(moving); length=max_arrows)))
        moving = moving[inds]
    end

    x  = Float64[_get_field(m, :x)  for m in moving]
    y  = Float64[_get_field(m, :y)  for m in moving]
    dx = Float64[_get_field(m, :dx) for m in moving]
    dy = Float64[_get_field(m, :dy) for m in moving]
    dr = Float64[_get_field(m, :dr) for m in moving]

    return (; moving, x, y, dx, dy, dr)
end

# ------------------------------------------------------------
# main plot function
# ------------------------------------------------------------

function plot_Bdiff_with_arrows(
    B_before::AbstractMatrix,
    B_after::AbstractMatrix,
    result;
    xlim::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
    ylim::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
    colorrange=nothing,
    auto_colorrange_percentile::Real=99.0,
    colormap=:balance,
    colorbar_label::AbstractString="B_after - B_before",
    title::AbstractString="B_after - B_before with vortex displacement",
    xlabel::AbstractString="x",
    ylabel::AbstractString="y",
    d_move::Real=0.0,
    arrow_scale::Real=1.0,
    max_arrows::Int=2000,
    transpose_field::Bool=true,
    reverse_y::Bool=false,
    show_start_points::Bool=true,
    marker_size::Real=4.0,
    arrow_color=:red,
    start_marker_color=(:black, 0.75),
    tipwidth::Real=10.0,
    tiplength::Real=12.0,
    shaftwidth::Real=3.0,
    figsize::Tuple{Int, Int}=(900, 760),
    outpath::Union{Nothing, AbstractString}="Bdiff_with_arrows.png",
)
    Z = _prepare_Bdiff(B_before, B_after; transpose_field=transpose_field)
    nx, ny = size(Z)

    if xlim === nothing
        xlim = (1.0, Float64(nx))
    end
    if ylim === nothing
        ylim = (1.0, Float64(ny))
    end

    xgrid = range(xlim[1], xlim[2], length=nx)
    ygrid = range(ylim[1], ylim[2], length=ny)

    crange = colorrange === nothing ? _auto_symmetric_colorrange(Z; percentile=auto_colorrange_percentile) : colorrange

    vecs = _moving_vectors(result; d_move=d_move, max_arrows=max_arrows)

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
        "arrows = $(length(vecs.moving)), d_move = $(d_move), colorrange = $(crange)";
        tellwidth=false,
        fontsize=13,
    )

    if outpath !== nothing
        save(outpath, fig)
    end

    return (;
        fig,
        dB=Z,
        vectors=vecs,
        colorrange=crange,
        outpath,
    )
end

# ------------------------------------------------------------
# helper: make all frame plots
# ------------------------------------------------------------

function plot_Bdiff_with_arrows_series(
    Bmat,
    analysis;
    outdir::AbstractString="Bdiff_arrow_output",
    xlim::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
    ylim::Union{Nothing, Tuple{<:Real, <:Real}}=nothing,
    colorrange=nothing,
    auto_colorrange_percentile::Real=99.0,
    colormap=:balance,
    colorbar_label::AbstractString="B_after - B_before",
    d_move::Real=0.0,
    arrow_scale::Real=1.0,
    max_arrows::Int=2000,
    transpose_field::Bool=true,
    reverse_y::Bool=false,
)
    mkpath(outdir)

    @assert length(Bmat) >= length(analysis.per_frame) + 1 "Bmat must contain one more frame than analysis.per_frame."

    results = Vector{Any}(undef, length(analysis.per_frame))

    for t in 1:length(analysis.per_frame)
        outpath = joinpath(outdir, "Bdiff_arrows_frame_$(lpad(t, 4, '0')).png")

        results[t] = plot_Bdiff_with_arrows(
            Bmat[t],
            Bmat[t+1],
            analysis.per_frame[t];
            xlim=xlim,
            ylim=ylim,
            colorrange=colorrange,
            auto_colorrange_percentile=auto_colorrange_percentile,
            colormap=colormap,
            colorbar_label=colorbar_label,
            title="B_after - B_before + arrows: frame $(t) -> $(t+1)",
            d_move=d_move,
            arrow_scale=arrow_scale,
            max_arrows=max_arrows,
            transpose_field=transpose_field,
            reverse_y=reverse_y,
            outpath=outpath,
        )
    end

    return results
end
