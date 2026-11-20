# moving_vortex_field_overlay_patch.jl
#
# Add this file after your original moving-vortex tracking script.
#
# Usage:
#   include("track_peak_move_arrow2d.jl")          # your original script
#   include("moving_vortex_field_overlay_patch.jl")
#
#   analysis = run_moving_vortex_analysis_with_field(
#       peaks,
#       Bmaps;
#       outdir = "moving_vortex_output",
#       d_static = 0.20,
#       d_move = 0.50,
#       dmax = 3.00,
#       background = :diff,       # :before, :after, or :diff
#       arrow_scale = 1.0,
#       field_arrow_scale = 1.0,
#   )
#
# Here:
#   peaks[t]  : peak data at frame t
#   Bmaps[t]  : magnetic field image matrix at frame t
#
# The functions below assume that your original script already defines:
#   common_xy_limits
#   analyze_peak_series
#   write_displacements_csv
#   plot_displacement_histogram
#   plot_frame_vectors
#   local_velocity_field
#   plot_local_velocity_field
#
# Required packages:
#   using CairoMakie
#   using JLD2
#   using Statistics
#   using Printf

using CairoMakie
using JLD2
using Statistics
using Printf

# ------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------

function _getv(v, name::Symbol)
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

function _moving_xyuv(moving)
    x  = Float64[_getv(m, :x)  for m in moving]
    y  = Float64[_getv(m, :y)  for m in moving]
    dx = Float64[_getv(m, :dx) for m in moving]
    dy = Float64[_getv(m, :dy) for m in moving]
    return x, y, dx, dy
end

function _sym_colorrange(A; percentile=99.0)
    vals = abs.(vec(skipmissing(A)))
    vals = vals[isfinite.(vals)]
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

function _normal_colorrange(A; percentile_low=1.0, percentile_high=99.0)
    vals = collect(vec(skipmissing(A)))
    vals = vals[isfinite.(vals)]
    if isempty(vals)
        return (-1.0, 1.0)
    end
    lo = quantile(vals, percentile_low / 100)
    hi = quantile(vals, percentile_high / 100)
    if lo == hi
        hi = lo + 1.0
    end
    return (lo, hi)
end

function _field_background(B_before, B_after; background::Symbol=:diff)
    if background == :before
        return B_before, "B before"
    elseif background == :after
        return B_after, "B after"
    elseif background == :diff
        return B_after .- B_before, "B after - B before"
    else
        error("background must be :before, :after, or :diff")
    end
end

function _heatmap_matrix(B; transpose_field::Bool=true)
    return transpose_field ? B' : B
end

# ------------------------------------------------------------
# Plot one frame: magnetic field image + moving vortex arrows
# ------------------------------------------------------------

function plot_frame_vectors_on_field(
    B_before::AbstractMatrix,
    B_after::AbstractMatrix,
    result;
    outpath::AbstractString,
    frame_label::AbstractString="",
    background::Symbol=:diff,
    xlim=nothing,
    ylim=nothing,
    arrow_scale::Real=1.0,
    max_arrows::Int=2000,
    d_move::Real=0.0,
    transpose_field::Bool=true,
    reverse_y::Bool=false,
    colormap_diff=:balance,
    colormap_field=:viridis,
    show_colorbar::Bool=true,
    show_moving_start::Bool=true,
    marker_size::Real=4.0,
)
    @assert size(B_before) == size(B_after) "B_before and B_after must have the same size."

    Bplot, bg_label = _field_background(B_before, B_after; background=background)

    nx = size(_heatmap_matrix(Bplot; transpose_field=transpose_field), 1)
    ny = size(_heatmap_matrix(Bplot; transpose_field=transpose_field), 2)

    if xlim === nothing
        xlim = (1.0, Float64(nx))
    end
    if ylim === nothing
        ylim = (1.0, Float64(ny))
    end

    xgrid = range(xlim[1], xlim[2], length=nx)
    ygrid = range(ylim[1], ylim[2], length=ny)

    Z = _heatmap_matrix(Bplot; transpose_field=transpose_field)

    if background == :diff
        crange = _sym_colorrange(Z)
        cmap = colormap_diff
    else
        crange = _normal_colorrange(Z)
        cmap = colormap_field
    end

    moving = filter(m -> _getv(m, :dr) >= d_move, result.moving)

    if length(moving) > max_arrows
        inds = unique(round.(Int, range(1, length(moving); length=max_arrows)))
        moving_plot = moving[inds]
    else
        moving_plot = moving
    end

    fig = Figure(size=(900, 760))
    ax = Axis(
        fig[1, 1],
        aspect=DataAspect(),
        xlabel="x",
        ylabel="y",
        title=isempty(frame_label) ? "Vortex displacement on $(bg_label)" : frame_label,
    )

    hm = heatmap!(
        ax,
        xgrid,
        ygrid,
        Z;
        colormap=cmap,
        colorrange=crange,
    )

    if show_colorbar
        Colorbar(fig[1, 2], hm, label=bg_label)
    end

    if !isempty(moving_plot)
        x, y, dx, dy = _moving_xyuv(moving_plot)

        if show_moving_start
            scatter!(
                ax,
                x,
                y;
                markersize=marker_size,
                color=(:black, 0.70),
                label="moving start",
            )
        end

        arrows2d!(
            ax,
            Point2f.(x, y),
            Vec2f.(dx, dy);
            lengthscale=arrow_scale,
            align=:tail,
            tipwidth=10,
            tiplength=12,
            shaftwidth=3,
            color=:red,
            label="displacement",
        )
    end

    xlims!(ax, xlim...)
    ylims!(ax, ylim...)
    if reverse_y
        ax.yreversed = true
    end

    if !isempty(moving_plot)
        axislegend(ax; position=:rt)
    end

    Label(
        fig[2, 1],
        "moving arrows = $(length(moving_plot)) / $(length(moving)); background = $(background)";
        tellwidth=false,
        fontsize=13,
    )

    save(outpath, fig)

    return (; fig, outpath, Bplot, moving=moving, moving_plot=moving_plot)
end

# ------------------------------------------------------------
# Main runner: original analysis + field-background arrow plots
# ------------------------------------------------------------

function run_moving_vortex_analysis_with_field(
    peaks,
    Bmaps;
    outdir::AbstractString="moving_vortex_output",
    dt::Real=1.0,
    d_static::Real=0.20,
    d_move::Real=0.50,
    dmax::Real=3.00,
    binsize::Real=10.0,
    min_count::Int=3,
    arrow_scale::Real=1.0,
    field_arrow_scale::Real=1.0,
    max_arrows::Int=2000,
    make_each_frame_plot::Bool=true,
    make_each_frame_field::Bool=true,
    make_each_frame_field_background::Bool=true,
    background::Symbol=:diff,
    transpose_field::Bool=true,
    reverse_y::Bool=false,
)
    @assert length(Bmaps) == length(peaks) "Bmaps and peaks must have the same length."

    mkpath(outdir)

    xlim, ylim = common_xy_limits(peaks)

    analysis = analyze_peak_series(
        peaks;
        dt=dt,
        d_static=d_static,
        d_move=d_move,
        dmax=dmax,
        verbose=true,
    )

    # Save numeric results.
    jldsave(joinpath(outdir, "moving_vortex_analysis.jld2"); analysis)
    write_displacements_csv(joinpath(outdir, "moving_vortex_vectors.csv"), analysis.all_moving)

    # Histogram for checking thresholds.
    plot_displacement_histogram(
        analysis;
        outpath=joinpath(outdir, "displacement_histogram.png"),
        d_static=d_static,
        d_move=d_move,
    )

    # Frame-by-frame vector plots on peak-only map.
    if make_each_frame_plot
        for t in 1:length(analysis.per_frame)
            result = analysis.per_frame[t]
            plot_frame_vectors(
                peaks[t],
                result;
                outpath=joinpath(outdir, "moving_vectors_frame_$(lpad(t, 4, '0')).png"),
                frame_label="frame $(t) -> $(t + 1)",
                arrow_scale=arrow_scale,
                max_arrows=max_arrows,
                xlim=xlim,
                ylim=ylim,
            )
        end
    end

    # New: frame-by-frame vector plots on magnetic field image.
    if make_each_frame_field_background
        for t in 1:length(analysis.per_frame)
            result = analysis.per_frame[t]
            if isempty(result.moving)
                continue
            end

            plot_frame_vectors_on_field(
                Bmaps[t],
                Bmaps[t + 1],
                result;
                outpath=joinpath(outdir, "field_vectors_frame_$(lpad(t, 4, '0')).png"),
                frame_label="field + vectors: frame $(t) -> $(t + 1)",
                background=background,
                arrow_scale=arrow_scale,
                max_arrows=max_arrows,
                d_move=d_move,
                xlim=xlim,
                ylim=ylim,
                transpose_field=transpose_field,
                reverse_y=reverse_y,
            )
        end
    end

    # Frame-by-frame local velocity fields.
    if make_each_frame_field
        for t in 1:length(analysis.per_frame)
            result = analysis.per_frame[t]
            if isempty(result.moving)
                continue
            end

            field = local_velocity_field(
                result.moving;
                xrange=xlim,
                yrange=ylim,
                binsize=binsize,
                min_count=min_count,
            )

            plot_local_velocity_field(
                peaks[t],
                field;
                outpath=joinpath(outdir, "local_velocity_frame_$(lpad(t, 4, '0')).png"),
                frame_label="local velocity field: frame $(t) -> $(t + 1)",
                arrow_scale=field_arrow_scale,
                xlim=xlim,
                ylim=ylim,
            )
        end
    end

    # Time-averaged local velocity field using all moving-vortex displacements.
    if !isempty(analysis.all_moving)
        field_all = local_velocity_field(
            analysis.all_moving;
            xrange=xlim,
            yrange=ylim,
            binsize=binsize,
            min_count=min_count,
        )

        plot_local_velocity_field(
            peaks[1],
            field_all;
            outpath=joinpath(outdir, "local_velocity_all_frames.png"),
            frame_label="local velocity field: all frames",
            arrow_scale=field_arrow_scale,
            xlim=xlim,
            ylim=ylim,
        )
    end

    return analysis
end
