# moving_vortex_field_overlay_patch_v3.jl
#
# Fixes:
#   1. Works even when the field image is transposed with B'.
#   2. Overrides the older AbstractMatrix method if you already included
#      moving_vortex_field_overlay_patch.jl before this file.
#   3. Bmaps[t] can be either a Matrix or a NamedTuple/Dict containing a Matrix.
#
# Usage:
#   include("track_peak_move_arrow2d.jl")
#   include("moving_vortex_field_overlay_patch_v3.jl")
#
#   analysis = run_moving_vortex_analysis_with_field(
#       peaks,
#       Bmaps;
#       field_key = :deltaf,      # use :B, :deltaf, etc. if Bmaps[t] is a NamedTuple/Dict
#       background = :diff,
#       outdir = "moving_vortex_output",
#   )

using CairoMakie
using JLD2
using Statistics
using Printf

# ------------------------------------------------------------
# Generic field/key access
# ------------------------------------------------------------

function _has_key_or_property(obj, key)
    skey = key isa Symbol ? key : Symbol(key)
    strkey = String(skey)

    if hasproperty(obj, skey)
        return true
    elseif obj isa AbstractDict && (haskey(obj, skey) || haskey(obj, strkey))
        return true
    else
        return false
    end
end

function _get_key_or_property(obj, key)
    skey = key isa Symbol ? key : Symbol(key)
    strkey = String(skey)

    if hasproperty(obj, skey)
        return getproperty(obj, skey)
    elseif obj isa AbstractDict && haskey(obj, skey)
        return obj[skey]
    elseif obj isa AbstractDict && haskey(obj, strkey)
        return obj[strkey]
    else
        error("Cannot get key/property $(key) from object of type $(typeof(obj))")
    end
end

function _getv(v, name::Symbol)
    return _get_key_or_property(v, name)
end

# ------------------------------------------------------------
# Extract magnetic field matrix from Matrix / NamedTuple / Dict
# ------------------------------------------------------------

function _matrix_candidates(obj)
    names = Symbol[]
    vals = Any[]

    if obj isa AbstractDict
        for (k, v) in obj
            if v isa AbstractMatrix
                push!(names, Symbol(k))
                push!(vals, v)
            end
        end
    else
        for k in propertynames(obj)
            v = getproperty(obj, k)
            if v isa AbstractMatrix
                push!(names, Symbol(k))
                push!(vals, v)
            end
        end
    end

    return names, vals
end

function extract_field_matrix(obj; field_key=nothing)
    if obj isa AbstractMatrix
        return obj
    end

    if field_key !== nothing
        v = _get_key_or_property(obj, field_key)
        if !(v isa AbstractMatrix)
            error("field_key=$(field_key) exists, but it is not a matrix. Its type is $(typeof(v)).")
        end
        return v
    end

    preferred_keys = [
        :B,
        :Bmap,
        :B_map,
        :field,
        :fieldmap,
        :field_map,
        :deltaf,
        :delta_f,
        :img,
        :image,
        :data,
    ]

    for k in preferred_keys
        if _has_key_or_property(obj, k)
            v = _get_key_or_property(obj, k)
            if v isa AbstractMatrix
                return v
            end
        end
    end

    names, vals = _matrix_candidates(obj)

    if length(vals) == 1
        return vals[1]
    elseif isempty(vals)
        error("""
        Bmaps element is not a matrix and no matrix field was found.

        typeof(Bmaps[t]) = $(typeof(obj))

        If your magnetic field is stored inside the object, pass for example:
            field_key = :deltaf
        or:
            field_key = :B
        """)
    else
        error("""
        Multiple matrix fields were found: $(names)

        Please specify which one is the magnetic field, for example:
            field_key = :deltaf
        or:
            field_key = :B
        """)
    end
end

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

function _moving_xyuv(moving)
    x  = Float64[_getv(m, :x)  for m in moving]
    y  = Float64[_getv(m, :y)  for m in moving]
    dx = Float64[_getv(m, :dx) for m in moving]
    dy = Float64[_getv(m, :dy) for m in moving]
    return x, y, dx, dy
end

# Important:
# Do not use vec(skipmissing(A)) here because A may be an Adjoint from B'.
function _finite_values(A)
    vals = Float64[]
    for v in A
        if isfinite(v)
            push!(vals, Float64(v))
        end
    end
    return vals
end

function _sym_colorrange(A; percentile=99.0)
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

function _normal_colorrange(A; percentile_low=1.0, percentile_high=99.0)
    vals = _finite_values(A)
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
    # collect(...) converts Adjoint into a normal Matrix.
    return transpose_field ? collect(B') : collect(B)
end

# ------------------------------------------------------------
# Plot one frame: magnetic field image + moving vortex arrows
# ------------------------------------------------------------

function plot_frame_vectors_on_field(
    B_before_src,
    B_after_src,
    result;
    outpath::AbstractString,
    field_key=nothing,
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
    B_before = extract_field_matrix(B_before_src; field_key=field_key)
    B_after  = extract_field_matrix(B_after_src;  field_key=field_key)

    @assert size(B_before) == size(B_after) "B_before and B_after must have the same size."

    Bplot, bg_label = _field_background(B_before, B_after; background=background)

    Z = _heatmap_matrix(Bplot; transpose_field=transpose_field)
    nx, ny = size(Z)

    if xlim === nothing
        xlim = (1.0, Float64(nx))
    end
    if ylim === nothing
        ylim = (1.0, Float64(ny))
    end

    xgrid = range(xlim[1], xlim[2], length=nx)
    ygrid = range(ylim[1], ylim[2], length=ny)

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

# This specific method intentionally overrides the older AbstractMatrix method
# if moving_vortex_field_overlay_patch.jl was included before this file.
function plot_frame_vectors_on_field(
    B_before_src::AbstractMatrix,
    B_after_src::AbstractMatrix,
    result;
    outpath::AbstractString,
    field_key=nothing,
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
    return invoke(
        plot_frame_vectors_on_field,
        Tuple{Any, Any, Any},
        B_before_src,
        B_after_src,
        result;
        outpath=outpath,
        field_key=field_key,
        frame_label=frame_label,
        background=background,
        xlim=xlim,
        ylim=ylim,
        arrow_scale=arrow_scale,
        max_arrows=max_arrows,
        d_move=d_move,
        transpose_field=transpose_field,
        reverse_y=reverse_y,
        colormap_diff=colormap_diff,
        colormap_field=colormap_field,
        show_colorbar=show_colorbar,
        show_moving_start=show_moving_start,
        marker_size=marker_size,
    )
end

# ------------------------------------------------------------
# Main runner: original analysis + field-background arrow plots
# ------------------------------------------------------------

function run_moving_vortex_analysis_with_field(
    peaks,
    Bmaps;
    outdir::AbstractString="moving_vortex_output",
    field_key=nothing,
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

    # Frame-by-frame vector plots on magnetic field image.
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
                field_key=field_key,
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
