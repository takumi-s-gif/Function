# local_vortex_fraction_from_global_matching.jl
#
# Local moving-vortex statistics using the NEW global-assignment matching.
#
# Works either from:
#   1) result = match_plot_save(...)
#   2) vortex_matching_frame001.csv
#
# Definition:
#   denominator = all vortices in BEFORE frame
#   numerator   = non-static matches with dr >= d_move
#
# Output CSV columns are compatible with the old workflow:
# t,ix,iy,xmin_um,xmax_um,ymin_um,ymax_um,
# total_vortices,moving_vortices,moving_fraction,moving_percent

using CSV
using DataFrames
using CairoMakie

function _peak_xy_local(p)
    if hasproperty(p, :x) && hasproperty(p, :y)
        x = Float64.(getproperty(p, :x))
        y = Float64.(getproperty(p, :y))
    elseif p isa AbstractDict && haskey(p, "x") && haskey(p, "y")
        x = Float64.(p["x"])
        y = Float64.(p["y"])
    elseif p isa AbstractDict && haskey(p, :x) && haskey(p, :y)
        x = Float64.(p[:x])
        y = Float64.(p[:y])
    else
        error("Peak object must contain x and y.")
    end
    @assert length(x) == length(y)
    return hcat(x, y)
end

function spatial_edges_global(range::Tuple{<:Real,<:Real}, binsize::Real)
    lo = Float64(range[1])
    hi = Float64(range[2])
    bs = Float64(binsize)
    @assert isfinite(lo) && isfinite(hi)
    @assert hi > lo
    @assert bs > 0
    edges = collect(lo:bs:hi)
    if isempty(edges)
        edges = [lo]
    end
    if edges[end] < hi
        push!(edges, hi)
    else
        edges[end] = hi
    end
    return edges
end

function spatial_bin_index_global(value::Real, edges::AbstractVector{<:Real})
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

function local_vortex_statistics_from_xy(
    peaks_before,
    moving_x,
    moving_y;
    t::Int = 2,
    xrange::Tuple{<:Real,<:Real} = (400.0, 1000.0),
    yrange::Tuple{<:Real,<:Real} = (300.0, 1000.0),
    binsize::Real = 50.0,
)
    xedges = spatial_edges_global(xrange, binsize)
    yedges = spatial_edges_global(yrange, binsize)
    xcenters = (xedges[1:end-1] .+ xedges[2:end]) ./ 2
    ycenters = (yedges[1:end-1] .+ yedges[2:end]) ./ 2
    nx = length(xedges) - 1
    ny = length(yedges) - 1

    total_count = zeros(Int, ny, nx)
    moving_count = zeros(Int, ny, nx)

    xy = _peak_xy_local(peaks_before)
    for k in axes(xy, 1)
        ix = spatial_bin_index_global(xy[k, 1], xedges)
        iy = spatial_bin_index_global(xy[k, 2], yedges)
        if ix !== nothing && iy !== nothing
            total_count[iy, ix] += 1
        end
    end

    @assert length(moving_x) == length(moving_y)
    for k in eachindex(moving_x)
        ix = spatial_bin_index_global(moving_x[k], xedges)
        iy = spatial_bin_index_global(moving_y[k], yedges)
        if ix !== nothing && iy !== nothing
            moving_count[iy, ix] += 1
        end
    end

    moving_fraction = fill(NaN, ny, nx)
    valid = total_count .> 0
    moving_fraction[valid] .= moving_count[valid] ./ total_count[valid]

    return (
        t=t,
        xrange=(Float64(xrange[1]), Float64(xrange[2])),
        yrange=(Float64(yrange[1]), Float64(yrange[2])),
        binsize=Float64(binsize),
        xedges=xedges,
        yedges=yedges,
        xcenters=xcenters,
        ycenters=ycenters,
        total_count=total_count,
        moving_count=moving_count,
        moving_fraction=moving_fraction,
        n_total_in_roi=sum(total_count),
        n_moving_in_roi=sum(moving_count),
    )
end

function local_vortex_statistics_from_result(
    peaks_before,
    result;
    t::Int = 2,
    d_move::Real = 1.0,
    xrange::Tuple{<:Real,<:Real} = (400.0, 1000.0),
    yrange::Tuple{<:Real,<:Real} = (300.0, 1000.0),
    binsize::Real = 50.0,
)
    moving_matches = [
        m for m in result.matches
        if m.kind != :static && m.dr >= d_move
    ]
    moving_x = Float64[m.x1 for m in moving_matches]
    moving_y = Float64[m.y1 for m in moving_matches]

    stat = local_vortex_statistics_from_xy(
        peaks_before,
        moving_x,
        moving_y;
        t=t,
        xrange=xrange,
        yrange=yrange,
        binsize=binsize,
    )

    return merge(stat, (
        d_move=Float64(d_move),
        n_moving_matches=length(moving_matches),
        source=:result,
    ))
end

function local_vortex_statistics_from_matching_csv(
    peaks_before,
    matching_csv_path::AbstractString;
    t::Int = 2,
    d_move::Real = 1.0,
    xrange::Tuple{<:Real,<:Real} = (400.0, 1000.0),
    yrange::Tuple{<:Real,<:Real} = (300.0, 1000.0),
    binsize::Real = 50.0,
)
    df = CSV.read(matching_csv_path, DataFrame)

    required = [:kind, :x1, :y1, :dr]
    missing_cols = [c for c in required if !(c in propertynames(df))]
    isempty(missing_cols) ||
        error("Matching CSV is missing columns: " * join(string.(missing_cols), ", "))

    moving_mask = [
        lowercase(string(k)) != "static" &&
        isfinite(Float64(dr)) &&
        Float64(dr) >= d_move
        for (k, dr) in zip(df.kind, df.dr)
    ]

    moving_df = df[moving_mask, :]
    moving_x = Float64.(moving_df.x1)
    moving_y = Float64.(moving_df.y1)

    stat = local_vortex_statistics_from_xy(
        peaks_before,
        moving_x,
        moving_y;
        t=t,
        xrange=xrange,
        yrange=yrange,
        binsize=binsize,
    )

    return merge(stat, (
        d_move=Float64(d_move),
        n_moving_matches=nrow(moving_df),
        source=:csv,
        matching_csv_path=String(matching_csv_path),
    ))
end

function print_local_vortex_statistics_global(stat)
    ny, nx = size(stat.total_count)
    println("t = ", stat.t)
    println("total vortices in ROI  = ", stat.n_total_in_roi)
    println("moving vortices in ROI = ", stat.n_moving_in_roi)
    println()
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

function write_local_vortex_statistics_global_csv(path::AbstractString, stat)
    outdir = dirname(path)
    if !isempty(outdir) && outdir != "."
        mkpath(outdir)
    end

    open(path, "w") do io
        println(
            io,
            "t,ix,iy,xmin_um,xmax_um,ymin_um,ymax_um,total_vortices,moving_vortices,moving_fraction,moving_percent",
        )

        ny, nx = size(stat.total_count)
        for iy in 1:ny, ix in 1:nx
            frac = stat.moving_fraction[iy, ix]
            percent = isfinite(frac) ? 100 * frac : NaN
            println(io, join((
                stat.t,
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
            ), ","))
        end
    end
    return path
end

function plot_local_vortex_statistics_global(
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
            ax_total, x, y;
            text=string(stat.total_count[iy, ix]),
            align=(:center, :center),
            color=:white,
        )

        text!(
            ax_moving, x, y;
            text=string(stat.moving_count[iy, ix]),
            align=(:center, :center),
            color=:white,
        )

        frac = stat.moving_fraction[iy, ix]
        frac_text = isfinite(frac) ? string(round(100 * frac; digits=1)) : "-"

        text!(
            ax_fraction, x, y;
            text=frac_text,
            align=(:center, :center),
            color=:white,
        )
    end

    frame_text = "frame transition t=$(stat.t)"
    Label(
        fig[0, 1:3],
        isempty(title_prefix) ? frame_text : title_prefix * " | " * frame_text;
        fontsize=22,
    )

    if outpath !== nothing
        outdir = dirname(outpath)
        if !isempty(outdir) && outdir != "."
            mkpath(outdir)
        end
        save(outpath, fig)
    end

    return fig
end

function analyze_local_vortex_fraction_from_result(
    peaks_before,
    result;
    t::Int=2,
    d_move::Real=1.0,
    xrange::Tuple{<:Real,<:Real}=(400.0, 1000.0),
    yrange::Tuple{<:Real,<:Real}=(300.0, 1000.0),
    binsize::Real=50.0,
    out_csv::Union{Nothing,AbstractString}=nothing,
    out_png::Union{Nothing,AbstractString}=nothing,
    title_prefix::AbstractString="",
    yreversed::Bool=false,
)
    stat = local_vortex_statistics_from_result(
        peaks_before,
        result;
        t=t,
        d_move=d_move,
        xrange=xrange,
        yrange=yrange,
        binsize=binsize,
    )

    if out_csv !== nothing
        write_local_vortex_statistics_global_csv(out_csv, stat)
        println("CSV saved to = ", out_csv)
    end

    if out_png !== nothing
        plot_local_vortex_statistics_global(
            stat;
            outpath=out_png,
            title_prefix=title_prefix,
            yreversed=yreversed,
        )
        println("figure saved to = ", out_png)
    end

    return stat
end

function analyze_local_vortex_fraction_from_csv(
    peaks_before,
    matching_csv_path::AbstractString;
    t::Int=2,
    d_move::Real=1.0,
    xrange::Tuple{<:Real,<:Real}=(400.0, 1000.0),
    yrange::Tuple{<:Real,<:Real}=(300.0, 1000.0),
    binsize::Real=50.0,
    out_csv::Union{Nothing,AbstractString}=nothing,
    out_png::Union{Nothing,AbstractString}=nothing,
    title_prefix::AbstractString="",
    yreversed::Bool=false,
)
    stat = local_vortex_statistics_from_matching_csv(
        peaks_before,
        matching_csv_path;
        t=t,
        d_move=d_move,
        xrange=xrange,
        yrange=yrange,
        binsize=binsize,
    )

    if out_csv !== nothing
        write_local_vortex_statistics_global_csv(out_csv, stat)
        println("CSV saved to = ", out_csv)
    end

    if out_png !== nothing
        plot_local_vortex_statistics_global(
            stat;
            outpath=out_png,
            title_prefix=title_prefix,
            yreversed=yreversed,
        )
        println("figure saved to = ", out_png)
    end

    return stat
end

# ============================================================
# Example for your current loop
# ============================================================

#=
for i = 1:Int(N / 3)

    analysis_folder_path =
        joinpath("B/peak/analysis_0826", filename[3 * i])

    matching_csv_path =
        joinpath(
            analysis_folder_path,
            "vortex_matching_frame001.csv",
        )

    local_csv_path =
        joinpath(
            analysis_folder_path,
            "local_vortex_statistics_50um.csv",
        )

    local_png_path =
        joinpath(
            analysis_folder_path,
            "local_vortex_statistics_50um.png",
        )

    local_stat =
        analyze_local_vortex_fraction_from_csv(
            peaks[3 * i - 1],
            matching_csv_path;
            t = 2,
            d_move = 1.00,
            xrange = (400.0, 1000.0),
            yrange = (300.0, 1000.0),
            binsize = 50.0,
            out_csv = local_csv_path,
            out_png = local_png_path,
            title_prefix = filename[3 * i],
        )

    print_local_vortex_statistics_global(local_stat)
end


# If `result` is still in memory:
#
# local_stat =
#     analyze_local_vortex_fraction_from_result(
#         peaks[3 * i - 1],
#         result;
#         t = 2,
#         d_move = 1.00,
#         xrange = (400.0, 1000.0),
#         yrange = (300.0, 1000.0),
#         binsize = 50.0,
#         out_csv = local_csv_path,
#         out_png = local_png_path,
#     )
=#
