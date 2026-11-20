# moving_vortex_vectors.jl
#
# Purpose:
#   Extract and visualize displacement vectors only for vortices that are judged to be moving.
#
# Required packages:
#   import Pkg
#   Pkg.add(["JLD2", "NearestNeighbors", "CairoMakie"])
#
# Expected peak data format:
#   peaks[t].x, peaks[t].y
#   or
#   peaks[t]["x"], peaks[t]["y"]
#
# Units:
#   d_static, d_move, dmax, x, y must be in the same unit.
#   If peaks.x and peaks.y are in um, then d_static, d_move, dmax are also in um.

using JLD2
using NearestNeighbors
using Statistics
using CairoMakie

# -----------------------------
# Data types
# -----------------------------

struct VortexMatch
    i::Int
    j::Int
    d::Float64
end

struct VortexDisplacement
    t::Int
    i1::Int
    i2::Int
    x::Float64
    y::Float64
    x2::Float64
    y2::Float64
    dx::Float64
    dy::Float64
    dr::Float64
    vx::Float64
    vy::Float64
    speed::Float64
end

# -----------------------------
# Basic utilities
# -----------------------------

function peak_get(p, name::Symbol)
    if hasproperty(p, name)
        return getproperty(p, name)
    elseif p isa AbstractDict && haskey(p, String(name))
        return p[String(name)]
    elseif p isa AbstractDict && haskey(p, name)
        return p[name]
    else
        error("Peak object does not have field/key: $(name)")
    end
end

function peak_xy(p)
    x = Float64.(peak_get(p, :x))
    y = Float64.(peak_get(p, :y))
    @assert length(x) == length(y) "length(x) and length(y) are different."
    return hcat(x, y)  # Npeak x 2
end

function load_peak_series(peak_jld_paths::Vector{<:AbstractString})
    @assert !isempty(peak_jld_paths) "peak_jld_paths is empty."
    return [load(path, "peaks") for path in peak_jld_paths]
end

function common_xy_limits(peaks; margin_fraction=0.03)
    xmin = Inf
    xmax = -Inf
    ymin = Inf
    ymax = -Inf

    for p in peaks
        xy = peak_xy(p)
        if size(xy, 1) == 0
            continue
        end
        xmin = min(xmin, minimum(xy[:, 1]))
        xmax = max(xmax, maximum(xy[:, 1]))
        ymin = min(ymin, minimum(xy[:, 2]))
        ymax = max(ymax, maximum(xy[:, 2]))
    end

    @assert isfinite(xmin) "No valid peak positions were found."

    dx = xmax - xmin
    dy = ymax - ymin
    mx = dx == 0 ? 1.0 : margin_fraction * dx
    my = dy == 0 ? 1.0 : margin_fraction * dy

    return (xmin - mx, xmax + mx), (ymin - my, ymax + my)
end

function thin_indices(n::Int, max_n::Int)
    if n <= max_n
        return collect(1:n)
    end
    return unique(round.(Int, range(1, n; length=max_n)))
end

# -----------------------------
# Matching
# -----------------------------

function mutual_nearest_matches(
    xy1::AbstractMatrix,
    xy2::AbstractMatrix;
    dmax::Real,
    idxs1=nothing,
    idxs2=nothing,
)
    n1 = size(xy1, 1)
    n2 = size(xy2, 1)

    if idxs1 === nothing
        idxs1 = collect(1:n1)
    end
    if idxs2 === nothing
        idxs2 = collect(1:n2)
    end

    if isempty(idxs1) || isempty(idxs2)
        return VortexMatch[]
    end

    sub1 = xy1[idxs1, :]
    sub2 = xy2[idxs2, :]

    # NearestNeighbors.jl expects data as dimension x number_of_points.
    tree2 = KDTree(permutedims(sub2))
    tree1 = KDTree(permutedims(sub1))

    idx12, dist12 = knn(tree2, permutedims(sub1), 1, true)
    idx21, _      = knn(tree1, permutedims(sub2), 1, true)

    matches = VortexMatch[]

    for a in eachindex(idxs1)
        b = idx12[a][1]
        d = Float64(dist12[a][1])

        # Mutual nearest neighbor condition.
        # xy1[a] -> xy2[b] and xy2[b] -> xy1[a]
        if idx21[b][1] == a && d <= dmax
            push!(matches, VortexMatch(idxs1[a], idxs2[b], d))
        end
    end

    return matches
end

function matches_to_displacements(matches, xy1, xy2; t::Int, dt::Real)
    disps = VortexDisplacement[]

    for m in matches
        x1 = xy1[m.i, 1]
        y1 = xy1[m.i, 2]
        x2 = xy2[m.j, 1]
        y2 = xy2[m.j, 2]

        dx = x2 - x1
        dy = y2 - y1
        dr = hypot(dx, dy)

        push!(disps, VortexDisplacement(
            t,
            m.i,
            m.j,
            x1,
            y1,
            x2,
            y2,
            dx,
            dy,
            dr,
            dx / dt,
            dy / dt,
            dr / dt,
        ))
    end

    return disps
end

function displacement_between_frames(
    p1,
    p2;
    t::Int=1,
    dt::Real=1.0,
    d_static::Real=0.20,
    d_move::Real=0.50,
    dmax::Real=3.00,
)
    @assert d_static <= d_move <= dmax "Require d_static <= d_move <= dmax."

    xy1 = peak_xy(p1)
    xy2 = peak_xy(p2)
    n1 = size(xy1, 1)
    n2 = size(xy2, 1)

    # Step 1:
    # First identify stationary vortices using a very small threshold.
    # These are removed before looking for moving vortices.
    static_matches = mutual_nearest_matches(xy1, xy2; dmax=d_static)
    static_disps = matches_to_displacements(static_matches, xy1, xy2; t=t, dt=dt)

    static_i = Set(m.i for m in static_matches)
    static_j = Set(m.j for m in static_matches)

    idxs1 = [i for i in 1:n1 if !(i in static_i)]
    idxs2 = [j for j in 1:n2 if !(j in static_j)]

    # Step 2:
    # Match only the non-stationary remaining vortices.
    # Ambiguous pairs that do not satisfy mutual nearest neighbor are discarded.
    candidate_matches = mutual_nearest_matches(xy1, xy2; dmax=dmax, idxs1=idxs1, idxs2=idxs2)
    candidate_disps = matches_to_displacements(candidate_matches, xy1, xy2; t=t, dt=dt)

    # Step 3:
    # Keep only vortices with displacement larger than d_move.
    moving_disps = [d for d in candidate_disps if d.dr >= d_move]

    return (; 
        t,
        n1,
        n2,
        static=static_disps,
        candidates=candidate_disps,
        moving=moving_disps,
        n_static=length(static_disps),
        n_candidate=length(candidate_disps),
        n_moving=length(moving_disps),
    )
end

function analyze_peak_series(
    peaks;
    dt::Real=1.0,
    d_static::Real=0.20,
    d_move::Real=0.50,
    dmax::Real=3.00,
    verbose::Bool=true,
)
    @assert length(peaks) >= 2 "Need at least two frames."

    per_frame = Vector{Any}(undef, length(peaks) - 1)
    all_moving = VortexDisplacement[]

    for t in 1:(length(peaks) - 1)
        result = displacement_between_frames(
            peaks[t],
            peaks[t + 1];
            t=t,
            dt=dt,
            d_static=d_static,
            d_move=d_move,
            dmax=dmax,
        )

        per_frame[t] = result
        append!(all_moving, result.moving)

        if verbose
            println(
                "frame ", t, " -> ", t + 1,
                " | n1=", result.n1,
                " n2=", result.n2,
                " static=", result.n_static,
                " candidate=", result.n_candidate,
                " moving=", result.n_moving,
            )
        end
    end

    return (; per_frame, all_moving)
end

# -----------------------------
# Local velocity field
# -----------------------------

function local_velocity_field(
    disps;
    xrange=nothing,
    yrange=nothing,
    binsize::Real=10.0,
    min_count::Int=3,
)
    if isempty(disps)
        error("disps is empty. No moving vortex was found.")
    end

    xs = [d.x for d in disps]
    ys = [d.y for d in disps]

    if xrange === nothing
        xrange = (minimum(xs), maximum(xs))
    end
    if yrange === nothing
        yrange = (minimum(ys), maximum(ys))
    end

    xedges = collect(xrange[1]:binsize:xrange[2])
    yedges = collect(yrange[1]:binsize:yrange[2])

    if length(xedges) < 2 || xedges[end] < xrange[2]
        push!(xedges, xedges[end] + binsize)
    end
    if length(yedges) < 2 || yedges[end] < yrange[2]
        push!(yedges, yedges[end] + binsize)
    end

    nx = length(xedges) - 1
    ny = length(yedges) - 1

    sumvx = zeros(Float64, ny, nx)
    sumvy = zeros(Float64, ny, nx)
    count = zeros(Int, ny, nx)

    for d in disps
        ix = searchsortedlast(xedges, d.x)
        iy = searchsortedlast(yedges, d.y)

        if 1 <= ix <= nx && 1 <= iy <= ny
            sumvx[iy, ix] += d.vx
            sumvy[iy, ix] += d.vy
            count[iy, ix] += 1
        end
    end

    vx_map = fill(NaN, ny, nx)
    vy_map = fill(NaN, ny, nx)
    speed_map = fill(NaN, ny, nx)

    for iy in 1:ny, ix in 1:nx
        if count[iy, ix] >= min_count
            vx_map[iy, ix] = sumvx[iy, ix] / count[iy, ix]
            vy_map[iy, ix] = sumvy[iy, ix] / count[iy, ix]
            speed_map[iy, ix] = hypot(vx_map[iy, ix], vy_map[iy, ix])
        end
    end

    xcenters = (xedges[1:end-1] .+ xedges[2:end]) ./ 2
    ycenters = (yedges[1:end-1] .+ yedges[2:end]) ./ 2

    return (; xedges, yedges, xcenters, ycenters, vx_map, vy_map, speed_map, count)
end

# -----------------------------
# Plotting
# -----------------------------

function plot_frame_vectors(
    p,
    result;
    outpath::AbstractString,
    frame_label::AbstractString="",
    arrow_scale::Real=1.0,
    max_arrows::Int=2000,
    xlim=nothing,
    ylim=nothing,
)
    xy = peak_xy(p)
    moving = result.moving

    fig = Figure(size=(900, 800))
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", aspect=DataAspect())

    scatter!(ax, xy[:, 1], xy[:, 2]; markersize=3, color=(:gray, 0.35), label="all vortices")

    if !isempty(moving)
        inds = thin_indices(length(moving), max_arrows)
        dsel = moving[inds]

        xs = [d.x for d in dsel]
        ys = [d.y for d in dsel]
        dxs = [d.dx for d in dsel]
        dys = [d.dy for d in dsel]

        scatter!(ax, xs, ys; markersize=6, color=:red, label="moving vortices")
        arrows!(ax, xs, ys, dxs, dys; lengthscale=arrow_scale, arrowsize=10, linewidth=1.5, color=:red)
    end

    if xlim !== nothing && ylim !== nothing
        limits!(ax, xlim[1], xlim[2], ylim[1], ylim[2])
    end

    title_text = isempty(frame_label) ? "moving vortex vectors" : frame_label
    ax.title = title_text * " | moving = $(length(moving))"
    axislegend(ax; position=:rt)

    save(outpath, fig)
    return fig
end

function plot_local_velocity_field(
    p,
    field;
    outpath::AbstractString,
    frame_label::AbstractString="",
    arrow_scale::Real=1.0,
    xlim=nothing,
    ylim=nothing,
)
    xy = peak_xy(p)

    fig = Figure(size=(900, 800))
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", aspect=DataAspect())

    scatter!(ax, xy[:, 1], xy[:, 2]; markersize=3, color=(:gray, 0.30), label="all vortices")

    xs = Float64[]
    ys = Float64[]
    vxs = Float64[]
    vys = Float64[]

    ny, nx = size(field.vx_map)
    for iy in 1:ny, ix in 1:nx
        vx = field.vx_map[iy, ix]
        vy = field.vy_map[iy, ix]
        if isfinite(vx) && isfinite(vy)
            push!(xs, field.xcenters[ix])
            push!(ys, field.ycenters[iy])
            push!(vxs, vx)
            push!(vys, vy)
        end
    end

    if !isempty(xs)
        arrows!(ax, xs, ys, vxs, vys; lengthscale=arrow_scale, arrowsize=12, linewidth=2.0, color=:blue)
        scatter!(ax, xs, ys; markersize=5, color=:blue, label="local mean velocity")
    end

    if xlim !== nothing && ylim !== nothing
        limits!(ax, xlim[1], xlim[2], ylim[1], ylim[2])
    end

    title_text = isempty(frame_label) ? "local velocity field of moving vortices" : frame_label
    ax.title = title_text
    axislegend(ax; position=:rt)

    save(outpath, fig)
    return fig
end

function plot_displacement_histogram(
    analysis;
    outpath::AbstractString,
    d_static::Real,
    d_move::Real,
    bins::Int=100,
)
    drs = Float64[]

    for result in analysis.per_frame
        append!(drs, [d.dr for d in result.static])
        append!(drs, [d.dr for d in result.candidates])
    end

    if isempty(drs)
        @warn "No displacement was found. Histogram was not created."
        return nothing
    end

    fig = Figure(size=(850, 500))
    ax = Axis(fig[1, 1]; xlabel="displacement", ylabel="count")
    hist!(ax, drs; bins=bins, color=(:gray, 0.65))
    vlines!(ax, [d_static]; color=:blue, linewidth=2, label="d_static")
    vlines!(ax, [d_move]; color=:red, linewidth=2, label="d_move")
    ax.title = "displacement histogram"
    axislegend(ax; position=:rt)
    save(outpath, fig)
    return fig
end

function write_displacements_csv(path::AbstractString, disps)
    open(path, "w") do io
        println(io, "t,i1,i2,x,y,x2,y2,dx,dy,dr,vx,vy,speed")
        for d in disps
            println(io, join((
                d.t, d.i1, d.i2,
                d.x, d.y, d.x2, d.y2,
                d.dx, d.dy, d.dr,
                d.vx, d.vy, d.speed,
            ), ","))
        end
    end
    return path
end

# -----------------------------
# Full pipeline
# -----------------------------

function run_moving_vortex_analysis(
    peaks;
    outdir::AbstractString="B/peak/analysis",
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
)
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

    # Frame-by-frame vector plots.
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

# -----------------------------
# Example main block
# -----------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    # Edit this part before running the script.
    # Example:
    # peak_jld_paths = [
    #     "/path/to/frame001_peaks.jld2",
    #     "/path/to/frame002_peaks.jld2",
    #     "/path/to/frame003_peaks.jld2",
    # ]

    peak_jld_paths = String[]

    if isempty(peak_jld_paths)
        @info "Edit peak_jld_paths in moving_vortex_vectors.jl, or include this file and call run_moving_vortex_analysis(peaks)."
    else
        peaks = load_peak_series(peak_jld_paths)

        run_moving_vortex_analysis(
            peaks;
            outdir="moving_vortex_output",
            dt=1.0,
            d_static=0.20,
            d_move=0.50,
            dmax=3.00,
            binsize=10.0,
            min_count=3,
            arrow_scale=1.0,
            field_arrow_scale=1.0,
            max_arrows=2000,
        )
    end
end
