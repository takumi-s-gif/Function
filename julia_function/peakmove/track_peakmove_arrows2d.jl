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

include("../B_calculation.jl")

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

    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1]; xlabel="x", ylabel="y", aspect=DataAspect())

    scatter!(ax , xy[:, 1], xy[:, 2]; markersize=6, color=(:black, 1), label="all vortices")

    if !isempty(moving)
        inds = thin_indices(length(moving), max_arrows)
        dsel = moving[inds]

        xs = [d.x for d in dsel]
        ys = [d.y for d in dsel]
        dxs = [d.dx for d in dsel]
        dys = [d.dy for d in dsel]

        scatter!(ax, xs, ys; markersize=6, color=:red, label="moving vortices")
        points = Point2f.(xs, ys)
        vectors = Vec2f.(dxs, dys)
        arrows2d!(ax, points, vectors; lengthscale=arrow_scale, align=:tail, tipwidth=10, tiplength=10, shaftwidth=3, color=:red)
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

function heatmap_with_movevector(
    Bmat,
    result;
    outpath::AbstractString,
    frame_label::AbstractString = "",
    arrow_scale::Real = 1.0,
    max_arrows::Int = 2000,
    xlim = nothing,
    ylim = nothing,
    colorrange = (-50, 50),
    px_um::Real = 0.65,
    # colormap = :balance,
    colormap = cgrad([:blue, :white, :red])
)
    moving = result.moving

    ny, nx = size(Bmat)

    # Bmat が 1024 x 1024 で、1 pixel = 0.65 um の場合
    xs_B = collect(0:nx-1) .* px_um
    ys_B = collect(0:ny-1) .* px_um

    fig = Figure(size = (800, 800))

    ax = Axis(
        fig[1, 1];
        xlabel = "x (μm)",
        ylabel = "y (μm)",
        aspect = DataAspect(),
        yreversed = true,
    )

    Bmat_plot = reverse(Bmat, dims = 1)
    hm = heatmap!(
        ax,
        xs_B,
        ys_B,
        permutedims(Float64.(Bmat_plot));
        colormap = colormap,
        colorrange = colorrange,
    )

    Colorbar(fig[1, 2], hm)

    if !isempty(moving)
        inds = thin_indices(length(moving), max_arrows)
        dsel = moving[inds]

        xs = [d.x for d in dsel]
        ys = [d.y for d in dsel]
        dxs = [d.dx for d in dsel]
        dys = [d.dy for d in dsel]

        points = Point2f.(xs, ys)
        vectors = Vec2f.(dxs, dys)

        # 黒い矢印
        arrows2d!(
            ax,
            points,
            vectors;
            # lengthscale = arrow_scale,
            lengthscale = 1.0,
            align = :tail,
            tipwidth = 20,
            tiplength = 20,
            shaftwidth = 10,
            color = :black,
        )
    end

    if xlim !== nothing && ylim !== nothing
        limits!(ax, xlim[1], xlim[2], ylim[1], ylim[2])
    end

    title_text = isempty(frame_label) ? "moving vortex vectors" : frame_label
    ax.title = title_text * " | moving = $(length(moving))"

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

    fig = Figure(size=(800, 800))
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
        points = Point2f.(xs, ys)
        vectors = Vec2f.(vxs, vys)
        arrows2d!(ax, points, vectors; lengthscale=arrow_scale, align=:tail, tipwidth=12, tiplength=12, shaftwidth=4, color=:blue)
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
    ax = Axis(fig[1, 1]; xlabel="displacement (um)", ylabel="count")
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
    peaks,
    diff_Bmat;
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
            heatmap_with_movevector(
                diff_Bmat,
                result;
                outpath=joinpath(outdir, "heatmap_with_movevector_$(lpad(t, 4, '0')).png"),
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

#=
このスクリプトの出力は、大きく分けると

1. **数値データ**
2. **確認用ヒストグラム**
3. **各フレーム間の moving vortex ベクトル図**
4. **局所平均速度場**
5. **標準出力のログ**

です。

以下でそれぞれ説明します。

---

# 1. 出力フォルダ

実行時に

```julia
analysis = run_moving_vortex_analysis(
    peaks;
    outdir = "moving_vortex_output",
    ...
)
```

とした場合、結果は

```text
moving_vortex_output/
```

の中に保存されます。

主な出力は以下です。

```text
moving_vortex_output/
├── moving_vortex_analysis.jld2
├── moving_vortex_vectors.csv
├── displacement_histogram.png
├── moving_vectors_frame_0001.png
├── moving_vectors_frame_0002.png
├── ...
├── local_velocity_frame_0001.png
├── local_velocity_frame_0002.png
├── ...
└── local_velocity_all_frames.png
```

---

# 2. `moving_vortex_analysis.jld2`

これは、解析結果をJuliaで再利用するためのJLD2ファイルです。

中には

```julia
analysis
```

という変数が保存されています。

読み込むには

```julia
using JLD2

analysis = load("moving_vortex_output/moving_vortex_analysis.jld2", "analysis")
```

です。

---

## `analysis` の中身

`analysis` は NamedTuple で、主に2つの要素を持っています。

```julia
analysis.per_frame
analysis.all_moving
```

---

## `analysis.per_frame`

これは、各フレーム間の解析結果です。

例えば

```julia
analysis.per_frame[1]
```

は

```text
frame 1 -> frame 2
```

の対応付け結果です。

```julia
analysis.per_frame[2]
```

は

```text
frame 2 -> frame 3
```

です。

---

## `analysis.per_frame[t]` の中身

例えば

```julia
r = analysis.per_frame[1]
```

とすると、`r` の中には次のような情報が入っています。

```julia
r.t
r.n1
r.n2
r.static
r.candidates
r.moving
r.n_static
r.n_candidate
r.n_moving
```

それぞれの意味は以下です。

| 変数              | 意味                                   |
| --------------- | ------------------------------------ |
| `r.t`           | フレーム番号。`t -> t+1` の解析であることを表す        |
| `r.n1`          | フレーム `t` に存在した vortex 数              |
| `r.n2`          | フレーム `t+1` に存在した vortex 数            |
| `r.static`      | 静止していると判定された vortex 対応               |
| `r.candidates`  | 静止vortexを除いた後、対応づけられた vortex         |
| `r.moving`      | `candidates` のうち、実際に動いたと判定された vortex |
| `r.n_static`    | `static` の数                          |
| `r.n_candidate` | `candidates` の数                      |
| `r.n_moving`    | `moving` の数                          |

---

## `static`, `candidates`, `moving` の違い

このスクリプトでは、対応付けを3段階で行っています。

### `static`

まず、非常に近い距離にある vortex を

```julia
dr <= d_static
```

なら静止 vortex とみなします。

例えば

```julia
d_static = 0.20
```

なら、フレーム間の変位が `0.20` 以下の vortex は「ほぼ動いていない」と判定されます。

これが

```julia
r.static
```

に入ります。

---

### `candidates`

次に、静止 vortex を除外します。

その後、残った vortex 同士で対応付けを行います。

このとき、対応距離が

```julia
dr <= dmax
```

を満たすものだけを候補として残します。

これが

```julia
r.candidates
```

です。

つまり `candidates` は、

> 静止 vortex ではないが、まだ moving と確定していない対応候補

です。

---

### `moving`

最後に、`candidates` の中から

```julia
dr >= d_move
```

を満たすものだけを moving vortex とします。

これが

```julia
r.moving
```

です。

例えば

```julia
d_move = 0.50
```

なら、フレーム間で `0.50` 以上動いたものだけを moving vortex として扱います。

---

# 3. `analysis.all_moving`

これは、すべてのフレーム間の moving vortex をまとめたものです。

例えば、フレームが10枚ある場合、

```text
frame 1 -> 2
frame 2 -> 3
...
frame 9 -> 10
```

のすべての moving vortex が

```julia
analysis.all_moving
```

にまとめられています。

全体の moving vortex 数は

```julia
length(analysis.all_moving)
```

で確認できます。

---

## moving vortex 1個分の中身

例えば

```julia
d = analysis.all_moving[1]
```

とすると、次の情報が入っています。

```julia
d.t
d.i1
d.i2
d.x
d.y
d.x2
d.y2
d.dx
d.dy
d.dr
d.vx
d.vy
d.speed
```

それぞれの意味は以下です。

| 変数        | 意味                                |
| --------- | --------------------------------- |
| `d.t`     | フレーム `t -> t+1` の移動であることを表す       |
| `d.i1`    | フレーム `t` における vortex の index      |
| `d.i2`    | フレーム `t+1` における対応先 vortex の index |
| `d.x`     | 移動前の x 座標                         |
| `d.y`     | 移動前の y 座標                         |
| `d.x2`    | 移動後の x 座標                         |
| `d.y2`    | 移動後の y 座標                         |
| `d.dx`    | x方向変位、`x2 - x`                    |
| `d.dy`    | y方向変位、`y2 - y`                    |
| `d.dr`    | 変位の大きさ、`sqrt(dx^2 + dy^2)`        |
| `d.vx`    | x方向速度、`dx / dt`                   |
| `d.vy`    | y方向速度、`dy / dt`                   |
| `d.speed` | 速度の大きさ、`dr / dt`                  |

座標の単位が µm なら、`dx`, `dy`, `dr` も µm です。

`dt` を秒で与えていれば、`vx`, `vy`, `speed` の単位は µm/s になります。

---

# 4. `moving_vortex_vectors.csv`

これは、moving vortex の変位ベクトルだけをまとめたCSVファイルです。

Excel、Python、Julia、Originなどで読み込めます。

中身の列は次の通りです。

```text
t,i1,i2,x,y,x2,y2,dx,dy,dr,vx,vy,speed
```

意味は `analysis.all_moving` と同じです。

| 列名      | 意味                     |
| ------- | ---------------------- |
| `t`     | フレーム `t -> t+1`        |
| `i1`    | 移動前フレームでの vortex index |
| `i2`    | 移動後フレームでの対応先 index     |
| `x`     | 移動前 x 座標               |
| `y`     | 移動前 y 座標               |
| `x2`    | 移動後 x 座標               |
| `y2`    | 移動後 y 座標               |
| `dx`    | x方向変位                  |
| `dy`    | y方向変位                  |
| `dr`    | 変位の大きさ                 |
| `vx`    | x方向速度                  |
| `vy`    | y方向速度                  |
| `speed` | 速度の大きさ                 |

例えばCSVをJuliaで読むなら、

```julia
using CSV, DataFrames

df = CSV.read("moving_vortex_output/moving_vortex_vectors.csv", DataFrame)
```

です。

---

# 5. `displacement_histogram.png`

これは、対応づけられた vortex の変位量 `dr` のヒストグラムです。

横軸は

```text
displacement
```

つまり

[
dr = \sqrt{dx^2 + dy^2}
]

です。

縦軸はカウントです。

---

## 青線：`d_static`

青い縦線は

```julia
d_static
```

を表します。

この線より左側、つまり

```julia
dr <= d_static
```

の vortex は、静止 vortex として扱われます。

---

## 赤線：`d_move`

赤い縦線は

```julia
d_move
```

を表します。

この線より右側、つまり

```julia
dr >= d_move
```

の vortex が moving vortex として採用されます。

---

## この図で見るべきこと

このヒストグラムは、しきい値が妥当か確認するために重要です。

理想的には、

```text
小さい dr の大きなピーク：位置決定誤差、または静止 vortex
大きい dr 側の裾・山：実際に動いた vortex
```

のように見えます。

もし静止ピークと moving 成分の間に谷があるなら、その谷付近を `d_move` にするとよいです。

例えば、

```text
dr = 0.0〜0.2 に大きなピーク
dr = 0.5 以降に長い裾
```

なら、

```julia
d_static = 0.2
d_move = 0.5
```

のような設定は自然です。

---

# 6. `moving_vectors_frame_0001.png`

これは、各フレーム間の moving vortex の変位ベクトル図です。

例えば

```text
moving_vectors_frame_0001.png
```

は

```text
frame 1 -> frame 2
```

の結果です。

```text
moving_vectors_frame_0002.png
```

は

```text
frame 2 -> frame 3
```

です。

---

## 図の見方

図には主に2種類の点・矢印が描かれます。

### 灰色の点

灰色の点は、そのフレームに存在するすべての vortex です。

具体的には、`peaks[t].x`, `peaks[t].y` の点です。

これは背景として表示しています。

---

### 赤い点

赤い点は、moving vortex と判定された vortex の移動前位置です。

つまり、`frame t` での位置です。

---

### 赤い矢印

赤い矢印は、moving vortex の変位ベクトルです。

始点は

```julia
(d.x, d.y)
```

で、矢印の向きと長さは

```julia
(d.dx, d.dy)
```

です。

つまり、矢印は

```text
frame t での位置 → frame t+1 での位置
```

を表します。

---

## `arrow_scale` について

図を作るときに

```julia
arrow_scale = 1.0
```

を指定しています。

これは矢印の表示スケールです。

例えば、矢印が短すぎて見えにくい場合は、

```julia
arrow_scale = 3.0
```

のようにすると、矢印が3倍長く表示されます。

ただし、これは**表示上の拡大率**です。
実際の数値データ `dx`, `dy`, `dr` は変わりません。

---

## `max_arrows` について

moving vortex が多すぎると図が見にくくなるため、

```julia
max_arrows = 2000
```

で、最大表示矢印数を制限しています。

例えば moving vortex が5000個ある場合、すべて描くのではなく、間引いて最大2000本だけ描きます。

これも表示上の都合であり、CSVやJLD2にはすべての moving vortex が保存されています。

---

# 7. `local_velocity_frame_0001.png`

これは、moving vortex だけから作った**局所平均速度場**です。

例えば

```text
local_velocity_frame_0001.png
```

は

```text
frame 1 -> frame 2
```

の moving vortex だけを使った局所速度場です。

---

## 何をしているか

個々の moving vortex の速度

[
v_x = \frac{dx}{dt}, \quad v_y = \frac{dy}{dt}
]

をそのまま全部描くと、密度が高い場合に見にくくなります。

そこで、視野を格子状に分割します。

例えば

```julia
binsize = 10.0
```

なら、10 µm × 10 µm の領域ごとに区切ります。

その各領域内で、moving vortex の速度を平均します。

[
\langle v_x \rangle, \quad \langle v_y \rangle
]

その平均速度を青い矢印として描いています。

---

## 図の見方

### 灰色の点

そのフレームに存在するすべての vortex です。

### 青い点

局所速度を計算した格子セルの中心です。

### 青い矢印

その領域内の moving vortex の平均速度ベクトルです。

つまり、

```text
その局所領域で、moving vortex が平均的にどちらへ流れているか
```

を表します。

---

## `binsize` の意味

```julia
binsize = 10.0
```

は、局所平均を取る空間サイズです。

小さくすると、細かい構造が見えます。

```julia
binsize = 5.0
```

ただし、1セルあたりの vortex 数が少なくなるのでノイズが増えます。

大きくすると、滑らかな速度場になります。

```julia
binsize = 20.0
```

ただし、局所構造は見えにくくなります。

---

## `min_count` の意味

```julia
min_count = 3
```

は、

> 1つの格子セル内に moving vortex が最低何個あれば平均速度を表示するか

を決める値です。

例えば

```julia
min_count = 3
```

なら、そのセル内に moving vortex が3個以上ある場合だけ青い矢印を描きます。

1個や2個しかないセルは、統計的に不安定なので表示しません。

---

# 8. `local_velocity_all_frames.png`

これは、全フレーム間の moving vortex をまとめて作った局所平均速度場です。

つまり、

```julia
analysis.all_moving
```

を使って作っています。

---

## `local_velocity_frame_0001.png` との違い

`local_velocity_frame_0001.png` は、

```text
frame 1 -> frame 2 だけ
```

を見ています。

一方で、

```text
local_velocity_all_frames.png
```

は、

```text
frame 1 -> 2
frame 2 -> 3
frame 3 -> 4
...
```

をすべてまとめています。

そのため、時間平均された流れを見る図になります。

---

## 何に使うか

これは、

```text
全体として moving vortex がどの方向へ流れやすいか
どの領域が vortex の通り道になっているか
pinning が強い領域を避けているか
境界から入ってどちらへ進むか
```

を見るのに使えます。

一方で、時間ごとの変化は平均されて消えるので、

```text
ある瞬間だけの急激な動き
時間依存の流れの変化
```

を見たい場合は、`local_velocity_frame_XXXX.png` を見る方がよいです。

---

# 9. 実行時に表示されるログ

実行すると、ターミナルに例えば次のような表示が出ます。

```text
frame 1 -> 2 | n1=1024 n2=1018 static=860 candidate=120 moving=35
frame 2 -> 3 | n1=1018 n2=1031 static=850 candidate=132 moving=42
frame 3 -> 4 | n1=1031 n2=1009 static=870 candidate=110 moving=28
```

それぞれの意味は以下です。

| 表示             | 意味                 |
| -------------- | ------------------ |
| `frame 1 -> 2` | フレーム1からフレーム2への解析   |
| `n1`           | フレーム1の vortex 数    |
| `n2`           | フレーム2の vortex 数    |
| `static`       | 静止と判定された対応数        |
| `candidate`    | 静止を除いた後、対応付けられた候補数 |
| `moving`       | 実際に moving と判定された数 |

このログを見ると、

```text
各フレームでどれくらいの vortex が静止しているか
どれくらいが moving と判定されているか
フレームによって異常に数が増減していないか
```

を確認できます。

---

# 10. 対応付けされなかった vortex について

このスクリプトでは、すべての vortex を無理に対応づけません。

対応づけの条件を満たさないものは捨てています。

捨てられる例は、

```text
視野外から入ってきた vortex
視野外へ出ていった vortex
近傍に複数候補があり曖昧な vortex
dmax より遠くへ移動している vortex
mutual nearest neighbor を満たさない vortex
```

です。

これは意図的です。

無理にすべてを対応づけると、別の vortex に誤対応してしまう危険が高くなります。

このスクリプトでは、

```text
信頼できる対応だけを使って moving vortex の速度場を見る
```

という方針にしています。

---

# 11. 一番重要な確認ポイント

まず見るべき順番はこの順です。

```text
1. displacement_histogram.png
2. moving_vectors_frame_XXXX.png
3. local_velocity_frame_XXXX.png
4. local_velocity_all_frames.png
5. moving_vortex_vectors.csv
```

特に最初は、`displacement_histogram.png` を見て、

```julia
d_static
d_move
dmax
```

が妥当か確認するのが重要です。

もし moving が多すぎるなら、

```julia
d_move
```

を大きくします。

もし moving が少なすぎるなら、

```julia
d_move
```

を小さくします。

もし明らかに離れた vortex が対応づけられているなら、

```julia
dmax
```

を小さくします。

逆に、本当に大きく動く vortex が取りこぼされているなら、

```julia
dmax
```

を大きくします。

=#