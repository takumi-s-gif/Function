using CairoMakie

# ============================================================
# Utilities
# ============================================================

"""
    peak_xy(peaks)

Create an N x 2 coordinate matrix [x y] from the return value of
extract_peaks(...). Units are the same as peaks.x and peaks.y, usually um.
"""
function peak_xy(peaks)
    x = Float64.(peaks.x)
    y = Float64.(peaks.y)

    @assert length(x) == length(y)

    return hcat(x, y)
end


# ============================================================
# Candidate pairs within a given radius
#
# Avoid constructing a huge full distance matrix. Only pairs
# within radius are collected using spatial cells.
# ============================================================

function candidate_graph(
    xy1::AbstractMatrix,
    xy2::AbstractMatrix;
    radius::Real,
)
    @assert radius > 0

    n1 = size(xy1, 1)
    n2 = size(xy2, 1)

    row_to_cols = [Int[] for _ in 1:n1]
    col_to_rows = [Int[] for _ in 1:n2]

    d2map = Dict{Tuple{Int,Int}, Float64}()

    cellsize = Float64(radius)

    cells = Dict{Tuple{Int,Int}, Vector{Int}}()

    for j in 1:n2
        cx = floor(Int, xy2[j, 1] / cellsize)
        cy = floor(Int, xy2[j, 2] / cellsize)

        push!(get!(cells, (cx, cy), Int[]), j)
    end

    r2 = Float64(radius)^2

    for i in 1:n1
        x1 = xy1[i, 1]
        y1 = xy1[i, 2]

        cx = floor(Int, x1 / cellsize)
        cy = floor(Int, y1 / cellsize)

        for dxcell in -1:1, dycell in -1:1
            key = (cx + dxcell, cy + dycell)

            js = get(cells, key, nothing)
            js === nothing && continue

            for j in js
                dx = xy2[j, 1] - x1
                dy = xy2[j, 2] - y1
                d2 = dx^2 + dy^2

                if d2 <= r2
                    push!(row_to_cols[i], j)
                    push!(col_to_rows[j], i)
                    d2map[(i, j)] = d2
                end
            end
        end
    end

    return row_to_cols, col_to_rows, d2map
end


# ============================================================
# Step 1:
# Mutual row/column minimum + d <= d_static
# ============================================================

function static_mutual_matches(
    xy1::AbstractMatrix,
    xy2::AbstractMatrix;
    d_static::Real,
    tie_tol::Real = 1e-12,
)
    row_to_cols, col_to_rows, d2map =
        candidate_graph(xy1, xy2; radius=d_static)

    n1 = size(xy1, 1)
    n2 = size(xy2, 1)

    row_best = zeros(Int, n1)
    col_best = zeros(Int, n2)

    # Row minimum
    for i in 1:n1
        js = row_to_cols[i]
        isempty(js) && continue

        vals = [d2map[(i, j)] for j in js]
        dmin = minimum(vals)

        candidates = [
            j for j in js
            if abs(d2map[(i, j)] - dmin) <= tie_tol
        ]

        # If there is a tie, do not fix it as static.
        if length(candidates) == 1
            row_best[i] = candidates[1]
        end
    end

    # Column minimum
    for j in 1:n2
        is = col_to_rows[j]
        isempty(is) && continue

        vals = [d2map[(i, j)] for i in is]
        dmin = minimum(vals)

        candidates = [
            i for i in is
            if abs(d2map[(i, j)] - dmin) <= tie_tol
        ]

        if length(candidates) == 1
            col_best[j] = candidates[1]
        end
    end

    matches = Tuple{Int,Int,Float64}[]

    for i in 1:n1
        j = row_best[i]

        if j != 0 && col_best[j] == i
            d2 = d2map[(i, j)]

            if d2 <= d_static^2
                push!(matches, (i, j, d2))
            end
        end
    end

    return matches
end


# ============================================================
# Hungarian algorithm for minimum-cost assignment
# ============================================================

function hungarian_min(cost::Matrix{Float64})
    n, m = size(cost)

    @assert n <= m

    u = zeros(Float64, n + 1)
    v = zeros(Float64, m + 1)

    p = zeros(Int, m + 1)
    way = zeros(Int, m + 1)

    for i in 1:n
        p[1] = i
        j0 = 1

        minv = fill(Inf, m + 1)
        used = falses(m + 1)

        while true
            used[j0] = true
            i0 = p[j0]

            delta = Inf
            j1 = 0

            for j in 2:(m + 1)
                if !used[j]
                    cur =
                        cost[i0, j - 1] -
                        u[i0 + 1] -
                        v[j]

                    if cur < minv[j]
                        minv[j] = cur
                        way[j] = j0
                    end

                    if minv[j] < delta
                        delta = minv[j]
                        j1 = j
                    end
                end
            end

            for j in 1:(m + 1)
                if used[j]
                    u[p[j] + 1] += delta
                    v[j] -= delta
                elseif j >= 2
                    minv[j] -= delta
                end
            end

            j0 = j1

            if p[j0] == 0
                break
            end
        end

        while true
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

            j0 == 1 && break
        end
    end

    assignment = zeros(Int, n)

    for j in 2:(m + 1)
        if p[j] != 0
            assignment[p[j]] = j - 1
        end
    end

    return assignment
end


# ============================================================
# Split dmax candidate graph into independent bipartite
# connected components.
# ============================================================

function bipartite_components(row_to_cols, col_to_rows)
    nrow = length(row_to_cols)
    ncol = length(col_to_rows)

    seen_row = falses(nrow)
    seen_col = falses(ncol)

    components = Vector{Tuple{Vector{Int},Vector{Int}}}()

    for start_row in 1:nrow
        seen_row[start_row] && continue
        isempty(row_to_cols[start_row]) && continue

        rows = Int[]
        cols = Int[]

        row_queue = [start_row]
        col_queue = Int[]

        seen_row[start_row] = true

        while !isempty(row_queue) || !isempty(col_queue)
            while !isempty(row_queue)
                i = popfirst!(row_queue)
                push!(rows, i)

                for j in row_to_cols[i]
                    if !seen_col[j]
                        seen_col[j] = true
                        push!(col_queue, j)
                    end
                end
            end

            while !isempty(col_queue)
                j = popfirst!(col_queue)
                push!(cols, j)

                for i in col_to_rows[j]
                    if !seen_row[i]
                        seen_row[i] = true
                        push!(row_queue, i)
                    end
                end
            end
        end

        push!(components, (rows, cols))
    end

    return components
end


# ============================================================
# Solve one local cluster by minimizing total squared
# displacement while allowing unmatched vortices.
# ============================================================

function solve_component_assignment(
    rows::Vector{Int},
    cols::Vector{Int},
    d2map;
    dmax::Real,
)
    nr = length(rows)
    nc = length(cols)

    N = nr + nc

    dmax2 = Float64(dmax)^2
    BIG = max(1.0, dmax2) * 1e9

    # One real-real match at exactly dmax has the same cost
    # as leaving both vortices unmatched.
    unmatched_cost = dmax2 / 2

    C = fill(BIG, N, N)

    # Real row -> real column
    for (a, i) in enumerate(rows)
        for (b, j) in enumerate(cols)
            d2 = get(d2map, (i, j), BIG)

            if d2 <= dmax2
                C[a, b] = d2
            end
        end
    end

    # Real row -> its dummy column
    for a in 1:nr
        C[a, nc + a] = unmatched_cost
    end

    # Dummy row -> real column
    for b in 1:nc
        C[nr + b, b] = unmatched_cost
    end

    # Dummy -> dummy
    C[(nr + 1):N, (nc + 1):N] .= 0.0

    assignment = hungarian_min(C)

    matches = Tuple{Int,Int,Float64}[]

    for a in 1:nr
        b = assignment[a]

        if 1 <= b <= nc
            i = rows[a]
            j = cols[b]

            d2 = get(d2map, (i, j), Inf)

            if d2 <= dmax2
                push!(matches, (i, j, d2))
            end
        end
    end

    return matches
end


# ============================================================
# Full matching
# ============================================================

function match_vortices(
    peaks1,
    peaks2;
    d_static::Real = 0.20,
    dmax::Real = 3.00,
    dmax_retry::Real = dmax,
)
    @assert 0 <= d_static < dmax
    @assert dmax_retry >= dmax

    xy1 = peak_xy(peaks1)
    xy2 = peak_xy(peaks2)

    n1 = size(xy1, 1)
    n2 = size(xy2, 1)

    # Stage 1: static vortices
    static_raw =
        static_mutual_matches(
            xy1,
            xy2;
            d_static=d_static,
        )

    static_i = Set(i for (i, _, _) in static_raw)
    static_j = Set(j for (_, j, _) in static_raw)

    # Stage 2: remove static vortices
    rem1 = [
        i for i in 1:n1
        if !(i in static_i)
    ]

    rem2 = [
        j for j in 1:n2
        if !(j in static_j)
    ]

    xy1_rem = xy1[rem1, :]
    xy2_rem = xy2[rem2, :]

    # Stage 3: candidate pairs within dmax
    row_to_cols, col_to_rows, d2map =
        candidate_graph(
            xy1_rem,
            xy2_rem;
            radius=dmax,
        )

    components =
        bipartite_components(
            row_to_cols,
            col_to_rows,
        )

    optimized_raw = Tuple{Int,Int,Float64}[]

    for (rows, cols) in components
        local_matches =
            solve_component_assignment(
                rows,
                cols,
                d2map;
                dmax=dmax,
            )

        for (ir, jr, d2) in local_matches
            i = rem1[ir]
            j = rem2[jr]

            push!(optimized_raw, (i, j, d2))
        end
    end

    function make_match(kind, i, j, d2)
        x1 = xy1[i, 1]
        y1 = xy1[i, 2]

        x2 = xy2[j, 1]
        y2 = xy2[j, 2]

        dx = x2 - x1
        dy = y2 - y1

        return (
            kind=kind,
            i1=i,
            i2=j,
            peak_id1=peaks1.peak_ids[i],
            peak_id2=peaks2.peak_ids[j],
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            dx=dx,
            dy=dy,
            dr=sqrt(d2),
            d2=d2,
        )
    end

    static_matches = [
        make_match(:static, i, j, d2)
        for (i, j, d2) in static_raw
    ]

    optimized_matches = [
        make_match(:optimized, i, j, d2)
        for (i, j, d2) in optimized_raw
    ]

    # ========================================================
    # Stage 4: retry unmatched vortices only
    # ========================================================

    first_matches = vcat(
        static_matches,
        optimized_matches,
    )

    used1_first = Set(m.i1 for m in first_matches)
    used2_first = Set(m.i2 for m in first_matches)

    unmatched1_first = [
        i for i in 1:n1
        if !(i in used1_first)
    ]

    unmatched2_first = [
        j for j in 1:n2
        if !(j in used2_first)
    ]

    retry_raw = Tuple{Int,Int,Float64}[]

    if !isempty(unmatched1_first) && !isempty(unmatched2_first)
        xy1_retry = xy1[unmatched1_first, :]
        xy2_retry = xy2[unmatched2_first, :]

        row_to_cols_retry, col_to_rows_retry, d2map_retry =
            candidate_graph(
                xy1_retry,
                xy2_retry;
                radius=dmax_retry,
            )

        components_retry =
            bipartite_components(
                row_to_cols_retry,
                col_to_rows_retry,
            )

        for (rows, cols) in components_retry
            local_matches =
                solve_component_assignment(
                    rows,
                    cols,
                    d2map_retry;
                    dmax=dmax_retry,
                )

            for (ir, jr, d2) in local_matches
                i = unmatched1_first[ir]
                j = unmatched2_first[jr]
                push!(retry_raw, (i, j, d2))
            end
        end
    end

    retry_matches = [
        make_match(:retry, i, j, d2)
        for (i, j, d2) in retry_raw
    ]

    matches = vcat(
        static_matches,
        optimized_matches,
        retry_matches,
    )

    used1 = Set(m.i1 for m in matches)
    used2 = Set(m.i2 for m in matches)

    unmatched1 = [
        i for i in 1:n1
        if !(i in used1)
    ]

    unmatched2 = [
        j for j in 1:n2
        if !(j in used2)
    ]

    return (
        matches=matches,
        static=static_matches,
        optimized=optimized_matches,
        retry=retry_matches,
        unmatched1=unmatched1,
        unmatched2=unmatched2,
        unmatched1_before_retry=unmatched1_first,
        unmatched2_before_retry=unmatched2_first,
        n1=n1,
        n2=n2,
        n_static=length(static_matches),
        n_optimized=length(optimized_matches),
        n_retry=length(retry_matches),
        n_matched=length(matches),
        d_static=Float64(d_static),
        dmax=Float64(dmax),
        dmax_retry=Float64(dmax_retry),
    )
end


# ============================================================
# Save matching result as CSV
# ============================================================

function save_matching_csv(path, result)
    open(path, "w") do io
        println(
            io,
            "kind,i1,i2,peak_id1,peak_id2,x1,y1,x2,y2,dx,dy,dr,d2"
        )

        for m in result.matches
            println(
                io,
                join(
                    (
                        m.kind,
                        m.i1,
                        m.i2,
                        m.peak_id1,
                        m.peak_id2,
                        m.x1,
                        m.y1,
                        m.x2,
                        m.y2,
                        m.dx,
                        m.dy,
                        m.dr,
                        m.d2,
                    ),
                    ",",
                )
            )
        end
    end

    return path
end


# ============================================================
# Plot:
# before = blue
# after  = red
# displacement = black arrows
# ============================================================

function plot_vortex_matching(
    peaks1,
    peaks2,
    result;
    outpath::AbstractString = "vortex_matching.png",
    title::AbstractString = "",
    markersize::Real = 8,
    arrow_scale::Real = 1.0,
    figure_size=(900, 850),
)
    xy1 = peak_xy(peaks1)
    xy2 = peak_xy(peaks2)

    fig = Figure(size=figure_size)

    ax = Axis(
        fig[1, 1];
        xlabel="x (um)",
        ylabel="y (um)",
        aspect=DataAspect(),
    )

    scatter!(
        ax,
        xy1[:, 1],
        xy1[:, 2];
        color=:blue,
        markersize=markersize+1.0,
        label="before",
    )

    scatter!(
        ax,
        xy2[:, 1],
        xy2[:, 2];
        color=:red,
        markersize=markersize,
        label="after",
    )

    # Draw arrows only for non-static matches.
    # Static vortices remain visible as blue/red points, but are not connected.
    arrow_matches = [
        m for m in result.matches
        if m.kind != :static
    ]

    if !isempty(arrow_matches)
        xs = [m.x1 for m in arrow_matches]
        ys = [m.y1 for m in arrow_matches]

        dxs = [m.dx for m in arrow_matches]
        dys = [m.dy for m in arrow_matches]

        points = Point2f.(xs, ys)

        vectors = Vec2f.(
            arrow_scale .* dxs,
            arrow_scale .* dys,
        )

        arrows2d!(
            ax,
            points,
            vectors;
            lengthscale=1.0,
            align=:tail,
            color=:black,
            shaftwidth=2.0,
            tipwidth=6,
            tiplength=6,
        )
    end

    if isempty(title)
        ax.title =
            "vortex matching | " *
            "before=$(result.n1), " *
            "after=$(result.n2), " *
            "static=$(result.n_static), " *
            "optimized=$(result.n_optimized), " *
            "retry=$(result.n_retry), " *
            "unmatched=$(length(result.unmatched1))/$(length(result.unmatched2))"
    else
        ax.title = title
    end

    axislegend(ax; position=:rt)

    root, ext = splitext(outpath)

    if isempty(ext)
        outpath = outpath * ".png"
        root = splitext(outpath)[1]
    end

    outdir = dirname(outpath)
    if !isempty(outdir) && outdir != "."
        mkpath(outdir)
    end

    save(outpath, fig)

    csvpath = root * ".csv"
    save_matching_csv(csvpath, result)

    return fig, outpath, csvpath
end


# ============================================================
# One-call wrapper
# ============================================================

function match_plot_save(
    peaks1,
    peaks2;
    d_static::Real = 0.20,
    dmax::Real = 3.00,
    dmax_retry::Real = dmax,
    outpath::AbstractString = "vortex_matching.png",
    arrow_scale::Real = 1.0,
    markersize::Real = 3,
    title::AbstractString = "",
)
    result =
        match_vortices(
            peaks1,
            peaks2;
            d_static=d_static,
            dmax=dmax,
            dmax_retry=dmax_retry,
        )

    fig, image_path, csv_path =
        plot_vortex_matching(
            peaks1,
            peaks2,
            result;
            outpath=outpath,
            arrow_scale=arrow_scale,
            markersize=markersize,
            title=title,
        )

    println("before           = ", result.n1)
    println("after            = ", result.n2)
    println("static           = ", result.n_static)
    println("optimized        = ", result.n_optimized)
    println("retry            = ", result.n_retry)
    println("matched          = ", result.n_matched)
    println("unmatched before retry (frame t)   = ", length(result.unmatched1_before_retry))
    println("unmatched before retry (frame t+1) = ", length(result.unmatched2_before_retry))
    println("final unmatched frame t            = ", length(result.unmatched1))
    println("final unmatched frame t+1          = ", length(result.unmatched2))
    println("figure saved to  = ", image_path)
    println("CSV saved to     = ", csv_path)

    return result
end


# ============================================================
# Convenience wrapper for peaks[t], peaks[t+1]
# ============================================================

function match_plot_save(
    peaks::AbstractVector;
    t::Int,
    kwargs...
)
    @assert 1 <= t < length(peaks)

    return match_plot_save(
        peaks[t],
        peaks[t + 1];
        kwargs...
    )
end


# ============================================================
# Example usage
# ============================================================

#=
include("minimal_peak_extraction.jl")
using .MinimalPeakExtraction

# Example:
# peaks[1] = extract_peaks(Bmat1; ...)
# peaks[2] = extract_peaks(Bmat2; ...)

result = match_plot_save(
    peaks;
    t=1,
    d_static=0.20,
    dmax=3.0,
    dmax_retry=4.5,
    outpath="result/vortex_matching_frame001.png",
    arrow_scale=1.0,
    markersize=8,
)

# Or:
#
# result = match_plot_save(
#     peaks[1],
#     peaks[2];
#     d_static=0.20,
#     dmax=3.0,
#     outpath="result/vortex_matching_frame001.png",
# )
=#
