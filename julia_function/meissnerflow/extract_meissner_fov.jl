# extract_meissner_fov.jl
#
# Extract a camera field of view from a whole-sample Meissner-current result.
#
# Coordinate conventions:
#   - Whole-sample coordinates: lower-left corner of the sample is (0, 0).
#   - FOV coordinates: lower-left corner of the extracted FOV is (0, 0).
#   - result.grid coordinates may remain centred at the sample centre internally.
#
# Expected result fields (from perfect_meissner_screening_fft.jl):
#   result.grid.x, result.grid.y, result.mask,
#   result.g, result.Kx, result.Ky, result.Kmag,
#   result.Jx, result.Jy, result.Jmag,
#   result.Bind_T, result.Btotal_T

using CairoMakie
using Statistics
using Printf

function _fov_positive_quantile(values, q::Real)
    vals = Float64[v for v in values if isfinite(v) && v >= 0]
    isempty(vals) && return 1.0
    vmax = quantile(vals, clamp(Float64(q), 0.0, 1.0))
    return vmax > 0 ? vmax : max(maximum(vals), eps(Float64))
end

"""
    extract_meissner_fov(result;
        sample_size_mm=(2.7, 1.7),
        lower_left_mm=(1.75, 0.46),
        fov_size_um=(650.0, 650.0))

Extract a rectangular FOV from a whole-sample screening-current result.

The whole-sample coordinate system has its origin at the sample lower-left.
The returned local coordinate system has its origin at the FOV lower-left.

For the default values, the FOV occupies
    x_sample = 1750--2400 um
    y_sample =  460--1110 um
and its local coordinates are approximately
    x_FOV = 0--650 um
    y_FOV = 0--650 um.

Because the whole-sample solution is discrete, the first and last selected
pixel centres need not be exactly 0 and 650 um. The requested plotting bounds
are retained separately in `fov.local_bounds_um`.
"""
function extract_meissner_fov(
    result;
    sample_size_mm::Tuple{<:Real,<:Real} = (2.7, 1.7),
    lower_left_mm::Tuple{<:Real,<:Real} = (1.75, 0.46),
    fov_size_um::Tuple{<:Real,<:Real} = (650.0, 650.0),
)
    sample_width_um  = Float64(sample_size_mm[1]) * 1000.0
    sample_height_um = Float64(sample_size_mm[2]) * 1000.0

    x0_um = Float64(lower_left_mm[1]) * 1000.0
    y0_um = Float64(lower_left_mm[2]) * 1000.0
    width_um  = Float64(fov_size_um[1])
    height_um = Float64(fov_size_um[2])
    x1_um = x0_um + width_um
    y1_um = y0_um + height_um

    @assert sample_width_um > 0 && sample_height_um > 0
    @assert width_um > 0 && height_um > 0
    @assert (0.0 <= x0_um < x1_um <= sample_width_um + 1e-9)         "The requested FOV is outside the sample in x."
    @assert (0.0 <= y0_um < y1_um <= sample_height_um + 1e-9)         "The requested FOV is outside the sample in y."

    grid = result.grid

    # The solver grid is centred on the sample centre. Shift it so that the
    # sample lower-left is the origin of the whole-sample coordinate system.
    x_sample_um = Float64.(grid.x) .* 1e6 .+ sample_width_um / 2
    y_sample_um = Float64.(grid.y) .* 1e6 .+ sample_height_um / 2

    ix = findall(x -> x0_um <= x <= x1_um, x_sample_um)
    iy = findall(y -> y0_um <= y <= y1_um, y_sample_um)

    isempty(ix) && error("No x-grid points were found inside the requested FOV.")
    isempty(iy) && error("No y-grid points were found inside the requested FOV.")

    # Coordinates referred to the FOV lower-left.
    x_local_um = x_sample_um[ix] .- x0_um
    y_local_um = y_sample_um[iy] .- y0_um

    # Corresponding centred solver coordinates, useful for debugging.
    x_solver_um = Float64.(grid.x[ix]) .* 1e6
    y_solver_um = Float64.(grid.y[iy]) .* 1e6

    return (;
        ix,
        iy,
        x_um = x_local_um,
        y_um = y_local_um,
        x_sample_um = x_sample_um[ix],
        y_sample_um = y_sample_um[iy],
        x_solver_um,
        y_solver_um,
        mask = BitMatrix(result.mask[iy, ix]),
        g = Matrix(result.g[iy, ix]),
        Kx = Matrix(result.Kx[iy, ix]),
        Ky = Matrix(result.Ky[iy, ix]),
        Kmag = Matrix(result.Kmag[iy, ix]),
        Jx = Matrix(result.Jx[iy, ix]),
        Jy = Matrix(result.Jy[iy, ix]),
        Jmag = Matrix(result.Jmag[iy, ix]),
        Bind_T = Matrix(result.Bind_T[iy, ix]),
        Btotal_T = Matrix(result.Btotal_T[iy, ix]),
        sample_size_um = (sample_width_um, sample_height_um),
        lower_left_sample_um = (x0_um, y0_um),
        upper_right_sample_um = (x1_um, y1_um),
        fov_size_um = (width_um, height_um),
        local_bounds_um = ((0.0, width_um), (0.0, height_um)),
        selected_pixel_bounds_um = (
            (first(x_local_um), last(x_local_um)),
            (first(y_local_um), last(y_local_um)),
        ),
    )
end

"""
    plot_whole_sample_with_fov(result; ...)

Plot the whole-sample current magnitude in lower-left-origin coordinates and
show the requested FOV as a rectangle.
"""
function plot_whole_sample_with_fov(
    result;
    sample_size_mm::Tuple{<:Real,<:Real} = (2.7, 1.7),
    lower_left_mm::Tuple{<:Real,<:Real} = (1.75, 0.46),
    fov_size_um::Tuple{<:Real,<:Real} = (650.0, 650.0),
    quantity::Symbol = :sheet,
    color_quantile::Real = 0.98,
    colorrange = nothing,
    colormap = :viridis,
    rectangle_color = :red,
    rectangle_linewidth::Real = 4.0,
    outpath = nothing,
    figure_size = (1050, 760),
)
    sample_width_um  = Float64(sample_size_mm[1]) * 1000.0
    sample_height_um = Float64(sample_size_mm[2]) * 1000.0
    x0 = Float64(lower_left_mm[1]) * 1000.0
    y0 = Float64(lower_left_mm[2]) * 1000.0
    w = Float64(fov_size_um[1])
    h = Float64(fov_size_um[2])

    grid = result.grid
    mask = result.mask
    x_sample_um = Float64.(grid.x) .* 1e6 .+ sample_width_um / 2
    y_sample_um = Float64.(grid.y) .* 1e6 .+ sample_height_um / 2

    if quantity == :sheet
        mag = result.Kmag
        cbar_label = "|K| (A/m)"
    elseif quantity == :volume
        mag = result.Jmag
        cbar_label = "|J| (A/m^2)"
    else
        error("quantity must be :sheet or :volume")
    end

    heat = fill(NaN, size(mag))
    heat[mask] .= mag[mask]

    if colorrange === nothing
        vmax = _fov_positive_quantile(mag[mask], color_quantile)
        colorrange = (0.0, vmax)
    end

    fig = Figure(size = figure_size)
    ax = Axis(
        fig[1, 1];
        xlabel = "sample x from lower-left (um)",
        ylabel = "sample y from lower-left (um)",
        aspect = DataAspect(),
        title = "Whole sample and selected FOV",
    )

    hm = heatmap!(
        ax,
        x_sample_um,
        y_sample_um,
        permutedims(heat);
        colormap = colormap,
        colorrange = colorrange,
    )
    Colorbar(fig[1, 2], hm; label = cbar_label)

    # Requested FOV rectangle.
    lines!(
        ax,
        [x0, x0 + w, x0 + w, x0, x0],
        [y0, y0, y0 + h, y0 + h, y0];
        color = rectangle_color,
        linewidth = rectangle_linewidth,
    )

    text!(
        ax,
        @sprintf("  FOV: (%.0f, %.0f) to (%.0f, %.0f) um", x0, y0, x0 + w, y0 + h);
        position = (x0, y0 + h),
        color = rectangle_color,
        align = (:left, :bottom),
        fontsize = 18,
    )

    limits!(ax, 0.0, sample_width_um, 0.0, sample_height_um)

    if outpath !== nothing
        save(outpath, fig)
    end
    return fig
end

"""
    plot_extracted_fov_current(fov; ...)

Plot the extracted FOV in a new coordinate system whose lower-left is (0, 0).
The heatmap shows current magnitude and optional arrows show current direction.
"""
function plot_extracted_fov_current(
    fov;
    quantity::Symbol = :sheet,
    show_arrows::Bool = true,
    stride::Int = 8,
    arrow_mode::Symbol = :direction,
    arrow_length_um = nothing,
    color_quantile::Real = 0.98,
    colorrange = nothing,
    colormap = :viridis,
    outpath = nothing,
    title::AbstractString = "Extracted 650 um x 650 um FOV",
    figure_size = (900, 760),
)
    @assert stride >= 1

    if quantity == :sheet
        vx, vy, mag = fov.Kx, fov.Ky, fov.Kmag
        cbar_label = "|K| (A/m)"
    elseif quantity == :volume
        vx, vy, mag = fov.Jx, fov.Jy, fov.Jmag
        cbar_label = "|J| (A/m^2)"
    else
        error("quantity must be :sheet or :volume")
    end

    heat = fill(NaN, size(mag))
    heat[fov.mask] .= mag[fov.mask]

    if colorrange === nothing
        vmax = _fov_positive_quantile(mag[fov.mask], color_quantile)
        colorrange = (0.0, vmax)
    end

    if arrow_length_um === nothing
        dx_um = length(fov.x_um) > 1 ? median(diff(fov.x_um)) : 1.0
        dy_um = length(fov.y_um) > 1 ? median(diff(fov.y_um)) : 1.0
        arrow_length_um = 0.75 * stride * min(dx_um, dy_um)
    end

    xs = Float64[]
    ys = Float64[]
    ux = Float64[]
    uy = Float64[]
    mags = Float64[]

    if show_arrows
        for jy in 1:stride:size(mag, 1), jx in 1:stride:size(mag, 2)
            fov.mask[jy, jx] || continue
            m = mag[jy, jx]
            if isfinite(m) && m > 0
                push!(xs, fov.x_um[jx])
                push!(ys, fov.y_um[jy])
                push!(ux, vx[jy, jx])
                push!(uy, vy[jy, jx])
                push!(mags, m)
            end
        end
    end

    uplot = Float64[]
    vplot = Float64[]
    if !isempty(mags)
        mref = _fov_positive_quantile(mags, 0.95)
        for i in eachindex(mags)
            factor = if arrow_mode == :direction
                1.0
            elseif arrow_mode == :magnitude
                min(mags[i] / mref, 1.5)
            else
                error("arrow_mode must be :direction or :magnitude")
            end
            push!(uplot, arrow_length_um * factor * ux[i] / mags[i])
            push!(vplot, arrow_length_um * factor * uy[i] / mags[i])
        end
    end

    fig = Figure(size = figure_size)
    ax = Axis(
        fig[1, 1];
        xlabel = "x in FOV (um)",
        ylabel = "y in FOV (um)",
        aspect = DataAspect(),
        title = title,
    )

    hm = heatmap!(
        ax,
        fov.x_um,
        fov.y_um,
        permutedims(heat);
        colormap = colormap,
        colorrange = colorrange,
    )
    Colorbar(fig[1, 2], hm; label = cbar_label)

    if !isempty(xs)
        points = Point2f.(xs, ys)
        vectors = Vec2f.(uplot, vplot)

        arrows2d!(
            ax,
            points,
            vectors;
            align = :tail,
            tipwidth = 16,
            tiplength = 16,
            shaftwidth = 6,
            color = :red,
        )
        # arrows2d!(
        #     ax,
        #     points,
        #     vectors;
        #     align = :tail,
        #     tipwidth = 10,
        #     tiplength = 10,
        #     shaftwidth = 3,
        #     color = :white,
        # )
    end

    (xlim, ylim) = fov.local_bounds_um
    limits!(ax, xlim[1], xlim[2], ylim[1], ylim[2])

    if outpath !== nothing
        save(outpath, fig)
    end
    return fig
end

using Statistics
using DataFrames
using CSV
using CairoMakie

"""
    average_current_in_bins(fov; kwargs...)

指定した範囲を正方形binに分割し、各bin内の遮蔽電流密度を平均する。

必要なfovの要素:
    fov.x_um, fov.y_um
    fov.Kx, fov.Ky, fov.Kmag
    fov.Jx, fov.Jy, fov.Jmag  # quantity=:volumeの場合
    fov.mask

戻り値:
    DataFrame
"""
function average_current_in_bins(
    fov;
    xrange::Tuple{<:Real,<:Real} = (250.0, 450.0),
    yrange::Tuple{<:Real,<:Real} = (150.0, 350.0),
    binsize_um::Real = 40.0,
    quantity::Symbol = :sheet,
    minimum_count::Int = 1,
)
    xmin, xmax = Float64.(xrange)
    ymin, ymax = Float64.(yrange)
    binsize = Float64(binsize_um)

    xmin < xmax || error("xrange must satisfy xmin < xmax.")
    ymin < ymax || error("yrange must satisfy ymin < ymax.")
    binsize > 0 || error("binsize_um must be positive.")

    x = Float64.(fov.x_um)
    y = Float64.(fov.y_um)
    mask = Bool.(fov.mask)

    if quantity == :sheet
        Cx = Float64.(fov.Kx)
        Cy = Float64.(fov.Ky)
        Cmag = Float64.(fov.Kmag)
        unit = "A/m"
    elseif quantity == :volume
        Cx = Float64.(fov.Jx)
        Cy = Float64.(fov.Jy)
        Cmag = Float64.(fov.Jmag)
        unit = "A/m²"
    else
        error("quantity must be :sheet or :volume.")
    end

    size(Cx) == (length(y), length(x)) ||
        error("Current arrays must have size (length(fov.y_um), length(fov.x_um)).")

    # 40 µm刻みのbin境界
    xedges = collect(xmin:binsize:xmax)
    yedges = collect(ymin:binsize:ymax)

    # 浮動小数点誤差などで最終境界が不足した場合
    if isempty(xedges) || xedges[end] < xmax
        push!(xedges, xmax)
    end
    if isempty(yedges) || yedges[end] < ymax
        push!(yedges, ymax)
    end

    # 指定範囲がbin幅で割り切れない場合も、最後を上限に合わせる
    xedges[end] = xmax
    yedges[end] = ymax

    rows = NamedTuple[]

    bin_id = 0

    for iy_bin in 1:(length(yedges) - 1)
        ylo = yedges[iy_bin]
        yhi = yedges[iy_bin + 1]

        # 最終binのみ右端を含める
        yinds = if iy_bin == length(yedges) - 1
            findall(v -> ylo <= v <= yhi, y)
        else
            findall(v -> ylo <= v < yhi, y)
        end

        for ix_bin in 1:(length(xedges) - 1)
            xlo = xedges[ix_bin]
            xhi = xedges[ix_bin + 1]

            xinds = if ix_bin == length(xedges) - 1
                findall(v -> xlo <= v <= xhi, x)
            else
                findall(v -> xlo <= v < xhi, x)
            end

            bin_id += 1

            values_x = Float64[]
            values_y = Float64[]
            values_mag = Float64[]

            for iy in yinds
                for ix in xinds
                    mask[iy, ix] || continue

                    cx = Cx[iy, ix]
                    cy = Cy[iy, ix]
                    cmag = Cmag[iy, ix]

                    if isfinite(cx) && isfinite(cy) && isfinite(cmag)
                        push!(values_x, cx)
                        push!(values_y, cy)
                        push!(values_mag, cmag)
                    end
                end
            end

            n = length(values_x)

            if n >= minimum_count
                mean_x = mean(values_x)
                mean_y = mean(values_y)

                # 平均ベクトルの大きさ
                magnitude_of_mean = hypot(mean_x, mean_y)

                # 各点での大きさを平均
                mean_magnitude = mean(values_mag)

                # 方向。x軸正方向から反時計回り
                angle_deg = rad2deg(atan(mean_y, mean_x))

                std_x = n > 1 ? std(values_x) : 0.0
                std_y = n > 1 ? std(values_y) : 0.0
                std_magnitude = n > 1 ? std(values_mag) : 0.0
            else
                mean_x = NaN
                mean_y = NaN
                magnitude_of_mean = NaN
                mean_magnitude = NaN
                angle_deg = NaN
                std_x = NaN
                std_y = NaN
                std_magnitude = NaN
            end

            push!(
                rows,
                (
                    bin_id = bin_id,
                    ix = ix_bin,
                    iy = iy_bin,

                    x_min_um = xlo,
                    x_max_um = xhi,
                    y_min_um = ylo,
                    y_max_um = yhi,

                    x_center_um = (xlo + xhi) / 2,
                    y_center_um = (ylo + yhi) / 2,

                    mean_x = mean_x,
                    mean_y = mean_y,

                    magnitude_of_mean = magnitude_of_mean,
                    mean_magnitude = mean_magnitude,
                    angle_deg = angle_deg,

                    std_x = std_x,
                    std_y = std_y,
                    std_magnitude = std_magnitude,

                    n_pixels = n,
                    quantity = String(quantity),
                    unit = unit,
                ),
            )
        end
    end

    return DataFrame(rows)
end

function plot_binned_current(
    bin_result::DataFrame;
    outpath = nothing,
    title::AbstractString = "Locally averaged shielding current",
    arrow_scale_um::Real = 25.0,
    arrow_reference = nothing,
    colormap = :viridis,
    show_values::Bool = false,
)
    ix_values = sort(unique(bin_result.ix))
    iy_values = sort(unique(bin_result.iy))

    nx = length(ix_values)
    ny = length(iy_values)

    mean_mag = fill(NaN, ny, nx)
    mean_x = fill(NaN, ny, nx)
    mean_y = fill(NaN, ny, nx)

    xcenters = zeros(nx)
    ycenters = zeros(ny)

    for row in eachrow(bin_result)
        mean_mag[row.iy, row.ix] = row.mean_magnitude
        mean_x[row.iy, row.ix] = row.mean_x
        mean_y[row.iy, row.ix] = row.mean_y

        xcenters[row.ix] = row.x_center_um
        ycenters[row.iy] = row.y_center_um
    end

    finite_values = filter(isfinite, vec(mean_mag))
    isempty(finite_values) && error("No finite current values to plot.")

    color_max = maximum(finite_values)

    if arrow_reference === nothing
        vector_magnitudes = hypot.(mean_x, mean_y)
        valid_vectors = filter(
            v -> isfinite(v) && v > 0,
            vec(vector_magnitudes),
        )

        arrow_reference_use = isempty(valid_vectors) ?
                              1.0 :
                              quantile(valid_vectors, 0.95)
    else
        arrow_reference_use = Float64(arrow_reference)
    end

    fig = Figure(size = (800, 700))

    ax = Axis(
        fig[1, 1];
        xlabel = "x in FOV (µm)",
        ylabel = "y in FOV (µm)",
        title = title,
        aspect = DataAspect(),
    )

    hm = heatmap!(
        ax,
        xcenters,
        ycenters,
        permutedims(mean_mag);
        colormap = colormap,
        colorrange = (0.0, color_max),
    )

    unit = first(bin_result.unit)

    Colorbar(
        fig[1, 2],
        hm;
        label = "Mean current magnitude ($unit)",
    )

    points = Point2f[]
    vectors = Vec2f[]

    for iy in 1:ny
        for ix in 1:nx
            mx = mean_x[iy, ix]
            my = mean_y[iy, ix]

            isfinite(mx) && isfinite(my) || continue

            # 95%点の電流をarrow_scale_umの長さにする
            ux = arrow_scale_um * mx / arrow_reference_use
            uy = arrow_scale_um * my / arrow_reference_use

            push!(
                points,
                Point2f(
                    xcenters[ix],
                    ycenters[iy],
                ),
            )

            push!(vectors, Vec2f(ux, uy))

            if show_values
                text!(
                    ax,
                    xcenters[ix],
                    ycenters[iy];
                    text = string(
                        round(mean_mag[iy, ix]; sigdigits = 3),
                    ),
                    align = (:center, :center),
                    fontsize = 12,
                    color = :white,
                )
            end
        end
    end

    # 黒縁
    # arrows2d!(
    #     ax,
    #     points,
    #     vectors;
    #     align = :center,
    #     color = :black,
    #     tipwidth = 15,
    #     tiplength = 15,
    #     shaftwidth = 6,
    # )

    # 赤い矢印
    arrows2d!(
        ax,
        points,
        vectors;
        align = :center,
        color = :red,
        tipwidth = 10,
        tiplength = 10,
        shaftwidth = 5,
    )

    # 40 µm binの境界線
    xedges = sort(unique(vcat(
        bin_result.x_min_um,
        bin_result.x_max_um,
    )))

    yedges = sort(unique(vcat(
        bin_result.y_min_um,
        bin_result.y_max_um,
    )))

    for xedge in xedges
        vlines!(
            ax,
            [xedge];
            color = (:white, 0.6),
            linewidth = 1,
        )
    end

    for yedge in yedges
        hlines!(
            ax,
            [yedge];
            color = (:white, 0.6),
            linewidth = 1,
        )
    end

    limits!(
        ax,
        minimum(xedges),
        maximum(xedges),
        minimum(yedges),
        maximum(yedges),
    )

    if outpath !== nothing
        save(outpath, fig)
    end

    return fig
end