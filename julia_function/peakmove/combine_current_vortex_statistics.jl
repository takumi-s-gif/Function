# combine_current_vortex_statistics.jl
#
# Utilities for combining local vortex statistics with
# local screening-current data.
#
# Expected vortex CSV columns:
#   t, ix, iy,
#   total_vortices,
#   moving_vortices,
#   moving_percent
#
# Expected current CSV columns:
#   ix, iy, mean_magnitude
#
# Main output columns:
#   ix, iy,
#   mean_magnitude_times_deltaI,
#   total_vortices,
#   moving_vortices,
#   moving_percent

using CSV
using DataFrames


"""
    prepare_vortex_statistics(
        vortex_csv_path;
        frame_t=2,
    )

Read the local-vortex-statistics CSV and keep only one frame transition.
"""
function prepare_vortex_statistics(
    vortex_csv_path::AbstractString;
    frame_t::Int = 2,
)
    df_vortex = CSV.read(vortex_csv_path, DataFrame)

    required = [
        :t,
        :ix,
        :iy,
        :total_vortices,
        :moving_vortices,
        :moving_percent,
    ]

    missing_cols = [
        c for c in required
        if !(c in propertynames(df_vortex))
    ]

    isempty(missing_cols) ||
        error(
            "Vortex CSV is missing columns: " *
            join(string.(missing_cols), ", ")
        )

    # t can sometimes be read as String because old files also contain "all".
    df_selected = filter(
        row -> string(row.t) == string(frame_t),
        df_vortex,
    )

    return select(
        df_selected,
        :ix,
        :iy,
        :total_vortices,
        :moving_vortices,
        :moving_percent,
    )
end


"""
    prepare_screening_current(
        current_csv_path;
        flip_ix=true,
    )

Read local screening-current data.

When `flip_ix=true`, transform
    ix -> minimum(ix) + maximum(ix) - ix

to match the vortex-bin coordinate convention.
"""
function prepare_screening_current(
    current_csv_path::AbstractString;
    flip_ix::Bool = true,
)
    df_current = CSV.read(current_csv_path, DataFrame)

    required = [
        :ix,
        :iy,
        :mean_magnitude,
    ]

    missing_cols = [
        c for c in required
        if !(c in propertynames(df_current))
    ]

    isempty(missing_cols) ||
        error(
            "Current CSV is missing columns: " *
            join(string.(missing_cols), ", ")
        )

    df_selected = select(
        df_current,
        :ix,
        :iy,
        :mean_magnitude,
    )

    # if flip_ix
    #     xmin = minimum(df_selected.ix)
    #     xmax = maximum(df_selected.ix)

    #     transform!(
    #         df_selected,
    #         :ix => ByRow(ix -> xmin + xmax - ix) => :ix,
    #     )
    # end

    return df_selected
end


"""
    combine_current_vortex(
        current_csv_path,
        vortex_csv_path;
        deltaI,
        frame_t=2,
        flip_ix=true,
    )

Join screening-current and local moving-vortex statistics by `(ix, iy)`.

The current magnitude is multiplied by `deltaI` and stored as
`mean_magnitude_times_deltaI`.
"""
function combine_current_vortex(
    current_csv_path::AbstractString,
    vortex_csv_path::AbstractString;
    deltaI::Real,
    frame_t::Int = 2,
    flip_ix::Bool = true,
)
    df_current = prepare_screening_current(
        current_csv_path;
        flip_ix = flip_ix,
    )

    df_vortex = prepare_vortex_statistics(
        vortex_csv_path;
        frame_t = frame_t,
    )

    df_result = innerjoin(
        df_current,
        df_vortex;
        on = [:ix, :iy],
    )

    transform!(
        df_result,
        :mean_magnitude =>
            ByRow(x -> x * deltaI) =>
            :mean_magnitude_times_deltaI,
    )

    select!(
        df_result,
        :ix,
        :iy,
        :mean_magnitude_times_deltaI,
        :total_vortices,
        :moving_vortices,
        :moving_percent,
    )

    sort!(
        df_result,
        [:ix, :iy],
    )

    return df_result
end


"""
    combine_current_vortex_save(
        current_csv_path,
        vortex_csv_path,
        output_csv_path;
        deltaI,
        frame_t=2,
        flip_ix=true,
    )

Run `combine_current_vortex(...)` and save the result as CSV.
"""
function combine_current_vortex_save(
    current_csv_path::AbstractString,
    vortex_csv_path::AbstractString,
    output_csv_path::AbstractString;
    deltaI::Real,
    frame_t::Int = 2,
    flip_ix::Bool = true,
)
    df_result = combine_current_vortex(
        current_csv_path,
        vortex_csv_path;
        deltaI = deltaI,
        frame_t = frame_t,
        flip_ix = flip_ix,
    )

    outdir = dirname(output_csv_path)

    if !isempty(outdir) && outdir != "."
        mkpath(outdir)
    end

    CSV.write(
        output_csv_path,
        df_result,
    )

    println("saved: ", output_csv_path)

    return df_result
end


# ============================================================
# Example
# ============================================================

#=
current_csv_path =
    "../Meissnerflow_x10/data/local_screening_current_50um_bins.csv"

vortex_csv_path =
    "B/peak/analysis_0826/example/local_vortex_statistics_50um.csv"

output_csv_path =
    "B/peak/analysis_0826/example/local_current_vortex_50um.csv"

df_result = combine_current_vortex_save(
    current_csv_path,
    vortex_csv_path,
    output_csv_path;
    deltaI = 2.0,
    frame_t = 2,
    flip_ix = true,
)
=#
