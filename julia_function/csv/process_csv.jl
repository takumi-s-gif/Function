using CSV
using DataFrames
using Statistics

function csv_grouping(df; group_size::Int = 5)
df.row_number = 1:nrow(df)
# group_size = 5
df.group = cld.(df.row_number, group_size)

df_grouped = combine(
groupby(df, :group),

# Row1~5などのラベル
:row_number =>
    (rows -> "Row$(first(rows))~$(last(rows))") =>
    :row_range,

# total_vorticesで重み付けしたscreening current
[
    :mean_magnitude_times_deltaI,
    :total_vortices,
] =>
    ((current, total) ->
        sum(current .* total) / sum(total)
    ) =>
    :mean_magnitude_times_deltaI,

# 全渦数の合計
:total_vortices =>
    sum =>
    :sum_total_vortices,

# 移動渦数の合計
:moving_vortices =>
    sum =>
    :sum_moving_vortices,
)

# 重み付き移動割合
df_grouped.percent =
    100 .* df_grouped.sum_moving_vortices ./
    df_grouped.sum_total_vortices

# 必要な列だけ、指定順に並べる
select!(
    df_grouped,
    :row_range,
    :mean_magnitude_times_deltaI,
    :sum_total_vortices,
    :sum_moving_vortices,
    :percent,
)

return df_grouped

end