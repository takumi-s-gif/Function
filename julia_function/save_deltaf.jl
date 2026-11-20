using JLD2

function save_deltaf(deltaf::Matrix{<:Real}, filename::String; dir = "deltaf/jld2")
    mkpath(dir)

    filepath = joinpath(dir, filename)
    jldsave(filepath; deltaf)
    return filepath
end