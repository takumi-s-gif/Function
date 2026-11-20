# include("function_share/pOptical_UT3_202606_20x_2bin.jl")
include("function_share/functions_share/ParamOptical.jl")

function roi(
    B::Matrix{<:Real}, 
    roi_B::Matrix{<:Real}; 
    xrange::Tuple{<:Real, <:Real} = (0,get_pxsize(pOptical) * size(Bmat)[1]), 
    yrange::Tuple{<:Real, <:Real} = (0,get_pxsize(pOptical) * size(Bmat)[1]))
    iy = floor(Int, xrange[1] / pxsize) + 1 : ceil(Int, xrange[2] / pxsize)
    ix = size(B)[1] - ceil(Int, yrange[2] / pxsize) + 1 : size(B)[1] - floor(Int, yrange[1] / pxsize) 
    
    roi_B[ix, iy] .= B[ix, iy]

    return roi_B
end