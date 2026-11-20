include("function_share/functions_share/base.jl")
include("function_share/functions_share/ParamOptical.jl")
include("function_share/functions_share/utilities.jl")
using CairoMakie
using LaTeXStrings

function B_calculation(data::Dict, data_zf::Dict; gamma_e = 28e-3)
    deltaf = data["deltaf"]
    deltaf_zf = data_zf["deltaf"]

    nijousa = (deltaf / 2).^2 - (deltaf_zf / 2).^2
    nijousa = max.(nijousa, 0)

    B = sqrt.(nijousa) / gamma_e

    return B
end

function B_imaging(
    B::Matrix{<:Real}, 
    pOptical::ParamOptical, 
    figname::String; 
    kwargs_Fig=(;), 
    kwargs_Axis=(;aspect=DataAspect()), 
    kwargs_Colorbar=(;), 
    kwargs_hm=(;), 
    xmin::Real=0, 
    ymin::Real=0)
    X, Y = get_XYrange(B, pOptical, xmin, ymin)
    fig = Figure(;kwargs_Fig...)
    ax = CairoMakie.Axis(fig[1, 1]; xlabel = "x [um]", ylabel = "y [um]", title = figname, kwargs_Axis...)
    hm = CairoMakie.heatmap!(ax, X, Y, B'[:, end:-1:1]; kwargs_hm...)
    if (@isdefined cb_off) && (cb_off != 0)
    else
        Colorbar(fig[1, 2], hm; label = L"B \text{(\mu T)}", kwargs_Colorbar...)
    end
    return fig
end

function save_B(B::Matrix, B_fig::Figure, name::String)
    mkpath("B/fig")
    mkpath("B/jld2")

    jld2path = joinpath("B/jld2", name * ".jld2")
    jldsave(jld2path; B)

    figname = joinpath("B/fig", name * ".png")
    save(figname, B_fig)
end