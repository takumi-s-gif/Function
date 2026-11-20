include("function_share/functions_share/ParamOptical.jl")
pOptical = ParamOptical(3.45 * 2 / 20, 100.0, 200.0)
pxsize = get_pxsize(pOptical)