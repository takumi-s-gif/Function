using MAT
function read_data(filepath)
    fid = matopen(filepath)
    Freq = vec(read(fid, "MW/Freq"))
    ImgRef = read(fid, "ODMR/Ref")
    ImgSig = read(fid, "ODMR/Sig")
    # MISC = read(fid, "MISC")
    # Loop = read(fid, "Loop")
    cam = read(fid, "cam")
    MW = read(fid, "MW")
    # PULSE = read(fid, "PULSE")
    # sg = read(fid, "sg")
    # awg = read(fid, "awg")
    close(fid)
    Expdata_path_vec = splitpath(filepath)
    # resdict = Dict("Freq" => Freq, "ImgRef" => ImgRef, "ImgSig" => ImgSig, "matfilepath" => filepath, "MISC" => MISC, "Loop" => Loop, "cam" => cam, "MW" => MW, "PULSE" => PULSE, "sg" => sg, "awg" => awg, "Expdata_path_vec" => Expdata_path_vec)
    resdict = Dict("Freq" => Freq, "ImgRef" => ImgRef, "ImgSig" => ImgSig, "cam" => cam, "MW" => MW)
    return resdict
end
