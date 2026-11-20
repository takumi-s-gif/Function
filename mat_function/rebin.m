function data1 = rebin(data0,Nbin)

    sizeH0 = size(data0,1);
    sizeW0 = size(data0,2);
    sizeH  = floor(sizeH0/Nbin);
    sizeW  = floor(sizeW0/Nbin);
    Nfreq  = size(data0,3);

    data1 = zeros(sizeH, sizeW, Nfreq);
    parfor reff = 1:Nfreq
        for refh = 1:sizeH
            for refw = 1:sizeW
                data1(refh,refw,reff) = sum(data0(Nbin*(refh-1)+(1:Nbin), Nbin*(refw-1)+(1:Nbin),reff) ,[1,2]);
            end
        end
    end
end