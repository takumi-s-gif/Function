function Deltaf = image_vortex(data, bin_size, colorbar_limit, filename)
    % lens_magnification = 100;
    % pixel_size = 13000 / lens_magnification  / 2048;

    % データ参照
    Ref = double(data.ODMR.Ref);
    Sig = double(data.ODMR.Sig);
    Freq = data.MW.Freq;
    % data_size = size(data.ODMR.Ref, 1);

    % パラメータ
    xlength = size(Ref, 1);
    ylength = size(Ref, 2);
    % zlength = size(Sig, 3); % ループ等で使用しない場合は省略可

    % % フィッティングパラメータを決める
    % squeeze_ref = rebin(Ref, data_size);
    % squeeze_sig = rebin(Sig, data_size);
    % 
    % Freq_size = size(Freq, 2);
    % contrast = zeros(1, Freq_size);
    % for i = 1:Freq_size
    %     contrast(i) = 1 - squeeze_sig(:,:,i) / squeeze_ref;
    % end
    % 
    % [~, locs] = findpeaks(contrast, Freq);

    % データのビニング
    bin_Ref = rebin(Ref, bin_size);
    bin_Sig = rebin(Sig, bin_size);
    binned_xlength = floor(xlength / bin_size);
    binned_ylength = floor(ylength / bin_size);

    % 各ピクセルでのDeltafを求める
    num_dips = 2;
    model_fun = @(Freq, x) lorentz_sum(Freq, x, num_dips);

    opts = optimoptions('lsqcurvefit','Display','off','MaxFunctionEvaluations',5000);
    
    Deltaf = zeros(binned_xlength, binned_ylength);

    fprintf('function:Image_vortex Calculating Deltaf (%s).\n', char(datetime))

    for ix = 1:binned_xlength
        for iy = 1:binned_ylength
            sig_bin = squeeze(bin_Sig(ix, iy, :));
            ref_bin = bin_Ref(ix, iy);
            contrast_xy = sig_bin(:) / ref_bin;
            
            xdata = Freq(:);
            ydata = contrast_xy;

            [~, locs] = findpeaks(-ydata, xdata, 'NPeaks', 2, 'SortStr', 'descend');

            if isempty(locs)
                locs = [NaN NaN];
            elseif isscalar(locs)
                locs = [locs(1) locs(1)];
            else
                locs = sort(locs);
            end

            % fprintf('%1d %1d: locs(1) = %5d, locs(2) = %5d. \n', ix, iy, locs(1), locs(2))

            init_params = [0.08, locs(1)-1, 3, 0.08, locs(2)+1, 3];
        
            [fit_params, ~, ~, ~, ~, ~, ~] = ...
                    lsqcurvefit(model_fun, init_params, xdata, ydata, [], [], opts);
            
            Deltaf(ix, iy) = fit_params(5) - fit_params(2);
        end
    end

    % Deltafのイメージング
    fprintf('Imaging Deltaf (%s).\n', char(datetime))

    figure;
    L = 130;
    imagesc([0 L], [L 0], Deltaf);
    axis square;
    xlabel('X'); 
    ylabel('Y');
    title(filename, 'Interpreter', 'none');
    colorbar;
    clim(colorbar_limit)

    xticks(0:100:L)
    yticks(0:100:L)
    
    xlabel('X [\mum]')
    ylabel('Y [\mum]')

    ax = gca;
    ax.TickDir = 'out';
    ax.YDir = 'normal';
    ax.Box = 'off';
    cb = colorbar;
    cb.Label.String = '\Delta f [MHz]';
end