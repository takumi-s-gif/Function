function locs = find_peak(data, bin_size, filename, minpeakheight, threshold)
    % image_vortex functionとDeltafの計算まではほとんど同じ。
    % データ参照
    Ref = double(data.ODMR.Ref);
    Sig = double(data.ODMR.Sig);
    Freq = data.MW.Freq;

    % パラメータ
    xlength = size(Ref, 1);
    ylength = size(Ref, 2);
    % zlength = size(Sig, 3); % ループ等で使用しない場合は省略可

    % データのビニング
    bin_Ref = rebin(Ref, bin_size);
    bin_Sig = rebin(Sig, bin_size);
    binned_xlength = floor(xlength / bin_size);
    binned_ylength = floor(ylength / bin_size);

    % 各ピクセルでのDeltafを求める
    num_dips = 2;
    f_1 = 2874.5;
    f_2 = 2880.5;
    model_fun = @(Freq, x) lorentz_sum(Freq, x, num_dips);
    init_params = [0.08, f_1, 3, 0.08, f_2, 3];
    opts = optimoptions('lsqcurvefit','Display','off','MaxFunctionEvaluations',5000);
    
    Deltaf = zeros(binned_xlength, binned_ylength);

    fprintf('function:find_peak Calculating Deltaf_subtracted (%s).\n', char(datetime))

    for ix = 1:binned_xlength
        for iy = 1:binned_ylength
            sig_bin = squeeze(bin_Sig(ix, iy, :));
            ref_bin = bin_Ref(ix, iy);
            contrast_xy = sig_bin(:) / ref_bin;
            
            xdata = Freq(:);
            ydata = contrast_xy;
        
            [fit_params, ~, ~, ~, ~, ~, ~] = ...
                    lsqcurvefit(model_fun, init_params, xdata, ydata, [], [], opts);
            
            df = fit_params(5) - fit_params(2);
            
            if (df > 4) && (df < 12)
                Deltaf(ix, iy) = df;
            else
                % Deltaf(ix, iy) = 8;
                Deltaf(ix, iy) = df;
            end
        end
    end

    % find peak
    fprintf('Finding peaks (%s).\n', char(datetime))

    [~, locs_y, locs_x] = peaks2(Deltaf, 'MinPeakHeight', minpeakheight, 'Threshold', threshold);
    figure;
    imagesc(Deltaf);
    hold on;
    % プロット時は x, y の順に入れ替える
    plot(locs_x, locs_y, 'ro', 'MarkerSize', 10, 'LineWidth', 2); 
    xlabel('X');
    ylabel('Y');
    title(filename, 'Interpreter', 'none');
    colorbar;

    locs = [locs_y, locs_x];
end