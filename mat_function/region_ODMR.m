function region_ODMR(data, bin_size, colorbar_limit, filename, xrange, yrange)
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
    % f_1 = 2870;
    % f_2 = 2870;
    model_fun = @(Freq, x) lorentz_sum(Freq, x, num_dips);
    init_params = [0.08, f_1, 3, 0.08, f_2, 3];
    opts = optimoptions('lsqcurvefit','Display','off','MaxFunctionEvaluations',5000);
    
    Deltaf = zeros(binned_xlength, binned_ylength);

    fprintf('function:region_ODMR Calculating Deltaf (%s).\n', char(datetime))

    for ix = 1:binned_xlength
        for iy = 1:binned_ylength
            sig_bin = squeeze(bin_Sig(ix, iy, :));
            ref_bin = bin_Ref(ix, iy);
            contrast_xy = sig_bin(:) / ref_bin;
            
            xdata = Freq(:);
            ydata = contrast_xy;
        
            [fit_params, ~, ~, ~, ~, ~, ~] = ...
                    lsqcurvefit(model_fun, init_params, xdata, ydata, [], [], opts);
            
            % Deltaf(ix, iy) = fit_params(5) - fit_params(2);
            df = fit_params(5) - fit_params(2);

            Deltaf(ix, iy) = df;
            % if (df > 4) && (df < 12)
            %     Deltaf(ix, iy) = df;
            % else
            %     % Deltaf(ix, iy) = 8;
            %     Deltaf(ix, iy) = df;
            % end
        end
    end

    figure;
    imagesc(Deltaf);
    xlabel('X'); 
    ylabel('Y'); 
    title(filename, 'Interpreter', 'none');
    colorbar;
    clim(colorbar_limit)
    
    fprintf('meaning ODMR in a certain region (%s).\n', char(datetime))

    idx_x = floor(xrange(1) * binned_xlength) : floor(xrange(2) * binned_xlength);
    idx_y = floor(yrange(1) * binned_ylength) : floor(yrange(2) * binned_ylength);

    sub_Ref = bin_Ref(idx_y, idx_x, :);
    sub_Sig = bin_Sig(idx_y, idx_x, :);

    contrast_map = sub_Sig ./ sub_Ref;
    avg_contrast = squeeze(mean(contrast_map, [1, 2]));

    x0 = idx_x(1);
    y0 = idx_y(1);
    w  = idx_x(end) - idx_x(1);
    h  = idx_y(end) - idx_y(1);
    
    hold on;
    rectangle('Position', [x0, y0, w, h], 'EdgeColor', 'r', 'LineWidth', 2);
    
    figure;
    hold on;
    plot(Freq, avg_contrast, 'b.-', 'DisplayName', 'Contrast Data', 'MarkerSize', 20);
    xlabel('Frequency'); ylabel('Average Contrast');
    grid on;
end