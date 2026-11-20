function locs = find_peak_B(B, filename, minpeakheight, threshold)
    [~, locs_y, locs_x] = peaks2(B, 'MinPeakHeight', minpeakheight, 'Threshold', threshold);
    B = real(B);
    
    figure;
    imagesc(B);
    hold on;
    % プロット時は x, y の順に入れ替える
    plot(locs_x, locs_y, 'ro', 'MarkerSize', 10, 'LineWidth', 2); 
    xlabel('X');
    ylabel('Y');
    title(filename, 'Interpreter', 'none');
    colorbar;

    locs = [locs_y, locs_x];
end