function B = B_map(Deltaf, Deltaf_zf, filename)
    gamma_e = 0.028; %MHz_uT
    B = 1 / gamma_e * sqrt( (Deltaf/2).^2 - (Deltaf_zf/2).^2 );
    realB = real(B);

    figure;
    imagesc(realB);
    colorbar;
    axis square;
    title(filename, 'Interpreter','none');
end