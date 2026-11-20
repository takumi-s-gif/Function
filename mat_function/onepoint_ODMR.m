function onepoint_ODMR(data, x, y)
Ref = double(data.ODMR.Ref);
Sig = double(data.ODMR.Sig);
Freq = data.MW.Freq;

zlength = size(Sig, 3);

contrast = zeros(1, zlength);
ref_xy = Ref(x, y);

for z = 1:zlength
    sig_xy = Sig(x, y, z);
    contrast(z) = sig_xy / ref_xy ;
end

num_dips = 2;
f_1 = 2875.5;
f_2 = 2879.5;
% f_1 = 2870;
% f_2 = 2870;

fprintf('function:onepoint_ODMR Fitting ODMRspectle (%s).\n', char(datetime))

model_fun = @(Freq, x) lorentz_sum(Freq, x, num_dips);
init_params = [0.08, f_1, 3, 0.08, f_2, 3];
opts = optimoptions('lsqcurvefit','Display','off','MaxFunctionEvaluations',5000);
    
[fit_params, ~, ~, ~, ~, ~, ~] = lsqcurvefit(model_fun, init_params, Freq, contrast, [], [], opts);
    
fprintf('Imaging ODMRspectle (%s).\n', char(datetime))

%---- 結果の表示 ----
figure;
plot(Freq, contrast, 'b.', 'DisplayName','Data', 'MarkerSize',20);
hold on;
params_detail = linspace(Freq(1), Freq(end), 1000);
plot(params_detail, model_fun(fit_params, params_detail), '-', 'LineWidth',2, 'DisplayName','Fitted');
legend show;
legend('Location','southeast');
xlabel('Freq (MHz)'); ylabel('Contrast'); title('2-Lorentzian Fit');
end