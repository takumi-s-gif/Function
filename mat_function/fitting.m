for id_B = 1:num_magpoints
    contrast = squeeze(contrast_list(1, id_B, :));
    
    % ---- 初期値とフィット ----
    f_min_guess = 2870 - B_list(id_B)*28*cos(10/180*pi);
    f_pls_guess = 2870 + B_list(id_B)*28*cos(10/180*pi);

    init_params = [0.08, f_min_guess, 3, 0.08, f_pls_guess, 3]; % 2dipのとき
    opts = optimoptions('lsqcurvefit','Display','iter','MaxFunctionEvaluations',5000);
    
    [fit_params, resnorm, residual, exitflag, output, lambda, jacobian] = ...
        lsqcurvefit(model_fun, init_params, params, contrast', [], [], opts);
    
    % 残差の標準偏差と共分散行列
    dof = length(contrast) - length(fit_params);
    mse = resnorm / dof;
    covariance = mse * inv(jacobian' * jacobian);
    
    % 信頼区間の計算
    alpha = 0.05;
    t_val = tinv(1 - alpha/2, dof);
    param_std = sqrt(diag(covariance));
    ci = [fit_params(:) - t_val * param_std, fit_params(:) + t_val * param_std];
end