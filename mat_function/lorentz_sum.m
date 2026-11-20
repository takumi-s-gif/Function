function y_fit = lorentz_sum(params, x, num_lorentz)
    y_fit = ones(size(x));
    for i = 1:num_lorentz
        A = params(3*(i-1)+1);
        x0 = params(3*(i-1)+2);
        gamma = params(3*(i-1)+3);
        y_fit = y_fit - A * (gamma ./ ((x - x0).^2 + gamma^2));
    end
end