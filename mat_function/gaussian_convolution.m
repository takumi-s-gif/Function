function binned_data = gaussian_convolution(data)
L = 130; %um
N = 2048;
sigma = 0.1; %um

dx = L / N;
sigma_px = sigma / dx;

% --- ① Gaussianカーネル作成 ---
sz = ceil(6 * sigma_px); % カーネルサイズ（±3σ）
% if mod(sz,2)==0
%     sz = sz + 1;
% end

% --- ② 畳み込み ---
h = fspecial('gaussian', sz, sigma_px);
binned_data = imfilter(data, h, 'replicate');
end