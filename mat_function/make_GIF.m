function make_GIF(folder, GIF_name)
    % 1. ファイルリストの取得
    file_list = dir(fullfile(folder, '*.png'));
    num_files = length(file_list);
    
    if num_files == 0
        error('指定されたフォルダに .png ファイルが見つかりません。');
    end

    % ファイル名を数値順（自然順）でソート
    % ファイル名が 'frame1.png', 'frame2.png' などの場合に有効
    filenames = {file_list.name};
    [~, idx] = sort(string(filenames)); 
    file_list = file_list(idx);

    % 出力パスの調整（.gif がなければ付与）
    [~, ~, ext] = fileparts(GIF_name);
    if ~strcmpi(ext, '.gif')
        GIF_name = [GIF_name, '.gif'];
    end
    output_path = fullfile(folder, GIF_name);

    delay_time = 1; % 1.0だとかなり遅いので、必要に応じて調整してください
    fprintf('function:make_GIF Making GIF of %d pictures (%s).\n', num_files, char(datetime));

    % 2. 画像の処理と書き込み
    for i = 1:num_files
        img_path = fullfile(folder, file_list(i).name);
        [img, map] = imread(img_path);
        
        % GIF用のインデックスカラー変換
        if ndims(img) == 3
            if i == 1
                % 最初のフレームで共通のカラーマップを作成
                [img_indexed, map_indexed] = rgb2ind(img, 256, 'nodither');
            else
                % 2枚目以降は1枚目のカラーマップに合わせる（色の点滅を防ぐ）
                img_indexed = rgb2ind(img, map_indexed, 'nodither');
            end
        else
            img_indexed = img;
            if isempty(map)
                map_indexed = gray(256);
            else
                map_indexed = map;
            end
        end

        % 3. 書き込み
        if i == 1
            imwrite(img_indexed, map_indexed, output_path, 'gif', ...
                'LoopCount', Inf, 'DelayTime', delay_time);
        else
            imwrite(img_indexed, map_indexed, output_path, 'gif', ...
                'WriteMode', 'append', 'DelayTime', delay_time);
        end
    end
    fprintf('Finished making GIF: %s (%s).\n', output_path, char(datetime));
end