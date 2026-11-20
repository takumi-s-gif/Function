function make_mp4(folder, Video_name)
    % 1. ファイルリストの取得とソート
    file_list = dir(fullfile(folder, '*.png'));
    num_files = length(file_list);
    
    if num_files == 0
        error('指定されたフォルダに .png ファイルが見つかりません。');
    end

    filenames = {file_list.name};
    [~, idx] = sort(string(filenames)); 
    file_list = file_list(idx);

    % 出力パスの調整（.mp4 を付与）
    output_path = fullfile(folder, [Video_name, '.mp4']);

    % 2. VideoWriter オブジェクトの作成
    % 'MPEG-4' プロファイルを使用（高圧縮・高画質）
    v = VideoWriter(output_path, 'MPEG-4');
    
    % フレームレートの設定（秒間何枚表示するか）
    % GIFの DelayTime = 0.1s は FrameRate = 10 に相当します
    v.FrameRate = 10; 
    v.Quality = 95; % 画質設定（0~100）

    open(v); % 書き込み開始
    fprintf('function:make_MP4 Creating video: %d frames (%s).\n', num_files, char(datetime));

    % 3. 画像の読み込みと書き込み
    for i = 1:num_files
        img_path = fullfile(folder, file_list(i).name);
        img = imread(img_path);
        
        % VideoWriterはRGB画像をそのまま扱えるため、変換は不要
        % ただし、画像サイズが全フレームで一致している必要があります
        writeVideo(v, img);
    end

    close(v); % ファイルを閉じる
    fprintf('Finished making MP4: %s (%s).\n', output_path, char(datetime));
end