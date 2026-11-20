function [data_list, name_list] = load_folder_data(folder)
    file_list = dir(fullfile(folder, '*.mat'));
    num_files = length(file_list);

    data_list = cell(1, num_files);
    name_list = cell(1, num_files);

    for i = 1:num_files
        % ファイル名（拡張子あり）を取得
        fname_with_ext = file_list(i).name;
        
        % フルパスを生成してロード
        fpath = fullfile(folder, fname_with_ext);
        data_list{i} = load(fpath);
        
        % 拡張子を除いた名前をリストに格納
        [~, name_only, ~] = fileparts(fname_with_ext);
        name_list{i} = name_only;
        
        fprintf('Loaded [%d/%d]: %s\n', i, num_files, fname_with_ext);
    end
end