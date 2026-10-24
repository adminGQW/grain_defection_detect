% 【用途】批量改图片文件名，不训练模型，也不改变图片所属类别文件夹。
% 【先检查】datasetRoot 仍是原作者电脑的绝对路径；运行前必须修改为自己确认过的数据目录。
% dryRun=false 会真实重命名；先用 true 预览映射，它仍会写出 dryrun CSV，不是完全无文件写入。
% 先把所有原图移到 UUID 临时名，再改目标名，防止相互占名；中途失败没有自动回滚，可能留下临时名。
% 映射 CSV 结束时才写出，同路径映射可能覆盖；操作前保留原始数据和已有映射。
% 前缀 1/2/3/4 是这里自定的文件名编码，不是 Python 的 0 起始类别编号。
% 每个集合/类别独立编号，不能靠新文件名恢复跨集合的原图身份；改名还可能丢失增强/来源后缀，谨慎再划分。
function rename_grain_dataset_files()
% rename_grain_dataset_files
% ------------------------------------------------------------
% 对 Task_data_30_10_60_aug 下 train / train_raw / val / test
% 四个数据子集中的四个类别文件夹进行重新编号命名。需要先增广再用这个代码
%重命名这一步可以不做。
%
% 命名规则：
%   normal                         -> 1_x.jpg
%   pad_surface_mark_defect         -> 2_x.jpg
%   probe_mark_offset_defect        -> 3_x.jpg
%   surface_contamination_defect    -> 4_x.jpg
%
% 注意：
%   1. 建议先备份数据集。
%   2. 本代码会真实修改文件名。
%   3. 会自动保存 rename_mapping.csv，用于追溯原文件名。

    % ==========================
    % 1. 修改为你的数据集路径
    % ==========================
    % 这是作者旧路径，不能照搬；修改前先确认目标是准备改名的副本而非唯一原始资料。
    datasetRoot = 'E:\文档\2025尖兵\图片数据\朗讯\5月20所有数据合并\项目指标代码v2\Task_data_30_10_60_aug';

    % 需要处理的数据子集
    splitNames = {'train', 'train_raw', 'val', 'test'};

    % 类别顺序，决定前缀编号
    classNames = { ...
        'normal', ...
        'pad_surface_mark_defect', ...
        'probe_mark_offset_defect', ...
        'surface_contamination_defect' ...
    };

    % 是否只预览不真正改名
    % 第一次建议 true，确认没问题后再改成 false
    % 【实际开关】当前是 false，会执行重命名；第一次预览需手动改为 true，不能只看上面的建议注释。
    dryRun = false;

    % ==========================
    % 2. 开始处理
    % ==========================
    if ~exist(datasetRoot, 'dir')
        error('数据集根目录不存在：%s', datasetRoot);
    end

    allRows = {};
    rowIdx = 1;

    fprintf('数据集根目录：%s\n', datasetRoot);
    fprintf('dryRun = %d\n\n', dryRun);

    % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
    for si = 1:numel(splitNames)
        splitName = splitNames{si};
        splitDir = fullfile(datasetRoot, splitName);

        if ~exist(splitDir, 'dir')
            warning('子集文件夹不存在，跳过：%s', splitDir);
            continue;
        end

        fprintf('\n==============================\n');
        fprintf('处理子集：%s\n', splitName);
        fprintf('==============================\n');

        for ci = 1:numel(classNames)
            className = classNames{ci};
            classDir = fullfile(splitDir, className);

            if ~exist(classDir, 'dir')
                warning('类别文件夹不存在，跳过：%s', classDir);
                continue;
            end

            % ci 是 classNames 列表中的位置，从 1 开始；此值只是文件名前缀，不参与 Python 分类标签推导。
            prefix = ci;

            files = collect_image_files(classDir);

            if isempty(files)
                warning('类别文件夹中没有图像：%s', classDir);
                continue;
            end

            % 按文件名排序，保证重命名顺序稳定
            [~, order] = sort(lower({files.name}));
            files = files(order);

            fprintf('类别 %-35s 文件数：%d\n', className, numel(files));

            % 第一步：先改成临时文件名，避免目标文件名冲突
            tempPaths = cell(numel(files), 1);
            oldPaths = cell(numel(files), 1);
            newPaths = cell(numel(files), 1);
            oldNames = cell(numel(files), 1);
            newNames = cell(numel(files), 1);

            for i = 1:numel(files)
                oldName = files(i).name;
                oldPath = fullfile(classDir, oldName);

                [~, ~, ext] = fileparts(oldName);

                % 如果你的图像全部都是 jpg，这里会生成 1_1.jpg
                % 如果原来是 png，则会生成 1_1.png
                % 不建议只改扩展名为 jpg，因为那不是格式转换
                newName = sprintf('%d_%d%s', prefix, i, lower(ext));

                tempName = sprintf('__tmp_rename_%s_%06d%s', char(java.util.UUID.randomUUID), i, lower(ext));
                tempPath = fullfile(classDir, tempName);
                newPath = fullfile(classDir, newName);

                oldPaths{i} = oldPath;
                tempPaths{i} = tempPath;
                newPaths{i} = newPath;
                oldNames{i} = oldName;
                newNames{i} = newName;
            end

            % 检查目标路径是否重复
            if numel(unique(newPaths)) ~= numel(newPaths)
                error('目标文件名存在重复，请检查：%s / %s', splitName, className);
            end

            if dryRun
                for i = 1:numel(files)
                    fprintf('[DryRun] %s  ->  %s\n', oldNames{i}, newNames{i});

                    allRows(rowIdx, :) = { ...
                        splitName, className, prefix, i, ...
                        oldNames{i}, newNames{i}, ...
                        oldPaths{i}, newPaths{i}, ...
                        'dryrun' ...
                    };
                    rowIdx = rowIdx + 1;
                end
            else
                % 先改成临时名
                for i = 1:numel(files)
                    movefile(oldPaths{i}, tempPaths{i});
                end

                % 再由临时名改成最终名
                for i = 1:numel(files)
                    movefile(tempPaths{i}, newPaths{i});

                    fprintf('[Renamed] %s  ->  %s\n', oldNames{i}, newNames{i});

                    allRows(rowIdx, :) = { ...
                        splitName, className, prefix, i, ...
                        oldNames{i}, newNames{i}, ...
                        oldPaths{i}, newPaths{i}, ...
                        'renamed' ...
                    };
                    rowIdx = rowIdx + 1;
                end
            end
        end
    end

    % ==========================
    % 3. 保存重命名映射表
    % ==========================
    if ~isempty(allRows)
        mappingTable = cell2table(allRows, ...
            'VariableNames', { ...
                'SplitName', ...
                'ClassName', ...
                'ClassPrefix', ...
                'FileIndex', ...
                'OldFileName', ...
                'NewFileName', ...
                'OldPath', ...
                'NewPath', ...
                'Status' ...
            });

        if dryRun
            mappingPath = fullfile(datasetRoot, 'rename_mapping_dryrun.csv');
        else
            mappingPath = fullfile(datasetRoot, 'rename_mapping.csv');
        end

        % 写出 CSV 核对数量/路径；同名报表会被覆盖，正式实验需另行保留来源和划分版本。
        writetable(mappingTable, mappingPath, 'Encoding', 'UTF-8');
        fprintf('\n映射表已保存：%s\n', mappingPath);
    end

    if dryRun
        fprintf('\n当前 dryRun = true，只预览，没有真正改名。\n');
        fprintf('确认输出无误后，把 dryRun 改成 false 再运行。\n');
    else
        fprintf('\n重命名完成。\n');
    end
end


% 收集支持的图片扩展名，合并大小写匹配并去重；如需新增格式要同步 Python 读取支持与统计规则。
function files = collect_image_files(folder)
% 收集常见图像格式

    patterns = { ...
        '*.jpg', '*.JPG', ...
        '*.jpeg', '*.JPEG', ...
        '*.png', '*.PNG', ...
        '*.bmp', '*.BMP', ...
        '*.tif', '*.TIF', ...
        '*.tiff', '*.TIFF' ...
    };

    files = [];

    for k = 1:numel(patterns)
        tmp = dir(fullfile(folder, patterns{k}));
        files = [files; tmp]; %#ok<AGROW>
    end

    if ~isempty(files)
        % 去重，防止 Windows 下大小写重复匹配
        names = {files.name};
        [~, ia] = unique(lower(names), 'stable');
        files = files(ia);
    end
end