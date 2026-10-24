% 【用途】K 折交叉验证：把图片/来源组划成 K 份，每次一份 test，其余 train，输出 split_0 到 split_(K-1)。
% 【接口边界】每折只有 train/test，没有 val，不能原样交给要求 train/val/test 的 Python 训练入口。
% 【危险操作】outputRoot 已存在会递归删除；一次生成多折会重复复制原图，占用约 K 倍图像存储。
% K 是折数，不是训练轮数；同一组不能拆散时只能近似平衡数量，K 大于某类组数可能出现空的测试类别。
% groupAware=true 的来源组依据仍是文件名后缀启发式，不能代替真实原图/批次编号。
function split_dataset_kfold(inputRoot, outputRoot, K, groupAware, seed)
% split_dataset_kfold
% ------------------------------------------------------------
% 将图像分类数据集划分为 K 折，用于 out-of-fold 训练/预测和标签清洗。
%
% 输入目录格式：
%   inputRoot/
%   ├── normal/
%   ├── pad_surface_mark_defect/
%   ├── probe_mark_offset_defect/
%   └── surface_contamination_defect/
%
% 输出目录格式：
%   outputRoot/
%   ├── split_0/
%   │   ├── train/
%   │   │   ├── normal/
%   │   │   └── ...
%   │   └── test/
%   │       ├── normal/
%   │       └── ...
%   ├── split_1/
%   ├── split_2/
%   └── ...
%
% 参数：
%   inputRoot  : 原始数据集根目录，下面直接放类别文件夹
%   outputRoot : K 折划分后的输出根目录
%   K          : 折数，例如 5
%   groupAware : 是否启用同源图像分组划分
%                true  -> 同一原图裁出的 patch 不跨 fold
%                false -> 普通逐图随机 K 折
%   seed       : 随机种子，默认 2026
%
% 示例：
%   split_dataset_kfold( ...
%       'D:\grain_dataset\Task_data_cleaned', ...
%       'D:\grain_dataset\Task_data_kfold', ...
%       5, ...
%       true, ...
%       2026);
%
% 说明：
%   1. 第 i 折中，split_i/test 是该折 holdout 数据；
%      split_i/train 是其余 K-1 折数据。
%   2. 每张图像会在且仅在一个 split_i/test 中出现；
%      同时会在其他 K-1 个 split_i/train 中出现。
%   3. groupAware=true 时，程序默认按文件名末尾 "_数字" 之前的部分识别同源 patch。
%      例如 000320_334_01.jpg、000320_334_02.jpg 会被视为同一组。
%   4. 输出 split_summary.csv 和 fold_assignment.csv，方便检查每折数量和样本归属。

    if nargin < 3 || isempty(K)
        K = 5;
    end

    if nargin < 4 || isempty(groupAware)
        groupAware = true;
    end

    if nargin < 5 || isempty(seed)
        seed = 2026;
    end

    if K < 2 || floor(K) ~= K
        error('K 必须是大于等于 2 的整数。');
    end

    % 固定本次划分随机过程；Python 训练 seed 与它独立。种子相同但文件列表变化时，划分也可能变化。
    rng(seed);

    if ~exist(inputRoot, 'dir')
        error('输入目录不存在：%s', inputRoot);
    end

    % 若输出目录已存在，删除后重新生成，防止旧文件残留
    if exist(outputRoot, 'dir')
        warning('输出目录已存在，将被删除并重新生成：%s', outputRoot);
        % 【停止确认路径】下一行会删除 outputRoot 整棵目录，包括现有图片与报表；不是只删除空文件夹。
        rmdir(outputRoot, 's');
    end
    mkdir(outputRoot);

    % 找到类别文件夹
    classDirs = dir(inputRoot);
    classDirs = classDirs([classDirs.isdir]);
    classDirs = classDirs(~ismember({classDirs.name}, {'.', '..'}));

    if isempty(classDirs)
        error('输入目录下未找到类别子文件夹：%s', inputRoot);
    end

    % 按名称排序，保证类别顺序稳定
    [~, order] = sort({classDirs.name});
    classDirs = classDirs(order);

    splitNames = cell(1, K);
    for fi = 1:K
        splitNames{fi} = sprintf('split_%d', fi - 1);
    end

    % 创建输出目录
    for fi = 1:K
        splitRoot = fullfile(outputRoot, splitNames{fi});
        mkdir(splitRoot);
        mkdir(fullfile(splitRoot, 'train'));
        mkdir(fullfile(splitRoot, 'test'));

        % 逐类别划分，尽量保持每个类别自己的训练/验证/测试比例；不是把所有类别混在一起切一刀。
        for ci = 1:numel(classDirs)
            className = classDirs(ci).name;
            mkdir(fullfile(splitRoot, 'train', className));
            mkdir(fullfile(splitRoot, 'test', className));
        end
    end

    fprintf('开始 K 折划分数据集：\n');
    fprintf('输入目录：%s\n', inputRoot);
    fprintf('输出目录：%s\n', outputRoot);
    fprintf('K = %d\n', K);
    fprintf('groupAware = %d\n', groupAware);
    fprintf('seed = %d\n\n', seed);

    % 用于汇总
    summaryRows = {};
    assignmentRows = {};

    % 逐类别划分，尽量保持每个类别自己的训练/验证/测试比例；不是把所有类别混在一起切一刀。
    for ci = 1:numel(classDirs)
        className = classDirs(ci).name;
        classInputDir = fullfile(inputRoot, className);

        files = collect_image_files(classInputDir);
        if isempty(files)
            warning('类别 %s 中未找到图像，跳过。', className);
            continue;
        end

        % true 分组后随机分配，false 每张图片独立分配；是否可防同源泄漏取决于命名规则能否识别真实来源。
        if groupAware
            units = build_units_from_filenames(files);
        else
            units = build_units_from_individual_files(files);
        end

        numUnits = numel(units);
        if numUnits < K
            warning('类别 %s 的分组/样本数量为 %d，小于 K=%d，部分折中该类 test 可能为空。', ...
                className, numUnits, K);
        end

        % 为该类别的 units 分配 fold，尽量使每折该类图片数量均衡
        foldId = assign_units_to_folds_balanced(units, K);

        % 记录 unit 归属
        for ui = 1:numUnits
            fileListStr = strjoin(units(ui).paths, '|');
            assignmentRows(end+1, :) = { ...
                className, units(ui).key, int32(foldId(ui)-1), int32(numel(units(ui).paths)), fileListStr}; %#ok<AGROW>
        end

        fprintf('类别：%s\n', className);

        for fi = 1:K
            testMask = (foldId == fi);
            trainMask = ~testMask;

            testUnits = units(testMask);
            trainUnits = units(trainMask);

            splitRoot = fullfile(outputRoot, splitNames{fi});
            trainDst = fullfile(splitRoot, 'train', className);
            testDst  = fullfile(splitRoot, 'test', className);

            nTrain = copy_units(trainUnits, trainDst);
            nTest  = copy_units(testUnits,  testDst);

            summaryRows(end+1, :) = { ...
                splitNames{fi}, className, int32(numel(trainUnits)), int32(numel(testUnits)), ...
                int32(nTrain), int32(nTest), int32(nTrain + nTest)}; %#ok<AGROW>

            fprintf('  %-8s | train: %6d | test: %6d\n', splitNames{fi}, nTrain, nTest);
        end

        fprintf('\n');
    end

    % 保存汇总表
    SummaryTable = cell2table(summaryRows, ...
        'VariableNames', {'SplitName', 'ClassName', 'TrainUnits', 'TestUnits', ...
                          'TrainImages', 'TestImages', 'TotalImagesInThisSplit'});
    summaryPath = fullfile(outputRoot, 'split_summary.csv');
    % 写出 CSV 核对数量/路径；同名报表会被覆盖，正式实验需另行保留来源和划分版本。
    writetable(SummaryTable, summaryPath, 'Encoding', 'UTF-8');

    % 保存每个 unit 的 fold 归属
    AssignmentTable = cell2table(assignmentRows, ...
        'VariableNames', {'ClassName', 'UnitKey', 'TestFold', 'NumImagesInUnit', 'ImagePaths'});
    assignmentPath = fullfile(outputRoot, 'fold_assignment.csv');
    % 写出 CSV 核对数量/路径；同名报表会被覆盖，正式实验需另行保留来源和划分版本。
    writetable(AssignmentTable, assignmentPath, 'Encoding', 'UTF-8');

    fprintf('K 折划分完成。\n');
    fprintf('统计表已保存：%s\n', summaryPath);
    fprintf('fold 归属表已保存：%s\n', assignmentPath);
end


% 收集支持的图片扩展名，合并大小写匹配并去重；如需新增格式要同步 Python 读取支持与统计规则。
function files = collect_image_files(classDir)
% 收集常见图像格式
    patterns = {'*.jpg', '*.JPG', '*.jpeg', '*.JPEG', ...
                '*.png', '*.PNG', '*.bmp', '*.BMP', ...
                '*.tif', '*.TIF', '*.tiff', '*.TIFF'};

    files = [];
    for i = 1:numel(patterns)
        tmp = dir(fullfile(classDir, patterns{i}));
        files = [files; tmp]; %#ok<AGROW>
    end

    % 防止 Windows 大小写搜索重复
    if ~isempty(files)
        names0 = {files.name};
        [~, ia] = unique(lower(names0), 'stable');
        files = files(ia);
    end

    % 补全完整路径
    for i = 1:numel(files)
        files(i).fullpath = fullfile(classDir, files(i).name);
    end
end


% 每张图建立一个划分单元，paths 字段仅装本图路径；不根据图像内容查找重复来源。
function units = build_units_from_individual_files(files)
% 每张图作为一个独立 unit
    units = struct('key', {}, 'paths', {});
    for i = 1:numel(files)
        units(i).key = file_key_from_name(files(i).name); %#ok<AGROW>
        units(i).paths = {files(i).fullpath}; %#ok<AGROW>
    end
end


% 根据去掉末尾 _数字 的文件名建立来源组；同组图片一起进某折的 test，其余折作为 train。
% 这里保留字符串组名，用 unique 稳定去重；仍需先核实命名规则对应真实来源。
function units = build_units_from_filenames(files)
% 将同一原图裁出的 patch 分为同一 unit
%
% 默认适配：
%   000320_334_01.jpg
%   000320_334_02.jpg
%   000320_334_03.jpg
%
% 去掉末尾 "_数字" 后作为 unit key：
%   000320_334

    keyList = cell(numel(files), 1);
    for i = 1:numel(files)
        [~, base, ~] = fileparts(files(i).name);
        key = regexprep(base, '_\d+$', '');
        if isempty(key)
            key = base;
        end
        keyList{i} = key;
    end

    uniqueKeys = unique(keyList, 'stable');

    units = struct('key', {}, 'paths', {});
    for k = 1:numel(uniqueKeys)
        key = uniqueKeys{k};
        idx = strcmp(keyList, key);

        paths = cell(1, sum(idx));
        fileIdx = find(idx);
        for j = 1:numel(fileIdx)
            paths{j} = files(fileIdx(j)).fullpath;
        end

        units(k).key = key; %#ok<AGROW>
        units(k).paths = paths; %#ok<AGROW>
    end
end


% 先随机打乱，再按组内图片数量从大到小分配到当前图片数最少的折；相同数量时随机选折。
% 这是贪心平衡（每一步先选当前较均衡的位置），不保证每一折最终图片数完全相等。
function foldId = assign_units_to_folds_balanced(units, K)
% 按 unit 中图片数量尽量均衡分配到 K 个 fold。
% 分配策略：
%   1. 先随机打乱 units
%   2. 再按 unit size 从大到小分配
%   3. 每次分配到当前图片数最少的 fold

    n = numel(units);
    foldId = zeros(n, 1);

    if n == 0
        return;
    end

    unitSizes = zeros(n, 1);
    for i = 1:n
        unitSizes(i) = numel(units(i).paths);
    end

    randomOrder = randperm(n)';
    shuffledSizes = unitSizes(randomOrder);

    [~, sortIdx] = sort(shuffledSizes, 'descend');
    orderedUnitIdx = randomOrder(sortIdx);

    foldImageCounts = zeros(K, 1);

    for t = 1:n
        ui = orderedUnitIdx(t);

        minCount = min(foldImageCounts);
        candidateFolds = find(foldImageCounts == minCount);

        % 若多个 fold 图片数一样，随机选择一个，避免固定偏置
        chosenFold = candidateFolds(randi(numel(candidateFolds)));

        foldId(ui) = chosenFold;
        foldImageCounts(chosenFold) = foldImageCounts(chosenFold) + unitSizes(ui);
    end
end


% 把一组组图片复制到当前折的 train 或 test；同一原图在各折中角色不同，不是多份独立实验来源。
function nCopied = copy_units(units, dstDir)
    nCopied = 0;

    for i = 1:numel(units)
        fileList = units(i).paths;

        for j = 1:numel(fileList)
            srcPath = fileList{j};
            [~, name, ext] = fileparts(srcPath);
            dstPath = fullfile(dstDir, [name ext]);

            if exist(dstPath, 'file')
                dstPath = make_unique_path(dstPath);
            end

            copyfile(srcPath, dstPath);
            nCopied = nCopied + 1;
        end
    end
end


% 从不含扩展名的文件名构造身份字符串；用于记录，不会读取图像内容判断同源。
function key = file_key_from_name(fileName)
    [~, base, ~] = fileparts(fileName);
    key = base;
end


% 单文件重名时追加 _dup 编号，不覆盖同名图；不保护整个输出根目录免于前面的递归删除。
function uniquePath = make_unique_path(filePath)
% 如果目标路径已存在，自动增加 _dup001, _dup002 等后缀
    [folder, name, ext] = fileparts(filePath);

    idx = 1;
    while true
        candidate = fullfile(folder, sprintf('%s_dup%03d%s', name, idx, ext));
        if ~exist(candidate, 'file')
            uniquePath = candidate;
            return;
        end
        idx = idx + 1;
    end
end
