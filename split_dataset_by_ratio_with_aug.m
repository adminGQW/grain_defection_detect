% 【用途】先划分原始图片，再只增强 train，保留 train_raw 作为未增强的训练原图。
% 数据流：各类原图→按文件/组随机划分→train_raw 与 train 各复制一份训练原图→仅在 train 加翻转图；val/test 不增强。
% 【危险操作】已有 outputRoot 会整目录删除！输入输出不可相同，也不要互相嵌套；函数目前没有自动保护这种误设。
% 【数据可信度】groupAware=true 只用“去掉末尾下划线数字”的文件名规则猜组；不等于掌握真实原图身份。
% false 时若同源图片跨 train/val/test，模型可能见过高度相似图，成绩偏乐观；是否应分组需先核对来源。
% 默认增强所有非 normal 类；augmentClassNames={} 或 augmentModes={} 会触发默认值，不代表禁用增强。
% 如只想划分不增强，用 split_dataset_by_ratio；修改增强方式前先确认翻转不会改变真实类别含义。
function split_dataset_by_ratio_with_aug(inputRoot, outputRoot, ratios, splitNames, groupAware, seed, augmentClassNames, augmentModes)
% split_dataset_by_ratio_with_aug
% ------------------------------------------------------------
% 按比例划分图像分类数据集，并且只对训练集进行离线数据增强。
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
%   ├── train_raw/     % 原始训练集，不含增强图
%   ├── train/         % 原始训练集 + 增强图，训练代码使用这个目录
%   ├── val/           % 原始验证集，不增强
%   ├── test/          % 原始测试集，不增强
%   ├── split_summary_raw.csv
%   ├── split_summary_augmented.csv
%   └── augmentation_summary.csv
%
% 默认方案：
%   train / val / test = 30 / 10 / 60
%   normal 不增强
%   其余类别做左右翻转 hflip 和上下翻转 vflip
%
% 参数：
%   inputRoot         原始数据集根目录
%   outputRoot        输出数据集根目录
%   ratios            划分比例，默认 [30 10 60]
%   splitNames        划分名称，默认 {'train','val','test'}，必须包含 'train'
%   groupAware        是否按同源 group 划分，默认 false
%                     true  : 同一组 patch 不跨 train/val/test
%                     false : 每张图独立随机划分
%   seed              随机种子，默认 2026
%   augmentClassNames 需要增强的类别名称 cell，默认除 normal 外全部类别
%   augmentModes      增强方式 cell，默认 {'hflip','vflip'}
%
% 示例：
%   split_dataset_by_ratio_with_aug( ...
%       'D:\grain_dataset\Task_data_cleaned', ...
%       'D:\grain_dataset\Task_data_30_10_60_aug', ...
%       [30 10 60], ...
%       {'train','val','test'}, ...
%       false, ...
%       2026);
%
% 说明：
%   1. 增强操作发生在划分之后，只作用于 train_raw 中的训练图像。
%   2. val 和 test 始终保持原始图像，不做任何增强，避免评估污染。
%   3. train_raw 保存原始训练图，train 保存原始训练图和增强图。
%   4. 如果你的文件名已经无法识别同源关系，建议 groupAware=false。
% 【补充边界】第 4 点只描述无法按文件名分组时的运行选择，不表示此时评估没有数据泄漏。
% 若同源关系丢失，应尽量找回原图/采集记录建立可靠分组；false 不能排除同源图跨集合。

    if nargin < 3 || isempty(ratios)
        ratios = [30 10 60];
    end

    if nargin < 4 || isempty(splitNames)
        splitNames = {'train', 'val', 'test'};
    end

    % nargin 是实际传入的参数个数；省略参数时采用下面默认值，显式传参则优先使用传入值。
    if nargin < 5 || isempty(groupAware)
        groupAware = false;
    end

    if nargin < 6 || isempty(seed)
        seed = 2026;
    end

    if nargin < 7
        augmentClassNames = {};
    end

    if nargin < 8 || isempty(augmentModes)
        augmentModes = {'hflip', 'vflip'};
    end

    % 固定本次划分随机过程；Python 训练 seed 与它独立。种子相同但文件列表变化时，划分也可能变化。
    rng(seed);

    % (:) 把比例展开成列，再转为行向量，保证后面累计边界计算形状一致。
    ratios = ratios(:)';
    % 把 [30 10 60] 变为 [0.3 0.1 0.6]；本增强版未完整检查负数/总和为 0，传参前自行确认非负且总和大于 0。
    ratios = ratios / sum(ratios);

    if numel(ratios) ~= numel(splitNames)
        error('ratios 的数量必须与 splitNames 的数量一致。');
    end

    if ~iscell(splitNames)
        error('splitNames 必须是 cell，例如 {''train'',''val'',''test''}。');
    end

    if numel(unique(splitNames)) ~= numel(splitNames)
        error('splitNames 中存在重复名称。');
    end

    % strcmp 区分大小写；必须准确叫 train，Python 读取端也依赖这个目录名。
    trainIdx = find(strcmp(splitNames, 'train'), 1);
    if isempty(trainIdx)
        error('splitNames 必须包含 ''train''，因为增强只对训练集执行。');
    end

    if ~exist(inputRoot, 'dir')
        error('输入目录不存在：%s', inputRoot);
    end

    if exist(outputRoot, 'dir')
        warning('输出目录已存在，将被删除并重新生成：%s', outputRoot);
        % 【停止确认路径】下一行会删除 outputRoot 整棵目录，包括现有图片与报表；不是只删除空文件夹。
        rmdir(outputRoot, 's');
    end
    mkdir(outputRoot);

    trainRawRoot = fullfile(outputRoot, 'train_raw');
    trainAugRoot = fullfile(outputRoot, 'train');
    mkdir(trainRawRoot);
    mkdir(trainAugRoot);

    % 创建 val/test 等非 train 输出目录
    % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
    for si = 1:numel(splitNames)
        if strcmp(splitNames{si}, 'train')
            continue;
        end
        mkdir(fullfile(outputRoot, splitNames{si}));
    end

    classDirs = dir(inputRoot);
    classDirs = classDirs([classDirs.isdir]);
    classDirs = classDirs(~ismember({classDirs.name}, {'.', '..'}));

    if isempty(classDirs)
        error('输入目录下未找到类别子文件夹：%s', inputRoot);
    end

    allClassNames = {classDirs.name};

    % 默认增强除 normal 外的全部类别
    % 空类别列表在这里被替换为“除 normal 外全部类别”；不是“所有类别都不增强”。
    if isempty(augmentClassNames)
        augmentClassNames = allClassNames(~strcmpi(allClassNames, 'normal'));
    end

    fprintf('开始划分并增强数据集：\n');
    fprintf('输入目录：%s\n', inputRoot);
    fprintf('输出目录：%s\n', outputRoot);
    fprintf('随机种子：%d\n', seed);
    fprintf('groupAware = %d\n', groupAware);
    fprintf('划分比例：\n');
    % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
    for si = 1:numel(splitNames)
        fprintf('  %-8s %.4f\n', splitNames{si}, ratios(si));
    end
    fprintf('增强类别：%s\n', strjoin(augmentClassNames, ', '));
    fprintf('增强方式：%s\n\n', strjoin(augmentModes, ', '));

    % 汇总信息
    summaryClass = {};
    summaryGroup = [];
    rawCountsAll = [];
    augTrainCounts = [];
    totalRawAll = [];

    augSummaryClass = {};
    augOriginalTrain = [];
    augHFlip = [];
    augVFlip = [];
    augHVFlip = [];
    augRot90 = [];
    augTotalGenerated = [];
    augTrainTotalAfter = [];

    % 逐类别划分，尽量保持每个类别自己的训练/验证/测试比例；不是把所有类别混在一起切一刀。
    for ci = 1:numel(classDirs)
        className = classDirs(ci).name;
        classInputDir = fullfile(inputRoot, className);

        % 类别目录
        classTrainRawDir = fullfile(trainRawRoot, className);
        classTrainAugDir = fullfile(trainAugRoot, className);
        mkdir(classTrainRawDir);
        mkdir(classTrainAugDir);

        splitClassDirs = cell(1, numel(splitNames));
        % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
        for si = 1:numel(splitNames)
            if strcmp(splitNames{si}, 'train')
                splitClassDirs{si} = classTrainRawDir;
            else
                splitClassDirs{si} = fullfile(outputRoot, splitNames{si}, className);
                mkdir(splitClassDirs{si});
            end
        end

        files = collect_image_files(classInputDir);
        if isempty(files)
            warning('类别 %s 中未找到图像，跳过。', className);
            continue;
        end

        % true 分组后随机分配，false 每张图片独立分配；是否可防同源泄漏取决于命名规则能否识别真实来源。
        if groupAware
            groups = build_groups_from_filenames(files);
            groupKeys = fieldnames(groups);
            numUnits = numel(groupKeys);
            % 生成 1..numUnits 的随机排列，每个编号只出现一次，不是有放回抽样。
            order = randperm(numUnits);
            unitList = groupKeys(order);
            unitSplits = split_units_by_ratio(unitList, ratios);

            rawCounts = zeros(1, numel(splitNames));
            trainRawCopiedPaths = {};

            % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
            for si = 1:numel(splitNames)
                if strcmp(splitNames{si}, 'train')
                    % train 原图复制两份：train_raw 和 train
                    [rawCounts(si), trainRawCopiedPaths] = copy_groups_to_train(groups, unitSplits{si}, classTrainRawDir, classTrainAugDir);
                else
                    rawCounts(si) = copy_groups(groups, unitSplits{si}, splitClassDirs{si});
                end
            end

            usedUnits = numUnits;

        else
            numUnits = numel(files);
            % 生成 1..numUnits 的随机排列，每个编号只出现一次，不是有放回抽样。
            order = randperm(numUnits);
            unitList = files(order);
            unitSplits = split_units_by_ratio(unitList, ratios);

            rawCounts = zeros(1, numel(splitNames));
            trainRawCopiedPaths = {};

            % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
            for si = 1:numel(splitNames)
                if strcmp(splitNames{si}, 'train')
                    [rawCounts(si), trainRawCopiedPaths] = copy_files_to_train(unitSplits{si}, classTrainRawDir, classTrainAugDir);
                else
                    copy_file_list(unitSplits{si}, splitClassDirs{si});
                    rawCounts(si) = numel(unitSplits{si});
                end
            end

            usedUnits = numUnits;
        end

        % 只对指定训练类别做增强
        doAugThisClass = any(strcmp(className, augmentClassNames));
        if doAugThisClass
            augInfo = augment_train_images(trainRawCopiedPaths, classTrainAugDir, augmentModes);
        else
            augInfo = make_empty_aug_info();
        end

        trainTotalAfter = rawCounts(trainIdx) + augInfo.totalGenerated;

        fprintf('类别 %-32s | ', className);
        % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
        for si = 1:numel(splitNames)
            fprintf('%s(raw): %6d | ', splitNames{si}, rawCounts(si));
        end
        fprintf('train(after aug): %6d | aug: %6d\n', trainTotalAfter, augInfo.totalGenerated);

        summaryClass{end+1,1} = className; %#ok<AGROW>
        summaryGroup(end+1,1) = usedUnits; %#ok<AGROW>
        rawCountsAll(end+1, :) = rawCounts; %#ok<AGROW>
        augTrainCounts(end+1,1) = trainTotalAfter; %#ok<AGROW>
        totalRawAll(end+1,1) = sum(rawCounts); %#ok<AGROW>

        augSummaryClass{end+1,1} = className; %#ok<AGROW>
        augOriginalTrain(end+1,1) = rawCounts(trainIdx); %#ok<AGROW>
        augHFlip(end+1,1) = augInfo.hflip; %#ok<AGROW>
        augVFlip(end+1,1) = augInfo.vflip; %#ok<AGROW>
        augHVFlip(end+1,1) = augInfo.hvflip; %#ok<AGROW>
        augRot90(end+1,1) = augInfo.rot90; %#ok<AGROW>
        augTotalGenerated(end+1,1) = augInfo.totalGenerated; %#ok<AGROW>
        augTrainTotalAfter(end+1,1) = trainTotalAfter; %#ok<AGROW>
    end

    % 原始划分统计表
    % 把组名转换为合法结构体字段；不同原名可能转换成同一字段，需检查是否误合并。
    safeSplitVarNames = matlab.lang.makeValidName(splitNames);
    SummaryRaw = table(summaryClass, summaryGroup, ...
        'VariableNames', {'ClassName', 'NumGroupsOrSamples'});

    % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
    for si = 1:numel(splitNames)
        SummaryRaw.(safeSplitVarNames{si}) = rawCountsAll(:, si);
    end
    SummaryRaw.TotalRaw = totalRawAll;

    rawSummaryPath = fullfile(outputRoot, 'split_summary_raw.csv');
    % 写出 CSV 核对数量/路径；同名报表会被覆盖，正式实验需另行保留来源和划分版本。
    writetable(SummaryRaw, rawSummaryPath);

    % 增强后统计表
    SummaryAug = SummaryRaw;
    SummaryAug.train = augTrainCounts;
    SummaryAug.TotalAfterAug = totalRawAll - rawCountsAll(:, trainIdx) + augTrainCounts;

    augSplitSummaryPath = fullfile(outputRoot, 'split_summary_augmented.csv');
    % 写出 CSV 核对数量/路径；同名报表会被覆盖，正式实验需另行保留来源和划分版本。
    writetable(SummaryAug, augSplitSummaryPath);

    % 增强明细表
    AugTable = table(augSummaryClass, augOriginalTrain, augHFlip, augVFlip, augHVFlip, augRot90, augTotalGenerated, augTrainTotalAfter, ...
        'VariableNames', {'ClassName', 'OriginalTrain', 'AugHFlip', 'AugVFlip', 'AugHVFlip', 'AugRot90', 'TotalGenerated', 'TrainTotalAfterAug'});

    augSummaryPath = fullfile(outputRoot, 'augmentation_summary.csv');
    % 写出 CSV 核对数量/路径；同名报表会被覆盖，正式实验需另行保留来源和划分版本。
    writetable(AugTable, augSummaryPath);

    fprintf('\n处理完成。\n');
    fprintf('原始划分统计：%s\n', rawSummaryPath);
    fprintf('增强后统计：%s\n', augSplitSummaryPath);
    fprintf('增强明细统计：%s\n', augSummaryPath);
    fprintf('\n训练代码请使用目录：%s\n', trainAugRoot);
    fprintf('原始训练集保存在：%s\n', trainRawRoot);
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

    if ~isempty(files)
        names0 = {files.name};
        [~, ia] = unique(lower(names0), 'stable');
        files = files(ia);
    end

    for i = 1:numel(files)
        files(i).fullpath = fullfile(classDir, files(i).name);
    end
end


% 用文件名去掉末尾 _数字 作为组名；同组整体分配，尽量避免同源图片跨集合。
% 这是假定命名规则，不能证明同源；例如 a_1 与 a_2 会同组，但 a_hflip 未必和 a 同组。
function groups = build_groups_from_filenames(files)
% 按文件名末尾 _数字 之前的部分构造同源组
% 例如：000001_xxx_04.jpg、000001_xxx_05.jpg -> group: 000001_xxx
    groups = struct();

    for i = 1:numel(files)
        [~, base, ~] = fileparts(files(i).name);
        groupKey = regexprep(base, '_\d+$', '');
        % 把组名转换为合法结构体字段；不同原名可能转换成同一字段，需检查是否误合并。
        groupKey = matlab.lang.makeValidName(groupKey);

        if ~isfield(groups, groupKey)
            groups.(groupKey) = {};
        end
        groups.(groupKey){end+1} = files(i).fullpath;
    end
end


% 把随机后的文件/组按归一化比例切片，累计边界取 round，所以小集合数量可能不严格等于百分比。
% 若 unit 是一组图片，边界按组数而非组内图片总数计算。
function unitSplits = split_units_by_ratio(unitList, ratios)
% 将列表按比例切分为若干份
    n = numel(unitList);
    numSplits = numel(ratios);
    boundaries = round(cumsum(ratios) * n);
    boundaries(end) = n;

    unitSplits = cell(1, numSplits);
    startIdx = 1;

    for si = 1:numSplits
        endIdx = boundaries(si);
        if endIdx < startIdx
            unitSplits{si} = unitList([]);
        else
            unitSplits{si} = unitList(startIdx:endIdx);
        end
        startIdx = endIdx + 1;
    end
end


% 训练原图各复制到 train_raw 和 train，返回 train_raw 路径供后续生成增强图；不移动原始文件。
function [nCopied, trainRawCopiedPaths] = copy_files_to_train(files, trainRawDir, trainAugDir)
% 训练集原图复制到 train_raw 和 train
    nCopied = 0;
    trainRawCopiedPaths = {};

    for i = 1:numel(files)
        srcPath = files(i).fullpath;
        rawDstPath = copy_one_file(srcPath, trainRawDir);
        copy_one_file(srcPath, trainAugDir);
        trainRawCopiedPaths{end+1,1} = rawDstPath; %#ok<AGROW>
        nCopied = nCopied + 1;
    end
end


% 展开被分到训练集的来源组，复制原图两份并累计图片数；组数不能直接当图片数。
function [nCopied, trainRawCopiedPaths] = copy_groups_to_train(groups, groupKeys, trainRawDir, trainAugDir)
% 分组训练集原图复制到 train_raw 和 train
    nCopied = 0;
    trainRawCopiedPaths = {};

    for i = 1:numel(groupKeys)
        key = groupKeys{i};
        fileList = groups.(key);

        for j = 1:numel(fileList)
            srcPath = fileList{j};
            rawDstPath = copy_one_file(srcPath, trainRawDir);
            copy_one_file(srcPath, trainAugDir);
            trainRawCopiedPaths{end+1,1} = rawDstPath; %#ok<AGROW>
            nCopied = nCopied + 1;
        end
    end
end


% 展开来源组并复制到本集合对应的类别目录，不修改图像内容。
function nCopied = copy_groups(groups, groupKeys, dstDir)
    nCopied = 0;
    for i = 1:numel(groupKeys)
        key = groupKeys{i};
        fileList = groups.(key);
        for j = 1:numel(fileList)
            copy_one_file(fileList{j}, dstDir);
            nCopied = nCopied + 1;
        end
    end
end


% 按列表逐张复制文件；本辅助函数不返回计数，原图仍保留。
function copy_file_list(files, dstDir)
    for i = 1:numel(files)
        copy_one_file(files(i).fullpath, dstDir);
    end
end


% 复制一张图，目标名撞车时调用唯一命名辅助函数；不保护整个输出目录免于前面的删除。
function dstPath = copy_one_file(srcPath, dstDir)
    [~, name, ext] = fileparts(srcPath);
    dstPath = fullfile(dstDir, [name ext]);
    dstPath = make_unique_path_if_needed(dstPath);
    copyfile(srcPath, dstPath);
end


% 只读取本次训练原图生成增强，不把已生成增强图再次作为输入。
% hflip=左右翻转，vflip=上下翻转，hvflip=两次翻转，rot90=旋转 90°；仅影响训练集合。
function augInfo = augment_train_images(trainRawCopiedPaths, dstTrainDir, augmentModes)
% 对 train_raw 中的图像做增强，增强图保存到 train 目录
    augInfo = make_empty_aug_info();

    for i = 1:numel(trainRawCopiedPaths)
        srcPath = trainRawCopiedPaths{i};
        [~, name, ext] = fileparts(srcPath);

        try
            img = imread(srcPath);
        catch ME
            warning('读取图像失败，跳过增强：%s\n原因：%s', srcPath, ME.message);
            continue;
        end

        for mi = 1:numel(augmentModes)
            mode = lower(augmentModes{mi});
            augImg = [];
            suffix = '';

            switch mode
                case 'hflip'
                    augImg = fliplr(img);
                    suffix = '_aug_hflip';
                    augInfo.hflip = augInfo.hflip + 1;

                case 'vflip'
                    augImg = flipud(img);
                    suffix = '_aug_vflip';
                    augInfo.vflip = augInfo.vflip + 1;

                case 'hvflip'
                    augImg = flipud(fliplr(img));
                    suffix = '_aug_hvflip';
                    augInfo.hvflip = augInfo.hvflip + 1;

                case 'rot90'
                    augImg = rot90(img);
                    suffix = '_aug_rot90';
                    augInfo.rot90 = augInfo.rot90 + 1;

                otherwise
                    warning('未知增强方式：%s，已跳过。', mode);
                    continue;
            end

            outPath = fullfile(dstTrainDir, [name suffix ext]);
            outPath = make_unique_path_if_needed(outPath);

            try
                imwrite(augImg, outPath);
                augInfo.totalGenerated = augInfo.totalGenerated + 1;
            catch ME
                warning('保存增强图失败：%s\n原因：%s', outPath, ME.message);
                % 如果保存失败，回滚已增加的模式计数
                augInfo = rollback_aug_counter(augInfo, mode);
            end
        end
    end
end


% 初始化增强计数结构；新增增强模式时同步这里、生成分支、失败回滚和汇总列。
function augInfo = make_empty_aug_info()
    augInfo.hflip = 0;
    augInfo.vflip = 0;
    augInfo.hvflip = 0;
    augInfo.rot90 = 0;
    augInfo.totalGenerated = 0;
end


% 增强图写入失败时撤回先前增加的计数，避免报表把未成功保存的图片算进去。
function augInfo = rollback_aug_counter(augInfo, mode)
    switch lower(mode)
        case 'hflip'
            augInfo.hflip = max(0, augInfo.hflip - 1);
        case 'vflip'
            augInfo.vflip = max(0, augInfo.vflip - 1);
        case 'hvflip'
            augInfo.hvflip = max(0, augInfo.hvflip - 1);
        case 'rot90'
            augInfo.rot90 = max(0, augInfo.rot90 - 1);
    end
end


% 若目标文件名已存在，追加 _dup 编号；只解决单文件同名，不阻止 outputRoot 的整目录清理。
function uniquePath = make_unique_path_if_needed(filePath)
% 如果目标文件已存在，自动增加 _dup001 后缀，避免覆盖
    if ~exist(filePath, 'file')
        uniquePath = filePath;
        return;
    end

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
