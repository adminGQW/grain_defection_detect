% 【用途】按比例复制原图到多个集合，不做增强；局部 helper 只供本文件调用。
% 【危险操作】输出目录存在时会 rmdir(...,'s') 删除全部内容再创建；不要与输入相同或放在输入内部。
% 【离线修改】优先改调用参数，不改后面的复制细节；Python 默认要求 train/val/test 三个名称。
% 本函数默认四集合与默认名称不完全等同于当前训练目录格式，调用时务必显式传 ratios 和 splitNames。
% 固定 rng(2026)，此函数没有 seed 形参；改变划分种子需定位该行。组划分的比例按组数近似而不是严格按图片数。
function split_dataset_by_ratio(inputRoot, outputRoot, ratios, splitNames, groupAware)
% split_dataset_by_ratio
% ------------------------------------------------------------
% 将图像分类数据集按用户指定比例划分为多个子集，并分别保存到不同文件夹。
%
% 输入目录格式：
%   inputRoot/
%   ├── class_1/
%   ├── class_2/
%   └── ...
%
% 输出目录格式：
%   outputRoot/
%   ├── train/
%   │   ├── class_1/
%   │   └── class_2/
%   ├── val/
%   ├── test/
%   └── final_test/
%
% 参数：
%   inputRoot  : 原始数据集根目录
%   outputRoot : 划分后的输出根目录
%   ratios     : 划分比例，例如 [45 15 20 20] 或 [0.45 0.15 0.20 0.20]
%                程序会自动归一化，所以不要求比例之和必须等于1
%   splitNames : 每一份对应的文件夹名称，例如 {'train','val','test','final_test'}
%   groupAware : 是否启用同源图像分组划分
%                true  -> 同一原图裁出的patch不跨集合
%                false -> 普通逐图随机划分
%
% 示例1：四份划分
%   split_dataset_by_ratio( ...
%       'D:\grain_dataset\Grain_4class', ...
%       'D:\grain_dataset\Grain_4class_split', ...
%       [45 15 20 20], ...
%       {'train','val','test','final_test'}, ...
%       true);
%
% 示例2：三份划分，同样支持
%   split_dataset_by_ratio( ...
%       'D:\grain_dataset\Grain_4class', ...
%       'D:\grain_dataset\Grain_4class_split', ...
%       [40 20 40], ...
%       {'train','val','test'}, ...
%       true);
%
% 说明：
%   1. 若你现在做的是 Normal + 3类缺陷的四分类任务，类别文件夹仍然放在 inputRoot 下。
%   2. splitNames 决定输出划分子集的数量和名称，不影响类别数量。
%   3. groupAware=true 时，程序会按文件名末尾 "_数字" 之前的部分识别同源patch。
%      例如 sample001_01.png、sample001_02.png 会被视为同一组。

    if nargin < 3 || isempty(ratios)
        ratios = [45 15 20 20];
    end

    if nargin < 4 || isempty(splitNames)
        if numel(ratios) == 4
            splitNames = {'train', 'val', 'test', 'final_test'};
        else
            splitNames = cell(1, numel(ratios));
            for i = 1:numel(ratios)
                splitNames{i} = sprintf('split_%d', i);
            end
        end
    end

    if nargin < 5
        groupAware = true;
    end

    % 固定本次划分随机过程；Python 训练 seed 与它独立。种子相同但文件列表变化时，划分也可能变化。
    rng(2026);  % 固定划分随机种子，保证划分可复现

    % (:) 把比例展开成列，再转为行向量，保证后面累计边界计算形状一致。
    ratios = ratios(:)';
    if numel(ratios) ~= numel(splitNames)
        error('ratios 的数量必须与 splitNames 的数量一致。');
    end

    if any(ratios < 0)
        error('划分比例不能为负数。');
    end

    if sum(ratios) <= 0
        error('划分比例之和必须大于0。');
    end

    % 自动归一化，兼容 [45 15 20 20] 和 [0.45 0.15 0.20 0.20]
    % 比例归一化，输入可用百分数或等比例权重；分配还受取整和分组影响。
    ratios = ratios / sum(ratios);

    % 检查 splitNames
    if ~iscell(splitNames)
        error('splitNames 必须是 cell，例如 {''train'',''val'',''test'',''final_test''}。');
    end

    if numel(unique(splitNames)) ~= numel(splitNames)
        error('splitNames 中存在重复名称，请修改。');
    end

    for i = 1:numel(splitNames)
        if isempty(splitNames{i}) || ~ischar(splitNames{i})
            error('splitNames 中每个名称必须是非空字符向量。');
        end
    end

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

    % 创建输出子集目录
    % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
    for si = 1:numel(splitNames)
        mkdir(fullfile(outputRoot, splitNames{si}));
    end

    % 找到类别文件夹
    classDirs = dir(inputRoot);
    classDirs = classDirs([classDirs.isdir]);
    classDirs = classDirs(~ismember({classDirs.name}, {'.', '..'}));

    if isempty(classDirs)
        error('输入目录下未找到类别子文件夹：%s', inputRoot);
    end

    % 汇总表
    summaryClass = {};
    summaryGroup = [];
    summaryCounts = [];
    summaryTotal = [];

    fprintf('开始划分数据集：\n');
    fprintf('输入目录：%s\n', inputRoot);
    fprintf('输出目录：%s\n', outputRoot);
    fprintf('groupAware = %d\n', groupAware);
    fprintf('划分方案：\n');
    % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
    for si = 1:numel(splitNames)
        fprintf('  %-15s %.4f\n', splitNames{si}, ratios(si));
    end
    fprintf('\n');

    % 逐类别划分，尽量保持每个类别自己的训练/验证/测试比例；不是把所有类别混在一起切一刀。
    for ci = 1:numel(classDirs)
        className = classDirs(ci).name;
        classInputDir = fullfile(inputRoot, className);

        % 创建每个划分下的类别目录
        splitClassDirs = cell(1, numel(splitNames));
        % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
        for si = 1:numel(splitNames)
            splitClassDirs{si} = fullfile(outputRoot, splitNames{si}, className);
            mkdir(splitClassDirs{si});
        end

        % 收集图像
        files = collect_image_files(classInputDir);

        if isempty(files)
            warning('类别 %s 中未找到图像，跳过。', className);
            continue;
        end

        % true 分组后随机分配，false 每张图片独立分配；是否可防同源泄漏取决于命名规则能否识别真实来源。
        if groupAware
            % 分组划分：同一原图衍生patch归为一组
            groups = build_groups_from_filenames(files);
            groupKeys = fieldnames(groups);
            numUnits = numel(groupKeys);

            % 随机打乱组
            % 生成 1..numUnits 的随机排列，每个编号只出现一次，不是有放回抽样。
            order = randperm(numUnits);
            unitList = groupKeys(order);

            % 根据比例切分组
            unitSplits = split_units_by_ratio(unitList, ratios);

            counts = zeros(1, numel(splitNames));
            % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
            for si = 1:numel(splitNames)
                counts(si) = copy_groups(groups, unitSplits{si}, splitClassDirs{si});
            end

            usedUnits = numUnits;

        else
            % 普通逐图随机划分
            numUnits = numel(files);
            % 生成 1..numUnits 的随机排列，每个编号只出现一次，不是有放回抽样。
            order = randperm(numUnits);
            unitList = files(order);

            % 根据比例切分文件
            unitSplits = split_units_by_ratio(unitList, ratios);

            counts = zeros(1, numel(splitNames));
            % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
            for si = 1:numel(splitNames)
                copy_file_list(unitSplits{si}, splitClassDirs{si});
                counts(si) = numel(unitSplits{si});
            end

            usedUnits = numUnits;
        end

        nTotal = sum(counts);

        fprintf('类别 %-28s | ', className);
        % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
        for si = 1:numel(splitNames)
            fprintf('%s: %6d | ', splitNames{si}, counts(si));
        end
        fprintf('total: %6d\n', nTotal);

        summaryClass{end+1,1} = className; %#ok<AGROW>
        summaryGroup(end+1,1) = usedUnits; %#ok<AGROW>
        summaryCounts(end+1, :) = counts; %#ok<AGROW>
        summaryTotal(end+1,1) = nTotal; %#ok<AGROW>
    end

    % 生成汇总表
    % 把组名转换为合法结构体字段；不同原名可能转换成同一字段，需检查是否误合并。
    safeSplitVarNames = matlab.lang.makeValidName(splitNames);
    SummaryTable = table(summaryClass, summaryGroup, ...
        'VariableNames', {'ClassName', 'NumGroupsOrSamples'});

    % 遍历集合名称；同一份原始图片在一次按比例划分中只进入一个集合。
    for si = 1:numel(splitNames)
        SummaryTable.(safeSplitVarNames{si}) = summaryCounts(:, si);
    end
    SummaryTable.Total = summaryTotal;

    summaryPath = fullfile(outputRoot, 'split_summary.csv');
    % 写出 CSV 核对数量/路径；同名报表会被覆盖，正式实验需另行保留来源和划分版本。
    writetable(SummaryTable, summaryPath);

    fprintf('\n划分完成。\n');
    fprintf('统计表已保存：%s\n', summaryPath);
end


% 把随机后的文件/组按归一化比例切片，累计边界取 round，所以小集合数量可能不严格等于百分比。
% 若 unit 是一组图片，边界按组数而非组内图片总数计算。
function unitSplits = split_units_by_ratio(unitList, ratios)
% 将任意列表按照比例切分为若干份，保证所有样本都会被分配一次且仅一次
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


% 用文件名去掉末尾 _数字 作为组名；同组整体分配，尽量避免同源图片跨集合。
% 这是假定命名规则，不能证明同源；例如 a_1 与 a_2 会同组，但 a_hflip 未必和 a 同组。
function groups = build_groups_from_filenames(files)
% 将同一原图裁出的 patch 分为同一组
%
% 默认适配此类命名：
%   000001_xxx_04.jpg
%   000001_xxx_05.jpg
%   000001_xxx_06.jpg
%
% 会去掉末尾 _数字 后缀作为 group key：
%   000001_xxx

    groups = struct();

    for i = 1:numel(files)
        [~, base, ~] = fileparts(files(i).name);

        % 去掉末尾的 _数字，例如 _04 / _5 / _123
        groupKey = regexprep(base, '_\d+$', '');

        % 若没有发生变化，说明无法识别同源组，则以自身作为独立组
        if strcmp(groupKey, base)
            % 把组名转换为合法结构体字段；不同原名可能转换成同一字段，需检查是否误合并。
            groupKey = matlab.lang.makeValidName(base);
        else
            % 把组名转换为合法结构体字段；不同原名可能转换成同一字段，需检查是否误合并。
            groupKey = matlab.lang.makeValidName(groupKey);
        end

        if ~isfield(groups, groupKey)
            groups.(groupKey) = {};
        end

        groups.(groupKey){end+1} = files(i).fullpath;
    end
end


% 展开来源组并复制到本集合对应的类别目录，不修改图像内容。
function nCopied = copy_groups(groups, groupKeys, dstDir)
    nCopied = 0;

    for i = 1:numel(groupKeys)
        key = groupKeys{i};
        fileList = groups.(key);

        for j = 1:numel(fileList)
            srcPath = fileList{j};
            [~, name, ext] = fileparts(srcPath);
            dstPath = fullfile(dstDir, [name ext]);
            copyfile(srcPath, dstPath);
            nCopied = nCopied + 1;
        end
    end
end


% 按列表逐张复制文件；本辅助函数不返回计数，原图仍保留。
function copy_file_list(files, dstDir)
    for i = 1:numel(files)
        srcPath = files(i).fullpath;
        [~, name, ext] = fileparts(srcPath);
        dstPath = fullfile(dstDir, [name ext]);
        copyfile(srcPath, dstPath);
    end
end
