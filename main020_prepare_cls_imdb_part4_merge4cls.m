clear
clc
close all;
%%
path_to_part1 = '/home/skong2/pollenProject_dataset_patch4cls';
path_to_part2 = '/home/skong2/pollenProject_dataset_patch4cls_part2';

cls_part1 = dir(path_to_part1); cls_part1 = {cls_part1(3:end).name}; cls_part1 = sort(cls_part1);
cls_part2 = dir(path_to_part2); cls_part2 = {cls_part2(3:end).name}; cls_part2 = sort(cls_part2);

clsNameList = unique([cls_part1 cls_part2]);

[I1,J1]=setdiff(cls_part1, cls_part2);
[I2,J2]=setdiff(cls_part2, cls_part1);

clsStat_part1 = zeros(1, length(clsNameList));
clsStat_part2 = zeros(1, length(clsNameList));
for i = 1:length(clsNameList)
    tmp = dir( fullfile(path_to_part1, clsNameList{i}) ); tmp = tmp(3:end);
    clsStat_part1(i) = length(tmp);
    tmp = dir( fullfile(path_to_part2, clsNameList{i}) ); tmp = tmp(3:end);
    clsStat_part2(i) = length(tmp);
end
%% show statistics, NO.104 is the reject class
thresholdX = 30;
figure(1);

subplot(3,1,1);
bar(clsStat_part1);
ylim([0 100]);
numGreaterThanX = sum(clsStat_part1>=thresholdX);
title(sprintf('part1 (grd) distr. confTresh>=%d, NO[>%d]=%d', 0, thresholdX, numGreaterThanX));

subplot(3,1,2);
bar(clsStat_part2);
ylim([0 100]);
numGreaterThanX = sum(clsStat_part2>=thresholdX);
title(sprintf('part2 (det) distr. confTresh>=%d, NO[>%d]=%d', 0, thresholdX, numGreaterThanX));


subplot(3,1,3);
clsStat_merge = clsStat_part1 + clsStat_part2;
bar(clsStat_merge);
ylim([0 100]);
numGreaterThanX = sum(clsStat_merge>=thresholdX);
title(sprintf('merge distr. confTresh>=%d, NO[>%d]=%d', 0, thresholdX, numGreaterThanX));
%% prepare imdb for part1 (the ground-truth)
fprintf('prepare imdb for part1 (the ground-truth)\n');
validClass = find(clsStat_part1>=thresholdX);
imdb.meta.className = clsNameList(validClass);
imdb.meta.className{end+1} = 'reject';
imdb.meta.className2Label = containers.Map;
imdb.labelList = [];
imdb.imgList = {};
imdb.maskList = {};
imdb.set = [];
rng(777);
for i = 1:length(imdb.meta.className)
    imdb.meta.className2Label(imdb.meta.className{i}) = i;
end
for i = 1:length(cls_part1)
    fprintf('%d %s\n', i, cls_part1{i});
    curImgList = dir( fullfile(path_to_part1, cls_part1{i}) );
    curImgList = curImgList(3:end);
    for j = 1:length(curImgList)
        if isKey(imdb.meta.className2Label, cls_part1{i})
            imdb.labelList(end+1) = imdb.meta.className2Label(cls_part1{i});
            imdb.imgList{end+1} = fullfile(path_to_part1, cls_part1{i}, curImgList(j).name);
            imdb.maskList{end+1} = fullfile('/home/skong2/pollenProject_dataset_patch4mask', cls_part1{i}, curImgList(j).name);
            if rand(1) > 0.3
                imdb.set(end+1) = 1;
            else
                imdb.set(end+1) = 2;
            end
        else
            curClass = 'reject';
            imdb.labelList(end+1) = imdb.meta.className2Label(curClass);
            imdb.imgList{end+1} = fullfile(path_to_part1, cls_part1{i}, curImgList(j).name);
            imdb.maskList{end+1} = fullfile('/home/skong2/pollenProject_dataset_patch4mask/', cls_part1{i}, curImgList(j).name);            
            if rand(1) > 0.3
                imdb.set(end+1) = 1;
            else
                imdb.set(end+1) = 2;
            end
        end
    end
end
save('imdb_gtPatch4cls.mat', 'imdb');
%% append part2 to the imdb struct
fprintf('append part2 to the imdb struct\n');
for i = 1:length(cls_part2)
    fprintf('%d %s\n', i, cls_part2{i});
    curImgList = dir( fullfile(path_to_part2, cls_part2{i}) );
    curImgList = curImgList(3:end);
    for j = 1:length(curImgList)
        if isKey(imdb.meta.className2Label, cls_part2{i})
            imdb.labelList(end+1) = imdb.meta.className2Label(cls_part2{i});
            imdb.imgList{end+1} = fullfile(path_to_part2, cls_part2{i}, curImgList(j).name);
            imdb.maskList{end+1} = fullfile('/home/skong2/pollenProject_dataset_patch4mask_part2', cls_part2{i}, curImgList(j).name);
            imdb.set(end+1) = 1;
        else
            curClass = 'reject';
            imdb.labelList(end+1) = imdb.meta.className2Label(curClass);
            imdb.imgList{end+1} = fullfile(path_to_part2, cls_part2{i}, curImgList(j).name);
            imdb.maskList{end+1} = fullfile('/home/skong2/pollenProject_dataset_patch4mask_part2', cls_part2{i}, curImgList(j).name);            
            imdb.set(end+1) = 1;
        end
    end
end
save('imdb_merge4cls.mat', 'imdb');





