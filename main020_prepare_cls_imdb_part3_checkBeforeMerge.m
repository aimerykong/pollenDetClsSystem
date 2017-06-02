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
thresholdX = 40;
figure(1);

subplot(3,1,1);
bar(clsStat_part1);
numGreaterThanX = sum(clsStat_part1>=thresholdX);
title(sprintf('part1 (grd) distr. confTresh>=%d, NO[>%d]=%d', 0, thresholdX, numGreaterThanX));

subplot(3,1,2);
bar(clsStat_part2);
numGreaterThanX = sum(clsStat_part2>=thresholdX);
title(sprintf('part2 (det) distr. confTresh>=%d, NO[>%d]=%d', 0, thresholdX, numGreaterThanX));


subplot(3,1,3);
clsStat_merge = clsStat_part1 + clsStat_part2;
bar(clsStat_merge);
numGreaterThanX = sum(clsStat_merge>=thresholdX);
title(sprintf('merge distr. confTresh>=%d, NO[>%d]=%d', 0, thresholdX, numGreaterThanX));



