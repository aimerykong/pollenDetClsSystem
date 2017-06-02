%LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:local matlab -nodisplay
%{
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-7.5', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda-7.5/cudnn-v5') ;

%}
clear
close all
clc;
rng(777);
addpath('./libs/exportFig')
addpath('./libs/layerExt')
addpath('./libs/myFunctions')
path_to_matconvnet = './libs/matconvnet';
path_to_model = '../models/';

run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

mean_r = 123.68; %means to be subtracted and the given values are used in our training stage
mean_g = 116.779;
mean_b = 103.939;
mean_bgr = reshape([mean_b, mean_g, mean_r], [1,1,3]);
mean_rgb = reshape([mean_r, mean_g, mean_b], [1,1,3]);
%% load imdb and dataset file
load('imdb_merge4cls_predSeg.mat');
saveRootFolder = './resultFigure';
if ~isdir(saveRootFolder)
    mkdir(saveRootFolder);
end
%% read matconvnet model
% set GPU
gpuId = 3; %[1, 2];
gpuDevice(gpuId);
flagSaveFig = false; % {true false} whether to store the result

% saveFolder = 'main030_cls_res50_gtOnly_avgPool';
% modelName = 'softmax_net-epoch-35.mat';
% imdb.meta.width = 672;
% imdb.meta.height = 672;

% saveFolder = 'main030_cls_res50_gtOnly_avgPool_masking';
% modelName = 'softmax_net-epoch-38.mat'; % 38

saveFolder = 'main030_cls_res50_alldata_avgPool_masking';
modelName = 'softmax_net-epoch-38.mat'; % 18 19 22 25 29

saveFolder = 'main041_cls_res50_alldata_avgPool_preSegMasking';
modelName = 'softmax_net-epoch-29.mat'; % 6 9 10 11 16 


saveToken = [strrep(saveFolder, '/', ''), '_', strrep(modelName, '/', '')];
load(sprintf('result_cls_%s.mat', saveToken));
%% confusion matrix for all class
acc = mean(predList(:) == grndList(:) );
[Conf_Mat, GORDER] = confusionmat(grndList, predList);

trNum = zeros(length(imdb.meta.className),1);
for i = 1:length(imdb.meta.className)
    trNum(i) = sum(grndList==i);
end
trNum = repmat(trNum, [1,length(imdb.meta.className)]);

c = Conf_Mat ./ trNum; % normalize into [0,1]

imgFig1 = figure(1);
set(imgFig1, 'Position', [100 100 1500 1000]) % [1 1 width height]
imagesc(c);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                
textStrings = num2str(c(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:size(Conf_Mat,1));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
    'HorizontalAlignment','center', 'FontSize', 5);
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range

textColors = repmat(c(:) > midValue,1,3);  %# Choose white or black for the
%#   text color of the strings so
%#   they can be easily seen over
%#   the background color
set(hStrings,{'Color'}, num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick', 1:size(Conf_Mat,1),...                         %# Change the axes tick marks
    'XTickLabel', imdb.meta.className,...  %#   and tick labels
    'YTick', 1:size(Conf_Mat,1),...
    'YTickLabel', imdb.meta.className,...
    'TickLength', [0 0]);

title( sprintf('confusion matrix on test set (acc=%.2f%%)', acc*100) );
ylabel('ground-truth label');
xlabel('predicted label');
xticklabel_rotate([], 45, [], 'Fontsize', 10);

export_fig( sprintf('%s/%s_confusionMatrix_allClass.jpg', saveRootFolder, saveToken) );
%% confusion matrix for all class, sorted by accuracy
tmpAccList = diag(c);

[sortedtmpAccList, sortedtmpAccListIdx] = sort(tmpAccList, 'descend');

c_sorted_by_acc = c(sortedtmpAccListIdx, sortedtmpAccListIdx);
imgFig2 = figure(2);
set(imgFig2, 'Position', [100 100 1500 1000]) % [1 1 width height]
imagesc(c_sorted_by_acc);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                
textStrings = num2str(c_sorted_by_acc(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:size(Conf_Mat,1));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
    'HorizontalAlignment','center', 'FontSize', 5);
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range

textColors = repmat(c_sorted_by_acc(:) > midValue,1,3);  %# Choose white or black for the
%#   text color of the strings so
%#   they can be easily seen over
%#   the background color
set(hStrings,{'Color'}, num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick', 1:size(Conf_Mat,1),...                         %# Change the axes tick marks
    'XTickLabel', imdb.meta.className(sortedtmpAccListIdx),...  %#   and tick labels
    'YTick', 1:size(Conf_Mat,1),...
    'YTickLabel', imdb.meta.className(sortedtmpAccListIdx),...
    'TickLength', [0 0]);

title( sprintf('confusion matrix (sorted by accuracy) on test set (acc=%.2f%%)', acc*100) );
ylabel('ground-truth label');
xlabel('predicted label');
xticklabel_rotate([],45,[],'Fontsize',10);

export_fig( sprintf('%s/%s_confusionMatrix_allClass_sortedByAccuracy.jpg', saveRootFolder, saveToken) );
%% confusion matrix for all class, sorted by number
[occurrenceLabel, uniqueLabel] = hist(grndList, unique(grndList));
[sortedoccurrenceLabel, sortedoccurrenceLabelIdx] = sort(occurrenceLabel, 'descend');

c_sorted_by_num = c(sortedoccurrenceLabelIdx, sortedoccurrenceLabelIdx);
imgFig3 = figure(3);
set(imgFig3, 'Position', [100 100 1500 1000]) % [1 1 width height]
imagesc(c_sorted_by_num);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                
textStrings = num2str(c_sorted_by_num(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:size(Conf_Mat,1));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
    'HorizontalAlignment','center', 'FontSize', 5);
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range

textColors = repmat(c_sorted_by_num(:) > midValue,1,3);  %# Choose white or black for the
%#   text color of the strings so
%#   they can be easily seen over
%#   the background color
set(hStrings,{'Color'}, num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick', 1:size(Conf_Mat,1),...                         %# Change the axes tick marks
    'XTickLabel', imdb.meta.className(sortedoccurrenceLabelIdx),...  %#   and tick labels
    'YTick', 1:size(Conf_Mat,1),...
    'YTickLabel', imdb.meta.className(sortedoccurrenceLabelIdx),...
    'TickLength', [0 0]);

title( sprintf('confusion matrix (sorted by image number) on test set (acc=%.2f%%)', acc*100) );
ylabel('ground-truth label');
xlabel('predicted label');
xticklabel_rotate([],45,[],'Fontsize',10);

export_fig( sprintf('%s/%s_confusionMatrix_allClass_sortedByNumber.jpg', saveRootFolder, saveToken) );
%%  sub confusion matrix by top 25 
%%{
accList = diag(c);
[sortedAccList, sortedAccListIdx] = sort(accList, 'descend');topNclass = 25;
subClassList = sortedAccListIdx(1:topNclass+2);

b = find(subClassList==48);
if isempty(b)
    subClassList = subClassList(1:topNclass+1);
else
    subClassList = subClassList([1:b-1, b+1:end]);
end

b = find(subClassList==23);
if isempty(b)
    subClassList = subClassList(1:topNclass+1);
else
    subClassList = subClassList([1:b-1, b+1:end]);
end


subpredList = [];
tmp = setdiff(1:length(imdb.meta.className), subClassList);
scoreMatTMP = scoreMat;
scoreMatTMP(tmp, :) = -Inf;
for i = 1:length(grndList)
    if isempty(find( subClassList==grndList(i) ))
        subpredList(i) = -1;
    else
        [~, tmp] = max(scoreMatTMP(:,i));
        subpredList(i) = tmp;
    end
end
tmp = find(subpredList~=-1);
subpredList = subpredList(tmp);
subgrndList = grndList(tmp);
subacc = mean(subpredList(:) == subgrndList(:) );
[subConf_Mat, subGORDER] = confusionmat(subgrndList, subpredList);

subClassList = sort(subClassList, 'ascend');
subtrNum = zeros(length(subClassList),1);
for i = 1:length(subClassList)
    subtrNum(i) = sum(subgrndList==subClassList(i));
end
subtrNum = repmat(subtrNum, [1,length(subClassList)]);
subc = subConf_Mat ./ subtrNum; % normalize into [0,1]


imgFig4 = figure(4);
set(imgFig4, 'Position', [100 100 1500 1000]) % [1 1 width height]
imagesc(subc);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                
textStrings = num2str(subc(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:size(subConf_Mat,1));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
    'HorizontalAlignment','center', 'FontSize', 9);
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(subc(:) > midValue,1,3);  %# Choose white or black for the
%#   text color of the strings so
%#   they can be easily seen over
%#   the background color
set(hStrings,{'Color'}, num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick', 1:size(subConf_Mat,1),...                         %# Change the axes tick marks
    'XTickLabel', imdb.meta.className(subClassList),...  %#   and tick labels
    'YTick', 1:size(subConf_Mat,1),...
    'YTickLabel', imdb.meta.className(subClassList),...
    'TickLength', [0 0]);

title( sprintf('confusion matrix on test set (acc=%.2f%%)', subacc*100) );
ylabel('ground-truth label');
xlabel('predicted label');
xticklabel_rotate([],45,[],'Fontsize',10)

export_fig( sprintf('%s/%s_confusionMatrix_top%dClassByAccuracy.jpg', saveRootFolder, saveToken, length(subClassList)) );
%}
%% accuracy vs. number
[occurrenceLabel, uniqueLabel] = hist(grndList, unique(grndList));
[~, b] = sort(uniqueLabel);
uniqueLabel = uniqueLabel(b);
occurrenceLabel = occurrenceLabel(b);
accList = diag(c);

[occurrenceLabel_sorted, b] = sort(occurrenceLabel, 'ascend');
accList_sorted = accList(b);
uniqueLabel_sorted = uniqueLabel(b);
nameList_sorted = imdb.meta.className(b);
for i = 1:length(occurrenceLabel_sorted)
    nameList_sorted{i} = sprintf('%s (%d)', nameList_sorted{i}, occurrenceLabel_sorted(i) );
end

b = find(uniqueLabel_sorted==48);
uniqueLabel_sorted = uniqueLabel_sorted([1:b-1, b+1:end]);
occurrenceLabel_sorted = occurrenceLabel_sorted([1:b-1, b+1:end]);
nameList_sorted = nameList_sorted([1:b-1, b+1:end]);
accList_sorted = accList_sorted([1:b-1, b+1:end]);

b = find(uniqueLabel_sorted==23);
uniqueLabel_sorted = uniqueLabel_sorted([1:b-1, b+1:end]);
occurrenceLabel_sorted = occurrenceLabel_sorted([1:b-1, b+1:end]);
nameList_sorted = nameList_sorted([1:b-1, b+1:end]);
accList_sorted = accList_sorted([1:b-1, b+1:end]);



imgFig5 = figure(5);
set(imgFig5, 'Position', [100 100 1700 700]) % [1 1 width height]
hold on;
plot( 1:length(occurrenceLabel_sorted), ((1:length(occurrenceLabel_sorted))-1)/(length(occurrenceLabel_sorted)-1), '.-r');

scatter( 1:length(occurrenceLabel_sorted), accList_sorted );
set(gca,'XTick', 1:length(occurrenceLabel_sorted),...                         %# Change the axes tick marks
    'XTickLabel', nameList_sorted );
xticklabel_rotate([],45,[],'Fontsize',10)
xlabel('species name with number of available data');
ylabel('accuracy (ratio of correctly classified data in each class)');
hold off;

export_fig( sprintf('%s/%s_acc_vs_number.jpg', saveRootFolder, saveToken) );
%% sub confusion matrix top 25 common species
%%{
topNclass = 25;

[occurrenceLabel, uniqueLabel] = hist(grndList, unique(grndList));

[~, b] = sort(uniqueLabel);
uniqueLabel = uniqueLabel(b);
occurrenceLabel = occurrenceLabel(b);


[sortedoccurrenceLabel, sortedoccurrenceLabelIdx] = sort(occurrenceLabel, 'descend');

subClassList = sortedoccurrenceLabelIdx(1:topNclass+2);
b = find(subClassList==48);
if isempty(b)
    subClassList = subClassList(1:topNclass+1);
else
    subClassList = subClassList([1:b-1, b+1:end]);
end

b = find(subClassList==23);
if isempty(b)
    subClassList = subClassList(1:topNclass);
else
    subClassList = subClassList([1:b-1, b+1:end]);
end

subpredList = [];
tmp = setdiff(1:length(imdb.meta.className), subClassList);
scoreMatTMP = scoreMat;
scoreMatTMP(tmp, :) = -Inf;
for i = 1:length(grndList)
    if isempty(find( subClassList==grndList(i) ))
        subpredList(i) = -1;
    else
        [~, tmp] = max(scoreMatTMP(:,i));
        subpredList(i) = tmp;
    end
end
tmp = find(subpredList~=-1);
subpredList = subpredList(tmp);
subgrndList = grndList(tmp);
subacc = mean(subpredList(:) == subgrndList(:) );
[subConf_Mat, subGORDER] = confusionmat(subgrndList, subpredList);

subClassList = sort(subClassList, 'ascend');
subtrNum = zeros(length(subClassList),1);
for i = 1:length(subClassList)
    subtrNum(i) = sum(subgrndList==subClassList(i));
end
subtrNum = repmat(subtrNum, [1,length(subClassList)]);
subc = subConf_Mat ./ subtrNum; % normalize into [0,1]


imgFig6 = figure(6);
set(imgFig6, 'Position', [100 100 1500 1000]) % [1 1 width height]
imagesc(subc);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                
textStrings = num2str(subc(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:size(subConf_Mat,1));   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
    'HorizontalAlignment','center', 'FontSize', 9);
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(subc(:) > midValue,1,3);  %# Choose white or black for the
%#   text color of the strings so
%#   they can be easily seen over
%#   the background color
set(hStrings,{'Color'}, num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick', 1:size(subConf_Mat,1),...                         %# Change the axes tick marks
    'XTickLabel', imdb.meta.className(subClassList),...  %#   and tick labels
    'YTick', 1:size(subConf_Mat,1),...
    'YTickLabel', imdb.meta.className(subClassList),...
    'TickLength', [0 0]);

title( sprintf('confusion matrix on test set (acc=%.2f%%)', subacc*100) );
ylabel('ground-truth label');
xlabel('predicted label');
xticklabel_rotate([],45,[],'Fontsize',10)

export_fig( sprintf('%s/%s_confusionMatrix_top%dClassByNumber.jpg', saveRootFolder, saveToken, length(subClassList)) );
%% leaving blank

