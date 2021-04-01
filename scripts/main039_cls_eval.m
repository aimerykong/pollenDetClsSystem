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
addpath('../libs/exportFig')
addpath('../libs/layerExt')
addpath('../libs/myFunctions')
path_to_matconvnet = '/home/skong2/project/autoSegClsSystem_BCI/libs/matconvnet';
path_to_model = '../models/';

run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

mean_r = 123.68; %means to be subtracted and the given values are used in our training stage
mean_g = 116.779;
mean_b = 103.939;
mean_bgr = reshape([mean_b, mean_g, mean_r], [1,1,3]);
mean_rgb = reshape([mean_r, mean_g, mean_b], [1,1,3]);
%% load imdb and dataset file
load('../imdb_files/imdb_merge4cls_predSeg.mat');
%% read matconvnet model
% set GPU
gpuId = 1; %[1, 2];
gpuDevice(gpuId);
flagSaveFig = false; % {true false} whether to store the result

% saveFolder = 'main030_cls_res50_gtOnly_avgPool';
% modelName = 'softmax_net-epoch-35.mat';
% imdb.meta.width = 672;
% imdb.meta.height = 672;

% saveFolder = 'main030_cls_res50_gtOnly_avgPool_masking';
% modelName = 'softmax_net-epoch-38.mat';

saveFolder = 'main030_cls_res50_alldata_avgPool_masking';
modelName = 'softmax_net-epoch-38.mat';

saveFolder = 'main041_cls_res50_alldata_avgPool_preSegMasking';
modelName = 'softmax_net-epoch-44.mat';


saveToken = [strrep(saveFolder, '/', ''), '_', strrep(modelName, '/', '')];
%% setup network
netMat = load( fullfile('./exp', saveFolder, modelName) );
netMat = netMat.net;
netMat = dagnn.DagNN.loadobj(netMat);

rmLayerName = 'obj_seg';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer

    baseName = 'res6';
    upsample_fac = 8;
    filters = single(bilinear_u(upsample_fac*2, 2, 2));
    crop = ones(1,4) * upsample_fac/2;
    deconv_name = [baseName, '_interp'];
    var_to_up_sample = sName;
    netMat.addLayer(deconv_name, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', 2, ...
        'hasBias', false), ...
        var_to_up_sample, deconv_name, {[deconv_name  '_f']}) ;
    ind = netMat.getParamIndex([deconv_name  '_f']) ;
    netMat.params(ind).value = filters ;
    sName = deconv_name;

    layerTop = sprintf('SoftMaxLayer');
    netMat.addLayer(layerTop, dagnn.SoftMax(),sName, layerTop);
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end

netMat.move('gpu');
netMat.mode = 'test' ;
% netMat.mode = 'normal' ;
netMat.conserveMemory = 1;
%% test 
testIdxList = find(imdb.set == 2);
grndList = zeros(1, length(testIdxList));
predList = zeros(1, length(testIdxList));
scoreMat = zeros(length(imdb.meta.className), length(testIdxList));
for imgIdx = 1:length(testIdxList)
    imgPathName = imdb.imgList{testIdxList(imgIdx)};
    imgPathName = strrep(imgPathName, 'skong2','skong2/local');
    imOrg = single(imread(imgPathName));  
    maskPathName = imdb.maskList{testIdxList(imgIdx)};
    maskPathName = strrep(maskPathName, 'skong2','skong2/local');    
    gtOrg = single(imread(maskPathName)); 
    
    imOrg = bsxfun(@minus, imOrg, mean_rgb);  
    grndList(imgIdx) = imdb.labelList(testIdxList(imgIdx));
    if imdb.meta.width<size(imOrg,1)
        tmp = zeros(imdb.meta.height, imdb.meta.width, 3, 'single');
        yxNew = floor((size(imOrg) - size(tmp))/2); yxNew(1) = max(yxNew(1),1); yxNew(2) = max(yxNew(2),1);
        imOrg = imOrg(yxNew(1):yxNew(1)+imdb.meta.width-1, yxNew(2):yxNew(2)+imdb.meta.width-1, :);  
        gtOrg = gtOrg(yxNew(1):yxNew(1)+imdb.meta.width-1, yxNew(2):yxNew(2)+imdb.meta.width-1, :);      
    elseif imdb.meta.width>size(imOrg,1)
        tmp = zeros(imdb.meta.height, imdb.meta.width, 3, 'single');
        yxNew = floor((size(tmp) - size(imOrg))/2); yxNew(1) = max(yxNew(1),1); yxNew(2) = max(yxNew(2),1);
        tmp(yxNew(1):yxNew(1)+size(imOrg,1)-1, yxNew(2):yxNew(2)+size(imOrg,2)-1, :) = imOrg;
        imOrg = tmp;        
        tmp = zeros(imdb.meta.height, imdb.meta.width, 1, 'single');
        tmp(yxNew(1):yxNew(1)+size(gtOrg,1)-1, yxNew(2):yxNew(2)+size(gtOrg,2)-1, :) = gtOrg;
        gtOrg = tmp; 
    else
    end  
    %% feed into the network 
    imOrg = bsxfun(@times, imOrg, gtOrg);
    inputs = {'data', gpuArray(imOrg)};
    netMat.eval(inputs) ;
    %% gather the output 
    res6_conv = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('res6_conv')).outputIndexes).value);        
    [~, predLabel] = max(squeeze(res6_conv));
    scoreMat(:, imgIdx) = res6_conv(:);
    predList(imgIdx) = predLabel;
    if mod(imgIdx, 10) == 0
        fprintf('%d/%d acc=%.2f %s\n', imgIdx, length(testIdxList), mean(grndList(1:imgIdx)==predList(1:imgIdx)), imgPathName);
    end
end
%% save result
save(sprintf('result_cls_%s.mat', saveToken), 'imdb', 'grndList', 'predList', 'scoreMat');
%% confusion matrix for all class
acc = mean(predList(:) == grndList(:) );
[Conf_Mat, GORDER] = confusionmat(grndList, predList);

trNum = zeros(length(imdb.meta.className),1);
for i = 1:length(imdb.meta.className)
    trNum(i) = sum(grndList==i);
end
trNum = repmat(trNum, [1,length(imdb.meta.className)]);

c = Conf_Mat ./ trNum; % normalize into [0,1]


imgFig = figure(1);
set(imgFig, 'Position', [100 100 1500 1000]) % [1 1 width height]
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
xticklabel_rotate([],45,[],'Fontsize',10);

export_fig( sprintf('./confusionMatrix_allClass_%s.jpg', saveToken) );
%% sub confusion matrix
accList = diag(c);
[sortedAccList, sortedAccListIdx] = sort(accList, 'descend');
topNclass = 25;
subClassList = sortedAccListIdx(1:topNclass);
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


imgFig2 = figure(2);
set(imgFig2, 'Position', [100 100 1500 1000]) % [1 1 width height]
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

export_fig( sprintf('./confusionMatrix_%dClass_%s.jpg', length(subClassList), saveToken) );
%% leaving blank
acc_per_class = [];
num_per_class = [];
for i = 1:47
    idx = find(grndList(:)==i);
    num_per_class(end+1) = length(idx);
    tmp = mean(predList(idx) == grndList(idx));
    name = imdb.meta.className{i};
    fprintf('%d\t%s\t%.2f\n', i, name, tmp);
    acc_per_class(end+1)=tmp;
end
disp(mean(acc_per_class));

figure();
scatter(num_per_class, acc_per_class);
xlabel('number per type');
ylabel('accuracy');
export_fig(sprintf('./acc_vs_num.jpg'));


figure();
scatter(log2(num_per_class), acc_per_class);
xlabel('number per type (log2 scale)');
ylabel('accuracy');
export_fig(sprintf('./acc_vs_num_log2.jpg'));
