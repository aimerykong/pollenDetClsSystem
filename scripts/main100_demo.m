%LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:local matlab -nodisplay
%{
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda', ...
               'cudaMethod', 'nvcc') ;

%}
clear
close all
clc;
rng(777);
addpath('../libs/exportFig')
addpath('../libs/layerExt')
addpath('../libs/myFunctions')
path_to_matconvnet = '../libs/matconvnet';
path_to_model = '../models/';

run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

mean_r = 123.68; %means to be subtracted and the given values are used in our training stage
mean_g = 116.779;
mean_b = 103.939;
mean_bgr = reshape([mean_b, mean_g, mean_r], [1,1,3]);
mean_rgb = reshape([mean_r, mean_g, mean_b], [1,1,3]);

gpuId = 1; %[1, 2];
gpuDevice(gpuId);
%% load detection&segmentation model
saveFolder = 'main011_segDet_res50';
modelName = 'softmax_net-epoch-10.mat';
if ~exist('netDetSeg', 'var')
    netDetSeg = load( fullfile('./exp', saveFolder, modelName) );
    netDetSeg = netDetSeg.net;
    netDetSeg = dagnn.DagNN.loadobj(netDetSeg);
end
rmLayerName = 'obj_seg';
if ~isnan(netDetSeg.getLayerIndex(rmLayerName))
    sName = netDetSeg.layers(netDetSeg.getLayerIndex(rmLayerName)).inputs{1};
    netDetSeg.removeLayer(rmLayerName); % remove layer
    
    baseName = 'res6';
    upsample_fac = 8;
    filters = single(bilinear_u(upsample_fac*2, 2, 2));
    crop = ones(1,4) * upsample_fac/2;
    deconv_name = [baseName, '_interp'];
    var_to_up_sample = sName;
    netDetSeg.addLayer(deconv_name, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', 2, ...
        'hasBias', false), ...
        var_to_up_sample, deconv_name, {[deconv_name  '_f']}) ;
    ind = netDetSeg.getParamIndex([deconv_name  '_f']) ;
    netDetSeg.params(ind).value = filters ;
    sName = deconv_name;
    
    layerTop = sprintf('SoftMaxLayer');
    netDetSeg.addLayer(layerTop, dagnn.SoftMax(),sName, layerTop);
    netDetSeg.vars(netDetSeg.layers(netDetSeg.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end

% netDetSeg.move('gpu');
netDetSeg.mode = 'test' ;
netDetSeg.conserveMemory = 1;
%% load fine segmentation model
saveFolder = 'main021_finegrained_segDet_res50';
modelName = 'softmax_net-epoch-27.mat';
if ~exist('netFineSeg', 'var')
    netFineSeg = load( fullfile('./exp', saveFolder, modelName) );
    netFineSeg = netFineSeg.net;
    netFineSeg = dagnn.DagNN.loadobj(netFineSeg);
end

rmLayerName = 'obj_seg';
if ~isnan(netFineSeg.getLayerIndex(rmLayerName))
    sName = netFineSeg.layers(netFineSeg.getLayerIndex(rmLayerName)).inputs{1};
    netFineSeg.removeLayer(rmLayerName); % remove layer
    
    baseName = 'res6';
    upsample_fac = 8;
    filters = single(bilinear_u(upsample_fac*2, 2, 2));
    crop = ones(1,4) * upsample_fac/2;
    deconv_name = [baseName, '_interp'];
    var_to_up_sample = sName;
    netFineSeg.addLayer(deconv_name, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', 2, ...
        'hasBias', false), ...
        var_to_up_sample, deconv_name, {[deconv_name  '_f']}) ;
    ind = netFineSeg.getParamIndex([deconv_name  '_f']) ;
    netFineSeg.params(ind).value = filters ;
    sName = deconv_name;
    
    layerTop = sprintf('SoftMaxLayer');
    netFineSeg.addLayer(layerTop, dagnn.SoftMax(),sName, layerTop);
    netFineSeg.vars(netFineSeg.layers(netFineSeg.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end

% netFineSeg.move('gpu');
netFineSeg.mode = 'test' ;
netFineSeg.conserveMemory = 1;
%% load classification model
saveFolder = 'main041_cls_res50_alldata_avgPool_preSegMasking';
modelName = 'softmax_net-epoch-20.mat';
if ~exist('netCls', 'var')
    netCls = load( fullfile('./exp', saveFolder, modelName) );
    netCls = netCls.net;
    netCls = dagnn.DagNN.loadobj(netCls);
end
% netCls.move('gpu');
netCls.mode = 'test' ;
netCls.conserveMemory = 1;
%% demo for a pile of slides
load('map_num2name.mat');
load('../imdb_files/imdb_cleanup.mat');
%path_to_dataset = '/home/skong2/pollenProject_dataset'; % /home/fowlkes/restore/pollen/tropical
%path_to_dataset = '/home/fowlkes/restore/pollen/tropical'; % 
path_to_dataset = '../data4demo';

demoFigure = './demoFigure';
if ~isdir(demoFigure)
    mkdir(demoFigure);
end


slideName = '*1994-1995-10-20.*.*.37000.15000.png'; % 3
slideName = '*2004-2004-15-25.*.*.13000.9000.png'; % 4
slideName = '*2004-2004-15-25.*.*.12000.2000.png'; %  5
slideName = '*2004-2004-15-25.*.*.11000.2000.png'; %  6
slideName = '*2004-2004-15-25.*.*.5000.16000.png';
slideName = '*2004-2004-15-35.*.*.46000.2000.png'; % 7
slideName = '*2005-2005-10-10.*.*.5000.15000.png'; % 11
%% ground-truth annotation
slideListOrg = slideHashtable(slideName);
slideList = {};
titleList = {};
for i = 1:length(slideListOrg)
    a = fullfile(path_to_dataset,  slideListOrg{i}{11}, slideListOrg{i}{end}) ;
    slideList{end+1} = a;
    titleList{end+1} = fullfile(slideListOrg{i}{11}, slideListOrg{i}{end});
end

disp(slideList(:))
imgList = cell(1,length(slideList));

% figure1 = figure(1);
% set(figure1, 'Position', [100 100 1500 1000]) % [1 1 width height]
imDemo = imread(slideList{1});
for i = 1:min(length(slideList),12)
    imgList{i} = imread(slideList{i});
    
    relativeY = slideListOrg{i}{7};
    relativeX = slideListOrg{i}{6};
    radius = slideListOrg{i}{9};
    thickness = 2;
    imDemo = drawCircle( imDemo, relativeY, relativeX, radius, thickness);
end

figure(1);
imshow(imDemo);
for i = 1:length(slideList)
    text(slideListOrg{i}{6}, slideListOrg{i}{7}, fileparts(titleList{i}), 'Color', 'red', 'FontSize', 12);
end

tmpname4save = sprintf('demoFig_%s_groundtruth.png', slideName);
export_fig( sprintf('%s/%s', demoFigure, tmpname4save) );
%% detection&segmentation
fprintf('detection-segmentation...\n');
[imgWindowList, maskWindowList, ulrbWindowList, centerWindowList] = genMask4DetSeg(netDetSeg, imgList, mean_rgb);
%% fine segmentation
fprintf('fine segmentation...\n');
fineMaskWindowList = maskWindowList;
netFineSeg.move('gpu');

for i = 1:length(imgWindowList)
    for j = 1:length(imgWindowList{i})
        imOrg = single(imgWindowList{i}{j});
        
        imFeed = bsxfun(@minus, imOrg, mean_rgb);
        inputs = {'data', gpuArray(imFeed)};
        netFineSeg.eval(inputs) ;
        %% gather the output
        SoftMaxLayer = gather(netFineSeg.vars(netFineSeg.layers(netFineSeg.getLayerIndex('SoftMaxLayer')).outputIndexes).value);
        [~, predSeg] = max(SoftMaxLayer,[],3);
        predSeg = imresize( predSeg, [size(imOrg,1) size(imOrg,2)], 'nearest');
        fineMaskWindowList{i}{j} = predSeg-1;
    end
end
netFineSeg.move('cpu');
%% classification
load('../imdb_files/imdb_merge4cls_predSeg.mat');
fprintf('classification...\n');
predLabelList = cell(1, length(imgWindowList));
predConfList = cell(1, length(imgWindowList));

netCls.move('gpu');

for i = 1:length(imgWindowList)
    predLabelList{i} = {};
    predConfList{i} = {};
    for j = 1:length(imgWindowList{i})
        imOrg = single(imgWindowList{i}{j});
        gtOrg = single(fineMaskWindowList{i}{j});
        
        imOrg = bsxfun(@minus, imOrg, mean_rgb);
        
        if 512 < size(imOrg,1)
            tmp = zeros(512, 512, 3, 'single');
            yxNew = floor((size(imOrg) - size(tmp))/2); yxNew(1) = max(yxNew(1),1); yxNew(2) = max(yxNew(2),1);
            imOrg = imOrg(yxNew(1):yxNew(1)+imdb.meta.width-1, yxNew(2):yxNew(2)+imdb.meta.width-1, :);
            gtOrg = gtOrg(yxNew(1):yxNew(1)+imdb.meta.width-1, yxNew(2):yxNew(2)+imdb.meta.width-1, :);
        elseif 512>size(imOrg,1)
            tmp = zeros(512, 512, 3, 'single');
            yxNew = floor((size(tmp) - size(imOrg))/2); yxNew(1) = max(yxNew(1),1); yxNew(2) = max(yxNew(2),1);
            tmp(yxNew(1):yxNew(1)+size(imOrg,1)-1, yxNew(2):yxNew(2)+size(imOrg,2)-1, :) = imOrg;
            imOrg = tmp;
            tmp = zeros(512, 512, 1, 'single');
            tmp(yxNew(1):yxNew(1)+size(gtOrg,1)-1, yxNew(2):yxNew(2)+size(gtOrg,2)-1, :) = gtOrg;
            gtOrg = tmp;
        else
        end
        
        imOrg = bsxfun(@times, imOrg, gtOrg);
        inputs = {'data', gpuArray(imOrg)};
        netCls.eval(inputs) ;
        %% gather the output
        res6_conv = gather(netCls.vars(netCls.layers(netCls.getLayerIndex('res6_conv')).outputIndexes).value);
        [~, predLabel] = max(squeeze(res6_conv));
        predLabelList{i}{j} = predLabel;
        predConfList{i}{j} = res6_conv;
    end
end
netFineSeg.move('cpu');
%% assemble
% ulrbWindowList
for i = 1:length(imgList)
%     figure('Name', sprintf('slide-%d', i));
    %     set(figure_assemble, 'Position', [100 100 1500 1000]) % [1 1 width height]
%     subplot(1,2,1);
%     imshow(imgList{i});
    mask = zeros(size(imgList{i},1), size(imgList{i},2));
    
    for j = 1:length(ulrbWindowList{i})
%         rectangle('Position',... % [UL(1),BR(1),UL(2),BR(2)]
%             [ulrbWindowList{i}{j}(3), ulrbWindowList{i}{j}(1), ulrbWindowList{i}{j}(4)-ulrbWindowList{i}{j}(3), ulrbWindowList{i}{j}(2)-ulrbWindowList{i}{j}(1)])
        mask(ulrbWindowList{i}{j}(1):ulrbWindowList{i}{j}(2), ulrbWindowList{i}{j}(3):ulrbWindowList{i}{j}(4), :) = fineMaskWindowList{i}{j};
    end
    %%
    CC = bwconncomp(mask);
    ccValidFlagList = ones(1, CC.NumObjects);
    ccLabelList = cell(1, CC.NumObjects);
    masktmp = zeros( size(imgList{i},1), size(imgList{i},2), 1 );
    for j = 1:CC.NumObjects
        ccLabelList{j} = [];
        if length(CC.PixelIdxList{j}) < 200
            ccValidFlagList(j) = 0;
        end
    end
    for j = 1:length(ulrbWindowList{i})
        masktmp = zeros( size(imgList{i},1), size(imgList{i},2), 1 );
        masktmp(ulrbWindowList{i}{j}(1):ulrbWindowList{i}{j}(2), ulrbWindowList{i}{j}(3):ulrbWindowList{i}{j}(4), :) = predLabelList{i}{j}*fineMaskWindowList{i}{j};
        for jj = 1:CC.NumObjects
            a = masktmp(CC.PixelIdxList{jj});
            a = a(a~=0);
            ccLabelList{jj} = [ccLabelList{jj}, a(:)'];
        end
    end
    
    for jj = 1:CC.NumObjects
        x = ccLabelList{jj};
        [a, b] = hist(x, unique(x));
        [~, a] = sort(a, 'descend');
        a = b(a(1));
        ccLabelList{jj} = a;
        if ccValidFlagList(jj) == 0
            mask(CC.PixelIdxList{jj}) = 0;
        end
    end
    %%
    demoFrame = single(imgList{i});
%     
    tmp = demoFrame(:,:,1);
    tmp(mask>0) = tmp(mask>0)/1.2;
    demoFrame(:,:,1) = tmp;
    
    tmp = demoFrame(:,:,2);
    tmp(mask>0) = tmp(mask>0)/1.2;
    demoFrame(:,:,2) = tmp;
        
    tmp = demoFrame(:,:,3);
    tmp(mask>0) = 255;
    demoFrame(:,:,3) = tmp;
    demoFrame = uint8(demoFrame);
    
    
    
    figure('Name', sprintf('slide-%d', i));
%     subplot(1,2,2);
    imshow(demoFrame);
    for j = 1:CC.NumObjects
        if ccValidFlagList(j)
            [yy, xx] = ind2sub(size(mask), CC.PixelIdxList{j});
            ccenter = [mean(yy), mean(xx)];
            text(ccenter(2), ccenter(1), imdb.meta.className{ccLabelList{j}}, 'Color','red', 'FontSize', 12);
        end
    end
    tmpname4save = sprintf('demoFig_%s_slide-%d.png', slideName, i);
    export_fig( sprintf('%s/%s', demoFigure, tmpname4save) );
end


% figure_assemble = figure('Name','Measured Data');
% set(figure_assemble, 'Position', [100 100 1500 1000]) % [1 1 width height]






%% leaving blank

