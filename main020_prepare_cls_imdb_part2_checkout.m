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
addpath(genpath('../libs'))
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
% load('imdb_segDet.mat');
load('imdb_cleanup.mat');

imdb.meta.classNum = 2;
imdb.meta.height = 1000;
imdb.meta.width = 1000;


imdb_cls_part2.path_to_image = '/home/skong2/pollenProject_dataset_patch4cls_part2';
imdb_cls_part2.path_to_mask = '/home/skong2/pollenProject_dataset_patch4mask_part2';
imdb_cls_part2.labelList = [];
imdb_cls_part2.imgList = {};

%% read matconvnet model
% set GPU
gpuId = 2; %[1, 2];
gpuDevice(gpuId);
flagSaveFig = false; % {true false} whether to store the result

saveFolder = 'main011_segDet_res50';
modelName = 'softmax_net-epoch-10.mat';
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
    filters = single(bilinear_u(upsample_fac*2, imdb.meta.classNum, imdb.meta.classNum));
    crop = ones(1,4) * upsample_fac/2;
    deconv_name = [baseName, '_interp'];
    var_to_up_sample = sName;
    netMat.addLayer(deconv_name, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', imdb.meta.classNum, ...
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
path_to_dataset = '/home/skong2/pollenProject_dataset';
slideNameList = keys(slideHashtable);
saveFolder = [strrep(saveFolder,'/', '') '_visualization'];
for slideIdx = 1:length(slideNameList)
    slideName = slideNameList{slideIdx};
    slideListOrg = slideHashtable(slideName);
    
    fprintf('%d/%d %s\n', slideIdx, length(slideNameList), slideName);
    
    
    slideList = {};
    titleList = {};
    imgList = {};
    yxList = zeros(2, length(slideListOrg));
    radiusList = zeros(1, length(slideListOrg));
    subfolderList = {};
    labelList = [];
    
    curMask = zeros(1000, 1000);
    curLabelMask = zeros(1000, 1000);
    for i = 1:length(slideListOrg)
        slideList{end+1} = fullfile(path_to_dataset,  slideListOrg{i}{11}, slideListOrg{i}{end}) ;
        titleList{end+1} = fullfile(slideListOrg{i}{11}, slideListOrg{i}{end});
        labelList(end+1) = slideListOrg{i}{12};
        subfolderList{end+1} = slideListOrg{i}{11};
        yxList(:, i) = [slideListOrg{i}{7} slideListOrg{i}{6}]';
        radiusList(i) = slideListOrg{i}{9};
        
        imgList{end+1} = single(imread(slideList{i}));
        
        cur_relativeY = yxList(1,i);
        cur_relativeX = yxList(2,i);
        if cur_relativeX<1
            cur_relativeX = 1;
        end
        if cur_relativeX>size(imgList{i},2);
            cur_relativeX=size(imgList{i},2);
        end
        if cur_relativeY<1
            cur_relativeY = 1;
        end
        if cur_relativeY>size(imgList{i},1);
            cur_relativeY=size(imgList{i},1);
        end
        
        radius = radiusList(i);
        curMask = drawDisk( curMask, round(cur_relativeY), round(cur_relativeX), radius);
        tmp = zeros(size(curMask));
        tmp = drawDisk( tmp, round(cur_relativeY), round(cur_relativeX), radius);
        curLabelMask(tmp==1) = labelList(i);
        
    end            
    %% feed into the network
    for i = 1:length(imgList)
%         fprintf('image: %s ... \n', titleList{i});
        imFeed = bsxfun(@minus, imgList{i}, mean_rgb);
        inputs = {'data', gpuArray(imFeed)};
        netMat.eval(inputs) ;
        %% gather the output
        SoftMaxLayer = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayer')).outputIndexes).value);
        respMap = SoftMaxLayer(:,:,2);
        [~, predSeg] = max(SoftMaxLayer,[],3);        
        %% NMS for attention-aware patches
        respMapMask = (respMap>0.2);
        se = strel('disk', 10);
        respMapMask = imdilate(respMapMask, se); % for connected component, get radius
        
        maxResp = imRespNMS(respMap);        
        maxResp(maxResp<0.35) = 0;
        maxResp = maxResp .* respMapMask; % for the grain centers
        
%         CC = bwconncomp(respMapMask); %% connected component
        ccLabel = bwlabel(respMapMask);
        centerList = find(maxResp);
        for centerIdx = 1:length(centerList)
            curCenter = centerList(centerIdx);             
            [cur_relativeY, cur_relativeX] = ind2sub(size(respMapMask), curCenter);            
%             curCenterCClabel = ccLabel(curCenter);
            cur_winSize = length(ccLabel==ccLabel(curCenter));
            cur_winSize = sqrt(cur_winSize)+10;
            if cur_winSize < 100
                cur_winSize = 100;
            elseif cur_winSize>500
                cur_winSize = 512;
            end
            UL = round([cur_relativeY, cur_relativeX]-cur_winSize/2);
            BR = UL + cur_winSize-1;
            if UL(1)<1
                UL(1) = 1;
                BR(1) = cur_winSize;
            end
            if UL(2)<1
                UL(2) = 1;
                BR(2) = cur_winSize;
            end
            if BR(1)>size(curMask,1)
                BR(1) = size(curMask,1);
                UL(1) = BR(1)-cur_winSize+1;
            end
            if BR(2)>size(curMask,2)
                BR(2) = size(curMask,2);
                UL(2) = BR(2)-cur_winSize+1;
            end
            
            curImgWindow = imgList{i}(UL(1):BR(1), UL(2):BR(2), :);
            curMaskWindow = curMask(UL(1):BR(1), UL(2):BR(2), :);
            curWindowLabel = curLabelMask(cur_relativeY, cur_relativeX);
            cur_imgpath = [char(datetime('now','Format','yyyyMMdd''_''HHmmss')), '.png'];
            
            if curWindowLabel~=0
                curWindowLabelName = imdb.className{curWindowLabel};
                if ~isdir(fullfile(imdb_cls_part2.path_to_image, curWindowLabelName))
                    mkdir(fullfile(imdb_cls_part2.path_to_image, curWindowLabelName));
                end
                if ~isdir(fullfile(imdb_cls_part2.path_to_mask, curWindowLabelName))
                    mkdir(fullfile(imdb_cls_part2.path_to_mask, curWindowLabelName));
                end
                
                imwrite(uint8(curImgWindow), fullfile(imdb_cls_part2.path_to_image, curWindowLabelName, cur_imgpath));
                imwrite(uint8(curMaskWindow), fullfile(imdb_cls_part2.path_to_mask, curWindowLabelName, cur_imgpath));
                imdb_cls_part2.labelList(end+1) = curWindowLabel;
                imdb_cls_part2.imgList{end+1} = fullfile(curWindowLabelName, cur_imgpath);                
%                 a = 1;
            else 
                curWindowLabelName = 'reject';
                curWindowLabel = 132;
                if ~isdir(fullfile(imdb_cls_part2.path_to_image, curWindowLabelName))
                    mkdir(fullfile(imdb_cls_part2.path_to_image, curWindowLabelName));
                end
                if ~isdir(fullfile(imdb_cls_part2.path_to_mask, curWindowLabelName))
                    mkdir(fullfile(imdb_cls_part2.path_to_mask, curWindowLabelName));
                end
                imwrite(uint8(curImgWindow), fullfile(imdb_cls_part2.path_to_image, curWindowLabelName, cur_imgpath));
                imwrite(uint8(curMaskWindow), fullfile(imdb_cls_part2.path_to_mask, curWindowLabelName, cur_imgpath));
                imdb_cls_part2.labelList(end+1) = curWindowLabel;
                imdb_cls_part2.imgList{end+1} = fullfile(curWindowLabelName, cur_imgpath);               
%                 a = 1;              
            end
%             figure(1);
%             subplot(2,1,1);
%             imagesc(uint8(curImgWindow)); title(fullfile(imdb_cls_part2.path_to_image, curWindowLabelName, cur_imgpath), 'Interpreter', 'None'); axis off image;
%             subplot(2,1,2); imagesc(curMaskWindow);           axis off image;
%             a = 1;
            %% visuzlize to check
%             imgFig = figure(1);
%             subWindowH = 1;
%             subWindowW = 2;
%             set(imgFig, 'Position', [100 100 1500 800]) % [1 1 width height]
%             windowID = 1;            
%             subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
%             imagesc(uint8(curImgWindow)); axis off image;
%             subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
%             imagesc(curMaskWindow); title(sprintf('curMaskWindow')); axis off image; caxis([0,1]);
        end  
    end
end
save('imdb_cls_part2.mat', 'imdb_cls_part2');
%% leaving blank

