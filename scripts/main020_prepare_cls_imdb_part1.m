clear 
close all
clc;

addpath('exportFig');
path_to_dataset = '/home/skong2/pollenProject_dataset';

load('validList.mat');
load('map_num2name.mat');
load('imdb_cleanup.mat');

imdb.path_to_mask = '/home/skong2/pollenProject_dataset_mask';
imdb.path_to_image = path_to_dataset;

imdb_cls.path_to_image = '/home/skong2/pollenProject_dataset_patch4cls';
imdb_cls.path_to_mask = '/home/skong2/pollenProject_dataset_patch4mask';
imdb_cls.labelList = [];
imdb_cls.imgList = {};
% imdb_cls.maskList = {};
%% get max pollen for crop window size
% [maxWinSize, idx] = max(imdb.radiusList);
winSize = 512;
for idx = 1:length(imdb.radiusList)
    if mod(idx,100) ==0
        fprintf('%d/%d\n', idx, length(imdb.radiusList));
    end    
    %% collect info
    cur_randNum = imdb.randNumList(idx);%: [1x16425 double]
    cur_slideNum = imdb.slideNumList(idx);%: [1x16425 double]
    cur_absX = imdb.absXList(idx);%: [1x16425 double]
    cur_absY = imdb.absYList(idx);%: [1x16425 double]
    cur_annotPollenNum = imdb.annotPollenNumList(idx);%: [1x16425 double]
    cur_relativeX = imdb.relativeXList(idx);%: [1x16425 double]
    cur_relativeY = imdb.relativeYList(idx);%: [1x16425 double]
    cur_zPlane = imdb.zPlaneList(idx);%: [1x16425 double]
    cur_radius = imdb.radiusList(idx);%: [1x16425 double]
    cur_conf = imdb.confList(idx);%: [1x16425 double]
        
    cur_label = imdb.labelList(idx);%: [1x16425 double]
    cur_imgpath = imdb.imgpathList{idx};%: {1x16425 cell}    
    cur_labelName = imdb.labelNameList{idx};%: {1x16425 cell}
%     cur_class = imdb.className{cur_label};%: {1x131 cell}
%     cur_label = imdb.classMapping_Name2Label(cur_class);%: [131x1 containers.Map]
    %% save        
    im = imread(fullfile(imdb.path_to_image, cur_labelName, cur_imgpath));
    
    if cur_relativeX<1
        cur_relativeX = 1;
    end
    if cur_relativeX>size(im,2);
        cur_relativeX=size(im,2);
    end
    if cur_relativeY<1
        cur_relativeY = 1;
    end
    if cur_relativeY>size(im,1);
        cur_relativeY=size(im,1);
    end
    
    curMask = zeros(size(im,1), size(im,2));
    curMask = drawDisk( curMask, round(cur_relativeY), round(cur_relativeX), cur_radius);
    if ~isdir(fullfile(imdb_cls.path_to_image, cur_labelName))
        mkdir(fullfile(imdb_cls.path_to_image, cur_labelName));
    end
    if ~isdir(fullfile(imdb_cls.path_to_mask, cur_labelName))
        mkdir(fullfile(imdb_cls.path_to_mask, cur_labelName));
    end
    
    cur_winSize = ceil((cur_radius*2+20)/8) *8;
    
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
    curImgWindow = im(UL(1):BR(1), UL(2):BR(2), :);
    curMaskWindow = curMask(UL(1):BR(1), UL(2):BR(2), :);
    imwrite(uint8(curImgWindow), fullfile(imdb_cls.path_to_image, cur_labelName, cur_imgpath));    
    imwrite(uint8(curMaskWindow), fullfile(imdb_cls.path_to_mask, cur_labelName, cur_imgpath));    
    imdb_cls.labelList(end+1) = cur_label;
    imdb_cls.imgList{end+1} = fullfile(cur_labelName, cur_imgpath);    
end
save('imdb_cls_part1.mat', 'imdb_cls');
%% leaving blank
% 1x16428 

