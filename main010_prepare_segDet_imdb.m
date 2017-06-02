clear
close all
clc

addpath('./libs/exportFig')
addpath('./libs/layerExt')
addpath('./libs/myFunctions')

path_to_dataset = '/home/skong2/pollenProject_dataset';

% load('validList.mat');
load('map_num2name.mat');
load('imdb_cleanup.mat');

slideNameList = keys(slideHashtable);
%% check every slides
%{
count = 0;
for i = 1:length(slideNameList)
    curSlide = slideHashtable(slideNameList{i});
    
    if length(curSlide)>1
        yxList = zeros(2, length(curSlide));
        radiusList = zeros(1, length(curSlide));
        annoLabelName = cell(1, length(curSlide));
        annoLabelID = zeros(1, length(curSlide));
        for j = 1:length(curSlide)
            curAnno = curSlide{j};
            yxList(1,j) = curAnno{7};
            yxList(2,j) = curAnno{6};
            radiusList(j) = curAnno{9};
            annoLabelName{j} = curAnno{11};
            annoLabelID(j) = imdb.classMapping_Name2Label( curAnno{11} );
        end
        %% check the annotation -- repeated-annotation? mis-annotation?
        distMat = sqrt( bsxfun(@plus, sum(yxList.^2, 1), sum(yxList.^2, 1)') - 2*(yxList'*yxList) );
        distMat(distMat==0) = inf;
        distMat(distMat>30) = 0;
        if sum(distMat(:)) ~= 0
            count = count + 1;
            fprintf('%03d -- %d %s, minDist=%.3f\n', count, i, slideNameList{i}, max(distMat(:)));
            
            for idx1 = 1:size(distMat,1)
                for idx2 = idx1:size(distMat,1)
                    distMat(idx1,idx2) = 0;
                end
            end
            [idx1, idx2] = find(distMat~=0);
            
            for k = 1:length(idx1)
                if abs(radiusList(idx1(k)) - radiusList(idx1(k))) < 10
                    fprintf('\t %s(%d) %s(%d) %d', curSlide{idx1(k)}{11}, curSlide{idx1(k)}{10}, curSlide{idx2(k)}{11}, curSlide{idx2(k)}{10}, distMat(idx1(k),idx2(k)));
                    if strcmp(curSlide{idx1(k)}{11}, curSlide{idx2(k)}{11})==1
                        fprintf('\n');
                    else
                        fprintf(' [!!!!!!!!!]\n');
                    end
                end
            end
        end
    end
end
%}
%% prepare imdb for segDet
imdb.path_to_mask = '/home/skong2/pollenProject_dataset_mask';
imdb.path_to_image = path_to_dataset;
imdb.imgList = {};
imdb.maskList = {};


if ~isdir(imdb.path_to_mask)
    mkdir(imdb.path_to_mask);
end


% 3909 3910 3935
for slideIdx = 1:length(slideNameList)
  
    slideName = slideNameList{slideIdx};
    slideListOrg = slideHashtable(slideName);
    
    slideList = {};
    titleList = {};
    yxList = zeros(2, length(slideListOrg));
    radiusList = zeros(1, length(slideListOrg));
    subfolderList = {};
    for i = 1:length(slideListOrg)
        slideList{end+1} = fullfile(path_to_dataset,  slideListOrg{i}{11}, slideListOrg{i}{end}) ;
        titleList{end+1} = fullfile(slideListOrg{i}{11}, slideListOrg{i}{end});
        
        subfolderList{end+1} = slideListOrg{i}{11};
        yxList(:, i) = [slideListOrg{i}{7} slideListOrg{i}{6}]';
        radiusList(i) = slideListOrg{i}{9};
    end
    
    % imList = cell(1,length(slideList));
    for i = 1:length(slideList)
        imdb.imgList{end+1} = slideList{i};
        curImg = imread(slideList{i});
        curMask = zeros(1000, 1000);
        for j = 1:size(yxList,2)
            relativeY = yxList(1,j);
            relativeX = yxList(2,j);
            radius = radiusList(j);
            
%             thickness = 4;
%             curImg = drawCircle( curImg, relativeY, relativeX, radius, thickness);
            curMask = drawDisk( curMask, round(relativeY), round(relativeX), radius);
        end
        %% visualize
%         figure(1);
%         subplot(1,2,1);
%         imshow(curImg);
%         subplot(1,2,2);
%         imshow(curMask);

        if ~isdir(fullfile(imdb.path_to_mask, subfolderList{i}))
            mkdir(fullfile(imdb.path_to_mask, subfolderList{i}));
        end
        [~, to_replaceToken] = fileparts(imdb.path_to_image);
        [~, with_replaceToken] = fileparts(imdb.path_to_mask);
        despath = strrep(slideList{i}, to_replaceToken, with_replaceToken);
        despath = strrep(despath, 'png', 'bmp');
        imdb.maskList{end+1} = despath;
        imwrite(curMask, despath);
%         disp(slideList{i});
    end
    if mod(slideIdx, 10) == 0
        fprintf('%d/%d\n', slideIdx, length(slideNameList));
    end
end
save('imdb_segDet.mat', 'imdb');


%% leaving blank




