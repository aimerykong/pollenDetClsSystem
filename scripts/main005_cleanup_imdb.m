clear 
close all
clc;

addpath('exportFig');

path_to_dataset = '/home/skong2/pollenProject_dataset';

load('validList.mat');
load('map_num2name.mat');
load('imdb_raw.mat');
%% extract info
slideHashtable = containers.Map;

flag_valid_list = zeros(1,length(imdb.randNumList));
imdb.imgpathList = cell(1,length(imdb.randNumList));
for i = 1:length(flag_valid_list)
    curLabelName = imdb.labelNameList{i};
    
    slideNumber = imdb.slideNumList(i);
    slideName = map_num2name{slideNumber};
    
    filename = ['*', slideName, '.*'... % 
        '.', int2str(imdb.zPlaneList(i)), '.', int2str(imdb.absXList(i)), '.', int2str(imdb.absYList(i)), '.png']; % z,x,y
    
    tmpItem = dir( fullfile(path_to_dataset, curLabelName, filename) );
    
    if ~isempty(tmpItem) && tmpItem(1).bytes~=0
        flag_valid_list(i) = 1;
        imdb.imgpathList{i} = tmpItem(1).name;
        
        curSlideName = ['*', slideName, '.*', '.', '*', '.', int2str(imdb.absXList(i)), '.', int2str(imdb.absYList(i)), '.png']; % z,x,y
        if isKey(slideHashtable, curSlideName)
            tmp = slideHashtable(curSlideName);
            tmp{end+1} = {imdb.randNumList(i), imdb.slideNumList(i), imdb.absXList(i), imdb.absYList(i), imdb.annotPollenNumList(i), ...
                imdb.relativeXList(i), imdb.relativeYList(i), imdb.zPlaneList(i), imdb.radiusList(i), imdb.confList(i),...
                imdb.labelNameList{i}, imdb.labelList(i), imdb.imgpathList{i} };
            
            slideHashtable(curSlideName) = tmp;
        else
            slideHashtable(curSlideName) = { {imdb.randNumList(i), imdb.slideNumList(i), imdb.absXList(i), imdb.absYList(i), imdb.annotPollenNumList(i), ...
                imdb.relativeXList(i), imdb.relativeYList(i), imdb.zPlaneList(i), imdb.radiusList(i), imdb.confList(i),...
                imdb.labelNameList{i}, imdb.labelList(i), imdb.imgpathList{i} } };
        end    	
%         disp(filename);    
    end
    if length(tmpItem)>1
        disp(filename);    
    end
end
%% cleanup
idx = find(flag_valid_list==1);
imdb.randNumList = imdb.randNumList(idx);
imdb.slideNumList = imdb.slideNumList(idx);
imdb.absXList = imdb.absXList(idx);
imdb.absYList = imdb.absYList(idx);
imdb.annotPollenNumList = imdb.annotPollenNumList(idx);
imdb.relativeXList = imdb.relativeXList(idx);
imdb.relativeYList = imdb.relativeYList(idx);
imdb.zPlaneList = imdb.zPlaneList(idx);
imdb.radiusList = imdb.radiusList(idx);
imdb.confList = imdb.confList(idx);
imdb.labelNameList = imdb.labelNameList(idx);
imdb.labelList = imdb.labelList(idx);
imdb.imgpathList = imdb.imgpathList(idx);
%% analysis
imgFig = figure(1);
set(imgFig, 'Position', [100 100 1500 800]) % [1 1 width height]

subplot(2, 3, 1);
structHist = histogram(imdb.labelList, length(imdb.className));
num65 = sum(structHist.Values>=65);
title(sprintf('whole datast distr. confTresh>=%d, NO[>65]=%d', 1, num65));

for confThresh = 3:7
    subplot(2, 3, confThresh-1);
    idx = find(imdb.confList>=confThresh);
    structHist = histogram(imdb.labelList(idx), length(imdb.className));
    num65 = sum(structHist.Values>=65);
    title(sprintf('datast distr. confTresh>=%d, NO[>65]=%d', confThresh, num65));
end
export_fig( sprintf('./distr_dataset_cleanup.png') );
save('imdb_cleanup.mat', 'imdb', 'slideHashtable');

%% leaving blank
% 1x16428 

