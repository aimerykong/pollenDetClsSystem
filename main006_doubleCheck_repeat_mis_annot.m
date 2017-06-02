clear
close all
clc

addpath('exportFig');

path_to_dataset = '/home/skong2/pollenProject_dataset';

% load('validList.mat');
load('map_num2name.mat');
load('imdb_cleanup.mat');
%% check every slides
slideNameList = keys(slideHashtable);
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
%             fprintf('%03d -- %d %s, repeated annotation (distBetweenCenters=%.3f)\n', count, i, slideNameList{i}, max(distMat(:)));
            fprintf('%03d -- %s [duplicate annotation]\n', i, slideNameList{i} );
            
            for idx1 = 1:size(distMat,1)
                for idx2 = idx1:size(distMat,1)
                    distMat(idx1,idx2) = 0;
                end
            end            
            [idx1, idx2] = find(distMat~=0);
            
            for k = 1:length(idx1)
                if abs(radiusList(idx1(k)) - radiusList(idx1(k))) < 10
                    fprintf('\t %s(%d) %s(%d) centerDist=%.2f', curSlide{idx1(k)}{11}, curSlide{idx1(k)}{10}, curSlide{idx2(k)}{11}, curSlide{idx2(k)}{10}, distMat(idx1(k),idx2(k)));
                    if strcmp(curSlide{idx1(k)}{11}, curSlide{idx2(k)}{11})==1
                        fprintf('\n');
                    else
                        fprintf(' [!!conflicting annotation!!]\n');
                    end
                end
            end
        else
            fprintf('%03d -- %s (#annotation=%d)\n', i, slideNameList{i}, length(curSlide));
        end
    end
    
end


%% leaving blank




