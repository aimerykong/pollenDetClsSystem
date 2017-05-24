clear 
close all
clc;

addpath('exportFig');

path_to_dataset = '/home/skong2/pollenProject_dataset';

load('validList.mat');
load('map_num2name.mat');
%% extract info
%{
1) The random image window number (this can be ignored)
2) The slide number (“198”) corresponds to the pollen sample. A list of all 199 pollen samples counted can be found in a separate file (SlideNumbersAndNames.txt). In this instance, 198 (viewable in the upper left corner of the image) corresponds to the pollen sample “2007-2007-15-35”
3) The (absolute) x coordinate of image within the original scanned image
4) The (absolute) y coordinate of image within the original scanned image 
5) Annotated pollen number within the tagged PNG. The numbers range from 1 to the total number of grains counted within a single image stack.
6) The (relative) x coordinate of the tagged pollen grain within the PNG stack
7) The (relative) y coordinate of the tagged pollen grain within the PNG stack
8) The z-plane where the pollen grain was identified (In this case -1; viewable in the upper right corner of the image). When gains span multiple depths, the center plane is tagged.
9) The radius of the circled pollen grain
10) The confidence level of the taxon identification (0-9)
11) The 3-letter identification code
%}
imdb.randNumList = zeros(1, length(validList));
imdb.slideNumList = zeros(1, length(validList));
imdb.absXList = zeros(1, length(validList));
imdb.absYList = zeros(1, length(validList));
imdb.annotPollenNumList = zeros(1, length(validList));
imdb.relativeXList = zeros(1, length(validList));
imdb.relativeYList = zeros(1, length(validList));
imdb.zPlaneList = zeros(1, length(validList));
imdb.radiusList = zeros(1, length(validList));
imdb.confList = zeros(1, length(validList));
imdb.labelNameList = cell(1,length(validList));

for i = 1:length(validList)    
    imdb.randNumList(i) = str2double(validList{i}{1});
    imdb.slideNumList(i) = str2double(validList{i}{2});
    imdb.absXList(i) = str2double(validList{i}{3});
    imdb.absYList(i) = str2double(validList{i}{4});
    imdb.annotPollenNumList(i) = str2double(validList{i}{5});
    imdb.relativeXList(i) = str2double(validList{i}{6});
    imdb.relativeYList(i) = str2double(validList{i}{7});
    imdb.zPlaneList(i) = str2double(validList{i}{8});
    imdb.radiusList(i) = str2double(validList{i}{9});
    imdb.confList(i) = str2double(validList{i}{10});
    imdb.labelNameList{i} = validList{i}{11};
end

imdb.className = unique(imdb.labelNameList);
imdb.classMapping_Name2Label = containers.Map;
for i = 1:length(imdb.className)
    imdb.classMapping_Name2Label(imdb.className{i}) = i;
%     disp(classMapping_Name2Label(className{i}));
end

imdb.labelList = ones(1, length(validList));
for i = 1:length(validList)
    imdb.labelList(i) = imdb.classMapping_Name2Label(validList{i}{11});
end
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




