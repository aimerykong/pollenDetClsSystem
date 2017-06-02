clear 
close all
clc

addpath('exportFig');

path_to_dataset = '/home/skong2/pollenProject_dataset';

% load('validList.mat');
load('map_num2name.mat');
load('imdb_cleanup.mat');
%% extract info
imgIdx = 15;

randNum = imdb.randNumList(imgIdx);
slideNum = imdb.slideNumList(imgIdx);
absX = imdb.absXList(imgIdx);
absY = imdb.absYList(imgIdx);
annotPollenNum = imdb.annotPollenNumList(imgIdx);
relativeX = imdb.relativeXList(imgIdx);
relativeY = imdb.relativeYList(imgIdx);
zPlane = imdb.zPlaneList(imgIdx);
radius = imdb.radiusList(imgIdx);
conf = imdb.confList(imgIdx);
labelName = imdb.labelNameList{imgIdx};
label = imdb.labelList(imgIdx);
imgpath = imdb.imgpathList{imgIdx};

im = imread(fullfile(path_to_dataset, labelName, imgpath));
im = drawCircle(im, relativeY, relativeX, radius);

imshow(im);
title(sprintf('%s, conf:%d', labelName, conf));


