%% add path and setup configuration
%LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:local matlab -nodisplay
clear
close all
clc;

rng(777);
addpath('./libs/exportFig')
addpath('./libs/layerExt')
addpath('./libs/myFunctions')

path_to_model = '/home/data/skong2/MarkovCNN/basemodels/';
path_to_dataset = '../dataset/';

path_to_matconvnet = './libs/matconvnet';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

% set GPU 
gpuId = 1; %[1, 2];
gpuDevice(gpuId);
%% load imdb and dataset file
load('imdb_cls_part1.mat');
imdb = imdb_cls;
imdb = rmfield(imdb, 'labelList');
imdb.imgList = strcat([imdb.path_to_image '/'], imdb.imgList(:));

numTrainImg = 15000;
randIdx = randperm(length(imdb.imgList));
imdb.train = imdb.imgList(randIdx(1:numTrainImg));
imdb.val = imdb.imgList(randIdx(1+numTrainImg:end));

%% part2
path_to_part2 = '/home/skong2/pollenProject_dataset_patch4cls_part2';
cls_part2 = dir(path_to_part2); cls_part2 = {cls_part2(3:end).name}; cls_part2 = sort(cls_part2);
for i = 1:length(cls_part2)
    tmp = dir( fullfile(path_to_part2, cls_part2{i}) ); tmp = tmp(3:end);
    tmp = strcat( fullfile(path_to_part2, cls_part2{i}), '/', {tmp.name} );
    imdb.train = [imdb.train; tmp(:)];
end
save('imdb_merge4seg.mat', 'imdb');




