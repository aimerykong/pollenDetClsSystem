clear 
close all
clc

addpath('exportFig');

path_to_dataset = '/home/skong2/pollenProject_dataset';

% load('validList.mat');
load('map_num2name.mat');
load('imdb_cleanup.mat');
%% extract info
slideName = '*1995-1996-15-30.100.*.27000.24000.png';
slideList = {};
titleList = {};
for i = 1:length(imdb.className)
    a = dir( fullfile(path_to_dataset, imdb.className{i}, slideName) );
    if length(a) > 0
        for j = 1:length(a)
            slideList{end+1} = fullfile(path_to_dataset, imdb.className{i}, a(j).name);
            titleList{end+1} = fullfile(imdb.className{i}, a(j).name);
        end
    end
end

disp(slideList(:))
imList = cell(1,length(slideList));

figure(1); 
for i = 1:length(slideList)
    imList{i} = imread(slideList{i});
    subplot(3,3,i);
    imshow(imList{i});
    title(titleList{i}(1:end-4));
end
