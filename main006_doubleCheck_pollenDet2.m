clear 
close all
clc

addpath('exportFig');

path_to_dataset = '/home/skong2/pollenProject_dataset';

% load('validList.mat');
load('map_num2name.mat');
load('imdb_cleanup.mat');
%% extract info
% slideName = '*1995-1996-10-05.*.*.35000.37000.png';
% slideName = '*1995-1996-15-15.*.*.18000.25000.png';
% slideName = '*1994-1995-10-30.*.*.32000.43000.png';
% slideName = '*1994-1995-10-15.*.*.42000.5000.png';
slideName = '*1996-1997-10-15.*.*.13000.4000.png';
% slideName = '*1995-1996-15-30.*.*.27000.24000.png';
%% repeated annotation
slideName = '*1994-1995-10-30.*.*.32000.43000.png';
slideName = '*1994-1995-10-35.*.*.23000.30000.png';
slideName = '*1994-1995-10-35.*.*.37000.24000.png';
slideName = '*1994-1995-15-10.*.*.58000.34000.png';
slideName = '*1994-1995-15-20.*.*.10000.25000.png';
slideName = '*1995-1996-10-05.*.*.35000.37000.png';
slideName = '*1995-1996-10-35.*.*.13000.37000.png';
slideName = '*1995-1996-15-15.*.*.18000.25000.png';
slideName = '*1995-1996-15-20.*.*.42000.29000.png';
slideName = '*1995-1996-15-30.*.*.27000.24000.png';
slideName = '*1995-1996-15-30.*.*.57000.31000.png';
slideName = '*1995-1996-15-30.*.*.57000.38000.png';
slideName = '*1996-1997-10-15.*.*.13000.4000.png';
slideName = '*1997-1997-10-20.*.*.41000.32000.png';
slideName = '*1995-1996-15-30.*.*.57000.38000.png';
%% conflick annotation
slideName = '*1994-1995-10-30.*.*.32000.43000.png';
slideName = '*1994-1995-10-35.*.*.23000.30000.png';
slideName = '*1994-1995-10-35.*.*.37000.24000.png';
slideName = '*1994-1995-15-10.*.*.58000.34000.png';
slideName = '*1994-1995-15-20.*.*.10000.25000.png';
slideName = '*1994-1995-15-30.*.*.27000.51000.png';
slideName = '*1995-1996-10-05.*.*.35000.37000.png';
slideName = '*1995-1996-10-35.*.*.13000.37000.png';
slideName = '*1995-1996-15-15.*.*.18000.25000.png';
slideName = '*1995-1996-15-20.*.*.42000.29000.png';
slideName = '*1995-1996-15-30.*.*.27000.24000.png';
slideName = '*1995-1996-15-30.*.*.57000.31000.png';
slideName = '*1995-1996-15-30.*.*.57000.38000.png';
slideName = '*1996-1997-10-01.*.*.11000.37000.png';
slideName = '*1996-1997-10-15.*.*.13000.4000.png';
slideName = '*1997-1997-10-20.*.*.41000.32000.png';
%%
slideListOrg = slideHashtable(slideName);
slideList = {};
titleList = {};
for i = 1:length(slideListOrg)
    a = fullfile(path_to_dataset,  slideListOrg{i}{11}, slideListOrg{i}{end}) ;
    
    slideList{end+1} = a;
    titleList{end+1} = fullfile(slideListOrg{i}{11}, slideListOrg{i}{end});
end

disp(slideList(:))
imList = cell(1,length(slideList));

figure;
imDemo = imread(slideList{1});
for i = 1:min(length(slideList),12)
    imList{i} = imread(slideList{i});
    
    relativeY = slideListOrg{i}{7};
    relativeX = slideListOrg{i}{6};
    radius = slideListOrg{i}{9};
    
    imDemo = drawCircle( imDemo, relativeY, relativeX, radius);
    thickness = 4;
    imList{i} = drawCircle( imList{i}, relativeY, relativeX, radius, thickness);
    subplot(4,3,i);
    imshow(imList{i});
    title(titleList{i}(1:end-4));
end

figure;
imshow(imDemo); 
% title('imDemo');

%% leaving blank




