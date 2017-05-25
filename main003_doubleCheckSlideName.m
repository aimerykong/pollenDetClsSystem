clear
close all force
clc;

% path to the images
path_to_save = '/home/skong2/pollenProject_dataset'; % where the images to save
path_to_fetch = '/media/skong2/Seagate Backup Plus Drive/BCIplot/allpng'; % where the images to search

load('validList.mat');
load('map_num2name.mat');

%% all the slides
slideNameList = {};
filename = 'meta3.txt';
fid = fopen(filename, 'r');
tline = fgets(fid);
i = 0;
while ischar(tline)
    tline = strtrim(tline);
    a = strfind(tline, '.');
    slideNameList{end+1} = tline(a(1)+1:end);    
    tline = fgets(fid);
end
%% png* directories
slidename2pngDir = containers.Map;
slidenamePrefix2pngDir = containers.Map;
for pngIdx = 1:10
    filename = sprintf('png%d.txt', pngIdx);
    fid = fopen(filename, 'r');
    tline = fgets(fid);
    i = 0;
    slideNameList = {};
    slideNamePrefixList = {};
    while ischar(tline)
        tline = strtrim(tline);
        a = strfind(tline, '.');
        slideNameList{end+1} = tline(a(1)+1:end);
        slideNamePrefixList{end+1} = tline;
        
        if isKey(slidenamePrefix2pngDir, tline)
%             fprintf('%d, %s\n', pngIdx, tline)
            slidenamePrefix2pngDir(tline) = [slidenamePrefix2pngDir(tline) pngIdx];
        else
            slidenamePrefix2pngDir(tline) = pngIdx;
        end
        
        if isKey(slidename2pngDir, tline(a(1)+1:end))
%             fprintf('%d, %s\n', pngIdx, tline)
            slidename2pngDir(tline(a(1)+1:end)) = [slidename2pngDir(tline(a(1)+1:end)), sprintf('png%d/%s', pngIdx, tline)];
        else
            slidename2pngDir(tline(a(1)+1:end)) = {sprintf('png%d/%s', pngIdx, tline)};
        end
        
        tline = fgets(fid);
    end
    slideNameList = sort(slideNameList);
    slideNamePrefixList = sort(slideNamePrefixList);

%     fprintf('%s\n', filename);
%     fprintf('\t #slideNameList=%d, #slideNamePrefixList=%d\n', length(unique(slideNameList)), length(unique(slideNamePrefixList)) );
end

slidenameList = keys(slidename2pngDir);
for i = 1:length(slidenameList)
    slidename2pngDir(slidenameList{i}) = unique(slidename2pngDir(slidenameList{i}));
    fprintf('%d', i);
    curmap = slidename2pngDir(slidenameList{i});
    for j = 1:length(slidename2pngDir(slidenameList{i}))
        fprintf('\t%s\n', curmap{j});
    end    
end
save('slidename2pngDir.mat', 'slidename2pngDir');
%% leaving blank
