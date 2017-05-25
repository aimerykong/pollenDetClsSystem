% clear
close all force
clc;

% path to the images
path_to_save = '/home/skong2/pollenProject_dataset'; % where the images to save
path_to_fetch = '/media/skong2/Seagate Backup Plus Drive/BCIplot/allpng'; % where the images to search

load('validList.mat');
load('map_num2name.mat');
load('slidename2pngDir.mat');
%% transfer the valid images
%{
For example, the image below was constructed from 41 PNG files with the file structure: 42.2007-2007-15-35.379.*.50000.39000.png
42.1994-1995-15-10.69.16.56000.26000.png

“42” refers to a randomized slide number index and is not included in the metadata (described later). This can be ignored. 
“2007-2007-15-35” refers to the sampling year and pollen trap coordinate
“379” represents the random image number (ranging from 1-1000)
“*” represents the Z-plane (-20 to 20 um) currently being viewed
“50000.39000” represents the x,y coordinates of the upper left corner of the current image stack within the larger, original scanned NGR file 


7638    155     50000   21000   1       19      667     0       4.079216e+01    5       mor

*155.0.50000.21000.png


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
pngFolder = dir(fullfile(path_to_fetch,'png*'));
if exist('pngFolderImgList.mat', 'file') && ~exist('pngFolderImgList', 'var')
    load('pngFolderImgList.mat');
elseif ~exist('pngFolderImgList.mat', 'file') && ~exist('pngFolderImgList', 'var')
    for i = 1:length(pngFolder)
        fprintf('png%d...\n', i);
        pngFolderImgList.(['png' int2str(i)]) = dir(fullfile(path_to_fetch, ['png' int2str(i)], '*png' ));
        pngFolderImgList.(['png' int2str(i)]) = {pngFolderImgList.(['png' int2str(i)]).name};
    end
    save('pngFolderImgList.mat', 'pngFolderImgList');
end
imdb.path_to_dataset = path_to_save;
imdb.imList = {};
imdb.labelname = {};
for i = 8300:length(validList) % [[[!!!!?? start point ??!!!!]]]
    curItem = validList{i};    
    className = curItem{end};
    if ~isdir( fullfile(path_to_save, className) )
        mkdir( fullfile(path_to_save, className) );
    end
    slideNumber = str2double(curItem{2});
    slideName = map_num2name{slideNumber};
    
    possiblePrefixList = slidename2pngDir(slideName);
    
    flag_found = 0;
    for pngIdx = 1:length(possiblePrefixList)
        filename = [possiblePrefixList{pngIdx}, '.*.', curItem{8}, '.', curItem{3}, '.', curItem{4}, '.png']; % z,x,y
        
        [curPngFolder, filename, ext] = fileparts(filename);
        filename = [filename ext];
        
        anchor = strfind(filename, '*');
        headmatch = strncmp( filename, pngFolderImgList.(curPngFolder), anchor-1);
        validIndex4head = find(headmatch);
        
        if isempty(validIndex4head)
            continue;
        end
        
        tailmatch = strfind( pngFolderImgList.(curPngFolder)(validIndex4head), filename(anchor+1:end) );
        tailmatch = not(cellfun('isempty', tailmatch));
        validIndex4tail_inhead = find(tailmatch);
        
        if isempty(validIndex4tail_inhead)
            continue;
        end
        
        validIndex = validIndex4head(validIndex4tail_inhead);
        
%         tailmatch = strfind( pngFolderImgList.(curPngFolder), filename(anchor+1:end) );
%         tailmatch = not(cellfun('isempty', tailmatch));
%         headmatch = headmatch & tailmatch;
%         validIndex = find(headmatch);
        
        if ~isempty(validIndex)
            flag_found = 1;
            srcpath = fullfile(path_to_fetch, curPngFolder, pngFolderImgList.(curPngFolder){validIndex(1)});
            desname = pngFolderImgList.(curPngFolder){validIndex(1)};
            break;
        end
    end
    
    if flag_found==1
        copyfile( srcpath,  fullfile(path_to_save, className, desname) );
        imdb.imList{end+1} =  fullfile(className, desname);
        imdb.labelname{end+1} = className;
    end
    if mod(i,10) == 0
        fprintf('%d/%d -- %s\n', i, length(validList), filename);        
    end
end

save('imdb.mat', 'imdb');
%% leaving blank






