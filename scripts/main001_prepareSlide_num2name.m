filename = './Slide_num2name.txt';

map_num2name = {};

fid = fopen(filename);
tline = fgets(fid);
i = 0;
while ischar(tline)
    disp(tline);
    
    a = strsplit(strtrim(tline), ',');
    i = i + 1;
    map_num2name{i} = a{2};
    
    tline = fgets(fid);
    if ~ischar(tline)
        break;
    end
end
fclose(fid);
%% save the hdf5 record file
save('map_num2name.mat','map_num2name');



