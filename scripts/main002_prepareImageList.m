filename = './ExampleMetaDataPart.txt';

validList = {};
invalidList = {};

fid = fopen(filename);
tline = fgets(fid);
i = 0;
while ischar(tline)
    if mod(i, 1000) == 0
        disp(tline)
    end
    a = strsplit(strtrim(tline), '	');
    if length(a) == 11
        validList{end+1} = a;
    else % incomplete annotation on confidence and label
        if str2double(a{5}) > 1 % missing confidence
            a{10} = '5'; % set the confidence to 5
            a{11} = validList{end}{end};
            validList{end+1} = a;
        else
            invalidList{end+1} = a;
        end
    end
    i = i + 1;
    
    tline = fgets(fid);
    if ~ischar(tline)
        break;
    end
end
fclose(fid);
%% save the hdf5 record file
save('validList.mat','validList');



