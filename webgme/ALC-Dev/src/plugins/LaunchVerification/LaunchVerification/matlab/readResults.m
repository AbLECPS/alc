function successMap = readResults(robustnessResultsFileName)
    successMap = containers.Map
    fid = fopen(robustnessResultsFileName)
    tline = fgetl(fid);
    while ischar(tline)
        imagePathAndCategory = strtrim(strsplit(tline, ','));
        [image_path, robustnessResults] = imagePathAndCategory{:};
        successMap(image_path) = str2double(robustnessResults)
        tline = fgetl(fid);
    end
    fclose(fid)
end