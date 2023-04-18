function successStruct = testImages2(net, imageDataFilePath, perImageFunction, reach_method, mean, std)

    successStruct = {};

    fid = fopen(imageDataFilePath);
    tline = fgetl(fid);
    while ischar(tline)
        imagePathAndCategory = strtrim(strsplit(tline, ','));
        [image_path, category] = imagePathAndCategory{:};

        % Load image
        % If saved as matlab format
        % load(image_path);
        % If saved as another format
        image = imread(image_path);
        image = double(image);

        %% Analysis
        im_target = str2double(category);

        [mean_for_image, std_for_image] = checkMeanStd(image, mean, std);

        inputSetStar = getImageStar(image, perImageFunction, mean_for_image, std_for_image);

        % Evaluation using selected method (timeout 0f 2 minutes for each image)
        r_nn = net.verifyRobustness(inputSetStar, im_target, reach_method, numCores);
        successStruct{end+1} = struct('image_path', image_path, 'value', r_nn);

        tline = fgetl(fid);
    end
    fclose(fid);
end