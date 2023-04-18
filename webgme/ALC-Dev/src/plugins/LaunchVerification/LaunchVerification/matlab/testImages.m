function successMap = testImages(net, imageDataFilePath, perImageFunction, reach_method, mean, std, robustness_results_file_name)
    successMap = containers.Map;

    %% Set up number of cores
    c = parcluster('local');
    numCores = c.NumWorkers; % specify number of cores used for verification

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
        successMap(image_path) = r_nn;

        tline = fgetl(fid);
    end
    fclose(fid);

    fid = fopen(robustness_results_file_name, 'w')
    for key = keys(successMap)
        fprintf(fid, "%s,%f", key{1}, successMap(key{1}))
    end
    fclose(fid)
end