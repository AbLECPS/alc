function successStruct = testImage3(net, image_path, im_target, perImageFunction, reach_method, mean, std)

        % Load image
        % If saved as matlab format
        % load(image_path);
        % If saved as another format
        image = imread(image_path);
        image = double(image);

        %% Analysis
        [mean_for_image, std_for_image] = checkMeanStd(image, mean, std);

        inputSetStar = getImageStar(image, perImageFunction, mean_for_image, std_for_image);

        % Evaluation using selected method (timeout 0f 2 minutes for each image)
        r_nn = net.verifyRobustness(inputSetStar, im_target, reach_method, numCores);
        successStruct = struct('image_path', image_path, 'value', r_nn);
end