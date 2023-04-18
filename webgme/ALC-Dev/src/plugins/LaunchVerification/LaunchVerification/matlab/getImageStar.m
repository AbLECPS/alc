function imageStar = getImageStar(image, perImageFunction, mean_for_image, std_for_image)

    perChannelFunction = perImageFunction(image);
    image_size = num2cell(size(image));
    size_dimensions = length(image_size);
    if size_dimensions == 2
        [w,h] = image_size{:};
        channels = 1;
    else
        [w, h, channels] = image_size{:};
    end

    % Image perturbation
    for c=1:channels

        [lb, ub] = perChannelFunction(c);

        % Normalize input set
        lb = reshape(lb,[w,h]);
        lb = ((lb./255)-mean_for_image(c))./std_for_image(c);
        ub = reshape(ub,[w,h]);
        ub = ((ub./255)-mean_for_image(c))./std_for_image(c);
        LB(:,:,c) = lb;
        UB(:,:,c) = ub;
    end
    imageStar = ImageStar(LB,UB);
end