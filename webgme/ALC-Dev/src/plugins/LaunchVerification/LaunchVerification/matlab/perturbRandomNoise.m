function perturbRandomNoisePerImageFunction = perturbRandomNoise(dataStruct)

    pixels = dataStruct.pixels;
    noise = dataStruct.noise;

    function perturbRandomNoisePerChannelFunction = perturbRandomNoisePerImage(image)

        image_size = num2cell(size(image));
        [w, h] = image_size{1:2};
        pixels_random = randi([1 h*w],channels,pixels);

        function [lb, ub] = perturbRandomNoisePerChannel(channel)

            IM = image(:, :, channel);
            lb = IM;
            ub = IM;
            for p=pixels_random(channel,:)
                lb(p) = max(0, IM(p) - rand*noise);
                ub(p) = min(255, IM(p)+ rand*noise);
            end
        end

        perturbRandomNoisePerChannelFunction = @perturbRandomNoisePerChannel;

    end

    perturbRandomNoisePerImageFunction = @perturbRandomNoisePerImage;
end