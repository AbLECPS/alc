function perturbDarkeningPerImageFunction = perturbDarkening(dataStruct)

    delta = dataStruct.delta;
    threshold = dataStruct.threshold;

    function perturbDarkeningPerChannelFunction = perturbDarkeningPerImage(image)

        image_size = num2cell(size(image));
        [w, h] = image_size{1:2};

        function [lb, ub] = perturbDarkeningPerChannel(channel)

            IM = image(:, :, channel);
            lb = IM;
            ub = IM;
            for p=1:(w*h)
                if  IM(p) >= threshold
                    lb(p) = 0;
                    ub(p) = IM(p)*delta;
                end
            end
        end

        perturbDarkeningPerChannelFunction = @perturbDarkeningPerChannel;

    end

    perturbDarkeningPerImageFunction = @perturbDarkeningPerImage;
end
