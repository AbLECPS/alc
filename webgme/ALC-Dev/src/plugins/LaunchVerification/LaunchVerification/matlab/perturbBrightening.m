function perturbBrighteningPerImageFunction = perturbBrightening(dataStruct)

    delta = dataStruct.delta;
    threshold = dataStruct.threshold;

    function perturbBrighteningPerChannelFunction = perturbBrighteningPerImage(image)

        image_size = num2cell(size(image));
        [w, h] = image_size{1:2};

        function [lb, ub] = perturbBrighteningPerChannel(channel)

            IM = image(:, :, channel);
            lb = IM;
            ub = IM;
            for p=1:(w*h)
                if  IM(p) >= threshold
                    lb(p) = 255-255*delta;
                    ub(p) = 255;
                end
            end
        end

        perturbBrighteningPerChannelFunction = @perturbBrighteningPerChannel;

    end

    perturbBrighteningPerImageFunction = @perturbBrighteningPerImage;
end
