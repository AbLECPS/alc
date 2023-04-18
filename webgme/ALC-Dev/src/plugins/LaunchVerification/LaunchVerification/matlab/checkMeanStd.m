function [mean, std] = checkMeanStd(image, mean, std)
        channels = size(image, 3);
        % Grayscale -> channels = 1;
        % RGB -> channels = 3
        if channels == 3 && length(mean) == 1
            mean = [mean mean mean];
        elseif channels == 3 && length(mean) ~= 3
            error('Please, select a valid mean vector');
        end
        % Check std
        if channels == 3 && length(std) == 1
            std = [std std std];
        elseif channels == 3 && length(std) ~= 3
            error('Please, select a valid std vector');
        end
end