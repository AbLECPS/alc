function yaw = convert_yaw(data)
%CONVERT_YAW Summary of this function goes here
%   Detailed explanation goes here
yaw = data(:,3);
drct = 0;
for i=2:length(yaw)
    if abs(yaw(i) - yaw(i-1)) > pi
        tt = yaw(i) + 2*pi*drct;
        yaw(i) = tt;
    end
    drct = sign(yaw(i)-yaw(i-1));
end
end

