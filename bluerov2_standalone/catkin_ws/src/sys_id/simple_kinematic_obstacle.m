function xp = simple_kinematic_obstacle(x,v,stepSize)
%SIMPLE_KINEMATIC_OBSTACLE Summary of this function goes here
%   Detailed explanation goes here
xt = [x(1);x(2)];
dx = [v(1);v(2)];
xp = (stepSize * dx) + xt;
end

