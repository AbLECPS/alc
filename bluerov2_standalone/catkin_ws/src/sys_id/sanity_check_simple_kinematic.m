num_steps = 101;
y0 = [0;0];
v = [-1.0;0.1];

% Initialize the time stp
timeStep = 0.02;
% create an array of zeros to store time steps and state vector
t=zeros(num_steps,1);
xe = {};
xe{1} = y0;
for i=1:num_steps
    t(i+1) = (i)*timeStep;
    xe{i+1} = simple_kinematic_obstacle(xe{i},v,timeStep);
end
% 
xe =cell2mat(xe);
xe = xe';


figure();
plot(xe(:,1),xe(:,2),'*');
xlabel('x') 
ylabel('y') 
xlim([-2 2])
ylim([-2 2])
title("Position of Vehicle")