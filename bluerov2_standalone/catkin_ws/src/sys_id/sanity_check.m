num_steps = 101;
y0 = [-1960.9;42.01;0.175];
u = [11.21;0.335];


% Initialize the time stp
timeStep = 0.02;
% create an array of zeros to store time steps and state vector
t=zeros(num_steps,1);
xe = {};
xe{1} = y0;
for i=1:num_steps
    t(i+1) = (i)*timeStep;
    xe{i+1} = simulate_model(xe{i},u,timeStep);
end
% 
xe =cell2mat(xe);
xe = xe';


figure();
plot(xe(:,1),xe(:,2),'*');
xlabel('x') 
ylabel('y') 
title("Position of Vehicle")


figure();
plot(t,xe(:,3));
xlabel('t') 
ylabel('heading') 
title("heading vehicle")

fprintf('final state [%f,%f,%f]\n',xe(end,1),xe(end,2),xe(end,3));
