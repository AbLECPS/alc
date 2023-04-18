function [err,xe] = validate_blue_rov_model(csvfilename)
%VALIDATE_QUAT_MODEL Summary of this function goes here
%   Detailed explanation goes here
data = load(csvfilename);
u = data(1:end-1,5:6); % heading change and speed input
%x = data(1:end-1,[1,2,3,4]); % x,y,yaw,speed
%x_t = data(2:end,[1,2,3,4]);
x = data(1:end-1,[1,2,3]); % x,y,yaw
x_t = data(2:end,[1,2,3]);

timeStep = 0.05;
% create an array of zeros to store time steps and state vector
t=zeros(length(u),1);
xe = {};
for i=1:length(x)
    t(i) = (i-1)*timeStep;
    xe{i} = simulate_blue_rov_model(x(i,:),u(i,:),timeStep);
end

xe =cell2mat(xe);
xe = xe';
err = immse(x_t,xe);

fig = figure();
set(gcf,'color','w');
set(gcf, 'Position',  [100, 100, 1200, 900])




plot(x(1,1),x(1,2),'*','Color', [38, 38, 38]/255,'DisplayName','init','LineWidth',2);
hold on 
plot(x(:,1),x(:,2),'-','Color', [157, 158, 157]/255,'DisplayName','ground-truth,gazebo','LineWidth',2) ;
plot(xe(:,1),xe(:,2),'--','DisplayName','prediction','Color', [70, 143, 199]/255,'LineWidth',2)



xlabel('x (meters)') 
ylabel('y (meters)') 
title("Vehicle Position (map frame)")
legend
hold off





ax = gca; % Get handle to current axes.
ax.XColor = [87, 93, 97]/255; % Red
ax.YColor = [87, 93, 97]/255; % Blue
xlabel('x (meters)') 
ylabel('y (meters)') 
t= title("Vehicle Position (map frame)",'Color',[87, 93, 97]/255);
set(t, 'horizontalAlignment', 'left')
set(t, 'units', 'normalized')
set(t, 'position', [0.01 1.01 0]);

legend('init','ground-truth,gazebo','prediction','Location','northwest')
legend boxoff 

set(gca,'box','off');
end

