data = load('Quat_Experiments/data_sys_id1.csv');
u = data(1:end-1,5:6); % heading change and speed input
x = data(1:end-1,[1,2,7,8,9,10]); % x,y,qx,qy,qz,qw
x_t = data(2:end,[1,2,7,8,9,10]);

fig = figure();
set(gcf,'color','w');
set(gcf, 'Position',  [100, 100, 1200, 900])

xe = {};
timeStep = 1.0;
% 
sim_time =2.0; 
step_size = 0.05;
simrange=0:step_size:sim_time;

xj = {};

for j=1:length(u)
    xe = {x(j,:)'};
    for i=1:length(simrange)
        xe{i+1} = simulate_quat_model(xe{i},u(j,:),step_size);
    end
    xj{j,1} = xe; 
end



hold on 

plot(x(1,1),x(1,2),'*','Color', [38, 38, 38]/255,'DisplayName','ground-truth,gazebo','LineWidth',2); 
plot(x(:,1),x(:,2),'-','Color', [157, 158, 157]/255,'DisplayName','ground-truth,gazebo','LineWidth',2); 



for k=1:length(u)
    x_sim = cell2mat(xj{k,:})';
    plot(x_sim(:,1),x_sim(:,2),'--','Color', [70, 143, 199]/255,'LineWidth',2)
end 

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

set(gca,'box','off')