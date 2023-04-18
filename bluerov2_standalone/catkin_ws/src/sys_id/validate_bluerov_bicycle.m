function err= validate_bluerov_bicycle(csv_filename,params)
% load the data from the csv 


T = load(csv_filename);
u = T(20:end-1,5:6); % heading change and speed input
x= T(20:end-1,[1,2,4,3]); % x,y,speed,yaw
x_t= T(21:end,[1,2,4,3]); % x,y,speed,yaw

ca = params{1}; %0.0916; %0.0677;
cm = params{2}; % 5.9942; %7.2439;	
ch =  params{3}; %0.7788; %0.7941;	
lf = 0.2285;	
lr = 0.1690;


% Initialize the time stp
timeStep = 0.05;
% create an array of zeros to store time steps and state vector
t=zeros(length(u),1);
xe = {};
for i=1:length(x)
    t(i) = (i-1)*timeStep;
    xe{i} = simulate_model_bicycle(x(i,:),u(i,:),ca,cm,ch,lf,lr,timeStep);

end
% 
xe =cell2mat(xe);
xe = xe';
err = immse(x_t,xe);

% fig = figure();
% subplot(2,2,[1,2]);
% set(gcf, 'Position',  [100, 100, 900, 900]);
% plot(x_t(:,1),x(:,2),'DisplayName','ground-truth,gazebo','LineWidth',2)
% hold on;
% plot(xe(:,1),xe(:,2),'--','DisplayName','prediction','LineWidth',2)
% xlabel('x (meters)') 
% ylabel('y (meters)') 
% title("Vehicle Position (map frame)")
% legend
% hold off
% 
% subplot(2,2,[3,4]);
% plot(t,x(:,3),'DisplayName','ground-truth,gazebo','LineWidth',2)
% hold on;
% plot(t,xe(:,3),'--','DisplayName','prediction','LineWidth',2)
% xlabel('t (seconds)') 
% ylabel('yaw (radians)') 
% title("Vehicle Orientation")
% legend


fig = figure();
set(gcf,'color','w');
set(gcf, 'Position',  [100, 100, 1200, 900])

xe = {};
% 
sim_time =0.5; 
step_size = 0.05;
simrange=0:step_size:sim_time;

xj = {};
for j=1:length(u)
    xe = {x(j,:)'};
    for i=1:length(simrange)
        xe{i+1} = simulate_model_bicycle(xe{i},-u(j,:),ca,cm,ch,lf,lr,step_size);
    end
    xj{j,1} = xe; 
end

hold on 

plot(x(1,1),x(1,2),'s','Color', [38, 38, 38]/255,'MarkerSize',10,'DisplayName','ground-truth,gazebo','LineWidth',2); 
plot(x(:,1),x(:,2),'-','Color', [157, 158, 157]/255,'DisplayName','ground-truth,gazebo','LineWidth',2); 




for k=1:length(u)
    x_sim = cell2mat(xj{k,:})';
    plot(x_sim(1,1),x_sim(1,2),'*','Color', [0, 0, 199]/255,'LineWidth',2)
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


% %change to the desired value     
% set(findobj(gcf,'type','axes'),'FontName','Calibri','FontSize',11,'FontWeight','Bold', 'LineWidth', 2,'layer','top');
% 
% sgt =sgtitle(strcat('Validation MSE=',string(err)));
% sgt.FontSize = 20;
% figname = split(strrep(csv_filename,'csv/',''),".");
% savename = strcat("../images/","val",".png");
% saveas(fig,savename);
end