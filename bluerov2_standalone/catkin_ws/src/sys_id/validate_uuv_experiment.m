function err= validate_uuv_experiment(csv_filename)
% load the data from the csv 
[x,u,x_t] = load_uuv_csv(csv_filename);


% Initialize the time stp
timeStep = 0.05;
% create an array of zeros to store time steps and state vector
t=zeros(length(u),1);
xe = {};
for i=1:length(x)
    t(i) = (i-1)*timeStep;
    xe{i} = simulate_model(x(i,:),u(i,:),timeStep);
end
% 
xe =cell2mat(xe);
xe = xe';
err = immse(x_t,xe);

fig = figure();
subplot(2,2,[1,2]);
set(gcf, 'Position',  [100, 100, 900, 900]);
plot(x_t(:,1),x(:,2),'DisplayName','ground-truth,gazebo','LineWidth',2)
hold on;
plot(xe(:,1),xe(:,2),'--','DisplayName','prediction','LineWidth',2)
xlabel('x (meters)') 
ylabel('y (meters)') 
title("Vehicle Position (map frame)")
legend
hold off

subplot(2,2,[3,4]);
plot(t,x(:,3),'DisplayName','ground-truth,gazebo','LineWidth',2)
hold on;
plot(t,xe(:,3),'--','DisplayName','prediction','LineWidth',2)
xlabel('t (seconds)') 
ylabel('yaw (radians)') 
title("Vehicle Orientation")
legend

%change to the desired value     
set(findobj(gcf,'type','axes'),'FontName','Calibri','FontSize',11,'FontWeight','Bold', 'LineWidth', 2,'layer','top');

sgt =sgtitle(strcat('Validation MSE=',string(err)));
sgt.FontSize = 20;
figname = split(strrep(csv_filename,'csv/',''),".");
savename = strcat("../images/","val",".png");
saveas(fig,savename);
end

