function xp = simulate_model_bicycle(x,u,ca,cm,ch,lf,lr,stepSize)

% states
% x(1): x position 
% x(2): y position
% x(3): vehcile speed
% x(4): vehicle heading

% u(1) is heading-change
% u(2) is speed

inner_state= [x(1);x(2);x(3); x(4)];
theta = x(4);


%deg2rad(u(1))
u1 = u(1);
if(u1<-30)
    u1 = -30;
elseif(u1>30)
    u1 = 30;
end
uh = (pi*u1)/180;
dx = [x(3)* cos(theta); 
      x(3)* sin(theta); 
      -ca *x(3) + (ca*cm)*(u(2)-ch);
      (x(3)/(lf+lr))*tan(uh)];
xp = (stepSize * dx) + inner_state;
end