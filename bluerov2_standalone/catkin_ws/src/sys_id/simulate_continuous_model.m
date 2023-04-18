function xp = simulate_model(x,u,stepSize)

% states
% x(1): x position 
% x(2): y position
% x(3): vehicle heading

% u(1) is heading-change
% u(2) is speed
% matrices = 


A = [1.56970837855349e-05, 0.0138347420972705, 0.0894261759462228;
-5.55466464489461e-05, -0.00462628281269562, 0.148150845874259;
3.04601124770537e-05,	-0.000222172003706165,	0.00857102289824863];

B = [ 0.000126689929278544, 0.507552834425425;
0.00347662933457233, 0.336562420408369;
2.67457034292762e-05, 0.299095809754267];

inner_state= [x(1);x(2);x(3)];
control = [u(1);u(2)];
dx = A* inner_state + B*control;
xp = (stepSize * dx) + inner_state;
end