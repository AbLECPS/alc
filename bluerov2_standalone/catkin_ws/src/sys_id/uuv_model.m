function [dx,y] = uuv_model(t,x,u,varargin)
% states
% x(1): x position 
% x(2): y position
% x(4): vehicle heading

% u(1) is heading-change
% u(2) is speed
% matrices = 
% A = [0.999683993461193,0.00107006380992925,-2.58895074646266;
%     3.93865064196743e-05,0.999865198478192,-0.290848899831410;
%     1.18474912533678e-06,-5.01931779057720e-06,0.583866645209362];
% B = [0.0369928875656454,0.844042316730837;
%     0.0163886082047795,0.446282095713212;
%     0.00728931565945887,0.00953122338961594];

A = [0.000110934757707122,	0.0176481545263511,	-0.361993031166798;
0.000113111902385930,	0.00162986090236893, 1.63321898403585;
2.82185649873063e-05,	-1.75925433470232e-05,	0.645980176182907];


B =[ 0.00588416436363635, 0.536394967123741;
-0.0240381022203886, 0.555059923037031;
-0.0112500920603916, 0.227799073448295];

y(1) = x(1);
y(2) = x(2);
y(3) = x(3);
xt = [x(1);x(2);x(3)];
ut = [u(1);u(2)];
dx = A* xt + B*ut;
end 