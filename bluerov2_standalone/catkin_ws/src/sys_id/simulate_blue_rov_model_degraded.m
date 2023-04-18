function xp = simulate_blue_rov_model_degraded(x,u,stepSize)
%SIMULATE_BLUE_ROV_MODEL Summary of this function goes here
%   Detailed explanation goes here

A = [-0.00956429693458404,0.00638189205403102,-0.178462917995277;-0.0101008766925151,0.0040599635078681,0.186029781068614;-0.00203540102846258,-0.000705517631239591,-0.0927455151714739];
B = [-0.00363284720934433,0.420989575738248;0.00051739611066663,0.338197692927455;-0.000373201726481037,0.141193147931726];
inner_state= [x(1);x(2);x(3)];
ut = [u(1);u(2)];
dxut = A * inner_state + B* ut;
xp = inner_state + (stepSize * dxut);
end


