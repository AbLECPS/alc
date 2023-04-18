function [sys, Fit] = QuatSysID(states,learnFocus,dispL,varargin)
%SYSID Learns a linear dynamical model with the speficied input parameters
%   Outputs:
%       sys = linear dynamical system learned
%       Fit = how accurate the model fits the data
%   Inputs:
%       outvar = 3 (x,y,yaw) 
%       learnFocus = 'prediction' or 'simulation'
%       dispL = 'on' or 'off', 'on' to see training progress, off for no displays
%       varargin = datasets we want to use for learning model
%% Learn the UUV dyamics as a black box model
% To begin we will use a simple linear state-space model with identifiable
% parameters
% This will do a 2-step system id. First, an initial estimate using n4sid, 
% and then the one we were already using
% Get the data in the format we need
out = 6; % number of outputs
% n = states; % number of states
j = 0;
Ts = 1; % time step
for i=1:length(varargin)
    % Load data (more data has been collected in this experiment)
    data = load(string(varargin{i}));
    % rows = x,y,yaw,speed, heading command, speed command
    j = j+1;
    inputs = data(:,5:6); % heading change and speed input
    outputs = data(:,[1,2,7,8,9,10]); % x,y,qx,qy,qz,qw
    if j == 1
        ida = iddata(outputs,inputs,Ts);
    else
        ida_t = iddata(outputs,inputs,Ts);
        ida = merge(ida,ida_t);
    end
end
%% Step 1. Initial estimate of the system
opts = n4sidOptions('Display','on',...
                    'InitialState','estimate', 'Focus',learnFocus,...
                    'EnforceStability',true);
sys_it = n4sid(ida,states,opts);
%% Step 2. Final step (ssest)
% Set options for estimating blackbox model
opt = ssestOptions('EnforceStability',false,'Focus',learnFocus,'Display',dispL,'SearchMethod','auto');
opt.SearchOptions.MaxIterations = 3000;
opt.InitialState = 'estimate';
% Fix some parameters
sys_it.Structure.C.Value = zeros(out,states);
sys_it.Structure.C.Value = eye(out);
sys_it.Structure.C.Free = false;
% Estimate system
sys = ssest(ida,sys_it,opt);
Fit = sys.Report.Fit.FitPercent;
%Name = ['sys_id'  '_rtreach_'  date];
%sys.Name = Name;
%save(Name,'sys','Fit','ida');
end