function data = load_experiment_data(csvFile)
% load the data
T = load(csvFile);
inputs = T(61:end-1,5:6); % heading change and speed input
outputs = T(62:end,[1,2,4,3]); % x,y,qx,qy,qz,qw

data = iddata(outputs,inputs,0.05);

data.OutputName ={'x';'y';'speed';'theta'};
data.InputName ={'heading';'speed'};
end

