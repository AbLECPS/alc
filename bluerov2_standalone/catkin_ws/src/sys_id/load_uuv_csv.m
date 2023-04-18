function [states,inputs,outputs] = load_uuv_csv(csvFile)
data = load(csvFile);

% inputs
inputs = [data(20:end-20,10) data(20:end-20,11)]; % heading change and speed
states = [data(20:end-20,12:13) data(20:end-20,23)];
outputs = [data(21:end-19,12:13) data(21:end-19,23)]; % x, y, yaw

% reshape data
inputs = reshape(inputs,[],2);
outputs = reshape(outputs,[],3);
states = reshape(states,[],3);
end


