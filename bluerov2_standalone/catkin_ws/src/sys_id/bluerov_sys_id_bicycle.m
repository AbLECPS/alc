% load the data
exp1 = load_experiment_data('SysIdData/sys_id_data37.csv');
exp2 = load_experiment_data('SysIdData/sys_id_data38.csv');
exp3 = load_experiment_data('SysIdData/sys_id_data39.csv');
exp4 = load_experiment_data('SysIdData/sys_id_data23.csv');
exp5 = load_experiment_data('SysIdData/sys_id_data34.csv');
exp6 = load_experiment_data('SysIdData/sys_id_data35.csv');
exp7 = load_experiment_data('SysIdData/sys_id_data26.csv');
exp8 = load_experiment_data('SysIdData/sys_id_data22.csv');
exp9 = load_experiment_data('SysIdData/sys_id_data36.csv');
exp10 = load_experiment_data('SysIdData/sys_id_data33.csv');
exp11 = load_experiment_data('SysIdData/sys_id_data32.csv');
exp12 = load_experiment_data('SysIdData/sys_id_data30.csv');
exp13 = load_experiment_data('SysIdData/sys_id_data31.csv');


% Merge the data
data_est = merge(exp11,exp12,exp8,exp4,exp1,exp2,exp3,exp4,exp5,exp6,exp7,exp9,exp10,exp11,exp13);

%data_est = merge(exp6);
num_experiments = 13;


% specify the initial parameters

 ca = 0.0677;
 cm = 7.2439;	
 ch = 0.7941;	
 lf = 0.2285;	
 lr = 0.169;
 parameters    = {ca,cm,ch,lf,lr};

 % SysID options 
opt = nlgreyestOptions;
opt.Display = 'Full';
opt.SearchOptions.FunctionTolerance = 1e-9;
opt.SearchOptions.MaxIterations = 200;
%opt.SearchMethod ='fmincon';
%opt.SearchMethod ='grad';

% not sure why they call it order
% it species the number of model outputs
% the model inputs, and states
% so for us its 2 inputs, 4 outputs, 4 states
order         = [4 2 4];
Ts = 0;

% Let's try the system identification now
for i=1:num_experiments
    
% get the dataset   
ds = getexp(data_est,i);
initial_states = reshape(ds.y(1,:),[],1);
nonlinear_model = idnlgrey('bicycle_model',order,parameters,initial_states,Ts);
%nonlinear_model.algorithm.SimulationOptions.Solver = 'ode45';
nonlinear_model.algorithm.SimulationOptions.MaxStep = 1e-1;
nonlinear_model.algorithm.SimulationOptions.InitialStep = 1e-4;
%nonlinear_model.SimulationOptions.AbsTol = 1e-6;
%nonlinear_model.SimulationOptions.RelTol = 1e-5;
setpar(nonlinear_model,'Fixed',{false, false,false,true,true});


nonlinear_model = nlgreyest(ds,nonlinear_model,opt);

params = nonlinear_model.Parameters;
params = [params.Value];
disp(params);
parameters = {params(1),params(2),params(3),lf,lr};


figure();
compare(ds,nonlinear_model)
end


close all 
% Let's see how the parameters do overall. 


for i=1:num_experiments
% get the dataset   
ds = getexp(data_est,i);
initial_states = reshape(ds.y(1,:),[],1);
nonlinear_model = idnlgrey('bicycle_model',order,parameters,initial_states,Ts);
figure();
compare(ds,nonlinear_model)
end 




