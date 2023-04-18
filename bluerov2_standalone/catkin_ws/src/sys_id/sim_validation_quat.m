files = {'Quat_Experiments/data_sys_id1.csv',...
    'Quat_Experiments/data_sys_id2.csv','Quat_Experiments/data_sys_id3.csv',...
    'Quat_Experiments/data_sys_id5.csv','Quat_Experiments/data_sys_id6.csv','Quat_Experiments/data_sys_id7.csv','Quat_Experiments/data_sys_id8.csv','Quat_Experiments/data_sys_id9.csv'};

for i=1:length(files)
    validate_quat_model(files{i});
    quat_validation(files{i});
end