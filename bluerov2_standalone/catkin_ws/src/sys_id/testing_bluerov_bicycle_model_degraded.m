clear
param_list = {{ 0.0445   14.7404    0.9544    0.2285    0.2285},...
        {0.0482   14.7404    0.9703    0.2285    0.2285}};


for i=1:length(param_list)
    err = validate_bluerov_bicycle('DegradedModel/sys_id_data2.csv',param_list{i}); 
end
%validate_bluerov_bicycle('SysIdData/sys_id_data30.csv',param_list{1});
%validate_bluerov_bicycle('SysIdData/sys_id_data31.csv',param_list{1});

