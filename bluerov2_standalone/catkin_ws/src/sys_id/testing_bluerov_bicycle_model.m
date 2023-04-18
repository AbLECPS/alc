%validate_bluerov_bicycle('SysIdData/sys_id_data27.csv');
%validate_bluerov_bicycle('SysIdData/sys_id_data20.csv');
%validate_bluerov_bicycle('SysIdData/sys_id_data23.csv');

param_list = {{0.0006  100.7298    0.9286},{0.0299    4.8538    1.0692},{0.0028   50.7746    0.9472},...
    {0.0021   63.0149    0.9444},{0.0033   53.1061    0.9411},{-0.0002   82.7936    0.9192},{-0.0060,0.8476,-0.4557},...
    {0.0916,5.9942,0.7788},{0.0677,7.2439,0.7941}...
    {0.0828   29.0014    1.0440},{0.0450   27.7782    1.0505}, {0.0450   27.7782    1.0505}...
    {0.0482   27.4953    1.0634},{0.0830   26.7002    1.0592}, {0.0815   25.6698    1.0582}...
    {0.0815   25.6698    1.0581},{0.0818   25.6699    1.0592}, {0.0874   27.5070    1.0569}...
    {0.0798   26.1081    1.0435},{0.0671   23.1977    1.0444}, {0.0690   23.1993    1.0416}...
    {0.0690   23.1993    1.0416}};


good_params = [1,8,9,11,12];


for i=1:length(param_list)
    if(ismember(i,good_params))
        param_list{i}
        err = validate_bluerov_bicycle('SysIdData/sys_id_data26.csv',param_list{i});
    end 
end
%validate_bluerov_bicycle('SysIdData/sys_id_data30.csv',param_list{1});
%validate_bluerov_bicycle('SysIdData/sys_id_data31.csv',param_list{1});

