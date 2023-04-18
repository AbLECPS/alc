[sys,fit] = runSysID_BlueROV(3,'prediction','on','DegradedModel/sys_id_data5.csv','DegradedModel/sys_id_data.csv','DegradedModel/sys_id_data2.csv','DegradedModel/sys_id_data3.csv','DegradedModel/sys_id_data4.csv');
cont_sys = d2c(sys,'tustin');
A = regexprep(mat2str(cont_sys.A), {'\[', '\]', '\s+'}, {'', '', ','});
B = regexprep(mat2str(cont_sys.B), {'\[', '\]', '\s+'}, {'', '', ','});

