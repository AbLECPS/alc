%[sys,fit] = runSysID_BlueROV(3,'prediction','on','blue_rov/sys_id_data6.csv','blue_rov/sys_id_data3.csv','blue_rov/sys_id_data4.csv','blue_rov/sys_id_data5.csv','blue_rov/sys_id_data6.csv');
%'blue_rov/sys_id_data.csv'
% [sys,fit] = runSysID_BlueROV(3,'prediction','on','SysIdData/sys_id_data10.csv',...
%     'SysIdData/sys_id_data11.csv','SysIdData/sys_id_data12.csv','SysIdData/sys_id_data13.csv',...
%     'SysIdData/sys_id_data14.csv','SysIdData/sys_id_data15.csv','SysIdData/sys_id_data16.csv',...
%     'SysIdData/sys_id_data17.csv','SysIdData/sys_id_data2.csv','SysIdData/sys_id_data3.csv',...
%     'SysIdData/sys_id_data4.csv','SysIdData/sys_id_data5.csv','SysIdData/sys_id_data6.csv',...
%     'SysIdData/sys_id_data7.csv','SysIdData/sys_id_data8.csv','SysIdData/sys_id_data9.csv',...
%     'SysIdData/sys_id_data18.csv','SysIdData/sys_id_data19.csv','SysIdData/sys_id_data.csv',...
%     'SysIdData/sys_id_data20.csv','SysIdData/sys_id_data22.csv','SysIdData/sys_id_data23.csv',...
%     'SysIdData/sys_id_data18.csv','SysIdData/sys_id_data19.csv','SysIdData/sys_id_data21.csv');

%[sys,fit] = runSysID_BlueROV(3,'prediction','on','SysIdData/sys_id_data20.csv','SysIdData/sys_id_data21.csv');

% [sys,fit] = runSysID_BlueROV(3,'prediction','on','SysIdData/sys_id_data22.csv');

% [sys,fit] = runSysID_BlueROV(3,'prediction','on','SysIdData/sys_id_data20.csv','SysIdData/sys_id_data22.csv','SysIdData/sys_id_data23.csv','SysIdData/sys_id_data18.csv','SysIdData/sys_id_data19.csv','SysIdData/sys_id_data21.csv');

% [sys,fit] = runSysID_BlueROV(3,'prediction','on','SysIdData/sys_id_data18.csv','SysIdData/sys_id_data21.csv','SysIdData/sys_id_data25.csv','SysIdData/sys_id_data26.csv');

[sys,fit] = runSysID_BlueROV(3,'prediction','on','SysIdData/sys_id_data27.csv','SysIdData/sys_id_data28.csv');%,,'SysIdData/sys_id_data29.csv');
cont_sys = d2c(sys,'tustin');
A = regexprep(mat2str(cont_sys.A), {'\[', '\]', '\s+'}, {'', '', ','});
B = regexprep(mat2str(cont_sys.B), {'\[', '\]', '\s+'}, {'', '', ','});