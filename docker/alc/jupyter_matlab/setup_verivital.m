cd /verivital/nnv/code/nnv
try
  install
catch
  disp('Exception caught!')
end
cd /verivital/nnv/code/nnv
startup_nnv
cd /verivital
savepath('/verivital/pathdef.m')
