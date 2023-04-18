alc_wd = getenv('ALC_WORKING_DIR')
cur_dir = fullfile(alc_wd,'{{result_dir}}')
setenv('temp_alc', cur_dir)
cd '/verivital/nnv/code/nnv'
startup_nnv
cur_dir = getenv('temp_alc')
cd(cur_dir)