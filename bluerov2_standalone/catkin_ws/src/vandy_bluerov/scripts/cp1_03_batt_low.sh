#!/bin/bash

exit_status=0

_term() {
  exit_status=$? # = 130 for SIGINT
  echo "Caught SIGINT signal!"
  kill -INT "$child" 2>/dev/null
}

trap _term SIGINT

while :
do

task_folder="task_"$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p /mnt/results/$task_folder

#let "enable_obstacles = $RANDOM % 3 == 0"
enable_obstacles=true
echo -e "\n\nenable obstacles: $enable_obstacles\n\n"

#let "load_specific_pipeline = true"
load_specific_pipeline=false
echo -e "\n\nloading specific pipeline: $load_specific_pipeline\n\n"

# random_val=27168
random_val=50
echo -e "\n\nrandom seed: $random_val\n\n"

fls_from_file=false

roslaunch vandy_bluerov start_bluerov_simulation.launch \
    record:=true \
    headless:=true gui:=false \
    bag_filename:="/mnt/results/$task_folder/recording.bag" \
    unpause_timeout:=15 \
    random_seed:=$random_val \
    enable_obstacles:=false \
    enable_debris:=false \
    box_size_x:=10 \
    box_size_y:=10 \
    box_size_z:=10 \
    box_max_cnt:=2 \
    results_directory:="/mnt/results/$task_folder/" \
    failsafe_rth_enable:=false \
    geofence_threshold:=450 \
    waypoint_filename:="search_pattern_cp4.yaml" \
    pipe_scale:=3 \
    sss_sonar_noise:=0.5 \
    fis_sonar_noise:=25 \
    use_rtreach:=false \
    ddlec_am_params:="{}"\
    generate_fdr:=false &
#    gz_debug:=true &

child=$!
wait "$child"
if [ $exit_status -eq 130 ]; then
    # SIGINT was captured meaning the user
    # wants full stop instead of start_simulation.launch
    # terminating normally from end of episode so...
    echo "stop looping"
    break
fi

done
