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
 
#let "load_specific_pipeline = true"
load_specific_pipeline=false
echo -e "\n\nloading specific pipeline: $load_specific_pipeline\n\n"

random_val=27168
echo -e "\n\nrandom seed: $random_val\n\n"

fls_from_file=false

roslaunch vandy_bluerov start_bluerov_uuv.launch \
    mission_file:="mission_05_straight.yaml" \
    record:=true \
    headless:=true gui:=false \
    disturbance_filename:="/aa/src/ng/vandy_bluerov/config/disturbances.yaml" \
    unpause_timeout:=15 \
    random_seed:=$random_val \
    enable_obstacles:=true \
    enable_dynamic_obstacles:=false \
    box_max_cnt:=10 \
    enable_debris:=false \
    box_size_x:=5 \
    box_size_y:=5 \
    box_size_z:=15 \
    timeout:=300 \
    generate_fdr:=false \
    use_obstacle_avoidance:=true \
    fls_in_view_limit:=100 \
    results_directory:="/mnt/results/$task_folder/" \
    load_specific_pipeline:=$load_specific_pipeline \
    pipeline_file:="premade_pipelines/pipeline_input.txt" \
    fls_from_file:=$fls_from_file \
    enable_waypoint_following:=true \
    generate_ais_data:=false \
    enable_fault_detection:=false \
    thruster_motor_failure:=false \
    thruster_thrust_force_efficiency:=0.79 \
    thruster_motor_fail_starting_time:=1 \
    failsafe_rth_enable:=false \
    geofence_threshold:=500 \
    failsafe_battery_low_threshold:=-9999 \
    enable_training_data_collection:=true \
    behaviour_tree:=true \
    obstacle_avoidance_source:="fls_pencilbeam" &

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
