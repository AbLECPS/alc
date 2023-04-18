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
 

roslaunch vandy_bluerov start_bluerov_simulation.launch \
    record:=true \
    headless:=true gui:=false \
    bag_filename:="/mnt/results/$task_folder/recording.bag" \
    unpause_timeout:=15 \
    timeout:=800 \
    geofence_threshold:=450 \
    batt_charge:=9999 \
    failsafe_battery_low_threshold:=0.15 \
    random_seed:=27168 \
    results_directory:="/mnt/results/$task_folder/" \
    mission_file:="mission_04.yaml" \
    enable_fault_detection:=true \
    thruster_motor_failure:=false \
    enable_obstacles:=true \
    box_max_cnt:=2 \
    box_distance_static:=65 \
    lambda_low_static:=60 \
    lambda_high_static:=60 \
    enable_emergency_stop:=false \
    parameter_uncertainty:=0.05 \
    uuv_sim_time:=20 \
    box_sim_time:=3 \
    x_unc:=0.01    \
    y_unc:=0.01    \
    use_rtreach:=false &


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
