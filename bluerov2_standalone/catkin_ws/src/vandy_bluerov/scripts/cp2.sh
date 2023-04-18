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
 
#let "enable_disturbances = 0 % 3 == 0"
enable_disturbances=false
echo -e "\n\nenable disturbances: $enable_disturbances\n\n"

#let "load_specific_pipeline = true"
load_specific_pipeline=false
echo -e "\n\nloading specific pipeline: $load_specific_pipeline\n\n"

random_val=27168
echo -e "\n\nrandom seed: $random_val\n\n"

fls_from_file=false

roslaunch vandy_bluerov start_bluerov_simulation.launch \
    record:=true bag_filename:=bluerov2_recording.bag \
    headless:=true gui:=false \
    bag_filename:="/mnt/results/$task_folder/recording.bag" \
    unpause_timeout:=15 \
    timeout:=800 \
    random_seed:=$random_val \
    enable_obstacles:=$enable_obstacles \
    results_directory:="/mnt/results/$task_folder/" \
    enable_fault_detection:=false \
    mission_file:="mission_04.yaml" \
    thruster_motor_failure:=true \
    thruster_thrust_force_efficiency:=0.59 \
    decision_source:="combination_am" \
    behaviour_tree:=true  &
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
