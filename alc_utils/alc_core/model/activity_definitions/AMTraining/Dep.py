# This contains deployment dictionary that is passed to the execution_runner
#

dep_dict = { "name": "am_training",
  "ros_master_image": None,
  "base_dir": "",
  "results_dir": ".",
  "timeout": 7200,
  "containers": [
    { "name": "am_training",
      "image": "alc_alc:latest",
      "command": "$ACTIVITY_HOME/training_runner.sh",
      "input_file": "launch_activity_output.json",
      "options": {
          "hostname": "am_training",
          "runtime": "nvidia"
          }
      }
  ]
}
