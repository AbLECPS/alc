# This contains deployment dictionary  that is passed to the execution_runner
#

dep_dict = { "name": "sl_training",
  "ros_master_image": None,
  "base_dir": "ver_example_job/example_timestamp",
  "results_dir": ".",
  "timeout": 7200,
  "containers": [
    { "name": "sl_training",
      "image": "alc_alc:latest",
      "command": "${ACTIVITY_HOME}/training_runner.sh",
      "input_file": "launch_activity_output.json",
      "options": {
          "hostname": "sl_training",
          "runtime": "nvidia"
          }
      }
  ]
}
