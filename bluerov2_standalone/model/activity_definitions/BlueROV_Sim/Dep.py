# This contains deployment dictionary for iver that is passed to the execution_runner
#

dep_dict = {
  "name": "bluerov_dep",
  "ros_master_image": "ros:kinetic-ros-core",
  "base_dir": "blue_example/example_timestamp",
  "results_dir": "results",
  "timeout": None,
  "containers": [
    {
      "name": "bluerov",
      "image": "alc:latest",
      "command": "${ACTIVITY_HOME}/bluerov_runner.sh",
      "input_file": "parameters.yml",
      "options": {
        "hostname": "bluerov",
        "runtime": "nvidia",
        "privileged": True,
        "volumes": {
          "$REPO_HOME/catkin_ws": {
            "bind": "/aa",
            "mode": "rw"
          }
        }
      }
    },
    {
      "name": "xpra",
      "image": "xpra_16.04:latest",
      "command": "${ACTIVITY_HOME}/xpra_runner.sh",
      "input_file": "parameters.yml",
      "options": {
        "hostname": "xpra",
        "runtime": "nvidia",
        "privileged": True,
        "ports":{10000:15001},
        "volumes": {
          "$REPO_HOME/catkin_ws": {
            "bind": "/aa",
            "mode": "rw"
          }
        }
      }
    }
  ]
}