# This contains deployment dictionary  that is passed to the execution_runner
#

dep_dict = {"name": "nnv_verification_execution",
            "ros_master_image": None,
            "base_dir": "ver_example_job/example_timestamp",
            "results_dir": ".",
            "timeout": 1000,
            "containers": [
                {"name": "ver_nnv_matlab",
                 "image": "alc_jupyter_matlab:latest",
                 "command": "$ALC_HOME/alc_utils/routines/Verification_Robustness/verification_runner.sh",
                 "input_file": "launch_activity_output.json",
                 "options": {
                     "hostname": "nnv_robustness",
                     "runtime": "nvidia",
                     "entrypoint": "/bin/bash",
                     "network": "host",
                     "volumes":  {"$MATLAB_ROOT": {"bind": "$MATLAB_ROOT", "mode": "rw"},
                                   "$MATLAB_ROOT": {"bind": "/usr/local/MATLAB/from-host", "mode": "rw"},
                                   "$MATLAB_SUPPORT_ROOT": {"bind": "$MATLAB_SUPPORT_ROOT", "mode": "rw"},
                                   "$ALC_HOME/matlab/matlab-logs": {"bind": "/var/log/matlab", "mode": "rw"},
                                   "$ALC_HOME/verivital": {"bind": "/verivital", "mode": "rw"}
                                  },
                     "environment":  {"$MATLAB_PATH": "$MATLAB_ROOT",
                                      "$MATLAB_ROOT": "/usr/local/MATLAB/from-host",
                                      "$MATLAB_SUPPORT_ROOT": "$MATLAB_SUPPORT_ROOT",
                                      "$MATLAB_LOGS": "/var/log/matlab",
                                      "$ALC_VERIVITAL_HOME": "/verivital"
                                      }
                 }
                 }
            ]
            }
