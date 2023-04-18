from . import config_util
import copy

# Parameters with names matching that used by Slurm API job info
JOB_NAME = "slurm_job"
SCRIPT = "run_job.slurm"
OUTPUT = "slurm_job_log.txt"
TIME = 720
PARTITION = "primary"
NODES = 1
NTASKS = 1
CPUS_PER_TASK = 4
#GRES = "gpu:1"

# Add parameter names to create default Slurm job info dict
DEFAULT_JOB_INFO = config_util.build_var_dict(globals(), convert_case="lower")

# Specific job-type variants of default job info
EXPERIMENT_DEFAULT_JOB_INFO = copy.copy(DEFAULT_JOB_INFO)
#EXPERIMENT_DEFAULT_JOB_INFO.update({"gres": "gpu:1"})

# FIXME: Apparently this is not needed. Confirm
# Handle special cases to exactly match Slurm format (eg. Python doesn't allow hyphen in variable name)
# DEFAULT_JOB_INFO["cpus-per-task"] = DEFAULT_JOB_INFO.pop("cpus_per_task")

# Additional custom parameters
UPDATE_MODEL_SCRIPT = "update_model.sh"
EXECUTE_JOB_SCRIPT = "run.sh"
JOB_OPTIONS_FILE = "slurm_job_opts.json"
# ALC_GROUP_ID = 10181
