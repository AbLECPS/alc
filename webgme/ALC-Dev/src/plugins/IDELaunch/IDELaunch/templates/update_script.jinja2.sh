#!/bin/bash

result_dir="$(pwd)"
UPDATE_SCRIPT_LOG=$result_dir/update_script_log.txt

# Useful functions
update_webgme_internal() {
  pushd /alc/webgme
  echo "Updating WebGME model..." >> $UPDATE_SCRIPT_LOG
  node ./node_modules/webgme-engine/src/bin/run_plugin.js ALCModelUpdater {{ project_name|string }} \
    -o {{ project_owner|string }} \
    -a /y \
    -u alc:alc \
    -l http://localhost:8888 \
    -j $result_dir \
    -n ALCMeta \
    >> $UPDATE_SCRIPT_LOG 2>&1
  local exit_status=$?
  echo "Done updating WebGME." >> $UPDATE_SCRIPT_LOG
  popd
  return $exit_status
}

# Useful functions
update_webgme() {
  local exit_status
  local count=0
  while [ $count -lt 100 ]; do
    update_webgme_internal
    exit_status=$?
    if [ $exit_status -eq 0 ]; then
      break
    fi
    count=$((count + 1))
    sleep 3
  done
}

get_job_status() {
  # Get job status and strip any whitespace
  # First try to get status from squeue. If that doesn't return a valid status, try sacct.
  local JOB_STATUS="$(squeue -j $JOB_ID -o %T -h | tr -d '[:space:]')"
  if [ -z "$JOB_STATUS" ]; then
    JOB_STATUS="$(sacct -j $JOB_ID -X -o STATE -n | tr -d '[:space:]')"
  fi

  if [ -z "$JOB_STATUS" ]; then
    # If both methods returned an empty status, exit
    echo "ERROR: Got empty status information. Exiting." >&2
    update_webgme
    exit 3
  fi
  echo "$JOB_STATUS"
}




# Make sure we got Job ID argument
if [ -z $1 ]; then
  echo "ERROR: Update model script did not receive a valid JOB_ID argument. Got ${1}." >> $UPDATE_SCRIPT_LOG
  echo "Finished_w_Errors" > $result_dir/slurm_exec_status.txt
  update_webgme
  exit 1
fi

# Wait for job to be queued. Should get a valid result from squeue.
JOB_ID=$1
echo "Waiting for job (id: $JOB_ID) to initialize..." >> $UPDATE_SCRIPT_LOG
sleep 1
until squeue -j $JOB_ID; do
  echo "Waiting for job to be queued...">> $UPDATE_SCRIPT_LOG
  sleep 1
done

# FIXME: Not sure why Slurm is re-using job ids. Maximum ID is supposed to default to 0x03ff0000 before rollover
# Brief wait after job has been queued to ensure Slurm has cleared any old job results at the same job ID
sleep 2
JOB_STATUS="$(get_job_status)"
echo "Job added to queue and initialized in state $JOB_STATUS." >> $UPDATE_SCRIPT_LOG

# FIXME: This may be redundant
JOB_STATUS="$(get_job_status)"
while [[ -z "$JOB_STATUS" ]]; do
  echo "Waiting for job to be valid status...">> $UPDATE_SCRIPT_LOG
  JOB_STATUS="$(get_job_status)"
  sleep 1
done

if [[ "$JOB_STATUS" == "PENDING" ]]; then
  echo "Job now in state \"PENDING\"." >> $UPDATE_SCRIPT_LOG
  # Wait for job to exit "PENDING" state
  while [[ "$JOB_STATUS" == "PENDING" ]]; do
    JOB_STATUS="$(get_job_status)"
    sleep 1
  done
  echo "Job has exited the \"PENDING\" state." >> $UPDATE_SCRIPT_LOG
fi

# Notify WebGME that job has started
if [[ "$JOB_STATUS" == "RUNNING" ]]; then
  echo "Job now in state \"RUNNING\"." >> $UPDATE_SCRIPT_LOG
  echo "Started" > $result_dir/slurm_exec_status.txt
  update_webgme
  # Wait for job to exit "RUNNING" state
  while [[ "$JOB_STATUS" == "RUNNING" ]]; do
    JOB_STATUS="$(get_job_status)"
    sleep 1
  done
  echo "Job has exited \"RUNNING\" state." >> $UPDATE_SCRIPT_LOG
fi

# Wait for job to exit "COMPLETING" state
if [[ "$JOB_STATUS" == "COMPLETING" ]]; then
  echo "Job now in state \"COMPLETING\"." >> $UPDATE_SCRIPT_LOG
  while [[ "$JOB_STATUS" == "COMPLETING" ]]; do
    JOB_STATUS="$(get_job_status)"
    sleep 1
  done
  echo "Job has exited \"COMPLETING\" state." >> $UPDATE_SCRIPT_LOG
fi

# Check if job completed successfully and notify WebGME
if [[ "$JOB_STATUS" == "COMPLETED" ]]; then
  echo "Job now in state \"COMPLETED\"." >> $UPDATE_SCRIPT_LOG
  echo "Finished" > $result_dir/slurm_exec_status.txt
else
  echo "Job now in state \"$JOB_STATUS\":  ERROR" >> $UPDATE_SCRIPT_LOG
  echo "Finished_w_Errors" > $result_dir/slurm_exec_status.txt
fi
update_webgme
