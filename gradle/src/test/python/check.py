from HistoryData import workflow_data, parameter_output


incorrect_usage_code = 1


JOB_NAME_1 = "DataGeneration_RB_Obstacle"
ACTIVITY_NAME_1 = "DataGen_RB_Obstacle"

JOB_NAME_2 = "DataGeneration_LB_Obstacle"
ACTIVITY_NAME_2 = "DataGen_LB_Obstacle"

PARAMETER_NAME = "pipe_posx"

def main():

    iteration_count = workflow_data.get_job_iteration()

    old_pipe_posx = -1

    job_data = workflow_data.get_job_data(JOB_NAME_1)
    if job_data is not None:

        activity_data = job_data.get_activity_data(ACTIVITY_NAME_1)
        if activity_data is not None:

            old_pipe_posx = activity_data.get_parameter_value(PARAMETER_NAME) or -1

    new_pipe_posx = (iteration_count + 1) * 2 + 30 if old_pipe_posx == -1 else old_pipe_posx + 2

    if new_pipe_posx <= 35:
        parameter_output.for_job(JOB_NAME_1).for_activity(ACTIVITY_NAME_1).set_parameter(PARAMETER_NAME, new_pipe_posx)
        parameter_output.for_job(JOB_NAME_2).for_activity(ACTIVITY_NAME_2).set_parameter(PARAMETER_NAME, new_pipe_posx)
        parameter_output.for_job(JOB_NAME_1).set_parameter("pipe_posy", 10)

if __name__ == "__main__":
    main()
