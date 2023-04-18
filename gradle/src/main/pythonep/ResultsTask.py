import json
from pathlib import Path
from ALCModelUpdaterTask import ALCModelUpdaterTask


class ResultsTask(ALCModelUpdaterTask):

    _result_output_command_string = "Result_Output"

    def _get_results_input_file_contents(self, get_results_task):

        return {
            self._result_output_command_string: {
                "name": self._result_output_command_string,
                "src": [
                    get_results_task.activity_node
                ],
                "dst": [],
                "dst_lec": "",
                "exec_name": "result-{0}".format(get_results_task.execution_name),
            },
            "Output_Name": str(self.output_file.resolve())
        }

    def __init__(self, directory, get_results_task, job_path):

        task_name = "results-{0}".format(get_results_task.task_name)
        results_directory = Path(directory, task_name)
        ALCModelUpdaterTask.__init__(self, results_directory, task_name, job_path)

        with self.input_file.open("w") as input_fp:
            json.dump(self._get_results_input_file_contents(get_results_task), input_fp, indent=4, sort_keys=True)

    def execute(self):

        is_complete = self.is_complete()

        ALCModelUpdaterTask.execute(self)

        if not is_complete:
            with self.output_file.open("r") as output_file_fp:
                output_json = json.load(output_file_fp)

            with self.output_file.open("w") as output_file_fp:
                json.dump(output_json, output_file_fp, indent=4, sort_keys=True)

        return self
