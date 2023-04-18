import json
from pathlib import Path
from ALCModelUpdaterTask import ALCModelUpdaterTask


class ConfigurationTask(ALCModelUpdaterTask):

    def _get_configuration_file_contents(self):
        return {
            "1": {
                "dst": self.destination_list,
                "dst_lec": self.destination_lec,
                "name": self.operation,
                "src": self.source_list
            }
        }

    def __init__(self, directory, job_path, unique_number_generator):

        task_name = "configuration-{0}".format(unique_number_generator.get_unique_number())
        configuration_directory = Path(directory, task_name)
        ALCModelUpdaterTask.__init__(self, configuration_directory, task_name, job_path)

        self.operation = None
        self.destination_list = None
        self.destination_lec = None
        self.source_list = None

    def set_operation(self, operation):

        self.operation = operation
        return self

    def get_operation(self):
        return self.operation

    def _check_operation(self):
        if not isinstance(self.operation, str):
            raise Exception(
                "ERROR:  ConfigurationTask (directory \"{0}\"):  operation not set".format(self.directory)
            )

    def set_source_list(self, source_list):

        self.source_list = source_list
        return self

    def get_source_list(self):
        return self.source_list

    def _check_source_list(self):
        if not isinstance(self.source_list, list):
            raise Exception(
                "ERROR:  ConfigurationTask (directory \"{0}\"):  source node list not set".format(self.directory)
            )

    def set_destination_list(self, destination_list):

        self.destination_list = destination_list
        return self

    def get_destination_list(self):
        return self.destination_list

    def _check_destination_list(self):
        if not isinstance(self.destination_list, list):
            raise Exception(
                "ERROR:  ConfigurationTask (directory \"{0}\"):  destination node list not set".format(self.directory)
            )

    def set_destination_lec(self, destination_lec):

        self.destination_lec = destination_lec
        return self

    def get_destination_lec(self):
        return self.destination_lec

    def _check_destination_lec(self):
        if not isinstance(self.destination_lec, str):
            raise Exception(
                "ERROR:  ConfigurationTask (directory \"{0}\"):  destination lec not set".format(self.directory)
            )

    def execute(self):

        if not self.is_complete():
            self._check_operation()
            self._check_destination_list()
            self._check_destination_lec()
            self._check_source_list()

            with self.input_file.open("w") as input_fp:
                json.dump(self._get_configuration_file_contents(), input_fp, indent=4, sort_keys=True)
        else:
            self.set_not_complete()  # OK TO REPEAT CONFIGURATION

        ALCModelUpdaterTask.execute(self)

        return self
