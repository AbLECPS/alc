from pathlib import Path
from BaseTask import BaseTask
import subprocess


class ExecTask(BaseTask):

    def __init__(self):
        BaseTask.__init__(self)
        self.process = None

    def execute(self):

        close_stdin, standard_input = (True, self.standard_input.open("r"))\
            if isinstance(self.standard_input, Path) else (False, self.standard_input)

        close_stdout, standard_output = (True, self.standard_output.open("w"))\
            if isinstance(self.standard_output, Path) else (False, self.standard_output)

        close_stderr, standard_error = (True, self.standard_error.open("w"))\
            if isinstance(self.standard_error, Path) else (False, self.standard_error)

        self.process = subprocess.run(
            self.command,
            cwd=str(self.working_dir),
            stdin=standard_input,
            stdout=standard_output,
            stderr=standard_error
        )

        close_stdin and standard_input.close()
        close_stdout and standard_output.close()
        close_stderr and standard_error.close()

        return self

    @staticmethod
    def get_read_file_pointer(file):
        if isinstance(file, Path):
            return file.open("r")
        elif isinstance(file, str):
            return Path(file).open("r")

        return file

    @staticmethod
    def get_write_file_pointer(file):
        if isinstance(file, Path):
            return file.open("w")
        elif isinstance(file, str):
            return Path(file).open("w")

        return file

    def get_exit_status(self):
        return self.process.returncode