import sys
from pathlib import Path
import subprocess


class ExecTask():

    def __init__(self):
        self.command = []
        self.working_dir = Path(".")
        self.standard_input = sys.stdin
        self.standard_output = sys.stdout
        self.standard_error = sys.stderr
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

    def set_command(self, list_of_args):
        self.command = list_of_args

    def set_working_dir(self, working_dir):
        if isinstance(working_dir, Path):
            self.working_dir = working_dir
        else:
            self.working_dir = Path(working_dir)
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

    def set_standard_input(self, standard_input):
        self.standard_input = standard_input
        return self

    def set_standard_output(self, standard_output):
        self.standard_output = standard_output
        return self

    def set_standard_error(self, standard_error):
        self.standard_error = standard_error
        return self

    def get_exit_status(self):
        return self.process.returncode