from pathlib import Path


class BaseTask:

    def __init__(
            self, working_dir=Path("."), command=None, standard_input=None, standard_output=None, standard_error=None
    ):
        if command is None:
            command=[]

        self.working_dir = None
        self.set_working_dir(working_dir)

        self.command = None
        self.set_command(command)

        self.standard_input = None
        self.set_standard_input(standard_input)

        self.standard_output = None
        self.set_standard_output(standard_output)

        self.standard_error = None
        self.set_standard_error(standard_error)

    def execute(self):
        raise Exception("not implemented")

    def set_working_dir(self, working_dir):
        if isinstance(working_dir, Path):
            self.working_dir = working_dir
        else:
            self.working_dir = Path(working_dir)
        return self

    def set_command(self, list_of_args):
        self.command = list_of_args

    def set_standard_input(self, standard_input):
        self.standard_input = Path(standard_input) if isinstance(standard_input, str) else standard_input
        return self

    def set_standard_output(self, standard_output):
        self.standard_output = Path(standard_output) if isinstance(standard_output, str) else standard_output
        return self

    def set_standard_error(self, standard_error):
        self.standard_error = Path(standard_error) if isinstance(standard_error, str) else standard_error
        return self

    def get_exit_status(self):
        return 0
