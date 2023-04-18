from pathlib import Path


class CompletionIndicator:

    _complete_job_file_name = "__COMPLETE__"

    def __init__(self, directory):
        self.complete_file_path = Path(directory, self._complete_job_file_name)

    def test_complete(self):
        return self.complete_file_path.exists()

    def set_complete(self):
        self.complete_file_path.touch()
