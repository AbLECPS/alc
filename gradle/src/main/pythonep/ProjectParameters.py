class ProjectParameters:
    _execution_dir_path = None
    _generic_active_node = "/y"
    _master_spec = "local[*]"
    _namespace = "ALCMeta"
    _owner = "alc"
    _project_name = None
    _status_node = None
    _script_path = None
    _stderr_file_path = None
    _stdout_file_path = None
    _failing_task_path_file_path = None
    _exceptions_file_path = None

    @staticmethod
    def set_master_spec(master_spec):
        ProjectParameters._master_spec = master_spec

    @staticmethod
    def get_master_spec():
        return ProjectParameters._master_spec

    @staticmethod
    def has_status_node():
        return ProjectParameters._status_node is not None and len(ProjectParameters._status_node) > 0

    @staticmethod
    def get_status_node():
        return ProjectParameters._status_node

    @staticmethod
    def set_status_node(status_node):
        ProjectParameters._status_node = status_node

    @staticmethod
    def get_project_name():
        return ProjectParameters._project_name

    @staticmethod
    def set_project_name(project_name):
        ProjectParameters._project_name = project_name

    @staticmethod
    def get_generic_active_node():
        return ProjectParameters._generic_active_node

    @staticmethod
    def set_generic_active_node(generic_active_node):
        ProjectParameters._generic_active_node = generic_active_node

    @staticmethod
    def get_execution_dir_path():
        return ProjectParameters._execution_dir_path

    @staticmethod
    def set_execution_dir_path(execution_dir_path):
        ProjectParameters._execution_dir_path = execution_dir_path

    @staticmethod
    def get_owner():
        return ProjectParameters._owner

    @staticmethod
    def set_owner(owner):
        ProjectParameters._owner = owner

    @staticmethod
    def get_project_id():
        return "{0}+{1}".format(ProjectParameters._owner, ProjectParameters._project_name)

    @staticmethod
    def get_namespace():
        return ProjectParameters._namespace

    @staticmethod
    def set_namespace(namespace):
        ProjectParameters._namespace = namespace

    @staticmethod
    def get_script_path():
        return ProjectParameters._script_path

    @staticmethod
    def set_script_path(script_path):
        ProjectParameters._script_path = script_path

    @staticmethod
    def get_stderr_file_path():
        return ProjectParameters._stderr_file_path

    @staticmethod
    def set_stderr_file_path(stderr_file_path):
        ProjectParameters._stderr_file_path = stderr_file_path

    @staticmethod
    def get_stdout_file_path():
        return ProjectParameters._stdout_file_path

    @staticmethod
    def set_stdout_file_path(stdout_file_path):
        ProjectParameters._stdout_file_path = stdout_file_path

    @staticmethod
    def get_failing_task_path_file_path():
        return ProjectParameters._failing_task_path_file_path

    @staticmethod
    def set_failing_task_path_file_path(failing_task_path_file_path):
        ProjectParameters._failing_task_path_file_path = failing_task_path_file_path

    @staticmethod
    def get_exceptions_file_path():
        return ProjectParameters._exceptions_file_path

    @staticmethod
    def set_exceptions_file_path(exceptions_file_path):
        ProjectParameters._exceptions_file_path = exceptions_file_path
