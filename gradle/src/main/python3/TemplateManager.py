from pathlib import Path
from jinja2 import Environment, FileSystemLoader

template_dir_path = Path(Path(__file__).absolute().parent.parent, "template")

file_loader = FileSystemLoader(str(template_dir_path))
environment = Environment(loader=file_loader)

add_activity_template = environment.get_template("add_activity.jinja2")
add_job_template = environment.get_template("add_job.jinja2")
add_parameter_updates_template = environment.get_template("add_parameter_updates.jinja2")
add_branch_point_template = environment.get_template("add_branch_point.jinja2")
configuration_template = environment.get_template("add_configuration.jinja2")
lambda_template = environment.get_template("lambda.jinja2")
named_argument_template = environment.get_template("named_argument.jinja2")
threaded_parameterized_loop_template = environment.get_template("threaded_parameterized_loop.jinja2")
root_template = environment.get_template("root_template.jinja2")
while_loop_template = environment.get_template("while_loop.jinja2")
