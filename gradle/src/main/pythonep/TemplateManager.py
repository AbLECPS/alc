from pathlib import Path
from jinja2 import Environment, FileSystemLoader

template_dir_path = Path(Path(__file__).absolute().parent.parent, "templateep")

file_loader = FileSystemLoader(str(template_dir_path))
environment = Environment(loader=file_loader)

add_activity_template = environment.get_template("add_activity.jinja2")
add_branch_point_template = environment.get_template("add_branch_point.jinja2")
add_branch_point_trivial_template = environment.get_template("add_branch_point_trivial.jinja2")
add_data_store_job_template = environment.get_template("add_data_store_job.jinja2")
add_configuration_template = environment.get_template("add_configuration.jinja2")
add_input_template = environment.get_template("add_input.jinja2")
add_launch_activity_job_template = environment.get_template("add_launch_activity_job.jinja2")
add_launch_activity_job_trivial_template = environment.get_template("add_launch_activity_job_trivial.jinja2")
add_launch_experiment_job_template = environment.get_template("add_launch_experiment_job.jinja2")
add_parameter_filter_template = environment.get_template("add_parameter_filter.jinja2")
add_parameter_updates_template = environment.get_template("add_parameter_updates.jinja2")
add_transform_template = environment.get_template("add_transform.jinja2")
add_transform_trivial_template = environment.get_template("add_transform_trivial.jinja2")
lambda_template = environment.get_template("lambda.jinja2")
module_imports_template = environment.get_template("function_modules.jinja2")
named_argument_template = environment.get_template("named_argument.jinja2")
threaded_parameterized_loop_template = environment.get_template("threaded_parameterized_loop.jinja2")
root_template = environment.get_template("root_template.jinja2")
while_loop_template = environment.get_template("while_loop.jinja2")
