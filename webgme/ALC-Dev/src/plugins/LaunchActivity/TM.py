from pathlib import Path
from jinja2 import Environment, FileSystemLoader

template_dir_path = Path(Path(__file__).absolute().parent, "LaunchActivity", "templates")

file_loader = FileSystemLoader(str(template_dir_path))
environment = Environment(loader=file_loader)

#image_perturbation_python_notebook_template = environment.get_template("image_perturbation_python.jinja2.ipynb")
python_main_template = environment.get_template("python_main.jinja2.py")
run_script_template = environment.get_template("run_script.jinja2.sh")
update_script_template = environment.get_template("update_script.jinja2.sh")
workflow_json_template = environment.get_template("workflow_camp.jinja2")
