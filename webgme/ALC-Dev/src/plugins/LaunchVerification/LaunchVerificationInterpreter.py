from pathlib import Path
import os
import re
import zipfile
import itertools
import imageio
import numpy
import matlab.engine
import nbformat
import subprocess
import traceback
import TemplateManager
from KeysAndAttributes import Keys, Attributes
from ActivityInterpreterBase import ActivityInterpreterBase
from TagRunPreprocessor import TagRunPreprocessor
from SourceReplacePreprocessor import SourceReplacePreprocessor

alc_port = 8000


class LaunchVerificationInterpreter(ActivityInterpreterBase):

    alc_home_env_var_name = "ALC_HOME"

    training_data_dir_name = "TrainingData"
    verification_data_dir_name = "VerificationData"
    notebook_dir_name = "Notebook"
    results_file_name = "results.json"
    robustness_notebook_file_name = "robustness.ipynb"

    attack_key_name = "attack"
    data_set_key_name = "DataSet"
    name_key_name = "name"
    method_key_name = "method"
    parameters_key_name = "parameters"
    params_table_key_name = "ParamsTable"

    input_lec_key_name = "inputLEC"
    input_training_data_key_name = "input_training_data"
    input_verification_data_key_name = "input_verification_data"

    brightening_attack_type_name = "brightening"
    darkening_attack_type_name = "darkening"
    random_noise_attack_type_name = "random_noise"

    image_path_key = "image_path"
    category_name_key = "category_name"
    category_number_key = "category_number"
    result_key = "result"

    # TEMPLATE KEYS
    active_node_named_path_template_key_name = "active_node_named_path"
    active_node_path_template_key_name = "active_node_path"
    mean_template_key_name = "mean"
    owner_name_template_key_name = "owner_name"
    project_name_template_key_name = "project_name"
    results_file_name_template_key_name = "results_file_name"
    standard_deviation_template_key_name = "standard_deviation"

    input_lec_file_name_template_key_name = "input_lec_file_name"
    input_lec_attribute_name_template_key_name = "input_lec_attribute_name"
    input_lec_node_named_path_template_key_name = "input_lec_node_named_path"
    input_lec_node_path_template_key_name = "input_lec_node_path"

    input_training_data_file_name_template_key_name = "input_training_data_file_name"
    input_training_data_attribute_name_template_key_name = "input_training_data_attribute_name"
    input_training_data_node_named_path_template_key_name = "input_training_data_node_named_path"
    input_training_data_node_path_template_key_name = "input_training_data_node_path"

    input_verification_data_file_name_template_key_name = "input_verification_data_file_name"
    input_verification_data_attribute_name_template_key_name = "input_verification_data_attribute_name"
    input_verification_data_node_named_path_template_key_name = "input_verification_data_node_named_path"
    input_verification_data_node_path_template_key_name = "input_verification_data_node_path"

    parameter_map_template_key_name = "parameter_map"

    attack_map = {
        brightening_attack_type_name: "perturbBrightening",
        darkening_attack_type_name: "perturbDarkening",
        random_noise_attack_type_name: "perturbRandomNoise"
    }

    def __init__(self, input_map):
        ActivityInterpreterBase.__init__(self, input_map)

        self.attributes = None
        self.current_choice = None

        self.lec_file_path = None
        self.lec_file_node_path = None
        self.lec_file_node_named_path = None

        self.input_training_data_zip_file_path = None
        self.input_training_data_node_path = None
        self.input_training_data_node_named_path = None
        self.input_training_dataset_class = None

        self.input_verification_data_zip_file_path = None
        self.input_verification_data_node_path = None
        self.input_verification_data_node_named_path = None
        self.input_verification_dataset_class = None

        self.method = None
        self.extra_parameter_map = None

        self.success_table = []

    @staticmethod
    def check_value(num_channels, value):
        retval = value

        if num_channels == 3:
            if isinstance(value, list):
                if len(value) == 1:
                    retval = value * 3
                elif len(value) != 3:
                    retval = None
            else:
                retval = [value] * 3

        return retval

    @staticmethod
    def check_mean_std(eng, image, mean, std):
        num_channels = eng.size(image, 3)

        return_mean = LaunchVerificationInterpreter.check_value(num_channels, mean)
        if return_mean is None:
            raise RuntimeError("Please specify a valid mean")
        return_mean = mean

        return_std = LaunchVerificationInterpreter.check_value(num_channels, std)
        if return_std is None:
            raise RuntimeError("Please specify a valid mean")

        return return_mean, return_std

    @staticmethod
    def get_dataset_class(dataset_class_string):
        match = re.search(r"class\s+(\w+)", dataset_class_string)
        class_name = match.group(1)
        new_class_name = class_name
        ix = 0
        while new_class_name in globals():
            new_class_name = class_name + str(ix)
            ix += 1
        dataset_class_string = re.sub(
            r"class\s+{0}".format(class_name), "class {0}".format(new_class_name), dataset_class_string, count=1
        )
        exec(dataset_class_string, globals())
        return globals().get(new_class_name)

    def setup(self):
        self.attributes = self.input_map[Keys.attributes_key_name]
        self.current_choice = self.attributes.get(Keys.current_choice_key_name)

        inputs = self.input_map[Keys.inputs_key_name]

        # GET INPUT LEC FILE INFO
        input_lec_map = inputs.get(self.input_lec_key_name)
        self.lec_file_path = Path(input_lec_map.get(Attributes.asset_attribute_name)).absolute()
        self.lec_file_node_path = input_lec_map.get(Keys.node_path_key_name)
        self.lec_file_node_named_path = input_lec_map.get(Keys.node_named_path_key_name)

        # GET TRAINING DATASET ZIP FILE INFO
        input_training_data_map = inputs.get(self.input_training_data_key_name)
        self.input_training_data_zip_file_path = Path(
            input_training_data_map.get(Attributes.asset_attribute_name)
        ).absolute()
        self.input_training_data_node_path = input_training_data_map.get(Keys.node_path_key_name)
        self.input_training_data_node_named_path = input_training_data_map.get(Keys.node_named_path_key_name)

        # GET "DataSet" CODE FOR TRAINING DATASET
        input_training_data_parameters = input_training_data_map.get(self.parameters_key_name)
        training_data_params_table = input_training_data_parameters.get(self.params_table_key_name)
        training_data_dataset_code = training_data_params_table.get(self.data_set_key_name)
        self.input_training_dataset_class = self.get_dataset_class(training_data_dataset_code)

        # GET TEST DATASET ZIP FILE INFO
        input_verification_data_map = inputs.get(self.input_verification_data_key_name)
        self.input_verification_data_zip_file_path = Path(
            input_verification_data_map.get(Attributes.asset_attribute_name)
        ).absolute()
        self.input_verification_data_node_path = input_verification_data_map.get(Keys.node_path_key_name)
        self.input_verification_data_node_named_path = input_verification_data_map.get(Keys.node_named_path_key_name)

        # GET "DataSet" CODE FOR VERIFICATION DATASET
        input_verification_data_parameters = input_verification_data_map.get(self.parameters_key_name)
        verification_data_params_table = input_verification_data_parameters.get(self.params_table_key_name)
        verification_data_dataset_code = verification_data_params_table.get(self.data_set_key_name)
        self.input_verification_dataset_class = self.get_dataset_class(verification_data_dataset_code)

        # GET METHOD
        parameters = self.input_map[self.parameters_key_name]
        misc_parameters = parameters.get(self.params_table_key_name)
        self.method = misc_parameters.get(self.method_key_name)

        self.extra_parameter_map = parameters.get(self.current_choice)

    def execute(self):

        project_name = self.attributes.get(Keys.project_name_key_name)
        project_owner = self.attributes.get(Keys.owner_key_name)
        active_node_path = self.attributes.get(Keys.node_path_key_name)
        active_node_named_path = self.attributes.get(Keys.node_named_path_key_name)
        temp_dir_path = self.attributes.get(Keys.temp_dir_key_name)

        training_data_dir_path = Path(temp_dir_path, self.training_data_dir_name).absolute()
        verification_data_dir_path = Path(temp_dir_path, self.verification_data_dir_name).absolute()

        #
        # UNZIP FILES
        #channel

        # TRAINING DATA
        training_data_dir_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(self.input_training_data_zip_file_path)) as training_data_zip:
            training_data_zip.extractall(str(training_data_dir_path))

        # VERIFICATION DATA
        verification_data_dir_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(self.input_verification_data_zip_file_path)) as verification_data_zip:
            verification_data_zip.extractall(str(verification_data_dir_path))

        # GET MEAN, STANDARD DEVIATION
        png_chain = itertools.chain(
            # *png_iterable_list,
            training_data_dir_path.glob("**/*.png")
        )

        intensity_initialized = False
        sum_intensity = None
        sum_intensity_squared = None
        total_pixels = 0
        for training_file in png_chain:
            image = imageio.imread(training_file)
            shape = image.shape
            if len(shape) == 2:
                image = numpy.expand_dims(image, axis=2)
                shape = image.shape
            total_pixels += shape[0] * shape[1]
            num_channels = shape[2]

            if not intensity_initialized:
                intensity_initialized = True
                sum_intensity = numpy.asarray([0] * num_channels)
                sum_intensity_squared = numpy.asarray([0] * num_channels)

            for channel in range(0, num_channels):
                norm_channel = image[:, :, channel] / 255.0
                sum_intensity[channel] += norm_channel.sum()
                sum_intensity_squared[channel] += numpy.square(norm_channel).sum()

        #
        # CALCULATE MEAN, STD
        #
        mean = sum_intensity / total_pixels
        variance = sum_intensity_squared / total_pixels - numpy.square(mean)
        std = numpy.sqrt(variance)

        mean_list = mean.tolist()
        std_list = std.tolist()

        category_dict = self.input_training_dataset_class(training_data_dir_path).get_category_dict()

        # START MATLAB ENGINE
        eng = matlab.engine.start_matlab()

        # ADD PATHS OF NEEDED FUNCTIONS TO MATLAB ENVIRONMENT
        matlab_function_path_list = []

        local_matlab_function_dir = str(Path(Path(__file__).absolute().parent, "LaunchVerification", "matlab"))
        matlab_function_path_list.append(local_matlab_function_dir)

        alc_home_path = Path(os.environ[self.alc_home_env_var_name])

        nnv_code_dir = str(Path(alc_home_path, "verivital", "nnv", "code", "nnv"))
        matlab_function_path_list.append(nnv_code_dir)

        nnv_engine_util_dir = str(Path(nnv_code_dir, "engine", "utils"))
        matlab_function_path_list.append(nnv_engine_util_dir)

        #
        # EXECUTE MATLAB ENGINE
        #
        eng.addpath(*matlab_function_path_list)
        eng.addpath(eng.genpath(nnv_code_dir))
        eng.cd(nnv_code_dir)
        eng.startup_nnv(nargout=0)

        # LOAD NETWORK
        net = eng.load(str(self.lec_file_path))
        net = eng.struct2cell(net)[0]
        # eng.workspace['net'] = net
        layers = eng.getfield(net, 'Layers')
        output_size = len(eng.getfield(layers[-1], 'Bias'))
        eng.setfield(net, 'OutputSize', output_size)

        # GET NUMBER OF CORES
        c = eng.parcluster('local')
        num_cores = eng.get(c).get('NumWorkers')  # specify number of cores used for verification

        #
        # GET NAMES OF ALL REQUIRED PARAMETERS
        #
        perturbation_function_name = self.attack_map.get(self.current_choice)
        perturbation_function = getattr(eng, perturbation_function_name)

        per_image_function = perturbation_function(self.extra_parameter_map)

        #
        # CREATE TEST-DATA SUCCESS TABLE
        #
        test_data_object = self.input_verification_dataset_class(
            [str(verification_data_dir_path)], category_map=category_dict
        )
        for image_file, category_name, category_number in test_data_object:
            im_target = matlab.double([category_number])

            image = eng.imread(str(image_file))
            image = matlab.double(image)

            mod_mean, mod_std = self.check_mean_std(eng, image, mean_list, std_list)
            mean_for_image = matlab.double(mod_mean)
            std_for_image = matlab.double(mod_std)

            input_set_star = eng.getImageStar(image, per_image_function, mean_for_image, std_for_image)

            # net.verifyRobustness(input_net_star, im_target, reach_method, num_cores)
            r_nn = eng.verifyRobustness(net, input_set_star, im_target, self.method, num_cores)

            self.success_table.append({
                self.image_path_key: str(image_file),
                self.category_name_key: category_name,
                self.category_number_key: category_number,
                self.result_key: r_nn
            })

        eng.exit()

        notebook_dir_path = Path(temp_dir_path, self.notebook_dir_name)
        if not notebook_dir_path.exists():
            notebook_dir_path.mkdir(parents=True)

        with Path(notebook_dir_path, self.results_file_name).open("w") as results_fp:
            json.dump(self.success_table, results_fp, indent=4, sort_keys=True)

        parameter_map = {
            self.method_key_name: self.method,
            self.attack_key_name: self.current_choice
        }
        parameter_map.update(self.extra_parameter_map)

        template_parameter_map = {
            self.active_node_named_path_template_key_name: active_node_named_path,
            self.active_node_path_template_key_name: active_node_path,
            self.owner_name_template_key_name: project_owner,
            self.project_name_template_key_name: project_name,
            self.mean_template_key_name: mean_list,
            self.standard_deviation_template_key_name: std_list,

            self.input_lec_file_name_template_key_name: str(self.lec_file_path.name),
            self.input_lec_attribute_name_template_key_name: Attributes.asset_attribute_name,
            self.input_lec_node_path_template_key_name: self.lec_file_node_path,
            self.input_lec_node_named_path_template_key_name: self.lec_file_node_named_path,

            self.input_training_data_file_name_template_key_name: str(self.input_training_data_zip_file_path.name),
            self.input_training_data_attribute_name_template_key_name: Attributes.asset_attribute_name,
            self.input_training_data_node_path_template_key_name: str(self.input_training_data_node_path),
            self.input_training_data_node_named_path_template_key_name: str(self.input_training_data_node_named_path),

            self.input_verification_data_file_name_template_key_name:
                str(self.input_verification_data_zip_file_path.name),
            self.input_verification_data_attribute_name_template_key_name: Attributes.asset_attribute_name,
            self.input_verification_data_node_path_template_key_name: str(self.input_verification_data_node_path),
            self.input_verification_data_node_named_path_template_key_name:
                str(self.input_verification_data_node_named_path),

            self.parameter_map_template_key_name: parameter_map,
            self.results_file_name_template_key_name: self.results_file_name
        }
        template = TemplateManager.image_perturbation_python_notebook_template

        notebook_string = template.render(**template_parameter_map)

        tag_run_preprocessor = TagRunPreprocessor(timeout=1800, kernel_name='python3')
        tag_run_preprocessor.execute_cell_tags = ("execute_cell",)

        source_replace_preprocessor = SourceReplacePreprocessor()
        source_replace_preprocessor.source_replace_tags = ("source_replace",)
        source_replace_preprocessor.regex_substitution_list = [
            ("AUTORUN ALL CELLS", "# Cells below should be executed automatically", True)
        ]

        nb = nbformat.reads(notebook_string, as_version=4)

        notebook_path = Path(notebook_dir_path, "robustness.ipynb")

        try:
            (nb, resources) = tag_run_preprocessor.preprocess(
                nb, {'metadata': {'path': str(notebook_path.parent.absolute())}}
            )
            source_replace_preprocessor.preprocess(nb, {})
        except Exception as e:
            foo = traceback.format_exc()
            print(foo)
            pass

        if notebook_path.exists():
            notebook_path.unlink()

        with notebook_path.open("w", encoding='utf-8') as notebook_fp:
            nbformat.write(nb, notebook_fp)

        subprocess.run(["jupyter", "trust", str(notebook_path)])
        notebook_path.chmod(0o444)

    def result(self):
        return self.success_table


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Specify json file from which to read program parameters")
    parser.add_argument("json_file")
    args = parser.parse_args()
    json_file = args.json_file
    with Path(json_file).open("r") as json_fp:
        input_json_map = json.load(json_fp)
    launch_verification_interpreter = LaunchVerificationInterpreter(input_json_map)
    launch_verification_interpreter.setup()
    launch_verification_interpreter.execute()
