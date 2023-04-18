import itertools
import json
import os
import matlab.engine
import imageio  # MUST APPEAR *AFTER* "import matlab.engine" or nasty error messages will occur
import nbformat
import numpy
from pathlib import Path
import re
import subprocess
import RobustnessKeys
from SourceReplacePreprocessor import SourceReplacePreprocessor
from TagRunPreprocessor import TagRunPreprocessor
import TemplateManager


class ComputeRobustness:

    training_data_directory_name = "TrainingData"

    stats_file_name = "stats.json"

    sum_key = "sum"
    sum_squared_key = "sum_squared"
    total_pixels_key = "total_pixels"

    results_json_file_name = "results.json"

    alc_home_env_var_name = "ALC_HOME"

    def __init__(self, template_parameters_file):
        self.template_parameters_file_path = Path(template_parameters_file)

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

        return_mean = ComputeRobustness.check_value(num_channels, mean)
        if return_mean is None:
            raise RuntimeError("Please specify a valid mean")
        return_mean = mean

        return_std = ComputeRobustness.check_value(num_channels, std)
        if return_std is None:
            raise RuntimeError("Please specify a valid mean")

        return return_mean, return_std

    def compute_and_create_notebook(self):

        with self.template_parameters_file_path.open("r") as template_parameters_fp:
            template_parameter_map = json.load(template_parameters_fp)

        #
        # GET DATASET SCRIPT FOR EXTRACTING TRAINING/TESTING DATA IMAGES, CATEGORY NAMES, CATEGORY VALUES
        #
        lec_dataset_script_text = template_parameter_map[RobustnessKeys.template_dataset_key]

        match = re.search(r"class\s*(\w+)", lec_dataset_script_text)
        class_name = match.group(1)
        exec(lec_dataset_script_text, globals())
        clazz = globals().get(class_name)

        #
        # GET CATEGORY DICT FROM TRAINING DATA
        #

        lec_directory = template_parameter_map[RobustnessKeys.template_lec_parent_directory_path_key]
        training_data_directory_path = Path(lec_directory, ComputeRobustness.training_data_directory_name)
        category_dict = clazz(training_data_directory_path).get_category_dict()

        #
        # GET STATS FOR CALCULATING MEAN, STD
        #
        stats_file_path = Path(training_data_directory_path, ComputeRobustness.stats_file_name)
        if stats_file_path.is_file():
            with stats_file_path.open("r") as stats_fp:
                stats_json = json.load(stats_fp)
            sum_intensity = stats_json[ComputeRobustness.sum_key]
            if isinstance(sum_intensity, list):
                sum_intensity = numpy.asarray(sum_intensity)
            sum_intensity_squared = stats_json[ComputeRobustness.sum_squared_key]
            if isinstance(sum_intensity_squared, list):
                sum_intensity_squared = numpy.asarray(sum_intensity_squared)
            total_pixels = stats_json[ComputeRobustness.total_pixels_key]
        else:
            png_chain = itertools.chain(
                # *png_iteratable_list,
                training_data_directory_path.glob("**/*.png")
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

            stats_json = {
                ComputeRobustness.sum_key: sum_intensity.tolist(),
                ComputeRobustness.sum_squared_key: sum_intensity_squared.tolist(),
                ComputeRobustness.total_pixels_key: total_pixels
            }
            with stats_file_path.open("w") as stats_fp:
                json.dump(stats_json, stats_fp, indent=4, sort_keys=True)

        #
        # CALCULATE MEAN, STD
        #
        mean = sum_intensity / total_pixels
        variance = sum_intensity_squared / total_pixels - numpy.square(mean)
        std = numpy.sqrt(variance)

        mean_list = mean.tolist()
        std_list = std.tolist()

        template_parameter_map[RobustnessKeys.template_mean_key] = mean_list
        template_parameter_map[RobustnessKeys.template_standard_deviation_key] = std_list

        # START MATLAB ENGINE
        eng = matlab.engine.start_matlab()

        # ADD PATHS OF NEEDED FUNCTIONS TO MATLAB ENVIRONMENT
        matlab_function_path_list = []

        local_matlab_function_dir = str(Path(Path(__file__).absolute().parent, "matlab"))
        matlab_function_path_list.append(local_matlab_function_dir)

        alc_home_path = Path(os.environ[ComputeRobustness.alc_home_env_var_name])

        nnv_code_dir = str(Path(alc_home_path, "verivital", "nnv", "code", "nnv"))
        matlab_function_path_list.append(nnv_code_dir)

        nnv_engine_util_dir = str(Path(nnv_code_dir, "engine", "utils"))
        matlab_function_path_list.append(nnv_engine_util_dir)

        #
        # EXECUTE MATLAB ENGINE
        #
        eng.addpath(*matlab_function_path_list)
        eng.cd(nnv_code_dir)
        eng.startup_nnv(nargout=0)

        # LOAD NETWORK
        mat_file_name = template_parameter_map[RobustnessKeys.template_lec_file_name_key]
        network_directory = template_parameter_map[RobustnessKeys.template_lec_directory_path_key]
        network_directory_path = Path(network_directory)
        mat_file = Path(network_directory_path, mat_file_name)

        net = eng.load(str(mat_file))
        net = eng.struct2cell(net)[0]
        # eng.workspace['net'] = net
        layers = eng.getfield(net, 'Layers')
        output_size = len(eng.getfield(layers[-1], 'Bias'))
        eng.setfield(net, 'OutputSize', output_size)

        # GET NUMBER OF CORES
        c = eng.parcluster('local')
        num_cores = eng.get(c).get('NumWorkers')  # specify number of cores used for verification

        parameter_map = template_parameter_map[RobustnessKeys.template_parameter_map_key]

        reach_method = parameter_map[RobustnessKeys.method_parameter_key]

        #
        # GET NAMES OF ALL REQUIRED PARAMETERS
        #
        attack_type = parameter_map[RobustnessKeys.attack_parameter_key]
        extra_parameter_set = RobustnessKeys.attack_map \
            .get(attack_type) \
            .get(RobustnessKeys.required_parameters_key)

        perturbation_function_name = RobustnessKeys.attack_map\
            .get(attack_type) \
            .get(RobustnessKeys.perturbation_function_name_key)
        perturbation_function = getattr(eng, perturbation_function_name)

        structure_data_map = {name: parameter_map[name] for name in extra_parameter_set}
        per_image_function = perturbation_function(structure_data_map)

        #
        # CREATE TEST-DATA CSV FILE
        #
        test_data_directory_list = template_parameter_map[RobustnessKeys.template_test_data_directory_list_key]
        success_table = []
        test_data_object = clazz(test_data_directory_list, category_map=category_dict)
        for image_file, category_name, category_number in test_data_object:
            im_target = matlab.double([category_number])

            image = eng.imread(str(image_file))
            image = matlab.double(image)

            mod_mean, mod_std = ComputeRobustness.check_mean_std(eng, image, mean_list, std_list)
            mean_for_image = matlab.double(mod_mean)
            std_for_image = matlab.double(mod_std)

            input_set_star = eng.getImageStar(image, per_image_function, mean_for_image, std_for_image)

            # net.verifyRobustness(input_net_star, im_target, reach_method, num_cores)
            r_nn = eng.verifyRobustness(net, input_set_star, im_target, reach_method, num_cores)

            success_table.append({
                RobustnessKeys.image_path_key: str(image_file),
                RobustnessKeys.category_name_key: category_name,
                RobustnessKeys.category_number_key: category_number,
                RobustnessKeys.result_key: r_nn
            })

        eng.exit()

        template_parameter_map[RobustnessKeys.template_results_file_name_key] = \
            ComputeRobustness.results_json_file_name

        template = TemplateManager.image_perturbation_python_notebook_template

        notebook_string = template.render(**template_parameter_map)

        specific_notebook_directory_path = Path(
            template_parameter_map[RobustnessKeys.template_specific_notebook_directory_key]
        )
        specific_notebook_directory_path.mkdir(parents=True, exist_ok=True)

        notebook_path = Path(specific_notebook_directory_path, "robustness.ipynb")

        results_json_file_path = Path(specific_notebook_directory_path, ComputeRobustness.results_json_file_name)
        with results_json_file_path.open("w") as results_fp:
            json.dump(success_table, results_fp, indent=4, sort_keys=True)

        tag_run_preprocessor = TagRunPreprocessor(timeout=1800, kernel_name='python3')
        tag_run_preprocessor.execute_cell_tags = ("execute_cell",)

        source_replace_preprocessor = SourceReplacePreprocessor()
        source_replace_preprocessor.source_replace_tags = ("source_replace",)
        source_replace_preprocessor.regex_substitution_list = [
            ("AUTORUN ALL CELLS", "# Cells below should be executed automatically", True)
        ]

        nb = nbformat.reads(notebook_string, as_version=4)

        try:
            (nb, resources) = tag_run_preprocessor.preprocess(
                nb, {'metadata': {'path': str(notebook_path.parent.absolute())}}
            )
            source_replace_preprocessor.preprocess(nb, {})
        except:
            pass

        with notebook_path.open("w", encoding='utf-8') as notebook_fp:
            nbformat.write(nb, notebook_fp)

        subprocess.run(["jupyter", "trust", str(notebook_path)])
        notebook_path.chmod(0o444)

        #
        # WRITE PATH TO NOTEBOOK HERE
        #


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Specify json file from which to read program parameters")
    parser.add_argument("json_file")
    args = parser.parse_args()
    json_file = args.json_file
    compute_robustness = ComputeRobustness(json_file)
    compute_robustness.compute_and_create_notebook()
