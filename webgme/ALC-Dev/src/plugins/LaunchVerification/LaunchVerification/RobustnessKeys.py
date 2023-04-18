template_dataset_key = "dataset"
template_eval_data_node_id_key = "eval_data_node_id"
template_eval_data_node_path_key = "eval_data_node_path"
template_lec_directory_path_key = "LEC_directory_path"
template_lec_parent_directory_path_key = "LEC_parent_directory_path"
template_lec_file_name_key = "LEC_file_name"
template_lec_node_id_key = "LEC_node_id"
template_lec_node_reference_id_key = "LEC_node_reference_id"
template_lec_node_reference_path_key = "LEC_node_reference_path"
template_lec_node_path_key = "LEC_node_path"
template_mean_key = "mean"
template_specific_notebook_directory_key = "specific_notebook_directory"
template_owner_name_key = "owner_name"
template_parameter_map_key = "parameter_map"
template_project_name_key = "project_name"
template_results_file_name_key = "results_file_name"
template_standard_deviation_key = "standard_deviation"
template_test_data_directory_list_key = "test_data_directory_list"
template_verification_node_id_key = "verification_node_id"
template_verification_node_path_key = "verification_node_path"

attack_parameter_key = "attack"
delta_parameter_key = "delta"
method_parameter_key = "method"
noise_parameter_key = "noise"
pixels_parameter_key = "pixels"
threshold_parameter_key = "threshold"

brightening_attack_type_name = "brightening"
darkening_attack_type_name = "darkening"
random_noise_attack_type_name = "random_noise"

required_parameters_key = "required_parameters"
perturbation_function_name_key = "perturbation_function_name"

attack_map = {
    brightening_attack_type_name: {
        required_parameters_key: {delta_parameter_key, threshold_parameter_key},
        perturbation_function_name_key: "perturbBrightening"
    },
    darkening_attack_type_name: {
        required_parameters_key: {delta_parameter_key, threshold_parameter_key},
        perturbation_function_name_key: "perturbDarkening"
    },
    random_noise_attack_type_name: {
        required_parameters_key: {noise_parameter_key, pixels_parameter_key},
        perturbation_function_name_key: "perturbRandomNoise"
    }
}

image_path_key = "image_path"
category_name_key = "category_name"
category_number_key = "category_number"
result_key = "result"

template_parameter_file_name = "template_parameters.json"
notebooks_directory_name = "notebooks"
