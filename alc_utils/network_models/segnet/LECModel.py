def get_model():
    model_parameters = {"img_norm": True,
                        "dataset": "aa_side_scan_sonar",
                        "n_classes": 4,
                        "img_size": (100, 512)}
    return "segnet", model_parameters
