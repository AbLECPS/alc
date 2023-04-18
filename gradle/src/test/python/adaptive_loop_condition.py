def adapt1(workflow_data):
    current_results = workflow_data.get_results_relative(-1, ["SLModelTraining"])
    if not current_results:
        return True

    current_loss = current_results[0].get_loss()

    if current_loss < 0.1:
        return False

    previous_results = workflow_data.get_results_relative(-2, ["SLModelTraining"])

    if not previous_results:
        return True

    previous_loss = previous_results[0].get_loss()

    return current_loss < previous_loss
