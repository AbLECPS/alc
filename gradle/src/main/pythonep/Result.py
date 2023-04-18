from addict import Dict


class Result(Dict):

    def __init__(self, result):
        Dict.__init__(self, result)

        if "parameters" not in self:
            parameters = self.get(ResultAux._params_key, Dict())
            if parameters:
                self.parameters = parameters

        if "loss" not in self:
            loss = result.get(ResultAux._results_key, Dict()) \
                .get(ResultAux._model_evaluation_key, Dict()) \
                .get(ResultAux._loss_key, None)
            if loss and (isinstance(loss, float) or isinstance(loss, int)):
                self.loss = loss


class ResultAux:
    _loss_key = "loss"
    _model_evaluation_key = "model_evaluation"
    _params_key = "exptParams"
    _results_key = "results"

    @staticmethod
    def get_list_paths(result_list):
        return [item.path for item in result_list]
