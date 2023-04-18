#from scripts.model_pytorch_y1a import RLController
import torch
import sys
sys.path.insert(0, "./pytorch_LRP")
from pytorch_LRP.innvestigator import InnvestigateModel

# Mods
# This one has been updated to accept a network instance as input
# It also includes a vector that the user can input for de-normalization


class SaliencyMapGenerator():
    def __init__(self, networks, norm):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.net = []
        self.inn_model = []
        for n in networks:
            net = n.to(self.device)
            net.eval()
            self.inn_model.append(InnvestigateModel(
                net, lrp_exponent=2, method="e-rule", beta=0.5))
        self.norm = [norm]

    def __call__(self, input_torch):
        input_torch = input_torch.to(self.device)
        #print ('input_torch_shape ',input_torch.shape)
        norm_torch = torch.tensor(self.norm)
        #print ('norm_torch_shape ',norm_torch.shape)
        norm_torch = norm_torch.to(self.device)
        input_torch = input_torch * norm_torch
        model_prediction = []
        true_relevance = []
        for m in self.inn_model:
            mp, tr = m.innvestigate(in_tensor=input_torch)
            model_prediction.append(mp)
            true_relevance.append(tr)
        return model_prediction, true_relevance
