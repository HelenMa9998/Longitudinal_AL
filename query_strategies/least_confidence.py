import numpy as np
import torch
import random
from .strategy import Strategy
class LeastConfidence(Strategy):
    def __init__(self, dataset, net):
        super(LeastConfidence, self).__init__(dataset, net)

    def query(self, n, param3, weight_maps, AL_method, seed):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data, AL_method, seed)

        uncertainties = probs.sum((1,2,3)) # probs.sum((1,2,3)).shape ([7250])
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
