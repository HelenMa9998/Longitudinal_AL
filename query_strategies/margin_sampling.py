import numpy as np
import torch
import random

from .strategy import Strategy

class MarginSampling(Strategy):
    def __init__(self, dataset, net):
        super(MarginSampling, self).__init__(dataset, net)

    def query(self, n, param3, weight_maps, AL_method, seed):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        probs = self.predict_prob(unlabeled_data, AL_method, seed).sum((1,2,3))
        probs_sorted, _ = probs.sort(descending=True)
        uncertainties = probs_sorted - (1-probs_sorted)#([7250])
        return unlabeled_idxs[uncertainties.sort()[1][:n]]
