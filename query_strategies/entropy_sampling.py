import numpy as np
import torch
from .strategy import Strategy

# # Use the prediction entropy as uncertainty
class EntropySampling(Strategy):
    def __init__(self, dataset, net):
        super(EntropySampling, self).__init__(dataset, net)

    def query(self, n, param3, weight_maps, AL_method, seed):
        if param3 == "our": 
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

            weight_maps = torch.tensor(weight_maps[unlabeled_idxs]) # (5520, 256, 256)
            probs = self.predict_prob(unlabeled_data, AL_method, seed)
            log_probs = torch.log(probs)
            uncertainties = probs*log_probs#torch.Size([5420, 1, 256, 256]) 
            uncertainties = uncertainties.view(uncertainties.size(0), -1)

            weight_maps = weight_maps.view(weight_maps.size(0), -1)
            print("uncertainties", uncertainties.shape)
            print("weight_maps", weight_maps.shape)

            weighted_uncertainty_per_sample = [torch.nan_to_num((uncertainty * weight_map).sum(), nan=0.0) for uncertainty, weight_map in zip(uncertainties, weight_maps)]

            weighted_sorted_indices = sorted(range(len(weighted_uncertainty_per_sample)), key=lambda k: weighted_uncertainty_per_sample[k])

            return unlabeled_idxs[weighted_sorted_indices[:n]]
        else: 
            
            unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
            # print("unlabeled_idxs",unlabeled_idxs)
            probs = self.predict_prob(unlabeled_data, AL_method, seed)
            log_probs = torch.log(probs)
            uncertainties = (probs*log_probs).sum((1,2,3))#([12384])
            
            return unlabeled_idxs[uncertainties.sort()[1][:n]]
    
