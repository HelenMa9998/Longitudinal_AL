import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from seed import setup_seed

setup_seed()
class Strategy:
    def __init__(self, dataset, net):
        self.dataset = dataset
        self.net = net

    # def query(self, n):
    #     pass
    
    # def query(self, n, index, pred):
    #     pass

    def query(self, n, rd, param2, AL_method, seed):
        pass
    #update training dataset: adding new labeled data to training dataset, and delete from unlabeled dataset
    def update(self, pos_idxs, neg_idxs=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if neg_idxs:
            self.dataset.labeled_idxs[neg_idxs] = False

    # Train on all label data (training dataset)
    def train(self, rd, training_method, AL_method, seed):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        val_data = self.dataset.get_val_data()
        if training_method == "supervised_train_acc":
            self.net.supervised_train_acc(labeled_data)
        elif training_method == "supervised_val_loss":
            self.net.supervised_val_loss(labeled_data, val_data, rd, AL_method, seed)
        elif training_method == "supervised_val_acc":
            self.net.supervised_val_acc(labeled_data, val_data)
        elif training_method == "supervised_train_epoch":
            self.net.supervised_train_epoch(labeled_data)
        elif training_method == "prop_train":
            self.net.prop_train(labeled_data, val_data)
        else:
            raise NotImplementedError

    # def efficient_train(self, rd, data):
    #     idx = self.net.get_underfit_idx(data)
    #     _, labeled_data = dataset.get_efficient_training_data(idx, query_idxs)
    #     # val_data = dataset.get_val_data()
    #     # self.net.supervised_train(rd, labeled_data, val_data)
    #     # self.net.supervised_train_loss(rd, labeled_data)
    #     self.net.supervised_train_acc(labeled_data)
    #
    # #         self.net.train(labeled_data)

    # predict on test dataset
    # def predict(self, data, full_test_imgs_list, x_test_slice, test_brain_images):  # 在test data上predict
    #     preds = self.net.predict(data, full_test_imgs_list, x_test_slice, test_brain_images)
    #     return preds
        
    def predict(self, data, slices_counts, AL_method, seed):  # 在test data上predict
        preds = self.net.predict(data, slices_counts, AL_method, seed)
        return preds
    
    # call predict_prob in net.py
    def predict_prob(self, data, AL_method, seed):
        probs = self.net.predict_prob(data, AL_method, seed)
        return probs

    # call predict_prob_dropout in net.py
    def predict_prob_dropout(self, data, AL_method, seed, n_drop=10):
        probs = self.net.predict_prob_dropout(data, AL_method, seed, n_drop=n_drop)
        return probs

    # call predict_prob_dropout_split in net.py
    def predict_prob_dropout_split(self, data, AL_method, seed, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, AL_method, seed, n_drop=n_drop)
        return probs

    def get_embeddings(self, data, AL_method, seed):
        embeddings = self.net.get_embeddings(data, AL_method, seed)
        return embeddings
    
    def compute_NKC(self, data, AL_method, seed, n_drop=10):
        probs = self.net.compute_NKC(data, AL_method, seed, n_drop=n_drop)
        return probs
    
    def predict_prob_embed(self, data, AL_method, seed):
        probs, embeddings = self.net.predict_prob_embed(data, AL_method, seed)
        return probs, embeddings
