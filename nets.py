import math
from turtle import shape
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import torchvision
from collections import OrderedDict
from tqdm import tqdm
from seed import setup_seed
import pdb
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim


setup_seed()
# used for getting prediction results for data
def recompone_overlap_3d(preds, slices_count, image_nums):
    final_full_prob = []
    a = 0
    for x in range(image_nums):
        img_d = slices_count[x]
        img_h, img_w = preds[0].shape
        
        full_prob = np.zeros((img_d, img_h, img_w))
        for i in range(img_d):
            full_prob[i] = preds[a]
            a += 1

        assert(np.max(full_prob) <= 1.0)  # 最大值不超过1.0
        assert(np.min(full_prob) >= 0.0)  # 最小值不小于0.0
        final_full_prob.append(full_prob)

    return final_full_prob


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class dice_coefficient(nn.Module):
    def __init__(self, epsilon=0.0001):
        super(dice_coefficient, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, targets, logits):
        batch_size = targets.shape[0]
        logits = (logits > 0.5).float()
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (logits * targets).sum(-1)
#         dice_score = 2. * (intersection + self.epsilon) / ((logits + targets).sum(-1) + self.epsilon)
        dice_score = (2. * intersection+ self.epsilon) / ((logits.sum(-1) + targets.sum(-1)) + self.epsilon)
        return torch.mean(dice_score)

# including different training method for active learning process (train acc=1, val loss, val acc, epoch)
class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.params = params
        self.device = device

    def supervised_val_loss(self, data, val_data, rd, AL_method = "RandomSampling", seed = 42):
        n_epoch = 200
        trigger = 0
        best = {'epoch': 1, 'loss': 10}
        train_loss=0
        validation_loss = 0
        train_dice=0
        val_dice=0
        self.clf = self.net().to(self.device)
        self.clf.train()
        if rd==0:
            self.clf = self.clf
        else:
            self.clf = torch.load(f'./result/{AL_method}_{seed}_model.pth')

        optimizer = optim.Adam(self.clf.parameters(), lr=0.0001)
        
        criterion = FocalLoss(alpha=1, gamma=2,logits=False)

        train_loader=DataLoader(data, **self.params['train_args'])
        val_loader=DataLoader(val_data, **self.params['val_args'])

        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            self.clf.train()
            for batch_idx, (x_base, x_follow, diff, target, idx) in enumerate(train_loader):

                x_base, x_follow, diff, target = x_base.to(self.device), x_follow.to(self.device), diff.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output,_ = self.clf(x_base, x_follow, diff)
                output = torch.sigmoid(output)
                loss = criterion(output.float(), target.float())
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                self.clf.eval()
                for valbatch_idx, (x_base, x_follow, diff, target, idx) in enumerate(val_loader):
                    x_base, x_follow, diff, target = x_base.to(self.device), x_follow.to(self.device), diff.to(self.device), target.to(self.device)
                    output,_ = self.clf(x_base, x_follow, diff)
                    output = torch.sigmoid(output)
                    validation_loss += criterion(output.float(), target.float())
                    
            trigger += 1
            # early stopping condition: if the acc not getting larger for over 10 epochs, stop
            if validation_loss / (valbatch_idx + 1) < best['loss']:
                trigger = 0
                best['epoch'] = epoch
                best['loss'] = validation_loss / (valbatch_idx + 1)
                # print(best['epoch'],best['loss'])
                torch.save(self.clf, f'./result/{AL_method}_{seed}_model.pth')
            # print("\n best performance at Epoch :{}, loss :{}".format(best['epoch'],best['loss']))
            validation_loss = 0
            # val_dice=0
            if trigger >= 5:
                break
        torch.cuda.empty_cache()
        
    
## restore to original dimensions
    def predict(self, data, slices_counts, AL_method, seed):
        self.clf = torch.load(f'./result/{AL_method}_{seed}_model.pth')
        self.clf.eval()
        preds = []
        labs= []
        loader = DataLoader(data, **self.params['test_args'])
        with torch.no_grad():
            for x_base, x_follow, diff, target, idxs in loader:
                x_base, x_follow, diff, target = x_base.to(self.device), x_follow.to(self.device), diff.to(self.device), target.to(self.device)
                output,_ = self.clf(x_base, x_follow, diff) # ([8, 1, 256, 256])
                output = torch.sigmoid(output)
                preds.append(output.data.cpu().numpy())
                labs.append(target.data.cpu().numpy())
        predictions = np.concatenate(preds, axis=0).squeeze()
        labels = np.concatenate(labs, axis=0).squeeze()
        predictions[predictions>=0.5] = 1
        predictions[predictions<0.5] = 0 # (7050, 256, 256)

        # print("predictions:", predictions.max(),predictions.min())
        
        pred_imgs_3d = recompone_overlap_3d(predictions, slices_counts, len(slices_counts))
        label_imgs_3d = recompone_overlap_3d(labels, slices_counts, len(slices_counts))
        return pred_imgs_3d, label_imgs_3d
    
    # def predict(self, data, num_slices_per_patient):
    #     self.clf = torch.load('./result/model.pth')
    #     self.clf.eval()
    #     preds = []
    #     loader = DataLoader(data, **self.params['test_args'])
    #     with torch.no_grad(): 
    #         for x, y, idxs in loader:
    #             # x, y = x.unsqueeze(1), y.unsqueeze(1)
    #             x, y = x.to(self.device), y.to(self.device)
    #             out = self.clf(x, phase='test')
    #             outputs = out.data.cpu().numpy()
    #             preds.append(outputs)

    #     predictions = np.concatenate(preds, axis=0)#(40617, 1, 128, 128)
    #     # pred_patches = np.expand_dims(predictions,axis=1)#(40617, 1, 1, 128, 128)
    #     # pred_patches[pred_patches>=0.5] = 1
    #     # pred_patches[pred_patches<0.5] = 0
    #     # pred_imgs = recompone_overlap(pred_patches.squeeze(), full_test_imgs_list, x_test_slice, stride=96, image_num=8251)
    #     # pred_imgs_3d = recompone_overlap_3d(np.array(pred_imgs), test_brain_images, image_num=38)
    #     pred_imgs_3d = merge_slices_to_3D_image(np.array(predictions), num_slices_per_patient)
    #     pred_imgs_3d = np.array(pred_imgs_3d)
    #     return pred_imgs_3d


    def predict_prob(self, data, AL_method, seed):
        self.clf = torch.load(f'./result/{AL_method}_{seed}_model.pth')
        self.clf.eval()
        probs = torch.zeros([len(data), 1, 256, 256])
        loader = DataLoader(data, **self.params['test_args'])
        with torch.no_grad():
            for x_base, x_follow, diff, target, idxs in loader:
                x_base, x_follow, diff, target = x_base.to(self.device), x_follow.to(self.device), diff.to(self.device), target.to(self.device)
                output,_ = self.clf(x_base, x_follow, diff)
                output = torch.sigmoid(output)
                probs[idxs] = output.cpu()
        return probs

    # Calculating 10 times probability for prediction, the mean used as uncertainty
    def predict_prob_dropout(self, data, AL_method, seed, n_drop=10):
        self.clf = torch.load(f'./result/{AL_method}_{seed}_model.pth')
        self.clf.train()
        probs = torch.zeros([len(data), 1, 256, 256])

        loader = DataLoader(data, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x_base, x_follow, diff, target, idxs in loader:

                    x_base, x_follow, diff, target = x_base.to(self.device), x_follow.to(self.device), diff.to(self.device), target.to(self.device)
                    output,_ = self.clf(x_base, x_follow, diff)
                    output = torch.sigmoid(output)
                    probs[idxs] += output.cpu()
        probs /= n_drop
        return probs

    # Used for Bayesian sampling
    def predict_prob_dropout_split(self, data, AL_method, seed, n_drop=10):
        self.clf = torch.load(f'./result/{AL_method}_{seed}_model.pth')

        self.clf.train()
        probs = torch.zeros([n_drop, len(data), 1, 256, 256])

        loader = DataLoader(data, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x_base, x_follow, diff, target, idxs in loader:

                    x_base, x_follow, diff, target = x_base.to(self.device), x_follow.to(self.device), diff.to(self.device), target.to(self.device)
                    output,_ = self.clf(x_base, x_follow, diff)
                    output = torch.sigmoid(output)
                    probs[i][idxs] += output.cpu()
        return probs

    def get_embeddings(self, data, AL_method, seed):
        self.clf = torch.load(f'./result/{AL_method}_{seed}_model.pth')
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, **self.params['test_args'])
        with torch.no_grad():
            for x_base, x_follow, diff, target, idxs in loader:

                x_base, x_follow, diff, target = x_base.to(self.device), x_follow.to(self.device), diff.to(self.device), target.to(self.device)
                _, e1 = self.clf(x_base, x_follow, diff)
                embeddings[idxs] = e1.cpu().reshape(len(x_base),-1)
        return embeddings
    


from glasses.models.segmentation.unet import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetC(nn.Module):
    def __init__(self, num_classes=1):
        super(UNetC, self).__init__()
        self.net = UNet(n_classes=num_classes, in_channels=3)
        self.encoder = self.net.encoder
        
    def forward(self, x1, x2, x3):
        x = torch.cat((x1, x2, x3),1)
        encoder_output = self.encoder(x)
        out = self.net(x)
        return out, encoder_output # [8, 1024, 16, 16]
    
    def get_embedding_dim(self):
        return 1024 * 16 *16
