from operator import index
import numpy as np
import torch
import glob
import os.path
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import cv2
import torch.nn as nn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

from seed import setup_seed
from data_func import *
setup_seed()

import numpy as np

# 3D sice coefficient

class dice_coefficient(nn.Module):
    def __init__(self, epsilon=0.0001):
        super(dice_coefficient, self).__init__()
        # smooth factor
        self.epsilon = epsilon

    def forward(self, targets, logits):
        batch_size = targets.shape[0]
        logits = logits.view(batch_size, -1).type(torch.FloatTensor)
        targets = targets.view(batch_size, -1).type(torch.FloatTensor)
        intersection = (logits * targets).sum(-1)
        dice_score = (2. * intersection+ self.epsilon) / ((logits.sum(-1) + targets.sum(-1)) + self.epsilon)
        return torch.mean(dice_score)
    
# def cal_subject_level_dice(prediction, target, class_num=2):# class_num是你分割的目标的类别个数
#     eps = 1e-10
#     empty_value = -1.0
#     dscs = empty_value * np.ones((class_num), dtype=np.float32)
#     for i in range(0, class_num):
#         if i not in target and i not in prediction:
#             continue
#         target_per_class = np.where(target == i, 1, 0).astype(np.float32)
#         prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

#         tp = np.sum(prediction_per_class * target_per_class)
#         fp = np.sum(prediction_per_class) - tp
#         fn = np.sum(target_per_class) - tp
#         dsc = 2 * tp / (2 * tp + fp + fn + eps)
#         dscs[i] = dsc
#     dscs = np.where(dscs == -1.0, np.nan, dscs)
#     subject_level_dice = np.nanmean(dscs[1:])
#     return subject_level_dice

def cal_subject_level_dice(prediction, target, class_num=2):
    eps = 1e-10
    empty_value = -1.0
    dscs = empty_value * np.ones((class_num), dtype=np.float32)
    
    for i in range(class_num):
        # 使用 np.any 检查类别是否存在于目标或预测中
        if np.any(target == i) or np.any(prediction == i):
            target_per_class = np.where(target == i, 1, 0).astype(np.float32)
            prediction_per_class = np.where(prediction == i, 1, 0).astype(np.float32)

            tp = np.sum(prediction_per_class * target_per_class)
            fp = np.sum(prediction_per_class) - tp
            fn = np.sum(target_per_class) - tp
            
            denominator = 2 * tp + fp + fn + eps
            if denominator > 0:
                dsc = 2 * tp / denominator
                dscs[i] = dsc

    dscs = np.where(dscs == empty_value, np.nan, dscs)
    subject_level_dice = np.nanmean(dscs)  # 根据需要是否跳过第0类
    return subject_level_dice



    
class Data:
    def __init__(self, train_images_A, train_images_B, train_labels, train_diffs, val_images_A, val_images_B, val_diffs, val_labels, test_images_A, test_images_B, test_diffs, test_labels, slices_per_3d_image, handler):
        self.train_images_A = train_images_A # used for maintaining original label
        self.train_images_B = train_images_B
        self.train_labels = train_labels # used for pseudo training
        self.train_diffs = train_diffs
        self.val_images_A = val_images_A
        self.val_images_B = val_images_B
        self.val_labels = val_labels
        self.val_diffs = val_diffs
        self.test_images_A = test_images_A 
        self.test_images_B = test_images_B 
        self.test_labels = test_labels 
        self.test_diffs = test_diffs
        
        self.slices_per_3d_image = slices_per_3d_image 

        self.handler = handler

        self.n_pool = len(train_images_A)
        self.n_test = len(test_images_A)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        # self.unlabeled_idxs = np.zeros(self.n_pool, dtype=bool)

    def supervised_training_labels(self):
        # used for supervised learning baseline, put all data labeled
        tmp_idxs = np.arange(self.n_pool)
        self.labeled_idxs[tmp_idxs[:]] = True

    def initialize_labels_random(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def initialize_labels_K(self, num_slices_per_patient, k):# 每个病人有多少slice
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        start_idx = 0
        non_blank_idx = []

        print("去除空白前X_train",self.X_train.shape) #(7750, 1, 240, 240)
        for i in range(len(num_slices_per_patient)):#([512, 512, 512, 512, 512, 256, 256, 256, 256, 256, 336, 336, 336, 336, 336])
            num_slices = num_slices_per_patient[i]
            num_full_segments = (num_slices // k)+1 # 每个病人多少初始化slice 25 
            last_segment_size = num_slices % k # 每个病人剩余多少slice 12 
            # print("num_full_segments",num_full_segments)
            selected_slices = [start_idx + k*j for j in range(num_full_segments)]#[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480]
            print("selected_slices",selected_slices)
            start_idx += num_slices
            self.labeled_idxs[selected_slices] = True
            # print("selected_slices",selected_slices)
            # print("num_slices_per_patient",num_slices_per_patient)
            for j in range(len(selected_slices)-1): #[0, 30, 60, 90, 120, 150]   0 
                
                if np.sum(self.Y_train[selected_slices[j]])==0 and np.sum(self.Y_train[selected_slices[j+1]])!=0:
                    start_blank_idx = selected_slices[j]
                if np.sum(self.Y_train[selected_slices[j]])!=0 and np.sum(self.Y_train[selected_slices[j+1]])==0:
                    end_blank_idx = selected_slices[j+1]
            # print(start_blank_idx,end_blank_idx)
            non_blank_idx.extend(range(start_blank_idx, end_blank_idx+1))
        return len(np.arange(self.n_pool, dtype=int)[self.labeled_idxs]),non_blank_idx

    def delete_black_slices(self, index):
        self.X_train = self.X_train[index]
        self.Y_train = self.Y_train[index]
        self.labeled_idxs = self.labeled_idxs[index]
        self.n_pool = len(self.X_train)
        print("去除空白后X_train",self.X_train.shape)
        print(len(self.labeled_idxs))
        print("labeled",self.X_train[self.labeled_idxs].shape)
        print(len(self.labeled_idxs))
    
    # def initialize_labels_K(self, num_slices_per_patient, k):# 每个病人有多少slice
    #     self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
    #     start_idx = 0
    #     cumulative_counts = []
    #     cumulative_sum = 0

    #     for num_slices in num_slices_per_patient:
    #         num_full_segments = (num_slices // k) + 1
    #         last_segment_size = num_slices % k
    #         selected_slices = [start_idx + k*j for j in range(num_full_segments)]
    #         start_idx += num_slices
    #         # selected_slices.append(start_idx-1)  # Add the last slice
    #         self.labeled_idxs[selected_slices] = True #[0, 20, 40, 60, 80, 100, 120, 140]
    #         # print("selected_slices",selected_slices)
    #     # for count in num_slices_per_patient:
    #     #     cumulative_sum += count
    #     #     cumulative_counts.append(cumulative_sum)
    #     # print("num_slices_per_patient",cumulative_counts)
    #     #     if last_segment_size > 0:
    #     #         last_selected_slices = [i+k*num_full_segments for i in range(last_segment_size)]
    #     #         self.X_train = self.X_train[~last_selected_slices]
    #     #         self.Y_train = self.Y_train[~last_selected_slices]
    #     # print("initialize_labels_K data",self.X_train.shape,self.Y_train.shape)

    #     return len(np.arange(self.n_pool, dtype=int)[self.labeled_idxs]) # 8*50


    # def initialize_labels_K(self, interval): #病人slice 不能整除K
    #     tmp_idxs = np.zeros(self.n_pool, dtype=bool)
    #     for i in range(self.n_patients):
    #         start_idx = i * self.n_slices
    #         tmp_idxs[start_idx:start_idx+self.n_slices:interval] = True
    #     return len(tmp_idxs[tmp_idxs])

    # def get_labeled_data(self):
    #     # get labeled data for training
    #     labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs]
    #     # print("labeled data", labeled_idxs.shape)
    #     # print("labeled_idxs ", labeled_idxs)
    #     return labeled_idxs, self.handler(self.X_train[labeled_idxs], self.Y_train[labeled_idxs],mode="train")

    def get_vis_labeled_data(self):
        # get labeled data for training
        labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs]
        # print("labeled data", labeled_idxs.shape)
        # print("labeled_idxs ", labeled_idxs)
        return labeled_idxs, self.train_images_A[labeled_idxs], self.train_images_B[labeled_idxs], self.train_labels[labeled_idxs], self.train_diffs[labeled_idxs]
    
    def get_vis_all_data(self):
        # get all data for training
        return self.train_images_A, self.train_images_B, self.train_labels, self.train_diffs

    def cal_target(self):
        target_num = []
        for i in range(len(self.Y_train)):
            target_num.append(np.sum(self.Y_train[i]))
        return target_num
    
    def get_labeled_data(self):
        # get labeled data for training
        labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs]
        # print("labeled data", labeled_idxs.shape)
        # print("labeled_idxs ", labeled_idxs)
        return labeled_idxs, self.handler(self.train_images_A[labeled_idxs], self.train_images_B[labeled_idxs], self.train_labels[labeled_idxs], self.train_diffs[labeled_idxs], mode="val")
  
    def get_data(self): 
        unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#24537
        labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs].tolist()
        return unlabeled_idxs,self.train_images_A[unlabeled_idxs], self.train_images_A[labeled_idxs]
    # def get_filtered_data(self, X_train, Y_train):
    #     return self.handler(X_train, Y_train, mode="train")

    
    # def get_data(self, pseudo_idxs, k, train_num_slices_per_patient): 

    #     labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs].tolist()
    #     print("normal",len(labeled_idxs))

    #     if pseudo_idxs != None:
    #         labeled_idxs.extend(pseudo_idxs) #把pseudo label加进去 进行采样
    #     labeled_idxs = np.array(labeled_idxs)
    #     # print(len(pseudo_idxs))
    #     print("pseudo",len(labeled_idxs))
    #     return labeled_idxs, self.handler(self.X_train, self.Y_train_pseudo, labeled_idxs, k, train_num_slices_per_patient)

    # def get_data(self, pseudo_idxs, k, train_num_slices_per_patient, handler): 
    #     # get labeled data for training
    #     labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs]
    #     if pseudo_idxs != None:
    #         labeled_idxs = np.concatenate((np.arange(self.n_pool, dtype=int)[self.labeled_idxs], pseudo_idxs), axis=0)
    #     print(len(labeled_idxs))
    #     return labeled_idxs, handler(self.X_train, self.Y_train_pseudo, labeled_idxs, k, train_num_slices_per_patient)


    # used for pseudo label filter remove blank patches
    def delete_black_patch(self, index, preds):
        black_index = []
        for i in range(preds.shape[0]):#24537
            idx = preds[i]
            index[i]
            pred = (preds[i][1] > 0.5).int()
            if torch.sum(pred)==0:
                black_index.append(idx)
        return black_index

    # def get_unlabeled_data(self, index=None): #index是空白patch
    #     # get unlabeled data for active learning selection process
    #     unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#24537
    #     # print("unlabeled_idxs",unlabeled_idxs.shape)
    #     if index!=None:
    #         self.labeled_idxs[index] = True #5486
    #         unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#19051 19255
    #         self.labeled_idxs[index] = False
    #     return unlabeled_idxs


    def get_unlabeled_data(self, rd=None, index=None): #index是空白patch
        # get unlabeled data for active learning selection process
        unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#24537
        # print("unlabeled_idxs",unlabeled_idxs.shape)
        if index!=None:
            self.labeled_idxs[index] = True #5486
            unlabeled_idxs = np.arange(self.n_pool, dtype=int)[~self.labeled_idxs]#19051 19255
            self.labeled_idxs[index] = False
        # if rd ==8: 
            # print("get_unlabeled_data_x, get_unlabeled_data_y", self.X_train[unlabeled_idxs].shape, self.Y_train[unlabeled_idxs].shape)
        return unlabeled_idxs, self.handler(self.train_images_A[unlabeled_idxs], self.train_images_B[unlabeled_idxs], self.train_labels[unlabeled_idxs], self.train_diffs[unlabeled_idxs], mode="val")
    

    # def get_embedding_data(self, index): #index是空白patch
    #     return self.handler(self.X_train[index], self.Y_train[index],mode="val"

    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.train_images_A, self.train_images_B, self.train_labels, self.train_diffs, mode="val")
    
    def get_all_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool, dtype=int)[self.labeled_idxs]
        return labeled_idxs, self.handler(self.train_images_A[labeled_idxs], self.train_images_B[labeled_idxs], self.train_labels[labeled_idxs], self.train_diffs[labeled_idxs], mode="val")

    def get_val_data(self):
        # get validation dataset if exist
        return self.handler(self.val_images_A, self.val_images_B, self.val_labels, self.val_diffs, mode="val")

    def get_test_data(self):
        # get test dataset if exist
        return self.handler(self.test_images_A, self.test_images_B, self.test_labels, self.test_diffs, mode="val")

    def cal_test_acc(self, logits, targets):
        # calculate accuracy for test dataset
        dscs = [] 
        recalls = []
        precisions = []
        for prediction, target in zip(logits, targets):
            dsc = cal_subject_level_dice(prediction, target, class_num=2)
            preds_np = prediction.flatten()
            targets_np = target.flatten()
            rec = recall_score(targets_np, preds_np)
            pre = precision_score(targets_np, preds_np)

            dscs.append(dsc)
            recalls.append(rec)
            precisions.append(pre)

            dice = np.mean(dscs)
            recall = np.mean(recalls)
            precision = np.mean(precisions)

        return dice, recall, precision

    # def cal_test_acc(self, logits, targets):
    #     preds = logits.max(1)[1] # 假设 threshold 是你设定的阈值

    #     # 确保 targets、preds 和 probs 都在 CPU 上并转换为 NumPy 数组
    #     targets_np = targets.cpu().numpy()
    #     preds_np = preds.cpu().numpy() # 类别
    #     probs_np = logits.cpu().numpy() # 概率

    #     # 计算各项指标
    #     print(targets_np)
    #     print(preds_np)

    #     acc = accuracy_score(targets_np, preds_np)
    #     recall = recall_score(targets_np, preds_np, average='micro')
    #     precision = precision_score(targets_np, preds_np, average='weighted')
    #     f1 = f1_score(targets_np, preds_np, average='weighted')
    #     auc = roc_auc_score(targets_np, probs_np, multi_class='ovr', average='macro')
    #     precision = precision_score(targets_np, preds_np, average='weighted')

    #     # 返回计算的指标
    #     return acc, recall, precision, f1, auc, precision

    def cal_target(self):
        target_num = []
        for i in range(len(self.train_labels)):
            target_num.append(np.sum(self.train_labels[i]))
        return target_num


def get_MSSEG(handler, param2, supervised = False):

    if supervised == False:
        train_images_A = np.load("../train_images_A_all.npy",allow_pickle=True)
        train_images_B = np.load("../train_images_B_all.npy",allow_pickle=True)
        train_labels = np.load("../train_labels_all.npy",allow_pickle=True)
        train_diffs = np.load("../train_diffs_all.npy",allow_pickle=True)

    else:

        train_images_A = np.load("../train_images_A_target.npy")
        train_images_B = np.load("../train_images_B_target.npy")
        train_labels = np.load("../train_labels_target.npy")
        train_diffs = np.load("../train_diffs_target.npy")
                            
    val_images_A = np.load("../val_images_A_all.npy",allow_pickle=True)
    val_images_B = np.load("../val_images_B_all.npy",allow_pickle=True)
    val_labels = np.load("../val_labels_all.npy",allow_pickle=True)
    val_diffs = np.load("../val_diffs_all.npy",allow_pickle=True)

    test_images_A = np.load("../test_images_A_all.npy",allow_pickle=True)
    test_images_B = np.load("../test_images_B_all.npy",allow_pickle=True)
    test_labels = np.load("../test_labels_all.npy",allow_pickle=True)
    test_diffs = np.load("../test_diffs_all.npy",allow_pickle=True)

    slices_per_3d_image = np.load("../slices_per_3d_image.npy",allow_pickle=True).item()


    if param2 == 'train_diff_f':
        weight_maps =  np.load("/mnt/recsys/siteng/MSSEG2/test/train_diff_f.npy")
    elif param2 == 'train_diff_f_thr20':
        weight_maps =  np.load("/mnt/recsys/siteng/MSSEG2/test/train_diff_f_thr20.npy")
    elif param2 == 'train_diff_f_thr50':
        weight_maps =  np.load("/mnt/recsys/siteng/MSSEG2/test/train_diff_f_thr50.npy")
    elif param2 == 'train_diff_f_thr80':
        weight_maps =  np.load("/mnt/recsys/siteng/MSSEG2/test/train_diff_f_thr80.npy")
    else:
        weight_maps = None

    if param2 != None: 
        weight_maps = weight_maps/255

    train_labels = (train_labels > 0).astype(int)
    val_labels = (val_labels > 0).astype(int)
    test_labels = (test_labels > 0).astype(int)

    train_images_A = np.expand_dims(train_images_A, axis=1)
    train_images_B = np.expand_dims(train_images_B, axis=1)
    train_labels = np.expand_dims(train_labels, axis=1)
    train_diffs = np.expand_dims(train_diffs, axis=1)

    val_images_A = np.expand_dims(val_images_A, axis=1)
    val_images_B = np.expand_dims(val_images_B, axis=1)
    val_labels = np.expand_dims(val_labels, axis=1)
    val_diffs = np.expand_dims(val_diffs, axis=1)

    test_images_A = np.expand_dims(test_images_A, axis=1)
    test_images_B = np.expand_dims(test_images_B, axis=1)
    test_labels = np.expand_dims(test_labels, axis=1)
    test_diffs = np.expand_dims(test_diffs, axis=1)

    # train_images_A = train_images_A/255
    # train_images_B = train_images_B/255
    # train_diffs = train_diffs/255

    # val_images_A = val_images_A/255
    # val_images_B = val_images_B/255
    # val_diffs = val_diffs/255

    # test_images_A = test_images_A/255
    # test_images_B = test_images_B/255
    # test_diffs = test_diffs/255

    print(train_images_A.shape, train_images_B.shape, train_labels.shape, train_diffs.shape)
    print(val_images_A.shape, val_images_B.shape, val_labels.shape, val_diffs.shape)
    print(test_images_A.shape, test_images_B.shape, test_labels.shape, test_diffs.shape)

    print(train_images_A.max(),train_images_A.min()) 
    print(train_images_B.max(),train_images_B.min())
    print(train_labels.max(),train_labels.min())
    print(train_diffs.max(),train_diffs.min())

    print(val_images_A.max(),val_images_A.min())
    print(val_images_B.max(),val_images_B.min())
    print(val_labels.max(),val_labels.min())
    print(val_diffs.max(),val_diffs.min())

    print(test_images_A.max(),test_images_A.min())
    print(test_images_B.max(),test_images_B.min())
    print(test_labels.max(),test_labels.min())
    print(test_diffs.max(),test_diffs.min())
    # print(slices_per_3d_image)

    return train_images_A, train_images_B, train_labels, train_diffs, val_images_A, val_images_B, val_diffs, val_labels, test_images_A, test_images_B, test_diffs, test_labels, slices_per_3d_image, weight_maps, handler
