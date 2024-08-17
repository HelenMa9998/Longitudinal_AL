import argparse
import numpy as np
import torch
import pandas as pd
from pprint import pprint
import random
from data import Data
from utils import get_dataset, get_net, get_strategy, get_handler
from config import parse_args
from seed import setup_seed
from visualization import visualiazation
import pdb

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import os

import numpy as np
import matplotlib.pyplot as plt

def vis(rd, labeled_idxs, train_images_A, train_images_B, train_labels, train_diffs):
    # 创建一个包含4个子图的图像，排列为1行4列
    for i in range(len(labeled_idxs)):
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 修改了子图的布局和尺寸

        # train_images_A 子图
        # 使用 np.squeeze 来移除单一颜色通道维度
        axs[0].imshow(np.squeeze(train_images_A[i]), cmap='gray')
        axs[0].set_title('Train Images A')
        
        # train_images_B 子图
        axs[1].imshow(np.squeeze(train_images_B[i]), cmap='gray')
        axs[1].set_title('Train Images B')
    
        # train_labels 子图
        axs[2].imshow(np.squeeze(train_labels[i]), cmap='gray')
        axs[2].set_title('Train Labels')
        
        # train_diffs 子图
        axs[3].imshow(np.squeeze(train_diffs[i]), cmap='gray')
        axs[3].set_title('Train Diffs')
        
        # 调整子图布局
        plt.tight_layout()
        
        # 保存图像
        plt.savefig(f'vis/entropy-100-50-0.0005/{rd}-{labeled_idxs[i]}.png')
        
        # 关闭图形显示
        plt.close()

    
def main(param1, param2, param3, param4):
    args = parse_args()
    pprint(vars(args))
    print()

    # fix random seed
    setup_seed(param4)
    import os
#    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#    os.environ['CUDA_LAUNCH_BLOCKINGs'] = '0'
#    torch.cuda.empty_cache()

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # get dataset
    train_images_A, train_images_B, train_labels, train_diffs, val_images_A, val_images_B, val_diffs, val_labels, test_images_A, test_images_B, test_diffs, test_labels, slices_per_3d_image, weight_maps, handler = get_dataset(args.dataset_name, param2, supervised = False )

    # get dataloader
    dataset = Data(train_images_A, train_images_B, train_labels, train_diffs, val_images_A, val_images_B, val_diffs, val_labels, test_images_A, test_images_B, test_diffs, test_labels, slices_per_3d_image, handler)

    slices_counts = np.array(list(slices_per_3d_image.values()))

    # start experiment
    dataset.initialize_labels_random(args.n_init_labeled)
    # init_num, non_blank_idx = dataset.initialize_labels_K(train_num_slices_per_patient, args.k_init_labeled) # 或者在load dataset的时候就保证整除
    # dataset.initialize_labels(strategy_init, args.n_init_labeled)
    print(f"number of labeled pool: {args.n_init_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool-args.n_init_labeled}")
    print(f"number of testing pool: {dataset.n_test}")
    print()

    # load prop network
    prop_net = get_net(args.dataset_name, device, prop=True) 
    # load AL strategy 这里有问题 train dataset是多个slice为一个样本
    strategy = get_strategy(param1)(dataset, prop_net) 
    # strategy = get_strategy(param1)(dataset, prop_net) # load strategy


    # Round 0 train 
    # 这里有另一种训练思路 即先训练一个模型（我理解是只有encoder和decoder?) 然后直接预测得到伪标签 然后使用伪标签和已标记数据进行标签传播 更新为标签
    print("Round 0")
    rd = 0
    # strategy.train(rd, args.training_name) #直接这里的train就是prop train
    dice = []
    recall = []
    precision = []

    size = []
    query_samples = []
    # Round 0 test 
    strategy.train(rd, args.training_name, param1, param4)
    test_preds,targets = strategy.predict(dataset.get_test_data(),slices_counts, param1, param4)
    dic, rec, prec = dataset.cal_test_acc(test_preds, targets)
    print(f"Round 0 testing accuracy: {dic, rec, prec}")  # get model performance for test dataset
    dice.append(dic)
    recall.append(rec)
    precision.append(prec)

    size.append(args.n_init_labeled)
    # active learning selection
    target_num = dataset.cal_target()

    for rd in range(1, args.n_round + 1):
        print(f"Round {rd}")
        # AL query
        


        query_idxs = strategy.query(args.n_query, param3, weight_maps, param1, param4)  # query_idxs为active learning请求标签的数据
        labeled_idxs, norm_train_loader = dataset.get_labeled_data()

        print("before",len(labeled_idxs))
        # update labels 这里也应该是slice为单位
        strategy.update(query_idxs)  # update training dataset and unlabeled dataset for active learning
        labeled_idxs, norm_train_loader = dataset.get_labeled_data()
        print("after",len(labeled_idxs))
        
        strategy.train(rd, args.training_name, param1, param4)
        # prop_net.prop_train(prop_train_loader,prop_val_loader,rd)#只根据labeled data去学习encoder、key val encoder、decoder
        # pseudo_idxs, pseudo_label = prop_net.prop(prop_loader) 

        # calculate accuracy 这就是正常的 需要把计算key val以及cross attention去掉
        test_preds, targets = strategy.predict(dataset.get_test_data(),slices_counts, param1, param4) # get model prediction for test dataset
        dic, rec, prec = dataset.cal_test_acc(test_preds, targets)
        query_samples.append(query_idxs)

        # if dic > 0.44412994:
        #     # labeled_idxs, train_images_A, train_images_B, train_labels, train_diffs = dataset.get_vis_labeled_data()
        #     # vis(rd, labeled_idxs, train_images_A, train_images_B, train_labels, train_diffs)
        #     all_train_images_A, all_train_images_B, _, _ = dataset.get_vis_all_data()

        #     visualiazation(all_train_images_A, all_train_images_B, query_samples, target_num, param1, rd) # all, query, 


        print(f"Round {rd} testing accuracy: {dic, rec, prec}")  # get model performance for test dataset
        dice.append(dic)
        recall.append(rec)
        precision.append(prec)
        # labeled_idxs, _ = dataset.get_labeled_data(eval_handler, pseudo_idxs=None)
        size.append(len(labeled_idxs))

    # save the result
    dataframe = pd.DataFrame(
        {'model': 'Unet', 'Method': param1, 'Training dataset size': size, 'Dice': dice, 'Recall': recall, 'Precision': precision})
    dataframe.to_csv(f"./{param4}-{param1}.csv", index=False, sep=',')

experiment_parameters = [
    # param1 具体方法；param2 JS; param3 具体选择策略
    # normal; hard; [0-0.2]; [0-0.2]+[0.2-0.4]
    # {'param1': "EntropySampling", 'param2':'train_diff_f', 'param3':'our'},
    # {'param1': "EntropySampling", 'param2':'train_diff_f_thr20', 'param3':'our'},
    # {'param1': "EntropySampling", 'param2':'train_diff_f_thr50', 'param3':'our'},
    # {'param1': "EntropySampling", 'param2':'train_diff_f_thr80', 'param3':'our'},


#     {'param1': "EntropySamplingDropout", 'param2': None, 'param3': None, 'param4': 1999},
#    {'param1': "BALDDropout", 'param2': None, 'param3': None, 'param4': 1999},
#   {'param1': "MarginSampling", 'param2': None, 'param3': None, 'param4': 1999},
#    {'param1': "LeastConfidence", 'param2': None, 'param3': None, 'param4': 1999},
#    {'param1': "RandomSampling", 'param2': None, 'param3': None, 'param4': 1999},
#    {'param1': "EntropySampling", 'param2': None, 'param3': None, 'param4': 1999},
    {'param1': "HybridSampling", 'param2': None, 'param3': None, 'param4': 98},
#{'param1': "ClusterMarginSampling", 'param2': None, 'param3': None, 'param4': 98},
#    {'param1': "KCenterGreedy", 'param2': None, 'param3': None, 'param4': 1999},
]

for params in experiment_parameters:
    main(params['param1'], params['param2'], params['param3'], params['param4'])
