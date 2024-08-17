import torch
from utils import get_dataset, get_net, get_strategy
from data import Data
from config import parse_args
import numpy as np
from seed import setup_seed
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKINGs'] = '0'

# fix random seed
seed = 98
setup_seed(seed)
#supervised learning baseline
args = parse_args()
# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# get dataset
param2 = None
train_images_A, train_images_B, train_labels, train_diffs, val_images_A, val_images_B, val_diffs, val_labels, test_images_A, test_images_B, test_diffs, test_labels, slices_per_3d_image, weight_maps, handler = get_dataset(args.dataset_name, param2, supervised = True)

# get dataloader
dataset = Data(train_images_A, train_images_B, train_labels, train_diffs, val_images_A, val_images_B, val_diffs, val_labels, test_images_A, test_images_B, test_diffs, test_labels, slices_per_3d_image, handler)
print(f"number of testing pool: {dataset.n_test}")
print()
slices_counts = np.array(list(slices_per_3d_image.values()))
# get network
net = get_net(args.dataset_name, device)

# start supervised learning baseline
dataset.supervised_training_labels()
labeled_idxs, labeled_data = dataset.get_labeled_data()
val_data = dataset.get_val_data()
print(f"number of labeled pool: {len(labeled_idxs)}")
net.supervised_val_loss(labeled_data,val_data,rd=0, AL_method = "supervised", seed = seed)

preds,labels= net.predict(dataset.get_test_data(),slices_counts, AL_method = "supervised", seed = seed) # get model prediction for test dataset
acc = dataset.cal_test_acc(preds, labels)
print(f"testing dice: {acc}")
