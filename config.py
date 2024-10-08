import argparse
# configurations

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=98, help="random seed")
    parser.add_argument('--n_init_labeled', type=int, default=500, help="number of init labeled samples")
    # parser.add_argument('--k_init_labeled', type=int, default=10, help="one init labeled sample every k")
    parser.add_argument('--n_query', type=int, default=200, help="number of queries per round")
    parser.add_argument('--n_round', type=int, default=5, help="number of rounds")
    parser.add_argument('--dataset_name', type=str, default="MSSEG", choices=["Messidor", "MSSEG"], help="dataset")
    parser.add_argument('--early-stop', default=5, type=int, help='early stopping')
    parser.add_argument('--training_name', type=str, default="supervised_val_loss",
                        choices=["supervised_train_acc",
                                 "supervised_val_loss",
                                 "supervised_val_acc",
                                 "supervised_train_epoch",], help="training method")
    parser.add_argument('--strategy_name', type=str, default="EntropySampling",
                        choices=["RandomSampling",
                                 "EntropySampling",
                                 "EntropySamplingDropout",
                                 "BALDDropout",
                                 "KCenterGreedy",
                                 "MarginSampling",
                                 "LeastConfidence",
                                 "HybridSampling",
                                 "ClusterMarginSampling"
                                 ], help="query strategy")

    parser.add_argument('--max_skip', type=list, default=[3, 2]) # max skip time length while training
    parser.add_argument('--train_batch', type=int, default=2) # training batchsize
    



    args = parser.parse_args()
    return args

