## Adaptive adversarial 
### Introduction
This project is the code for paper Is Complete Labeling Necessary? An Active Learning Framework for Longitudinal Medical Imaging based on python and pytorch framework.
  

### Requirements  
The main package and version of the python environment are as follows
```
# Name                    Version         
python                    3.8.5                    
pytorch                   1.10.1         
torchvision               0.11.2         
cudatoolkit               10.2.89       
cudnn                     7.6.5           
matplotlib                3.3.2              
numpy                     1.19.2        
opencv                    4.6.0.66         
pandas                    1.1.3               
scikit-learn              0.23.2                
tqdm                      4.50.2             
```  

The above environment is successful when running the code of the project. Pytorch has very good compatibility. Thus, I suggest that try to use the existing pytorch environment firstly.

---  
## Usage 
### 1) Download Project 

Running```git clone https://github.com/activelearning2022/adversarial_active_learning.git```  
The project structure and intention are as follows : 
```
Adversarial active learning			# Source code		
    ├── seed.py			 	                                          # Set up random seed
    ├── query_strategies		                                    # All query_strategies
    │   ├── bayesian_active_learning_disagreement_dropout.py	  # Deep bayesian query method
    │   ├── margin_sampling.py      # Margin-based query method
    │   ├── least_confidence.py      # least_confidence-based query method
    │   ├── entropy_sampling.py		                              # Entropy based query method
    │   ├── entropy_sampling_dropout.py		                      # Entropy based MC dropout query method
    │   ├── random_sampling.py		                              # Random selection
    │   ├── kcenter_greedy.py                                      # Coreset selection
    │   ├── cluster_margin_sampling.py                                      # Cluster-margin selection
    │   ├── hybrid_sampling.py                                      # Hybrid selection
    │   ├── strategy.py                                         # Functions needed for query strategies
    ├── data.py	                                                # Prepare the dataset & initialization and update for training dataset
    ├── handlers.py                                             # Get dataloader for the dataset
    ├── main.py			                                            # An example for code utilization, including the whole process of active learning
    ├── nets.py		                                              # Training models and methods needed for query method
    ├── config.py                                    # Configurations
    ├── visualization.py                                    # Visualization for AL sample selection
    ├── supervised_baseline.py	                                # An example for supervised learning traning process
    └── utils.py			                                          # Important setups including network, dataset, hyperparameters...
```
### 2) Datasets preparation 
1. Download the datasets from the official address:
   
   Messidor: https://portal.fli-iam.irisa.fr/msseg-2/
   
   
2. Modify the data folder path for specific dataset in `data.py`

### 3) Run Active learning process 
Please confirm the configuration information in the [`utils.py`]
```
  python main.py \
      --n_round 100 \
      --n_query 50 \
      --n_init_labeled 100 \
      --traning_method supervised_val_loss \
      --strategy_name RandomSampling \
      --seed 42
```
The training results will be saved to the corresponding directory(save name) in `performance.csv`.  
You can also run `supervised_baseline.py` by
```
python supervised_baseline.py
```

## Visualization
1 Active learning performance visualization  
Set the criterion for visualization then you can run `visualization.py` to visualize the selected AL sample distribution.

