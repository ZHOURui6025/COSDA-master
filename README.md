# COSDA-master
Pytorch Implementation of **COSDA: Counterfactual-based Susceptibility Risk Framework for Open-Set Domain Adaptation**.

# Introduction
Open-Set Domain Adaptation (OSDA) aims to transfer knowledge from the labeled source domain to the unlabeled target domain that contains unknown categories, thus facing the challenges of domain shift and unknown category recognition. While recent works have demonstrated the potential of causality for domain alignment, little exploration has been conducted on causal-inspired theoretical frameworks for OSDA. To fill this gap, we introduce the concept of _Susceptibility_ and propose a novel **C**ounterfactual-based susceptibility risk framework for **OSDA**, termed **COSDA**. 
   Specifically, COSDA consists of three novel components: (i) a _Susceptibility Risk Estimator (SRE)_ for capturing causal information, along with comprehensive derivations of the computable theoretical upper bound, forming a risk minimization framework under the OSDA paradigm; (ii) a _Contrastive Feature Alignment (CFA)_ module, which is theoretically proven based on mutual information to satisfy the \textit{Exogeneity} assumption and facilitate cross-domain feature alignment; (iii) a _Virtual Multi-Unknown Category Prototype (VMP)_ pseudo-labeling strategy, providing label information by measuring how similar samples are to known and multiple virtual unknown category prototypes, thereby assisting in open-set recognition and intra-category discriminative feature learning. Extensive experiments on both benchmark and synthetic datasets demonstrate that our approach achieves state-of-the-art performance.
![image](https://github.com/ZHOURui6025/COSDA-master/blob/master/method.png)


# Requirements
- Python 3.7
- pytorch 1.10.2
- numpy 1.21.2
- tqdm 4.62.3
- faiss

To install dependencies run
 ```
conda env create -f environment.yml
```

# Methods

- COSDA
- COSDA-CLIP

# Commands
 Run Office-31
 ```
python train_a.py --dataset Office --a_idx 0 --lr 0.01 --epochs 30
python train_a.py --dataset Office --a_idx 1 --lr 0.01 --epochs 30
python train_a.py --dataset Office --a_idx 2 --lr 0.01 --epochs 30

python train_b.py --dataset Office --a_idx 0 --b_idx 1 --lr 0.001 --batch_size 32 --epochs 50  --lambda_exo 1 --lambda_beta_d 1 --lambda_beta_e 0.2 
python train_b.py --dataset Office --a_idx 0 --b_idx 2 --lr 0.001 --batch_size 32 --epochs 50  --lambda_exo 1  --lambda_beta_d 1 --lambda_beta_e 0.2 
python train_b.py --dataset Office --a_idx 1 --b_idx 0 --lr 0.001 --batch_size 32 --epochs 50 --lambda_exo 1 --lambda_beta_d 1 --lambda_beta_e 0.2 
python train_b.py --dataset Office --a_idx 1 --b_idx 2 --lr 0.001 --batch_size 32 --epochs 50 --lambda_exo 1  --lambda_beta_d 1 --lambda_beta_e 0.2 
python train_b.py --dataset Office --a_idx 2 --b_idx 0 --lr 0.001 --batch_size 32 --epochs 50  --lambda_exo 1 --lambda_beta_d 1  --lambda_beta_e 0.2 
python train_b.py --dataset Office --a_idx 2 --b_idx 1 --lr 0.001 --batch_size 32 --epochs 50  --lambda_exo 1 --lambda_beta_d 1  --lambda_beta_e 0.2 
```

Run DomainNet
 ```
python train_a.py --dataset DomainNet --a_idx 0 --lr 0.001 --epochs 30
python train_a.py --dataset DomainNet --a_idx 1 --lr 0.001 --epochs 30
python train_a.py --dataset DomainNet --a_idx 2 --lr 0.001 --epochs 30
python train_b.py --dataset DomainNet --a_idx 0 --b_idx 1 --lr 0.0001 --batch_size 32 --epochs 50  --lambda_exo 1  --lambda_beta_d 1 --lambda_beta_e 0.2 
python train_b.py --dataset DomainNet --a_idx 0 --b_idx 2 --lr 0.0001 --batch_size 32 --epochs 50  --lambda_exo 1  --lambda_beta_d 1 --lambda_beta_e 0.2 
python train_b.py --dataset DomainNet --a_idx 1 --b_idx 0 --lr 0.0001 --batch_size 32 --epochs 50 --lambda_exo 1 --lambda_beta_d 1 --lambda_beta_e 0.2 
python train_b.py --dataset DomainNet --a_idx 1 --b_idx 2 --lr 0.0001 --batch_size 32 --epochs 50 --lambda_exo 1  --lambda_beta_d 1 --lambda_beta_e 0.2
python train_b.py --dataset DomainNet --a_idx 2 --b_idx 0 --lr 0.0001 --batch_size 32 --epochs 50  --lambda_exo 1  --lambda_beta_d 1  --lambda_beta_e 0.2 
python train_b.py --dataset DomainNet --a_idx 2 --b_idx 1 --lr 0.0001 --batch_size 32 --epochs 50  --lambda_exo 1  --lambda_beta_d 1  --lambda_beta_e 0.2
```




Run Distributed Training
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 2 --master_port 10000  python train_a_ddp.py --dataset DomainNet --a_idx 0 --lr 0.01 --epochs 30 --backbone_arch clip
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 2 --master_port 10001  train_b_ddp.py --dataset DomainNet --a_idx 0 --b_idx 1 --lr 0.0001 --batch_size 64 --warm_up_epoch 0 --epochs 50 --lambda_exo 1 --lambda_beta_d 1 --da_method MO --lambda_beta_e 0.2 --seed 0 --backbone_arch clip
```
