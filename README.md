# COSDA-master
Pytorch Implementation of **COSDA: Counterfactual-based Susceptibility Risk Framework for Open-Set Domain Adaptation**.

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

python train_b.py --dataset Office --a_idx 0 --b_idx 1 --lr 0.001 --batch_size 32 --epochs 50 
python train_b.py --dataset Office --a_idx 0 --b_idx 2 --lr 0.001 --batch_size 32 --epochs 50
python train_b.py --dataset Office --a_idx 1 --b_idx 0 --lr 0.001 --batch_size 32 --epochs 50 
python train_b.py --dataset Office --a_idx 1 --b_idx 2 --lr 0.001 --batch_size 32 --epochs 50 
python train_b.py --dataset Office --a_idx 2 --b_idx 0 --lr 0.001 --batch_size 32 --epochs 50  
python train_b.py --dataset Office --a_idx 2 --b_idx 1 --lr 0.001 --batch_size 32 --epochs 50 
```

Run DomainNet
 ```
python train_a.py --dataset DomainNet --a_idx 0 --lr 0.005 --epochs 30
python train_a.py --dataset DomainNet --a_idx 1 --lr 0.005 --epochs 30
python train_a.py --dataset DomainNet --a_idx 2 --lr 0.005 --epochs 30
python train_b_noddp.py --dataset DomainNet --a_idx 0 --b_idx 1 --lr 0.0005 --batch_size 32 --epochs 50 
python train_b_noddp.py --dataset DomainNet --a_idx 0 --b_idx 2 --lr 0.0005 --batch_size 32 --epochs 50 
python train_b_noddp.py --dataset DomainNet --a_idx 1 --b_idx 0 --lr 0.0005 --batch_size 32 --epochs 50 
python train_b_noddp.py --dataset DomainNet --a_idx 1 --b_idx 2 --lr 0.0005 --batch_size 32 --epochs 50
python train_b_noddp.py --dataset DomainNet --a_idx 2 --b_idx 0 --lr 0.0005 --batch_size 32 --epochs 50 
python train_b_noddp.py --dataset DomainNet --a_idx 2 --b_idx 1 --lr 0.0005 --batch_size 32 --epochs 50 
```




Run COSDA with CLIP
```
CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.run --nproc_per_node 2 --master_port 10018  train_a_ddp.py --dataset tracking6 --a_idx 0 --target_label_type OSDA --lr 0.005 --epochs 10   --backbone_arch clip
CUDA_VISIBLE_DEVICE=4 python train_b_noddp.py --dataset tracking6 --a_idx 0 --b_idx 1 --lr 0.0005 --epochs 50  --backbone_arch clip
CUDA_VISIBLE_DEVICE=4 python train_b_noddp.py --dataset tracking6 --a_idx 0 --b_idx 2 --lr 0.0005 --epochs 50 --backbone_arch clip

```
