# COSDA-master
Pytorch Implementation of **COSDA: Counterfactual-based Susceptibility Risk Framework for Open-Set Domain Adaptation**.

## Introduction
Open-Set Domain Adaptation (OSDA) aims to transfer knowledge from the labeled source domain to the unlabeled target domain that contains unknown categories, thus facing the challenges of domain shift and unknown category recognition. While recent works have demonstrated the potential of causality for domain alignment, little exploration has been conducted on causal-inspired theoretical frameworks for OSDA. To fill this gap, we introduce the concept of _Susceptibility_ and propose a novel **C**ounterfactual-based susceptibility risk framework for **OSDA**, termed **COSDA**. 
   Specifically, COSDA consists of three novel components: (i) a _Susceptibility Risk Estimator (SRE)_ for capturing causal information, along with comprehensive derivations of the computable theoretical upper bound, forming a risk minimization framework under the OSDA paradigm; (ii) a _Contrastive Feature Alignment (CFA)_ module, which is theoretically proven based on mutual information to satisfy the \textit{Exogeneity} assumption and facilitate cross-domain feature alignment; (iii) a _Virtual Multi-Unknown Category Prototype (VMP)_ pseudo-labeling strategy, providing label information by measuring how similar samples are to known and multiple virtual unknown category prototypes, thereby assisting in open-set recognition and intra-category discriminative feature learning. Extensive experiments on both benchmark and synthetic datasets demonstrate that our approach achieves state-of-the-art performance.
![image](https://github.com/ZHOURui6025/COSDA-master/blob/master/method.png)


## Requirements
- Python 3.7
- pytorch 1.10.2
- numpy 1.21.2
- tqdm 4.62.3
- faiss

**Full version of code is available after publication.**
