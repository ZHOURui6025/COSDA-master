import shutil
import numpy as np
from sklearn.metrics import accuracy_score
import torch
import logging
import torch.nn as nn
import gc
import torch.nn.functional as F
from collections import Counter
from scipy.optimize import linear_sum_assignment


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        """

        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, targets):
        """

        #计算weight
        category_counts = Counter(targets.cpu().detach().numpy())
        weight = [len(targets) / float(j) if j > 0 else 0.0 for i, j in category_counts.items()]
        weight = [i /max(weight) for i in weight]
        weight = [weight[i] for i in targets]
        print(len(weight))

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1).cuda()
        loss = -(targets * log_probs).sum(dim=-1) #* weight
        print(loss.shape)
        loss = weight *loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class AccuracyCounter:
    def __init__(self):
        self.Ncorrect = 0.0
        self.Ntotal = 0.0
        
    def addOntBatch(self, predict, label):
        assert predict.shape == label.shape
        correct_prediction = np.equal(np.argmax(predict, 1), np.argmax(label, 1))
        Ncorrect = np.sum(correct_prediction.astype(np.float32))
        Ntotal = len(label)
        self.Ncorrect += Ncorrect
        self.Ntotal += Ntotal
        return Ncorrect / Ntotal
    
    def reportAccuracy(self):
        return np.asarray(self.Ncorrect, dtype=float) / np.asarray(self.Ntotal, dtype=float)

def cal_acc(gt_list, predict_list, num, writer, epoch, name):
    acc_sum = 0
    accu_set = {}
    for n in range(num):
        y = []
        pred_y = []
        for i in range(len(gt_list)):
            gt = gt_list[i]
            predict = predict_list[i]
            if gt == n:
                y.append(gt)
                pred_y.append(predict)
        print('{}: {:4f} {}/{}'.format(n if n != (num - 1) else 'Unk', accuracy_score(y, pred_y), round(accuracy_score(y, pred_y) * len(y)), len(y)))
        if n != (num - 1):
            accu_set[name[n]] = accuracy_score(y, pred_y)
        if n == (num - 1):
            print ('OS*: {:4f}'.format(acc_sum / (num - 1)))
            OS_star = acc_sum / (num - 1)
            unk = accuracy_score(y, pred_y)
            writer.add_scalar('Known_ave', acc_sum / (num - 1), epoch)
            writer.add_scalar('Unknown', accuracy_score(y, pred_y), epoch)
        acc_sum += accuracy_score(y, pred_y)
    writer.add_scalars('Known_class', accu_set, epoch)
    print ('OS: {:4f}'.format(acc_sum / num))
    writer.add_scalar('Acc_ave', acc_sum / num, epoch)
    print ('Overall Acc : {:4f}'.format(accuracy_score(gt_list, predict_list)))
    writer.add_scalar('Over_all', accuracy_score(gt_list, predict_list), epoch)
    return OS_star, unk

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def aToBSheduler(step, A, B, gamma=10, max_iter=10000):
    ans = A + (2.0 / (1 + np.exp(- gamma * step * 1.0 / max_iter)) - 1.0) * (B - A)
    return float(ans)

def list2tensor(list):
    return torch.cat(list, dim=0)

def process_dict(dict, points=3):
    for key in dict.keys():
        dict[key] = np.round(dict[key].item(), points)
    return dict

@torch.no_grad()
def ema_model_update(model,ema_model,ema_m):
    for param_train, param_eval in zip(model.parameters(), ema_model.parameters()):
        param_eval.copy_(param_eval*ema_m + param_train.detach()*(1 - ema_m))
    for buffer_train, buffer_eval in zip(model.buffers(), ema_model.buffers()):
        buffer_eval.copy_(buffer_train)

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def enable_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def opt_step(opt_list):
    for opt in opt_list:
        opt.step()
    for opt in opt_list:
        opt.zero_grad()

def print_and_log(data):
    print(data)
    logging.info(data)


def condition_prior(scale, label, dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean = ((label-scale[0])/(scale[1]-scale[0])).reshape(-1, 1).repeat(1, dim) #torch.ones(label.size()[0], dim)*label
    var = torch.ones(label.size()[0], dim)
    return mean.to(device), var.to(device)

def kl_normal(qm, qv, pm, pv):
    """
        Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
        sum over the last dimension

        Args:
            qm: tensor: (batch, dim): q mean
            qv: tensor: (batch, dim): q variance
            pm: tensor: (batch, dim): p mean
            pv: tensor: (batch, dim): p variance

        Return:
            kl: tensor: (batch,): kl between each sample
        """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1) #(qm - pm).pow(2)#
    kl = element_wise.mean()
    #print("log var1", qv)
    return kl

def intervention_loss(intervention, args):
    return torch.norm(torch.pow(intervention, 2)-args.int_epsilon)

# def intervention_loss(intervention, args):
#     return

def kl_loss(m, v, y, args):
    if args.prior_type == 'conditional':
        pm, pv = condition_prior([0, args.known_class+1], y, m.size()[1])
    else:
        pm, pv = torch.zeros_like(m), torch.ones_like(m)
    return kl_normal(m, v * 0.0001, pm, pv * 0.0001)

class SoftLoss(object):
    def __call__(self, outputs_u, targets_u, epoch, max_epochs=30, lambda_u=75):
        probs_u = torch.softmax(outputs_u, dim=1)

        # Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lu

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)

        if self.reduction:
            return loss.mean()
        else:
            return loss

class LeastSquareLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(LeastSquareLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = torch.mean((log_probs - targets)**2)

        return loss

def CalculateMean(features, labels, class_num):
    N = features.size(0)#
    C = class_num#
    A = features.size(1)#

    avg_CxA = torch.zeros(C, A).cuda()#
    NxCxFeatures = features.view(N, 1, A).expand(N, C, A)#

    onehot = torch.zeros(N, C).cuda()#[N,C]
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1.0

    del onehot
    gc.collect()
    for c in range(class_num):
        c_temp = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
        c_temp = torch.sum(c_temp, dim=0)
        avg_CxA[c] = c_temp / Amount_CxA[c]
    return avg_CxA.detach()



def optimal_assignment(prob_matrix):
    cost_matrix = -prob_matrix.cpu().numpy()  # 转换为NumPy数组并取负值
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    return col_indices



def MO(mean_source_up1, features_target1, hard_label_bank, class_num):

    ce_criterion = nn.CrossEntropyLoss()
    N = features_target1.size(0)
    C = class_num
    A = features_target1.size(1)

    norm_features_target1 = features_target1.norm(dim=1, keepdim=True)  # (N, 1)
    norm_features_target1[norm_features_target1 == 0] = 1
    features_target = features_target1 / norm_features_target1           # (N, A)

    norm_mean_source_up1 = mean_source_up1.norm(dim=1, keepdim=True)
    norm_mean_source_up1[norm_mean_source_up1 == 0] = 1
    mean_source_up = mean_source_up1 / norm_mean_source_up1

    predict_gnn_target = hard_label_bank # (N,)

    ## calculate g_mu
    sourceMean_NxCxA = mean_source_up.expand(N, C, A)        # (N, C, A)
    sourceMean_NxAxC = sourceMean_NxCxA.permute(0, 2, 1)     # (N, A, C)
    features_target_Nx1xA = features_target.unsqueeze(1)     # (N, 1, A)
    g_mu = torch.bmm(features_target_Nx1xA, sourceMean_NxAxC).squeeze(1)  # (N, C)

    aug_result = g_mu# 


    loss = ce_criterion(aug_result, predict_gnn_target)

    return loss

def spld(loss, group_member_ship, lam, gamma):
    groups_labels = np.array(list(set(group_member_ship)))
    b = len(groups_labels)
    selected_idx = []
    selected_score = [0] * len(loss)
    for j in range(b):
        idx_in_group = np.where(group_member_ship == groups_labels[j])[0]
        # print(idx_in_group)
        loss_in_group = []
        # print(type(idx_in_group))
        for idx in idx_in_group:
            loss_in_group.append(loss[idx])
        idx_loss_dict = dict()
        for i in idx_in_group:
            idx_loss_dict[i] = loss[int(i)]
        sorted_idx_in_group = sorted(idx_loss_dict.keys(), key=lambda s: idx_loss_dict[s])
        sorted_idx_in_group_arr = np.array(sorted_idx_in_group)

        # print(sorted_idx_in_group_arr)

        for (i, ii) in enumerate(sorted_idx_in_group_arr):
            if loss[ii] < (lam + gamma / (np.sqrt(i + 1) + np.sqrt(i))):
                selected_idx.append(ii)
            else:
                pass
            selected_score[ii] = loss[ii] - (lam + gamma / (np.sqrt(i + 1) + np.sqrt(i)))
    selected_idx_arr = np.array(selected_idx)
    selected_idx_and_new_loss_dict = dict()
    for idx in selected_idx_arr:
        selected_idx_and_new_loss_dict[idx] = selected_score[idx]

    sorted_idx_in_selected_samples = sorted(selected_idx_and_new_loss_dict.keys(),
                                            key=lambda s: selected_idx_and_new_loss_dict[s])

    sorted_idx_in_selected_samples_arr = np.array(sorted_idx_in_selected_samples)
    return sorted_idx_in_selected_samples_arr


def spl(max_cls, lam):
    selected_idx = torch.zeros_like(max_cls)
    for idx, val in enumerate(max_cls):
        if val >= lam:
            selected_idx[idx] = 1
    # print(sum(selected_idx))
    return selected_idx

def ce_criterion(args, hard_label, output, domain):
    if args.adaptation_type == "smooth":
        return CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1, reduction=True)(output, hard_label)
    elif args.adaptation_type == "vanilla":
        return CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.0, reduction=True)(output, hard_label)

def bce_loss(output, target):
    output_neg = 1 - output
    target_neg = 1 - target
    result = torch.mean(target * torch.log(output + 1e-6))
    result += torch.mean(target_neg * torch.log(output_neg + 1e-6))
    return -torch.mean(result)


def kl_anneal_function(anneal_cap, epoch, times, step, total_annealing_step=10000):

    return min(1, 2*(epoch*times+step)/ total_annealing_step)

def cl_anneal_function(anneal_cap, step, total_annealing_step=10000):

    return min(anneal_cap, step / total_annealing_step)


def spl_anneal_function(anneal_cap, step, total_annealing_step):

    return max(0, 1- (step / total_annealing_step))
import os
import psutil




def get_cpu_mem_info():
    """
    """
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    return mem_total, mem_free, mem_process_used

def lr_scheduler(args, optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer
