import os
import shutil
import torch
import numpy as np 
from tqdm import tqdm
from model.COSDA import COSDA
from dataset.dataset import SFUniDADataset
from torch.utils.data.dataloader import DataLoader 

from config.model_config import build_args
from utils.net_utils import set_logger, set_random_seed
from utils.net_utils import compute_h_score, CrossEntropyLabelSmooth
from utils_me import *
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def forward_BETA_ce(v, y, intervention, int_v,  int_y, target, args, domain='source'):
    loss_beta_dict = {}
    if domain =='source':
        nll_1= ce_criterion(args, target, y, domain)
        int_nll_1 = -ce_criterion(args, target, int_y, domain='source')
        # loss_kl_s = kl_loss(rois_c, v, y, args).mean() + kl_loss(int_rois, v, y, args).mean()
        loss_beta_dict.update(loss_kl_nll=args.lambda_kl*v)
        loss_beta_dict.update(loss_kl_int_nll=args.lambda_kl*int_v)

        loss_sta_s = intervention_loss(intervention, args).mean()
        loss_beta_dict.update(loss_sta_s=args.lambda_sta*loss_sta_s)
        # beta = torch.ones_like(nll_1)#get_beta_divergence()
        loss_beta_dict.update(loss_e_s=args.lambda_beta_e*((nll_1+args.lambda_int*int_nll_1).mean()))
    else:
        nll_1= ce_criterion(args, target, y, domain)
        int_nll_1 = -ce_criterion(args, target, int_y, domain)
        loss_sta_t = intervention_loss(intervention, args).mean()
        loss_beta_dict.update(loss_sta_t=args.lambda_sta*loss_sta_t)
        loss_beta_dict.update(loss_kl_nll=args.lambda_kl*v)
        loss_beta_dict.update(loss_kl_int_nll=args.lambda_kl*int_v)
        loss_beta_dict.update(loss_d_t=args.lambda_beta_d*((nll_1+args.lambda_int*int_nll_1).mean()))

    return loss_beta_dict

def train(args, model, dataloader, criterion, optimizer, epoch_idx=0.0):
    model.train()
    loss_stack = []
    
    iter_idx = epoch_idx * len(dataloader)
    iter_max = args.epochs * len(dataloader)
    index = -1
    for imgs_train, _, imgs_label, batch_idx in tqdm(dataloader, ncols=60):
        index +=1
        kld_weight = kl_anneal_function(1, epoch_idx, len(dataloader), index)
        args.lambda_kl = kld_weight
        iter_idx += 1
        imgs_train = imgs_train.cuda()
        imgs_label = imgs_label.cuda()
        soft_source_psd_label_bank = torch.zeros(imgs_train.shape[0], args.known_class+1).scatter_(1, imgs_label.unsqueeze(1).cpu(), 1)
        rois_s, v_s, rois_c_s, y_s, intervention_s, int_rois_s, int_v_s, int_y_s = model(imgs_train, sample_gaussian=True, apply_softmax=False)

        # print(target_s, soft_source_psd_label_bank)
        loss_beta_s = forward_BETA_ce(v_s, y_s, intervention_s, int_v_s, int_y_s, imgs_label, args, domain='source')
        
        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.zero_grad()
        loss = sum(loss for loss in loss_beta_s.values())
        loss.backward()
        optimizer.step()

        loss_stack.append(loss.cpu().item())
        
    train_loss = np.mean(loss_stack)
    
    return train_loss

@torch.no_grad()
def test(args, model, dataloader, src_flg=True):
    model.eval()
    gt_label_stack = []
    pred_cls_stack = []

    if src_flg:
        class_list = args.source_class_list
        open_flg = False
    else:
        class_list = args.target_class_list
        open_flg = args.target_private_class_num > 0

    for _, imgs_test, imgs_label, _ in tqdm(dataloader, ncols=60):

        imgs_test = imgs_test.cuda()
        rois = model.backbone_layer(imgs_test)
        feature, _ = model.feat_embed_layer(rois)
        pred_cls = model.class_layer(feature, apply_softmax=True)
        # print('test', pred_cls.shape)
        gt_label_stack.append(imgs_label)
        pred_cls_stack.append(pred_cls.cpu())

    gt_label_all = torch.cat(gt_label_stack, dim=0) #[N]
    pred_cls_all = torch.cat(pred_cls_stack, dim=0) #[N, C]

    h_score, all_acc, known_acc, unknown_acc, _ = compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flg, open_thresh=0.50)

    return h_score, all_acc, known_acc, unknown_acc

def main(args):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    this_dir = os.path.join(os.path.dirname(__file__), ".")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = COSDA(args)
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        save_dir = os.path.dirname(args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])

    save_dir = os.path.join(this_dir, "checkpoints_glc", args.dataset, "source_{}".format(args.s_idx),
                            "source_{}_{}_{}".format(args.source_train_type, args.target_label_type, args.bn_type))
    pth_dir = os.path.join(this_dir, "models_pth", args.dataset, "source_{}".format(args.s_idx),
                            "source_{}_{}_{}".format(args.source_train_type, args.target_label_type, args.bn_type))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(pth_dir):
        os.makedirs(pth_dir)
            
    model = model.to(device)
    args.save_dir = save_dir     
    logger = set_logger(args, log_name=args.backbone_arch+"log_source_training.txt")
    
    params_group = []
    for k, v in model.backbone_layer.named_parameters():
        params_group += [{"params":v, 'lr':args.lr*0.1}]
    for k, v in model.intervener.named_parameters():
        params_group += [{"params":v, 'lr':args.lr}]
    for k, v in model.feat_embed_layer.named_parameters():
        params_group += [{"params":v, 'lr':args.lr}]
    for k, v in model.class_layer.named_parameters():
        params_group += [{"params":v, 'lr':args.lr}]
    
    optimizer = torch.optim.SGD(params_group)
    optimizer = op_copy(optimizer)
    
    source_data_list = open(os.path.join(args.sourcelist_data_dir, "image_unida_list.txt"), "r").readlines()
    source_dataset = SFUniDADataset(args, args.source_data_dir, source_data_list, d_type="source", preload_flg=True)
    source_dataloader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True, pin_memory=False)
    
    target_dataloader_list = []
    for idx in range(len(args.targetlist_domain_dir_list)):
        targetlist_data_dir = args.targetlist_domain_dir_list[idx]
        target_data_dir = args.target_domain_dir_list[idx]
        target_data_list = open(os.path.join(targetlist_data_dir, "image_unida_list.txt"), "r").readlines()
        target_dataset = SFUniDADataset(args, target_data_dir, target_data_list, d_type="target", preload_flg=False)
        target_dataloader_list.append(DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, drop_last=False, pin_memory=False))
    
    if args.source_train_type == "smooth":
        criterion = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1, reduction=True)
    elif args.source_train_type == "vanilla":
        criterion = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.0, reduction=True)
    else:
        raise ValueError("Unknown source_train_type:", args.source_train_type) 
    
    notation_str =  "\n=================================================\n"
    notation_str += "    START TRAINING ON THE SOURCE:{} == {}         \n".format(args.s_idx, args.target_label_type)
    notation_str += "================================================="
    
    logger.info(notation_str)
    
    for epoch_idx in tqdm(range(args.epochs), ncols=60):
        
        train_loss = train(args, model, source_dataloader, criterion, optimizer, epoch_idx)
        logger.info("Epoch:{}/{} train_loss:{:.3f}".format(epoch_idx, args.epochs, train_loss))
        
        if epoch_idx % 1 == 0:
            # EVALUATE ON SOURCE
            source_h_score, all_acc, source_known_acc, source_unknown_acc = test(args, model, source_dataloader, src_flg=True)
            logger.info("EVALUATE ON SOURCE: H-Score:{:.3f}, KnownAcc:{:.3f}, UnknownAcc:{:.3f}".\
                            format(source_h_score, source_known_acc, source_unknown_acc))
            if args.dataset == "VisDA":
                logger.info("VISDA PER_CLS_ACC:")
                # logger.info(src_per_cls_acc)
           
        checkpoint_file = args.backbone_arch+"_latest_source_checkpoint.pth"
        torch.save({
            "epoch":epoch_idx,
            "model_state_dict":model.state_dict()}, os.path.join(pth_dir, checkpoint_file))
        
    for idx_i, item in enumerate(args.target_domain_list):
        notation_str =  "\n=================================================\n"
        notation_str += "        EVALUATE ON THE TARGET:{}                  \n".format(item)
        notation_str += "================================================="
        logger.info(notation_str)
        
        hscore, allacc, knownacc, unknownacc= test(args, model, target_dataloader_list[idx_i], src_flg=False)
        logger.info("H-Score:{:.3f}, AllACC: {:.3f} KnownAcc:{:.3f}, UnknownACC:{:.3f}".format(hscore, allacc, knownacc, unknownacc))
    
if __name__ == "__main__":
    args = build_args()
    print(args.bn_type)
    set_random_seed(args.seed)
    main(args)
