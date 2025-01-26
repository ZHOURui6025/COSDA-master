
from tqdm import tqdm
from model.COSDA import COSDA
from dataset.dataset import COSDAataset
from torch.utils.data.dataloader import DataLoader

from config.model_config import build_args
from utils.net_utils import set_logger, set_random_seed
from utils.net_utils import compute_h_score, Entropy
from utils_me import *
from pseudo_labelling import *

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

def initalize_memory(model, dataloader):
    # model.eval()
    memory_source_features = torch.zeros(args.source_len, args.embed_feat_dim).cuda()
    # print(memory_source_features.shape)
    memory_source_labels = torch.zeros(args.source_len).long().cuda()
    memory_domain_labels = torch.zeros(args.source_len).long().cuda()

    flag = False
    begin_index = 0
    for imgs_train, _, imgs_label, _ in tqdm(dataloader, ncols=60):
        #load
        images = imgs_train.cuda()
        label = imgs_label.cuda()
        index = [i for i in range(begin_index,  begin_index+images.shape[0])]
        begin_index = begin_index + images.shape[0]
        with torch.no_grad():
            rois = model.backbone_layer(images)
            features_temp, _ = model.feat_embed_layer(rois)
            del _
            if flag:
                memory_source_features[index] = features_temp[0].unsqueeze(0)
                memory_source_labels[index] = label
                flag = False
            else:
                memory_source_features[index] = features_temp
                memory_source_labels[index] = label

            del features_temp
    memory_source_features = memory_source_features[:index[-1]]
    memory_source_labels = memory_source_labels[:index[-1]]
    print("memory module initialization has finished!")
    # memory
    if memory_source_features.shape[0] > args.lll:
        args.lll = memory_source_features.shape[0]
    return memory_source_features, memory_source_labels


def forward_BETA_ce(v, y, intervention, int_v,  int_y, target, args, domain='source'):
    loss_beta_dict = {}
    if domain =='source':
        nll_1= ce_criterion(args, target, y, domain)
        int_nll_1 = -ce_criterion(args, target, int_y, domain='source')
        loss_beta_dict.update(loss_kl_nll=args.lambda_kl*v)
        loss_beta_dict.update(loss_kl_int_nll=args.lambda_kl*int_v)

        loss_sta_s = intervention_loss(intervention, args).mean()
        loss_beta_dict.update(loss_sta_s=args.lambda_sta*loss_sta_s)
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

def forward_EXO(model, memory_source_features, memory_source_labels, mean_unk_proto, hard_label_bank, target_s, train_s, features_target, NUM_K):
    model.train()
    loss_exo_dict = {}

    # delete Top-batch size memory_source_features and memory_source_labels
    memory_source_features = memory_source_features[args.batch_size:]
    memory_source_labels = memory_source_labels[args.batch_size:]
    # add features_source  labels_source to the memory tail
    with torch.no_grad():
            rois = model.backbone_layer(train_s)
            features_temp, _ = model.feat_embed_layer(rois)
    memory_source_features = torch.cat((memory_source_features, features_temp), dim=0)
    memory_source_labels = torch.cat((memory_source_labels, target_s), dim=0)

    # addxu
    # print(memory_source_labels)
    mean_source = CalculateMean(memory_source_features, memory_source_labels, args.known_class)
    #### addxu   # *** MO-Method ***
    if args.da_method == 'MO':
        mean_all = torch.cat((mean_source, mean_unk_proto), dim=0)
        # print(mean_all.shape)
        trans_loss = MO(mean_all, features_target, hard_label_bank, NUM_K)
    else:
        raise ValueError('Method cannot be recognized.')
    loss_exo_dict.update(loss_cl=args.lambda_exo*trans_loss)
    return memory_source_features, memory_source_labels, loss_exo_dict

def train(args, source_train_dataloader, target_train_dataloader, test_dataloader, optimizer, logger, epoch_idx, memory_source_features, memory_source_labels):

    source_loader_iter = iter(source_train_dataloader)
    target_loader_iter = iter(target_train_dataloader)
    all_proto, neg_proto, NUM_K = get_pseudo_label(args, model, target_train_dataloader, new_epoch=True)
    model.train()

    for batch_idx in range(max(len(target_train_dataloader), len(source_train_dataloader))):
        optimizer.zero_grad()
        if batch_idx >= len(source_train_dataloader):
            source_loader_iter = iter(source_train_dataloader)
        if batch_idx >= len(target_train_dataloader):
            target_loader_iter = iter(target_train_dataloader)
        train_s, test_s, target_s, _ = source_loader_iter.next()
        train_t, test_t, target_t, _ = target_loader_iter.next()
        train_s, target_s = train_s.cuda(), target_s.long().cuda(non_blocking=True)
        train_t, target_t = train_t.cuda(), target_t.long().cuda(non_blocking=True)
        test_s = test_s.cuda()
        #source domain：e_s、kl、sta
        loss_dict = {}
        kld_weight = kl_anneal_function(1, epoch_idx, len(source_train_dataloader), batch_idx)
        args.lambda_kl = kld_weight

        rois_s, v_s, rois_c_s, y_s, intervention_s, int_rois_s, int_v_s, int_y_s = model(train_s, sample_gaussian=True, apply_softmax=False)
        loss_beta_s = forward_BETA_ce(v_s, y_s, intervention_s, int_v_s, int_y_s, target_s, args, domain='source')
        loss_dict.update(loss_beta_s)
        model.train()
        rois_t, v_t, rois_c_t, y_t, intervention_t, int_rois_t, int_v_t, int_y_t = model(train_t, sample_gaussian=True, apply_softmax=False)
        virtual_label_bank = get_pseudo_label(args, model, target_train_dataloader, feature_batch=rois_c_t, targets = target_t, new_epoch= False, all_proto=all_proto)
        hard_label_bank = virtual_label_bank
        hard_label_bank[hard_label_bank >= args.known_class] = args.known_class
        model.train()
        if epoch_idx < args.warm_up_epoch:
            loss_beta_t = forward_BETA_ce(v_t, y_t, intervention_t, int_v_t, int_y_t, hard_label_bank, args, domain='target')
            print('----------no cl', model.training)
        else:
            loss_beta_t = forward_BETA_ce(v_t, y_t, intervention_t, int_v_t, int_y_t, hard_label_bank, args, domain='target')
            target_score, predict_target = torch.max(y_t.softmax(-1), 1)
            # args.logger.info('score {}'.format(target_score))
            # print(target_score)
            idx_pseudo1 = target_score > args.confidence_th
            # args.logger.info('mask1 {}'.format(idx_pseudo1))
            idx_pseudo2 = predict_target == hard_label_bank
            combined_mask = idx_pseudo1 & idx_pseudo2
            hard_label_bank = hard_label_bank[combined_mask]
            features_target = rois_c_t[combined_mask]
            true_rest = target_t[combined_mask]
            mask_label = (hard_label_bank < args.known_class)
            hard_label_bank1 = hard_label_bank[mask_label]
            features_target1 = features_target[mask_label]
            true_rest =true_rest[mask_label]
            args.logger.info('number for cl all: {}, number for cl only known: {}, accurate rate: {:.3f}'.format(features_target.shape[0], features_target1.shape[0], sum(hard_label_bank1 == true_rest)/(features_target1.shape[0]+1e-8)))
            if args.if_true_label == 1:
                print('true label')
                label_for_exo = true_rest
            else:
                label_for_exo = hard_label_bank1
                print('pseudo label')
            memory_source_features, memory_source_labels, loss_align_t = forward_EXO(model, memory_source_features, memory_source_labels, neg_proto, label_for_exo, target_s, train_s, features_target1, NUM_K)
            print('----------with cl', model.training)
            loss_dict.update(loss_align_t)
        loss_dict.update(loss_beta_t)

        loss_all = sum(loss for loss in loss_dict.values())

        loss_all.backward()

        optimizer.step()


        iter_idx = epoch_idx * max(len(target_train_dataloader), len(source_train_dataloader))
        lr_scheduler(optimizer, iter_idx, args.epochs*max(len(target_train_dataloader),len(source_train_dataloader)))

        if batch_idx % args.log_interval == 0 and batch_idx != 0:

            args.logger.info(
                '[Epoch: {} {}/{} ({:.0f}%)] [Loss_all: {:.3f}], [lr:{:.4f}] {}'.format(
                    epoch_idx,
                    batch_idx * args.batch_size, args.batch_size * max(len(target_train_dataloader), len(source_train_dataloader)),
                    100. * batch_idx / max(len(target_train_dataloader), len(source_train_dataloader)),
                    loss_all.item(),
                    optimizer.param_groups[0]['lr'],
                    {**process_dict(loss_dict)}
                )
            )
    return memory_source_features, memory_source_labels
    

@torch.no_grad()
def valid(args, model, dataloader, src_flg=False):
    
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
        # print(pred_cls.shape)
        gt_label_stack.append(imgs_label)
        pred_cls_stack.append(pred_cls.cpu())
    
    gt_label_all = torch.cat(gt_label_stack, dim=0) #[N]
    pred_cls_all = torch.cat(pred_cls_stack, dim=0) #[N, C]

    h_score, all_acc, known_acc, unknown_acc, _ = compute_h_score(args, class_list, gt_label_all, pred_cls_all, open_flg, open_thresh=args.w_0)
    return h_score, all_acc, known_acc, unknown_acc


if __name__ == "__main__":
    args = build_args()

    set_random_seed(args.seed)
    criterion_bce = nn.BCELoss()
    criterion_bce_red = nn.BCELoss(reduction='none')
    # SET THE CHECKPOINT     
    args.checkpoint = os.path.join("models_pth", args.dataset, "source_{}".format(args.s_idx),\
                    "source_{}_{}_{}".format(args.source_train_type, args.target_label_type, args.bn_type),
                    "source_checkpoint.pth")
    ##main
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    this_dir = os.path.join(os.path.dirname(__file__), ".")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = COSDA(args)
    model = model.to(device)
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(args.checkpoint)
        raise ValueError("NO SOURCE CHECKPOINT!!!")
    print(args.s_idx, args.t_idx)
    save_dir = os.path.join(this_dir, "checkpoints_glc", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                            args.target_label_type, args.note)
    pth_dir = os.path.join(this_dir, "models_pth", args.dataset, "s_{}_t_{}".format(args.s_idx, args.t_idx),
                            args.target_label_type, args.note)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(pth_dir):
        os.makedirs(pth_dir)
    args.save_dir = save_dir
    args.logger = set_logger(args, log_name="log_adaptation.txt")
    args.logger.info(
                args.note
            )
    param_group = []
    for k, v in model.backbone_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*0.1}]

    for k, v in model.intervener.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    for k, v in model.feat_embed_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    for k, v in model.class_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]

    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    source_data_list = open(os.path.join(args.sourcelist_data_dir, "osda_list.txt"), "r").readlines()
    # print(len(source_data_list))
    source_dataset = COSDAataset(args, args.source_data_dir, source_data_list, d_type="source", preload_flg=True)
    source_dataloader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, drop_last=True)

    target_data_list = open(os.path.join(args.targetlist_data_dir, "osda_list.txt"), "r").readlines()
    target_dataset = COSDAataset(args, args.target_data_dir, target_data_list, d_type="target", preload_flg=True)
    target_train_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=args.num_workers, drop_last=True)
    target_test_dataloader = DataLoader(target_dataset, batch_size=args.batch_size*2, shuffle=False,
                                        num_workers=args.num_workers, drop_last=False)

    args.source_len = len(source_dataset.data_list)
    args.target_len = len(target_dataset.data_list)
    print(args.source_len)

    notation_str =  "\n=======================================================\n"
    notation_str += "   START TRAINING ON THE TARGET:{} BASED ON SOURCE:{}  \n".format(args.t_idx, args.s_idx)
    notation_str += "======================================================="

    args.logger.info(notation_str)
    best_h_score = 0.0
    best_known_acc = 0.0
    best_unknown_acc = 0.0
    best_epoch_idx = 0
    best_all_acc = 0.0
    memory_source_features, memory_source_labels= initalize_memory(model, source_dataloader)
    criterion_bce = nn.BCELoss()
    for epoch_idx in tqdm(range(args.epochs), ncols=60):

        args.epoch_idx = epoch_idx
        # Train on target

        memory_source_features, memory_source_labels = train(args, source_dataloader, target_train_dataloader, target_test_dataloader, optimizer, args.logger, epoch_idx, memory_source_features, memory_source_labels)
        # args.logger.info("Epoch: {}/{}, train_e_s_loss:{:.3f}, train_d_t_loss:{:.3f},\n\
        #                    train_kl_loss:{:.3f}, train_cl_loss:{:.3f}, ".format(epoch_idx+1, args.epochs,
        #                                 loss_dict["e_s_loss"], loss_dict["d_t_loss"], loss_dict["kl_loss"], loss_dict['cl_loss']))

        # Evaluate on target
        hscore, all_acc, knownacc, unknownacc = valid(args, model, target_test_dataloader, src_flg=False)
        args.logger.info("Epoch {} Current: H-Score:{:.3f}, All{:.3f} KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(epoch_idx, hscore, all_acc, knownacc, unknownacc))
        # hscore, all_acc, knownacc, unknownacc = valid1(args, model, target_test_dataloader, src_flg=False)
        # args.logger.info("Epoch {} Current1111: H-Score:{:.3f}, OS* {:.3f} KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(epoch_idx, hscore,all_acc, knownacc, unknownacc))

        if hscore >= best_h_score:
            best_h_score = hscore
            best_known_acc = knownacc
            best_unknown_acc = unknownacc
            best_epoch_idx = epoch_idx
            best_all_acc = all_acc

            checkpoint_file = "{}_best_target_checkpoint.pth".format(args.dataset)
            # torch.save({
            #     "epoch": epoch_idx,
            #   "model_state_dict": model.state_dict()}, os.path.join(pth_dir, checkpoint_file))

        args.logger.info("Best Epoch {}  : H-Score:{:.3f}, AllACC:{:.3f} KnownAcc:{:.3f}, UnknownAcc:{:.3f}".format(best_epoch_idx,best_h_score, best_all_acc, best_known_acc, best_unknown_acc))
