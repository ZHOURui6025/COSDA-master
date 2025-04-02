import os 
import argparse
import torch


def build_args():
    
    parser = argparse.ArgumentParser("This script is used to Source-free Universal Domain Adaptation")
    
    parser.add_argument("--dataset", type=str, default="Office")
    parser.add_argument("--backbone_arch", type=str, default="resnet50")
    parser.add_argument("--embed_feat_dim", type=int, default=256)
    parser.add_argument("--a_idx", type=int, default=0)
    parser.add_argument("--b_idx", type=int, default=1)
    parser.add_argument("--s_idx", type=int, default=0)
    parser.add_argument("--t_idx", type=int, default=1)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--epochs", default=10, type=int)
    
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    # we set lam_psd to 0.3 for Office and VisDA, 1.5 for OfficeHome and DomainNet
    # parser.add_argument("--lam_psd", default=0.3, type=float)
    parser.add_argument("--lam_knn", default=1.0, type=float)
    parser.add_argument("--local_K", default=4, type=int)
    parser.add_argument("--w_0", default=0.55, type=float)
    parser.add_argument("--rho", default=0.75, type=float)
    
    parser.add_argument("--source_train_type", default="smooth", type=str, help="vanilla, smooth")
    parser.add_argument("--target_label_type", default="OSDA", type=str)
    parser.add_argument("--target_private_class_num", default=None, type=int)
    parser.add_argument("--note", default="COSDA")
    parser.add_argument('--int_epsilon', type=float, default=0.1, help='L_sta intervention below bound (default: 0.1)')
    parser.add_argument('--lambda_sta', type=float, default=0.001, help='hyperparameter for loss-sta weight (default: 1)')
    parser.add_argument('--lambda_beta_e', type=float, default=0.2, help='hyperparameter for loss-beta weight (default: 1)')
    parser.add_argument('--lambda_beta_d', type=float, default=1, help='hyperparameter for loss-beta weight (default: 1)')
    parser.add_argument('--lambda_int', type=float, default=0.0001, help='hyperparameter for sus (default: 0.001)')
    # parser.add_argument('--lambda_reg', type=float, default=0.1, help='hyperparameter for intervention (default: 0.1)')
    parser.add_argument('--lambda_kl', type=float, default=0.1, help='hyperparameter for loss-kl weight (default: 1)')
    parser.add_argument('--lambda_exo', type=float, default=1, help='hyperparameter for loss-exo weight (default: 1)')
    parser.add_argument('--mlp_width', type=int,default=128, help='dimension for MLP')
    parser.add_argument('--mlp_dropout', type=float,default=0., help='dropput parameter for MLP', choices=[0., 0.1, 0.5])
    parser.add_argument('--mlp_depth', type=int,default=3, help='layer number for MLP')
    parser.add_argument('--pl_method', type=str,default='Topk3_distance', help='Top2k_distance', choices=['temperature_scalling','Topk3_distance','Top2k_distance', 'glc'])
    parser.add_argument('--T', type=float,default=0.5, help='Temperature')
    parser.add_argument('--da_method', type=str, default='MO', help='method for domain alignment',choices=['MO','MODU'])
    # parser.add_argument('--dim', type=int, default=256, help='dimension for feature')
    parser.add_argument('--lll', type=int, default=1000, help='length for memory bank')
    parser.add_argument('--aply_softmax',type=bool, default=True, help='if apply softmax')
    parser.add_argument('--sample_gaussian',type=bool, default=True, help="if apply the feature-level noise")
    parser.add_argument('--source_len',type=int, default=300, help='length of source data')
    parser.add_argument('--target_len',type=int, default=600, help='length of target data')
    parser.add_argument('--prior_type', type=str, default='no conditional', help='if use conditional variational inference')
    parser.add_argument('--smooth', type=float, default=0.1 ,help='hyperparameter for label smooth')
    parser.add_argument('--tao', type=float, default=0.1 ,help='hyperparameter for label smooth')
    parser.add_argument('--K_times', type=float, default=3 ,help='hyperparameter for top K')
    parser.add_argument('--log_interval', type=int, default=10, help='interval for loss print')
    parser.add_argument('--warm_up_epoch', type=int, default=0, help='train without cl')
    parser.add_argument('--V_times', type=float, default=1.4, help='train without cl')
    parser.add_argument('--bn_type', type=int, default=0, help='wheter use multi-bn', choices=[0, 1])
    parser.add_argument('--confidence_th', type=float, default=0.7, help='the thresold of filtering target sample')
    parser.add_argument('--if_true_label', type=int, default=0, help='if exo use true target_t')
    parser.add_argument('--adaptation_type', default="vanilla", type=str, help="vanilla, smooth")
    parser.add_argument('--finetune', default=True, type=bool, help="if finetuning clip")
    parser.add_argument('--clip_model_name', default="./ckpt/clip/ViT-L-14-336px.pt", type=str, help="if finetuning clip")
    parser.add_argument('--dist_url', default='env://')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--training', default=True)





    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.note = 'arch'+args.backbone_arch+'seed'+str(args.seed)+'_epo'+str(args.epochs)+'_bs'+str(args.batch_size)+'_Ktimes'+str(args.K_times)+'_exo'+str(args.lambda_exo)+ '_V'+str(args.V_times)+'_pl'+args.pl_method+'_kl'+str(args.lambda_kl)+'_itv'+str(args.log_interval)+'_lr'+str(args.lr)+'_da'+args.da_method+'_dt'+str(args.lambda_beta_d)+'_bn'+str(args.bn_type)+'_warm'+str(args.warm_up_epoch)+'_es'+str(args.lambda_beta_e)+'_th'+str(args.confidence_th)+'_label'+str(args.if_true_label)+'_atype'+str(args.adaptation_type)
    print(args.note)
    if args.dataset  == 'tracking1':
        args.dataset = 'OfficeHome'
    elif args.dataset == 'tracking2':
        args.dataset = 'Office'
    elif args.dataset == 'tracking3':
        args.dataset = 'image_CLEF'
    elif args.dataset == 'tracking4':
        args.dataset = 'image_app'
    elif args.dataset == 'tracking5':
        args.dataset = 'VisDA'
    elif args.dataset == 'tracking6':
        args.dataset = 'DomainNet'
    args.s_idx = args.a_idx
    args.t_idx = args.b_idx

    '''
    assume classes across domains are the same.
    [0 1 ............................................................................ N - 1]
    |---- common classes --||---- source private classes --||---- target private classes --|

    |-------------------------------------------------|
    |                DATASET PARTITION                |
    |-------------------------------------------------|
    |DATASET    |  class split(com/sou_pri/tar_pri)   |
    |-------------------------------------------------|
    |DATASET    |    PDA    |    OSDA    | OPDA/UniDA |
    |-------------------------------------------------|
    |Office-31  |  10/21/0  |  10/0/11   |  10/10/11  |
    |-------------------------------------------------|
    |OfficeHome |  25/40/0  |  25/0/40   |  10/5/50   |
    |-------------------------------------------------|
    |VisDA-C    |   6/6/0   |   6/0/6    |   6/3/3    |
    |-------------------------------------------------|  
    |DomainNet  |           |            | 150/50/145 |
    |-------------------------------------------------|
    '''
    
    if args.dataset == "Office":
        domain_list = ['amazon', 'dslr', 'webcam']
        args.source_data_dir = os.path.join("../dataset/office31", domain_list[args.s_idx])
        args.target_data_dir = os.path.join("../dataset/office31", domain_list[args.t_idx])
        args.sourcelist_data_dir = os.path.join("./data/Office", domain_list[args.s_idx])
        args.targetlist_data_dir = os.path.join("./data/Office", domain_list[args.t_idx])
        args.target_domain_list = [domain_list[idx] for idx in range(3) if idx != args.s_idx]
        args.target_domain_dir_list = [os.path.join("../dataset/office31", item) for item in args.target_domain_list]
        args.targetlist_domain_dir_list = [os.path.join("./data/Office", item) for item in args.target_domain_list]
         
        args.shared_class_num = 10
        
        if args.target_label_type == "PDA":
            args.source_private_class_num = 21
            args.target_private_class_num = 0
        
        elif args.target_label_type == "OSDA":
            args.source_private_class_num = 0
            if args.target_private_class_num is None:
                args.target_private_class_num = 10
            
        elif args.target_label_type == "OPDA":
            args.source_private_class_num = 10
            if args.target_private_class_num is None:
                args.target_private_class_num = 11
        
        elif args.target_label_type == "CLDA":
            args.shared_class_num = 31 
            args.source_private_class_num = 0
            args.target_private_class_num = 0
        
        else:
            raise NotImplementedError("Unknown target label type specified")
 
    elif args.dataset == "OfficeHome":
        domain_list = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.source_data_dir = os.path.join("../dataset/OfficeHomeDataset_10072016", domain_list[args.s_idx])
        args.target_data_dir = os.path.join("../dataset/OfficeHomeDataset_10072016", domain_list[args.t_idx])
        args.sourcelist_data_dir = os.path.join("./data/OfficeHome", domain_list[args.s_idx])
        args.targetlist_data_dir = os.path.join("./data/OfficeHome", domain_list[args.t_idx])
        args.target_domain_list = [domain_list[idx] for idx in range(4) if idx != args.s_idx]
        args.target_domain_dir_list = [os.path.join("../dataset/OfficeHomeDataset_10072016", item) for item in args.target_domain_list]
        args.targetlist_domain_dir_list = [os.path.join("./data/OfficeHome", item) for item in args.target_domain_list]
        
        if args.target_label_type == "PDA":
            args.shared_class_num = 25
            args.source_private_class_num = 40
            args.target_private_class_num = 0
            
        elif args.target_label_type == "OSDA":
            args.shared_class_num = 25
            args.source_private_class_num = 0
            if args.target_private_class_num is None:
                args.target_private_class_num = 40
        
        elif args.target_label_type == "OPDA":
            args.shared_class_num = 10
            args.source_private_class_num = 5
            if args.target_private_class_num is None:
                args.target_private_class_num = 50
        
        elif args.target_label_type == "CLDA":
            args.shared_class_num = 65 
            args.source_private_class_num = 0
            args.target_private_class_num = 0
        else:
            raise NotImplementedError("Unknown target label type specified")

    elif args.dataset == "VisDA":
        args.source_data_dir = "../dataset/visda2017/train/"
        args.target_data_dir = "../dataset/visda2017/validation/"
        args.sourcelist_data_dir = "./data/visda2017/train/"
        args.targetlist_data_dir = "./data/visda2017/validation/"
        args.target_domain_list = ["validataion"]
        args.target_domain_dir_list = [args.target_data_dir]
        args.targetlist_domain_dir_list = [args.targetlist_data_dir]
        
        args.shared_class_num = 6
        if args.target_label_type == "PDA":
            args.source_private_class_num = 6
            args.target_private_class_num = 0
        
        elif args.target_label_type == "OSDA":
            args.source_private_class_num = 0
            args.target_private_class_num = 6
        
        elif args.target_label_type == "OPDA":
            args.source_private_class_num = 3
            args.target_private_class_num = 3
            
        elif args.target_label_type == "CLDA":
            args.shared_class_num = 12 
            args.source_private_class_num = 0
            args.target_private_class_num = 0
            
        else:
            raise NotImplementedError("Unknown target label type specified", args.target_label_type)
        
    elif args.dataset == "DomainNet":
        domain_list = ["painting", "real", "sketch"]
        args.source_data_dir = os.path.join("../dataset/DomainNet", domain_list[args.s_idx])
        args.target_data_dir = os.path.join("../dataset/DomainNet", domain_list[args.t_idx])
        args.sourcelist_data_dir = os.path.join("./data/DomainNet", domain_list[args.s_idx])
        args.targetlist_data_dir = os.path.join("./data/DomainNet", domain_list[args.t_idx])
        args.target_domain_list = [domain_list[idx] for idx in range(3) if idx != args.s_idx]
        args.target_domain_dir_list = [os.path.join("../dataset/DomainNet", item) for item in args.target_domain_list]
        args.targetlist_domain_dir_list = [os.path.join("./data/DomainNet", item) for item in args.target_domain_list]
        args.embed_feat_dim = 512 # considering that DomainNet involves more than 256 categories.
        
        args.shared_class_num = 150
        if args.target_label_type == "OSDA":
            args.source_private_class_num = 0
            args.target_private_class_num = 172
        else:
            raise NotImplementedError("Unknown target label type specified")


    elif args.dataset == "image_CLEF":
        domain_list = ['b', 'c', 'i', 'p']
        args.source_data_dir = os.path.join("../dataset/image_CLEF", domain_list[args.s_idx])
        args.target_data_dir = os.path.join("../dataset/image_CLEF", domain_list[args.t_idx])
        args.sourcelist_data_dir = os.path.join("./data/image_CLEF", domain_list[args.s_idx])
        args.targetlist_data_dir = os.path.join("./data/image_CLEF", domain_list[args.t_idx])
        args.target_domain_list = [domain_list[idx] for idx in range(4) if idx != args.s_idx]
        args.target_domain_dir_list = [os.path.join("../dataset/image_CLEF", item) for item in args.target_domain_list]
        args.targetlist_domain_dir_list = [os.path.join("./data/image_CLEF", item) for item in args.target_domain_list]

        if args.target_label_type == "PDA":
            args.shared_class_num = 6
            args.source_private_class_num = 6
            args.target_private_class_num = 0

        elif args.target_label_type == "OSDA":
            args.shared_class_num = 6
            args.source_private_class_num = 0
            if args.target_private_class_num is None:
                args.target_private_class_num = 6

        elif args.target_label_type == "OPDA":
            args.shared_class_num = 6
            args.source_private_class_num = 3
            if args.target_private_class_num is None:
                args.target_private_class_num = 3

        elif args.target_label_type == "CLDA":
            args.shared_class_num = 12
            args.source_private_class_num = 0
            args.target_private_class_num = 0
        else:
            raise NotImplementedError("Unknown target label type specified")
    elif args.dataset == "image_app":
        domain_list = ['A','B','C']
        args.source_data_dir = os.path.join("../dataset/image_app", domain_list[args.s_idx])
        args.target_data_dir = os.path.join("../dataset/image_app", domain_list[args.t_idx])
        args.sourcelist_data_dir = os.path.join("./data/image_APP", domain_list[args.s_idx])
        args.targetlist_data_dir = os.path.join("./data/image_APP", domain_list[args.t_idx])
        args.target_domain_list = [domain_list[idx] for idx in range(3) if idx != args.s_idx]
        args.target_domain_dir_list = [os.path.join("../dataset/image_app", item) for item in args.target_domain_list]
        args.targetlist_domain_dir_list = [os.path.join("./data/image_APP", item) for item in args.target_domain_list]

        if args.target_label_type == "PDA":
            args.shared_class_num = 5
            args.source_private_class_num = 2
            args.target_private_class_num = 0

        elif args.target_label_type == "OSDA":
            args.shared_class_num = 5
            args.source_private_class_num = 0
            if args.target_private_class_num is None:
                args.target_private_class_num = 2

        elif args.target_label_type == "OPDA":
            args.shared_class_num = 5
            args.source_private_class_num = 1
            if args.target_private_class_num is None:
                args.target_private_class_num = 1

        elif args.target_label_type == "CLDA":
            args.shared_class_num = 7
            args.source_private_class_num = 0
            args.target_private_class_num = 0
        else:
            raise NotImplementedError("Unknown target label type specified")

    args.source_class_num = args.shared_class_num + args.source_private_class_num
    args.target_class_num = args.shared_class_num + args.target_private_class_num
    args.class_num = args.source_class_num+1
    args.known_class = args.source_class_num
    args.source_class_list = [i for i in range(args.source_class_num)]
    args.target_class_list = [i for i in range(args.shared_class_num)]
    if args.target_private_class_num > 0:
        args.target_class_list.append(args.source_class_num)

    return args
