from utils import *
import torch
from sklearn.cluster import KMeans
import faiss
import time
import numpy as np
import gc
def get_pseudo_label(args, device, model, dataloader, feature_batch=None, targets=None, new_epoch= True, all_proto=None, test=False):
    if args.pl_method == 'Topk3_distance':
        return Topk_distance3(args, device, model, dataloader, feature_batch=feature_batch, targets=targets, new_epoch=new_epoch, all_proto=all_proto, test=test)
    else:
        raise ValueError("INVALID PSEUDO LABELLING METHOD NAME.")
    # elif args.pl_method == 'glc':
    #     return glc(args, model, dataloader,  feature_batch=feature_batch, targets=targets, new_epoch=new_epoch, all_proto=all_proto, pos_proto = pos_proto)



def Topk_distance3(args, device, model, dataloader, feature_batch=None, targets=None, new_epoch=True, all_proto=None, test=False):
    model.eval()
    print('Using local clustering generating pesudo label')
    class_list = [i for i in range(args.known_class+1)]
    KK = int(args.known_class) # K
    # print(new_epoch)
    if new_epoch:
        print('generating new proto with local clustering pesudo label')
        dim = args.embed_feat_dim
        embed_feat_bank= torch.zeros(args.target_len, args.embed_feat_dim).cuda()
        # print(embed_feat_bank.shape)
        gt_label_bank = torch.zeros(args.target_len).long().cuda()
        pred_cls_bank = torch.zeros(args.target_len, args.known_class+1).cuda()
        class_list = [i for i in range(args.known_class+1)]
        home_loader_iter = iter(dataloader)

        begin_index = 0
        for batch_idx in range(len(dataloader)):

            data_t, _, target_t, _ = home_loader_iter.next()
            images = data_t.cuda()  #
            # print()
            label = target_t.cuda()#
            index = [i for i in range(begin_index,  begin_index+images.shape[0])]
            begin_index = begin_index + images.shape[0]
            flag = False
            # print(batch_idx, index[-1], args.target_len)
            if images.size(0) == 1:
                temp_iter = iter(dataloader)
                images_a, target_s, _, _, _ = next(temp_iter)
                images = torch.cat((images, images_a.cuda() ), dim=0)
                flag = True
                del temp_iter
                del _
            with torch.no_grad():
                rois = model.image_encoder(images)
                features_temp, _ = model.feat_embed_layer(rois)
                pred_cls = model.class_layer(features_temp, apply_softmax=True)
                # print(pred_cls.shape)
                del _
                if flag:
                    embed_feat_bank[index] = features_temp[0].unsqueeze(0)
                    gt_label_bank[index] = label
                    pred_cls_bank[index] = pred_cls
                    flag = False
                else:
                    embed_feat_bank[index] = features_temp
                    gt_label_bank[index] = label
                    pred_cls_bank[index] = pred_cls
        embed_feat_bank = embed_feat_bank[:index[-1]]
        gt_label_bank = gt_label_bank[:index[-1]]
        pred_cls_bank = pred_cls_bank[:index[-1]]
        embed_feat_bank = embed_feat_bank / torch.norm(embed_feat_bank, p=2, dim=1, keepdim=True)#

        KK = int(args.known_class) # K
        data_num = pred_cls_bank.shape[0]
        pos_topk_num = int(data_num / (KK*args.K_times)) #topk
        new_pos_topk_num = pos_topk_num#int((data_num- pos_topk_num*KK))
        print(pos_topk_num)
        # print(new_pos_topk_num)
        sorted_pred_cls, sorted_pred_cls_idxs = torch.sort(pred_cls_bank, dim=0, descending=True) #dim=0,
        pos_topk_idxs = sorted_pred_cls_idxs[:pos_topk_num, :-1].t() #[C+1, pos_topk_num]

        A_flat = pos_topk_idxs.flatten().cpu().numpy()
        # print(A_flat.shape)
        # print(A_flat)
        mask = ~np.isin(np.array([i for i in range(index[-1])]), A_flat)
        # print(mask)
        neg_embed_feat_bank = embed_feat_bank[mask]
        # print(neg_embed_feat_bank.shape)

        # print(pos_topk_idxs.shape)
        pos_topk_idxs = pos_topk_idxs.unsqueeze(2)
        # print(pos_topk_idxs.shape)
        pos_topk_idxs = pos_topk_idxs.expand([-1, -1, dim]) #[C+1, pos_topk_num, D]
        # print(pos_topk_idxs.shape)
        embed_feat_bank_expand = embed_feat_bank.unsqueeze(0).expand([KK, -1, -1]) #[C+1, N, D]
        # print(pos_topk_idxs.shape)
        # print(embed_feat_bank_expand.shape)
        pos_feat_sample = torch.gather(embed_feat_bank_expand, 1, pos_topk_idxs)
        # print(pos_feat_sample.shape)
        pos_feat_proto = torch.mean(pos_feat_sample, dim=1, keepdim=True) #[C+1, 1, D]
        # print(pos_feat_proto.shape)
        pos_feat_proto = pos_feat_proto / torch.norm(pos_feat_proto, p=2, dim=-1, keepdim=True)


        ######################Top-k centric feature_batch
        NUM_K = int(KK*(args.V_times))
        faiss_kmeans = faiss.Kmeans(dim, NUM_K, niter=100, verbose=False, min_points_per_centroid=1, gpu=False)
        faiss_kmeans.train(neg_embed_feat_bank.cpu().numpy())
        neg_feat_proto = torch.from_numpy(faiss_kmeans.centroids).cuda() #
        neg_feat_proto = neg_feat_proto / torch.norm(neg_feat_proto, p=2, dim=-1, keepdim=True)#[K+1, D]
        # print(cls_feat_proto.squeeze(1).shape)

        all_proto = torch.cat([pos_feat_proto, neg_feat_proto.unsqueeze(1)], dim=0)
        if test:
            psd_label_prior_simi = torch.einsum("nd, cd -> nc", embed_feat_bank, all_proto.squeeze(1))#[N,K] #
            feat_proto_simi = psd_label_prior_simi
            psd_label_prior_idxs = torch.max(feat_proto_simi, dim=-1, keepdim=True)[1].squeeze(1) #[N] ~ (0, class_num-1)
            # print(psd_label_prior_idxs.shape)
            need_hard_label = psd_label_prior_idxs
            return need_hard_label
        else:
            return all_proto, neg_feat_proto, args.known_class+neg_feat_proto.shape[0]

    else:

        psd_label_prior_simi = torch.einsum("nd, cd -> nc", feature_batch, all_proto.squeeze(1))#[b,K]
        feat_proto_simi = psd_label_prior_simi
        psd_label_prior_idxs = torch.max(feat_proto_simi, dim=-1, keepdim=True)[1].squeeze(1) #[N]
        # print(psd_label_prior_idxs.shape)
        need_hard_label = psd_label_prior_idxs
        psd_label_prior_idxs[psd_label_prior_idxs >= KK] = KK
        # print(psd_label_prior_idxs)
        psd_label_prior = torch.zeros(feat_proto_simi.shape).scatter(1, psd_label_prior_idxs.unsqueeze(1).cpu(), 1.0).cuda() # one_hot prior #[N, C]
        # print(psd_label_prior.shape)
        hard_label = psd_label_prior_idxs#torch.argmax(psd_label_prior, dim=1) #[N]
        per_class_num = np.zeros((len(class_list)))
        pre_class_num = np.zeros_like(per_class_num)
        per_class_correct = np.zeros_like(per_class_num)
        # print(hard_label)
        # print(targets)
        for i, label in enumerate(class_list):
            label_idx = torch.where(targets == label)[0]
            correct_idx = torch.where(hard_label[label_idx] == label)[0]
            # pre_class_num[i] = float(len(torch.where(hard_label == label)[0]))
            per_class_num[i] = float(len(label_idx))
            per_class_correct[i] = float(len(correct_idx))
        per_class_acc = per_class_correct / (per_class_num + 1e-5)
        if True:#batch_index % args.log_interval == 0:
            print("PSD AVG ACC:\t" + "{:.3f}".format(sum(per_class_acc)/np.count_nonzero(per_class_num)))
            print("PSD PER ACC:\t" + "\t".join(["{:.3f}".format(item) for item in per_class_acc]))
            print("PER CLS NUM:\t" + "\t".join(["{:.0f}".format(item) for item in per_class_num]))
            # print_and_log("PRE CLS NUM:\t" + "\t".join(["{:.0f}".format(item) for item in pre_class_num]))
            print("PRE ACC NUM:\t" + "\t".join(["{:.0f}".format(item) for item in per_class_correct]))

        # all_unk_proto = torch.mean(all_proto.squeeze(1)[KK:], dim=0)
        gc.collect()
        torch.cuda.empty_cache()
        return need_hard_label

