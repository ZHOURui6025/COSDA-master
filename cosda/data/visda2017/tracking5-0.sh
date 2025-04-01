
# VisDA
python train_a.py \
       --gpu 0 \
       --backbone_arch vgg19 \
       --dataset tracking5 \
       --s_idx 0  \
       --target_label_type OSDA \
       --epochs 10 \
       --lr 0.001

python train_b.py \
       --gpu 0 \
       --dataset tracking5 \
       --backbone_arch vgg19 \
       --a_idx 0 \
       --b_idx 1 \
       --lr 0.0001 \
       --batch_size 32 \
       --warm_up_epoch 10 \
       --K_times 3 \
       --epochs 20 \
       --pl_method Topk3_distance \
       --V_times 1 \
       --log_interval 10 \
       --lambda_beta_d 1 \
       --da_method MO \
       --lambda_kl 0.1 \
       --bn_type 0 \
       --seed 1 \
       --source_train_type smooth \
       --confidence_th 0.7 \
       --if_true_label 0 \
       --lambda_exo 0.2  \
       --lambda_beta_e 0.2
