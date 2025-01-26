import torch.nn.functional as F
import torch
import numpy as np 
import torch.nn as nn
from torchvision import models
from torch.distributions.normal import Normal

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, 
            "vgg16":models.vgg16, "vgg19":models.vgg19, 
            "vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn,
            "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn} 

class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    # self.in_features = model_vgg.classifier[6].in_features
    self.backbone_feat_dim = model_vgg.classifier[6].in_features
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, 
            "resnet50":models.resnet50, "resnet101":models.resnet101,
            "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d,
            "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name, args):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.backbone_feat_dim = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        rois = x.view(x.size(0), -1)
        return rois

class Embedding(nn.Module):
    
    def __init__(self, feature_dim, embed_dim=256, type="bn"):
    
        super(Embedding, self).__init__()
        self.bn1 = nn.BatchNorm1d(embed_dim, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, embed_dim)
        self.bottleneck.apply(init_weights)
        self.bn_type = type
        self.fc1_mu = nn.Linear(embed_dim, embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim, affine=True)
        self.fc1_sig = nn.Linear(embed_dim, embed_dim)
        self.bn3 = nn.BatchNorm1d(embed_dim, affine=True)
        self.threshold = 0.9
        self.sig_act = F.softplus
        self.swish1 = nn.SiLU()
        self.swish2 = nn.SiLU()
        self.swish3 = nn.SiLU()
        self.swish4 = nn.SiLU()
        self.bn4 = nn.BatchNorm1d(embed_dim, affine=True)
        print('----------: multiple bn',self.bn_type)


    def dist_ent(self, x):
        dist = torch.mean(F.relu(self.threshold - self.base(x).entropy()))
        return dist

    def forward(self, x):
        x = self.bottleneck(x)
        if self.bn_type == 1:
            x = self.bn1(x)
            x = self.swish1(x)

        x_mu = self.fc1_mu(x)
        if self.bn_type == 1:
            x_mu = self.swish2(self.bn2(x_mu))
        x_sig = self.fc1_sig(x)
        if self.bn_type == 1:
            x_sig = self.bn3(x_sig)
        x_sig = self.sig_act(x_sig) + 1e-8
        x = Normal(x_mu, x_sig)
        dist = torch.mean(self.relu2(self.threshold - x.entropy()))
        for i in range(1):
            x = x.rsample()
        x = self.swish4(self.bn4(x))
        return x, dist
    
class Classifier(nn.Module):
    def __init__(self, embed_dim, class_num, type="linear"):
        super(Classifier, self).__init__()

        self.type = type
        if type == 'wn':
            self.fc = nn.utils.weight_norm(nn.Linear(embed_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(embed_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x, apply_softmax):
        x = self.fc(x)
        if apply_softmax:
            # print('apply_softmax')
            cls_out = torch.softmax(x, dim=1)
        else:
            cls_out = x
        return cls_out

# class Classifier(nn.Module):
#     def __init__(self, args, unit_size=100):
#         super(Classifier, self).__init__()
#         self.linear1 = nn.Linear(args.embed_feat_dim, unit_size)
#         self.bn1 = nn.BatchNorm1d(unit_size, affine=True, track_running_stats=True)
#         self.linear2 = nn.Linear(unit_size, unit_size)
#         self.bn2 = nn.BatchNorm1d(unit_size, affine=True, track_running_stats=True)
#         self.classifier = nn.Linear(unit_size, args.known_class+1)
#         # self.drop = nn.Dropout(p=0.3)
#         self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#
#     def forward(self, rois, apply_softmax):
#
#         x = F.relu(self.bn1(self.linear1(rois)))
#         x = F.relu(self.bn2(self.linear2(x)))
#         logits = self.classifier(x)
#
#         if apply_softmax:
#             # print('apply_softmax')
#             cls_out = torch.softmax(logits, dim=1)
#         else:
#             cls_out = logits
#         return cls_out


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams.mlp_width)
        self.dropout = nn.Dropout(hparams.mlp_dropout)
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams.mlp_width, hparams.mlp_width)
            for _ in range(hparams.mlp_depth-2)])
        self.output = nn.Linear(hparams.mlp_width, n_outputs)
        self.n_outputs = n_outputs
        self.fc1_mu = nn.Linear(n_outputs, n_outputs)
        self.fc1_sig = nn.Linear(n_outputs, n_outputs)
        self.threshold = 0.9
        self.sig_act = F.softplus
        self.bn_type = hparams.bn_type
        self.swish1 = nn.SiLU()
        self.swish2 = nn.SiLU()
        self.swish3 = nn.SiLU()
        self.swish4 = nn.SiLU()
        self.bn1 = nn.BatchNorm1d(n_outputs, affine=True)
        self.bn2 = nn.BatchNorm1d(n_outputs, affine=True)
        self.bn3 = nn.BatchNorm1d(n_outputs, affine=True)
        self.bn4 = nn.BatchNorm1d(n_outputs, affine=True)
        print('----------: multiple bn', self.bn_type)


    def forward(self, x1):
        x = self.input(x1)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        intervention = self.output(x)
        int_rois = x1 + intervention
        if self.bn_type == 1:
            int_rois = self.bn1(int_rois)
            int_rois = self.swish1(int_rois)
        x_mu = self.fc1_mu(int_rois)
        if self.bn_type == 1:
            x_mu = self.swish2(self.bn2(x_mu))
        x_sig = self.fc1_sig(int_rois)
        if self.bn_type == 1:
            x_sig = self.bn3(x_sig)
        x_sig = self.sig_act(x_sig) + 1e-8
        x = Normal(x_mu, x_sig)
        dist = torch.mean(F.relu(self.threshold - x.entropy()))
        for i in range(1):
            x = x.rsample()
        x = self.swish4(self.bn4(x))
        return intervention, dist, x

class COSDA(nn.Module):
    def __init__(self, args):

        super(COSDA, self).__init__()

        self.backbone_arch = args.backbone_arch   # resnet50
        self.embed_feat_dim = args.embed_feat_dim # 256
        self.class_num = args.class_num           # shared_class_num + source_private_class_num

        if "resnet" in self.backbone_arch:
            self.backbone_layer = ResBase(self.backbone_arch, args)
        elif "vgg" in self.backbone_arch:
            self.backbone_layer = VGGBase(self.backbone_arch, args)
        else:
            raise ValueError("Unknown Feature Backbone ARCH of {}".format(self.backbone_arch))

        self.backbone_feat_dim = self.backbone_layer.backbone_feat_dim
        self.feat_embed_layer = Embedding(self.backbone_feat_dim, self.embed_feat_dim, type=args.bn_type)
        self.intervener = MLP(self.embed_feat_dim, self.embed_feat_dim, args)
        self.class_layer = Classifier(self.embed_feat_dim, self.class_num, type='wn')

    def get_embed_feat(self, input_imgs):
        # input_imgs [B, 3, H, W]
        backbone_feat = self.backbone_layer(input_imgs)
        # embed_feat = self.feat_embed_layer(backbone_feat)
        return backbone_feat

    def forward(self, input_imgs, sample_gaussian, apply_softmax=True):
        rois = self.backbone_layer(input_imgs)
        rois_c, v = self.feat_embed_layer(rois)
        y = self.class_layer(rois_c, apply_softmax)

        intervention, int_v, int_rois = self.intervener(rois_c)

        int_y = self.class_layer(int_rois, apply_softmax)
        return rois, v, rois_c, y, intervention, int_rois, int_v, int_y
