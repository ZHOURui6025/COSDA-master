import os
from tqdm import tqdm 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import clip
from torch.utils.data.dataloader import DataLoader
def train_transform(resize_size=256, crop_size=224,):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]) 
    
def test_transform(resize_size=256, crop_size=224,):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


# def clip_train_transform(resize=336,crop_size=336 ):
#     normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
#                                [0.26862954, 0.26130258, 0.27577711])
#     return transforms.Compose([
#             transforms.Resize((resize, resize)),  # CLIP-336模型需要336x336输入
#             transforms.RandomCrop(crop_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize
#         ])
#
# def clip_test_transform(resize=336,crop_size=336 ):
#     normalize = transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
#                                [0.26862954, 0.26130258, 0.27577711])
#     return transforms.Compose([
#             transforms.Resize((resize, resize)),  # CLIP-336模型需要336x336输入
#             transforms.CenterCrop(crop_size),
#             transforms.ToTensor(),
#             normalize
#         ])

'''
assume classes across domains are the same.
[0 1 ............................................................................ N - 1]
|---- common classes --||---- source private classes --||---- target private classes --|

|-------------------------------------------------|
|                DATASET PARTITION                |
|-------------------------------------------------|
|DATASET    |  class split(com/sou_pri/tar_pri)   |
|-------------------------------------------------|
|DATASET    |    PDA    |    OSDA    |   UniDA    |
|-------------------------------------------------|
|Office-31  |  10/21/0  |  10/0/11   |  10/10/11  |
|-------------------------------------------------|
|OfficeHome |  25/40/0  |  25/0/40   |  10/5/50   |
|-------------------------------------------------|
|VisDA-C    |           |   6/0/6    |   6/3/3    |
|-------------------------------------------------|  
|DomainNet  |           |            | 150/50/145 |
|-------------------------------------------------|
'''

class SFUniDADataset(Dataset):
    
    def __init__(self, args, device, data_dir, data_list, d_type, preload_flg=True) -> None:
        super(SFUniDADataset, self).__init__()
        
        self.d_type = d_type
        self.dataset = args.dataset
        self.preload_flg = preload_flg
        
        self.shared_class_num = args.shared_class_num
        self.source_private_class_num = args.source_private_class_num
        self.target_private_class_num = args.target_private_class_num 
        
        self.shared_classes = [i for i in range(args.shared_class_num)]
        self.source_private_classes = [i + args.shared_class_num for i in range(args.source_private_class_num)]
        
        if args.dataset == "Office" and args.target_label_type == "OSDA":
            self.target_private_classes = [i + args.shared_class_num + args.source_private_class_num + 10 for i in range(args.target_private_class_num)]
        else:
            self.target_private_classes = [i + args.shared_class_num + args.source_private_class_num for i in range(args.target_private_class_num)]
            
        self.source_classes = self.shared_classes + self.source_private_classes
        self.target_classes = self.shared_classes + self.target_private_classes
        
        self.data_dir = data_dir 
        self.data_list = [item.strip().split() for item in data_list]
        # Filtering the data_list
        if self.d_type == "source":
            # self.data_dir = args.source_data_dir
            self.data_list = [item for item in self.data_list if int(item[1]) in self.source_classes]
        else:
            # self.data_dir = args.target_data_dir
            self.data_list = [item for item in self.data_list if int(item[1]) in self.target_classes]
            
        # self.pre_loading()
        
        self.train_transform = train_transform()
        self.test_transform = test_transform()
        _, self.preprocess = clip.load("./ckpt/clip/ViT-L-14-336px.pt")
        
    def pre_loading(self):
        if "Office" in self.dataset and self.preload_flg:
            self.resize_trans = transforms.Resize((256, 256))
            print("Dataset Pre-Loading Started ....")
            self.img_list = [self.resize_trans(Image.open(os.path.join(self.data_dir, item[0])).convert("RGB")) for item in tqdm(self.data_list, ncols=60)]
            print("Dataset Pre-Loading Done!")
        else:
            pass
    
    def load_img(self, img_idx):
        img_f, img_label = self.data_list[img_idx]
        if "Office" in self.dataset and self.preload_flg:
            img = self.img_list[img_idx]
        else:
            img = Image.open(os.path.join(self.data_dir, img_f)).convert("RGB")        
        return img, img_label
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, img_idx):
        
        img, img_label = self.load_img(img_idx)
        
        if self.d_type == "source":
            img_label = int(img_label)
        else:
            img_label = int(img_label) if int(img_label) in self.source_classes else len(self.source_classes)
        
        img_train = self.preprocess(img)
        img_test = self.preprocess(img)

        return img_train, img_test, img_label, img_idx
class ContinuousDataloader:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)
