import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import random
import os
import numpy as np
from PIL import Image

class Base_Dataset(data.Dataset):
    def __init__(self, root, partition):
        super(Base_Dataset, self).__init__()
        # set dataset info
        self.root = root
        self.partition = partition
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        # self.source_len =
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        if self.partition == 'train':
            self.transformer = transforms.Compose([transforms.Resize([224, 224]),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   normalize])
        # transforms.RandomCrop(224) # augmentation for tiny-scale datsets, e.g., office31                                               
        else:
            self.transformer = transforms.Compose([transforms.Resize([224, 224]),  # try resize to 224, instead of crop
                                                   transforms.ToTensor(),
                                                   normalize])
        # transforms.CenterCrop(224) # augmentation for tiny-scale datsets, e.g., office31

    def __len__(self):
        return min(len(self.target_image), len(self.source_image))

    def load_dataset(self, openness, label_flag=None):
        source_image_list = []
        source_label_list = []
        target_image_list = []
        target_label_list = []
        target_num = np.zeros(self.num_class)
        with open(self.source_path) as f:
            for ind, line in enumerate(f.readlines()):
                image_dir, label = ' '.join(line.split(' ')[0:-1]), line.split(' ')[-1]
                label = label.strip()
                source_image_list.append(self.root + '/' + image_dir)
                source_label_list.append(int(label))
        source_len = len(source_label_list)
        with open(self.target_path) as f:
            for ind, line in enumerate(f.readlines()):
                image_dir, label = ' '.join(line.split(' ')[0:-1]), line.split(' ')[-1]
                label = int(label.strip())
                if openness is not None:
                    if label >= self.num_class + openness - 1:
                        continue
                if label >= self.num_class:
                    label = self.num_class - 1
                target_num[label] += 1
                target_image_list.append(self.root + '/' + image_dir)
                target_label_list.append(label)
        target_len = len(target_label_list)
        if self.partition == 'train' and len(source_image_list) > len(target_image_list):
            k = int(len(source_image_list) / len(target_image_list)) + 1
            target_image_list *= k
            target_label_list *= k
        else:
            multi = int(len(target_image_list) / len(source_image_list)) + 1
            source_image_list *= multi
            source_label_list *= multi
        return source_image_list, source_label_list, target_image_list, target_label_list, target_num, source_len, target_len

    def __getitem__(self, item):
        name_s = self.source_image[item]
        name_t = self.target_image[item]
        lbl_s = self.source_label[item]
        lbl_t = self.target_label[item]
        img_s = Image.open(name_s).convert('RGB')
        img_t = Image.open(name_t).convert('RGB')

        img_s = self.transformer(img_s)
        img_t = self.transformer(img_t)

        if self.label_flag is not None:
            pse_lbl = self.label_flag[item]
        else:
            pse_lbl = self.num_class
        if self.confidence is not None:
            confidence = self.confidence[item]
        else:
            confidence = 0
        return img_s, lbl_s, img_t, lbl_t, (pse_lbl, confidence)

class Home_Dataset(Base_Dataset):
    def __init__(self, root, partition, source='A', target='R', label_flag=None, confidence=None, openness=None):
        # openness: In fact refer to [1 - known_num / (known_num + unknown_num)], here refer to the number of unknown
        super(Home_Dataset, self).__init__(root, partition)
        src_name, tar_name = self.getFilePath(source, target)
        self.source_path = os.path.join('data', src_name)
        self.target_path = os.path.join('data', tar_name)
        self.class_name = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
                           'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
                           'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder',
                           'Fork', 'unk']
        self.num_class = len(self.class_name)
        self.label_flag = label_flag
        self.confidence = confidence
        self.source_image, self.source_label, self.target_image, self.target_label, self.target_number, self.source_len, self.target_len = self.load_dataset(
            openness, label_flag)

        if self.label_flag is not None:
            self.label_flag = label_flag

    def getFilePath(self, source, target):

        if source == 'A':
            src_name = 'office_home/art_s.txt'
        elif source == 'C':
            src_name = 'office_home/clipart_s.txt'
        elif source == 'P':
            src_name = 'office_home/product_s.txt'
        elif source == 'R':
            src_name = 'office_home/realworld_s.txt'
        else:
            print("Unknown Source Type, only supports A C P R.")
        if target == 'A':
            tar_name = 'office_home/art_t.txt'
        elif target == 'C':
            tar_name = 'office_home/clipart_t.txt'
        elif target == 'P':
            tar_name = 'office_home/product_t.txt'
        elif target == 'R':
            tar_name = 'office_home/realworld_t.txt'
        else:
            print("Unknown Target Type, only supports A C P R.")
        return src_name, tar_name

class CLEF_Dataset(Base_Dataset):
    def __init__(self, root, partition, source='B', target='C', label_flag=None, confidence=None, openness=None):
        # openness: In fact refer to [1 - known_num / (known_num + unknown_num)], here refer to the number of unknown
        super(CLEF_Dataset, self).__init__(root, partition)
        src_name, tar_name = self.getFilePath(source, target)
        self.source_path = os.path.join('data', src_name)
        self.target_path = os.path.join('data', tar_name)
        self.class_name = [str(i) for i in range(6)]+['unk']
        self.num_class = len(self.class_name)
        self.label_flag = label_flag
        self.confidence = confidence
        self.source_image, self.source_label, self.target_image, self.target_label, self.target_number, self.source_len, self.target_len = self.load_dataset(
            openness, label_flag)

        if self.label_flag is not None:
            self.label_flag = label_flag

    def getFilePath(self, source, target):

        if source == 'B':
            src_name = 'image_clef/b_s.txt'
        elif source == 'C':
            src_name = 'image_clef/c_s.txt'
        elif source == 'P':
            src_name = 'image_clef/p_s.txt'
        elif source == 'I':
            src_name = 'image_clef/i_s.txt'
        else:
            print("Unknown Source Type, only supports B C P I.")
        if target == 'B':
            tar_name = 'image_clef/b_t.txt'
        elif target == 'C':
            tar_name = 'image_clef/c_t.txt'
        elif target == 'P':
            tar_name = 'image_clef/p_t.txt'
        elif target == 'I':
            tar_name = 'image_clef/i_t.txt'
        else:
            print("Unknown Target Type, only supports B C P I.")
        return src_name, tar_name
class office31_Dataset(Base_Dataset):
    def __init__(self, root, partition, source='B', target='C', label_flag=None, confidence=None, openness=None):
        # openness: In fact refer to [1 - known_num / (known_num + unknown_num)], here refer to the number of unknown
        super(office31_Dataset, self).__init__(root, partition)
        src_name, tar_name = self.getFilePath(source, target)
        self.source_path = os.path.join('data', src_name)
        self.target_path = os.path.join('data', tar_name)
        self.class_name = ["backpack", "bike", "bike_helmet", "bookcase", "bottle",
                           "calculator", "desk_chair", "desk_lamp", "desktop_computer",
                           "file_cabinet", "unk"]
        self.num_class = len(self.class_name)
        self.label_flag = label_flag
        self.confidence = confidence
        self.source_image, self.source_label, self.target_image, self.target_label, self.target_number, self.source_len, self.target_len = self.load_dataset(
            openness, label_flag)

        if self.label_flag is not None:
            self.label_flag = label_flag

    def getFilePath(self, source, target):

        if source == 'A':
            src_name = 'office31/amazon_0-9_train_all.txt'
        elif source == 'D':
            src_name = 'office31/dslr_0-9_train_all.txt'
        elif source == 'W':
            src_name = 'office31/webcam_0-9_train_all.txt'

        else:
            print("Unknown Source Type, only supports B C P I.")
        if target == 'A':
            tar_name = 'office31/amazon_0-9_20-30_test.txt'
        elif target == 'D':
            tar_name = 'office31/dslr_0-9_20-30_test.txt'
        elif target == 'W':
            tar_name = 'office31/webcam_0-9_20-30_test.txt'
        else:
            print("Unknown Target Type, only supports B C P I.")
        return src_name, tar_name


class vlcs_Dataset(Base_Dataset):
    def __init__(self, root, partition, source='C', target='L', label_flag=None, confidence=None, openness=None):
        # openness: In fact refer to [1 - known_num / (known_num + unknown_num)], here refer to the number of unknown
        super(vlcs_Dataset, self).__init__(root, partition)
        src_name, tar_name = self.getFilePath(source, target)
        self.source_path = os.path.join('data', src_name)
        self.target_path = os.path.join('data', tar_name)
        self.class_name = ['bird', 'car', 'unk']
        self.num_class = len(self.class_name)
        self.label_flag = label_flag
        self.confidence = confidence
        self.source_image, self.source_label, self.target_image, self.target_label, self.target_number, self.source_len, self.target_len = self.load_dataset(
            openness, label_flag)

        if self.label_flag is not None:
            self.label_flag = label_flag

    def getFilePath(self, source, target):

        if source == 'C':
            src_name = 'vlcs/Caltech101_s.txt'
        elif source == 'L':
            src_name = 'vlcs/LabelMe_s.txt'
        elif source == 'S':
            src_name = 'vlcs/SUN09_s.txt'
        elif source == 'V':
            src_name = 'vlcs/VOC2007_s.txt'
        else:
            print("Unknown Source Type, only supports C L S V.")
        if target == 'C':
            tar_name = 'vlcs/Caltech101_t.txt'
        elif target == 'L':
            tar_name = 'vlcs/LabelMe_t.txt'
        elif target == 'S':
            tar_name = 'vlcs/SUN09_t.txt'
        elif target == 'V':
            tar_name = 'vlcs/VOC2007_t.txt'
        else:
            print("Unknown Target Type, only supports C L S V.")
        return src_name, tar_name
