import os 
import torch 
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np 
from PIL import Image, ImageFilter
import json
import random 
import pickle


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class EvalPASCAL(data.Dataset):
    VOC_CATEGORY_NAMES = ['background',
                          'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow',
                          'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    GOOGLE_DRIVE_ID = '1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH'
    FILE = 'PASCAL_VOC.tgz'
    DB_NAME = 'VOCSegmentation'
    def __init__(self, root, split='val', res=224, transform_list=[], download=False):
        
        self.root  = root 
        self.split = split
        self.res = res
        self.transform_list = transform_list  
        
        if download:
            self._download()
        

        with open(os.path.join(self.root, self.DB_NAME, 'sets', '{}.txt'.format(self.split)), 'r') as f:
            lines = f.read().splitlines()
        
        # self.reshuffle()
        _image_dir = os.path.join(self.root, self.DB_NAME, 'images')
        _sal_dir = os.path.join(self.root, self.DB_NAME, 'saliency_unsupervised_model')
        _label_dir = os.path.join(self.root, self.DB_NAME, 'SegmentationClass')
        
        self.images = []
        self.sals = []
        self.labels = []
        for ii, line in enumerate(lines):
            _image = os.path.join(_image_dir, line + ".jpg")
            _sal = os.path.join(_sal_dir, line + ".png")
            _label = os.path.join(_label_dir, line + ".png")
            if os.path.isfile(_image) and os.path.isfile(_sal) and os.path.isfile(_label):
                self.images.append(_image)
                self.sals.append(_sal)
                self.labels.append(_label)

        assert(len(self.images) == len(self.sals))
        assert(len(self.images) == len(self.labels))

        ignore_classes = []
        self.ignore_classes = [self.VOC_CATEGORY_NAMES.index(class_name) for class_name in ignore_classes]

        self.reshuffle()


    def _download(self):
        
    
        _fpath = os.path.join(self.root, self.FILE)

        if os.path.isfile(_fpath):
            print('Files already downloaded')
            return
        else:
            print('Downloading dataset from google drive')
            mkdir_if_missing(os.path.dirname(_fpath))
            download_file_from_google_drive(self.GOOGLE_DRIVE_ID, _fpath)

        # extract file
        cwd = os.getcwd()
        print('\nExtracting tar file')
        tar = tarfile.open(_fpath)
        os.chdir(os.path.join(self.root))
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('Done!')

    def reshuffle(self):
        """
        Generate random floats for all image data to deterministically random transform.
        This is to use random sampling but have the same samples during clustering and 
        training within the same epoch. 
        """
        self.shuffled_indices = np.arange(len(self.images))
        np.random.shuffle(self.shuffled_indices)
    


    def load_data(self, index):
        _image = Image.open(self.images[index]).convert('RGB')
        _sal = Image.open(self.sals[index])
        _semseg = Image.open(self.labels[index])
        # _semseg = np.array(Image.open(self.labels[index]))
        return _image, _sal, _semseg
    
    def __getitem__(self, index):
        
        index = self.shuffled_indices[index]
        
        image, sal, label = self.load_data(index)
        image, sal, label = self.transform_data(image, sal, label)

        return index, image, sal.squeeze().long(), label

   
    def transform_data(self, image, sal, label):

        # 1. Resize
        image = TF.resize(image, self.res, Image.BILINEAR)
        sal = TF.resize(sal, self.res, Image.NEAREST)
        label = TF.resize(label, self.res, Image.NEAREST)
        
        # 2. CenterCrop
        w, h = image.size
        left = int(round((w - self.res) / 2.))
        top  = int(round((h - self.res) / 2.))

        image = TF.crop(image, top, left, self.res, self.res)
        sal = TF.crop(sal, top, left, self.res, self.res)
        label = TF.crop(label, top, left, self.res, self.res)

        # 3. Transformation
        transform_image, totensor = self._get_data_transformation()
        image = transform_image(image)
        sal= totensor(sal)
        label = np.array(label)
        label = torch.from_numpy(label).long()

        return image, sal, label



    def _get_data_transformation(self):
        trans_list = []
        if 'jitter' in self.transform_list:
            trans_list.append(transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.8))
        if 'grey' in self.transform_list:
            trans_list.append(transforms.RandomGrayscale(p=0.2))
        if 'blur' in self.transform_list:
            trans_list.append(transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5))
        
        # Base transformation
        trans_list += [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        
        return transforms.Compose(trans_list), transforms.ToTensor()
    
    def __len__(self):
        return len(self.images)


  
if __name__ == '__main__':
    testset    = EvalPASCAL(root='PASCAL_VOC', res=224, split='val', transform_list=['jitter', 'blur', 'grey'])
    
    index, img, sal, label = testset[0]
    print(img.shape)
    print(sal.shape)
    # print(label.shape)
    # label.show()
    # x = np.array(label)
    # print(np.unique(x))
    print(torch.unique(label))
    topil = transforms.ToPILImage()
    # topil(img).show()
    topil(label.float()).show()
    
    # print(torch.unique(sal))
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=32,
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True)

    for i, (index, img, sal, label) in enumerate(testloader):
        print(img.shape)
        print(sal.shape)
        print(label.shape)
        print(index.shape)
        print(label[0])
        break
    
