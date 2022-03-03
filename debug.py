# from train_pascal import train
# from data.pascal_train_dataset import TrainPASCAL
# from data.pascal_eval_dataset import EvalPASCAL
# from utils import get_transform_params, collate_train, worker_init_fn, collate_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
# from commons import run_mini_batch_kmeans
# from torchvision.transforms import ToPILImage
# import numpy as np
# import matplotlib.pyplot as plt


feats = torch.randn((32, 64, 128, 128)) # Bx Cx H x W
sal = torch.randn((32, 128, 128))
sal = sal.unsqueeze(1)
feats = feats * sal # need to fixed

print(feats.shape)
# VOC_CLASSES = [
#     'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
#     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
#     'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

# def _fast_hist(label_true, label_pred, n_class):
#     mask = (label_true >= 0) & (label_true < n_class) # Exclude unlabelled data.
#     hist = np.bincount(n_class * label_true[mask] + label_pred[mask],\
#                        minlength=n_class ** 2).reshape(n_class, n_class)
    
#     return hist
# def scores(label_trues, label_preds, n_class):
#     hist = np.zeros((n_class, n_class))
#     for lt, lp in zip(label_trues, label_preds):
#         hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
#     return hist
# if __name__ == '__main__':
#     # inv_list = ['blur', 'gray', 'brightness', 'contrast', 'saturation', 'hue']
#     # eqv_list = ['h_flip', 'v_flip', 'random_crop']
#     inv_list = ['jiter', 'blur', 'gray']
#     # inv_list = []
#     eqv_list = ['h_flip', 'random_crop']
#     # eqv_list=[]
#     trainset = TrainPASCAL(root='./PASCAL_VOC',
#                             mode='compute',
#                             split='trainaug',
#                             labeldir='',
#                             res1=224,
#                             res2=448,
#                             inv_list=inv_list,
#                             eqv_list=eqv_list)
#     evalset = EvalPASCAL(root='./PASCAL_VOC', mode='test', split='val', res=224)
    

   
#     trainloader = torch.utils.data.DataLoader(trainset, 
#                                             batch_size=4,
#                                             shuffle=False, 
#                                             num_workers=1,
#                                             pin_memory=True,
#                                             collate_fn=collate_train,
#                                             worker_init_fn=worker_init_fn(16))


#     testloader = torch.utils.data.DataLoader(evalset,
#                                              batch_size=4,
#                                              shuffle=False,
#                                              num_workers=1,
#                                              pin_memory=True,
#                                              collate_fn=collate_eval,
#                                              worker_init_fn=worker_init_fn(16))
#     toPIL = ToPILImage()
#     # classifier = nn.Conv2d(128, 21, kernel_size=1, stride=1, padding=0, bias=True)
#     # histogram = np.zeros((21, 21))
#     for i, (_, image, label) in enumerate(testloader):
#         print(image.shape)
#         print(label.shape)
#         print(torch.unique(label[0]))
#         toPIL(image[0].squeeze(0)).show()
#         toPIL(label[0].float()).show()
#         break
#     #     feats = torch.randn(size=(32, 128, 56, 56))
#     #     feats = F.normalize(feats, dim=1, p=2)
        
#     #     B, C, H, W = feats.size()

#     #     probs = classifier(feats)
#     #     probs = F.interpolate(probs, label.shape[-2:], mode='bilinear', align_corners=False)
#     #     preds = probs.topk(1, dim=1)[1].view(B, -1).cpu().numpy()
#     #     label = label.view(B, -1).cpu().numpy()
#     #     print(preds.shape)
#     #     print(label.shape)
#     #     # histogram += scores(label, preds, 21)
#     #     hist = np.zeros((21, 21))
#     #     for lt, lp in zip(label, preds):
#     #         print(lt.flatten().shape)
#     #         print(lp.shape)
#     #         mask = (lt >= 0) & (lt < 21) # Exclude unlabelled data.
            
#     #         hist = np.bincount(21 * lt[mask] + lp[mask],\
#     #                         minlength=21 ** 2).reshape(21, 21)
#     #         # hist += _fast_hist(lt.flatten(), lp.flatten(), 21)
#     #     # return hist
#     #     break
    

    
#     # plt.imshow(toPIL(trainset[0][1]))
#     # plt.imsave("./test.jpg", toPIL(trainset[0][1]))
#     # # print(trainset[0][1])

    