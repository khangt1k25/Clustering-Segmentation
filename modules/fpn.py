import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from . import backbone
import torch
import torch.nn as nn
import torch.nn.functional as F


class PanopticFPN(nn.Module):
    def __init__(self, args):
        super(PanopticFPN, self).__init__()
        self.backbone = backbone.__dict__[args.arch](pretrained=args.pretrain)
        self.decoder  = FPNDecoder(args)

    def forward(self, x):
        feats = self.backbone(x)
        outs  = self.decoder(feats) 

        return outs 

class FPNDecoder(nn.Module):
    def __init__(self, args):
        super(FPNDecoder, self).__init__()
        if args.arch == 'resnet18':
            mfactor = 1
            out_dim = 128 
        else:
            mfactor = 4
            out_dim = 256

        self.layer4 = nn.Conv2d(512*mfactor//8, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(512*mfactor//4, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(512*mfactor//2, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer1 = nn.Conv2d(512*mfactor, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        o1 = self.layer1(x['res5'])
        o2 = self.upsample_add(o1, self.layer2(x['res4']))
        o3 = self.upsample_add(o2, self.layer3(x['res3']))
        o4 = self.upsample_add(o3, self.layer4(x['res2']))

        return o4

    def upsample_add(self, x, y):
        _, _, H, W = y.size()

        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y 




# Code referenced from https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/deeplabv3.py





class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu = orig_resnet.relu
        
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x) 
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
    
class ContrastiveSegmentationModel(nn.Module):
    def __init__(self, backbone, decoder, head, upsample=True, use_classification_head=True):
        super(ContrastiveSegmentationModel, self).__init__()
        self.backbone = backbone
        self.upsample = upsample
        self.use_classification_head = use_classification_head
        
        if head == 'linear': 
            # Head is linear.
            # We can just use regular decoder since final conv is 1 x 1.
            self.head = decoder[-1]
            decoder[-1] = nn.Identity()
            self.decoder = decoder

        else:
            raise NotImplementedError('Head {} is currently not supported'.format(head))


        if self.use_classification_head: # Add classification head for saliency prediction
            self.classification_head = nn.Conv2d(self.head.in_channels, 1, 1, bias=False)

    def forward(self, x):
        # Standard model
        input_shape = x.shape[-2:]
        x = self.backbone(x)
        embedding = self.decoder(x)

        # Head
        x = self.head(embedding)
        if self.use_classification_head:
            sal = self.classification_head(embedding)

        # Upsample to input resolution
        if self.upsample: 
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            if self.use_classification_head:
                sal = F.interpolate(sal, size=input_shape, mode='bilinear', align_corners=False)

        # Return outputs
        if self.use_classification_head:
            return x, sal.squeeze()
        else:
            return x

def load_pretrained_weights(args, model):
    if args.pretraining == 'imagenet_classification':
        try:
            from torch.hub import load_state_dict_from_url
        except ImportError:
            from torch.utils.model_zoo import load_url as load_state_dict_from_url
            
        print('Loading ImageNet pre-trained classification (supervised) weights for backbone')
        model_urls = {
            'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
        }
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=True)
        msg = model.load_state_dict(state_dict, strict=False)        
        print(msg)

    elif args.pretraining == 'imagenet_moco':
        # Load pre-trained ImageNet MoCo weights
        print('Loading MoCo pre-trained weights for backbone')
        state_dict = torch.load(args.moco_state_dict, map_location='cpu')['state_dict']
        new_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder_q'):
                new_dict[k.rsplit('encoder_q.')[1]] = v
        msg = model.load_state_dict(new_dict, strict=False)   
        assert(all(['fc' in k for k in msg[0]])) 
        assert(all(['fc' in k for k in msg[1]])) 

    else:
        raise ValueError('Invalid value {}'.format(args.moco_state_dict))
        
def get_model(args):
    import torchvision.models.resnet as resnet
    backbone = resnet.__dict__['resnet50'](pretrained=False)
    backbone_channels = 2048
    load_pretrained_weights(args, backbone)
    backbone = ResnetDilated(backbone)
    deplabhead = DeepLabHead(backbone_channels, args.ndim)
    return ContrastiveSegmentationModel(backbone, deplabhead, head='linear')
