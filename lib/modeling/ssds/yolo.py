import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os


class YOLO(nn.Module):
    """Single Shot Multibox Architecture
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    Args:
        phase: (string) Can be "eval" or "train" or "feature"
        base: base layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        feature_layer: the feature layers for head to loc and conf
        num_classes: num of classes 
    """
    def __init__(self, base, extras, head, feature_layer, num_classes):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        # SSD network
        self.base = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(dim=-1)

        self.feature_layer = [ f for feature in feature_layer[0] for f in feature]
        # self.feature_index = [ len(feature) for feature in feature_layer[0]]
        self.feature_index = list()
        s = -1
        for feature in feature_layer[0]:
            s += len(feature)
            self.feature_index.append(s)
        
        print(self.feature_index)
    def forward(self, x, phase='eval'):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]

            feature:
                the features maps of the feature extractor
        """
        cat = dict()
        sources, loc, conf = [list() for _ in range(3)]
        #print("end")
        # apply bases layers and cache source layer outputs
        for k in range(len(self.base)):
            x = self.base[k](x)
            if k in self.feature_layer:
                cat[k] = x
        #print(cat[23])
        # apply extra layers and cache source layer outputs
        #print(self.feature_layer)
        #print(len(self.extras))
        for k, v in enumerate(self.extras):
            
            #print(k)
            #if is int which means concate
            if isinstance(self.feature_layer[k], int):
                #print(self.feature_layer[k])
                x = v(x, cat[self.feature_layer[k]])
            else:
            #else is B, apply convolution
                x = v(x)
            if k in self.feature_index:
                sources.append(x)
        #print(self.feature_index)
        if phase == 'feature':
            return sources

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if phase == 'eval':
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output


class _dilated_residual_bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(_dilated_residual_bottleneck, self).__init__()
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride,  groups=inp * expand_ratio, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        
        self.conv2 = nn.Sequential(
            # pw
            nn.Conv2d(inp, oup, 1, stride, 0, bias=False),
            nn.BatchNorm2d(oup),
                )
    def forward(self, x):
        return self.conv2(x) + self.conv(x)



def add_extras(base, feature_layer, mbox, num_classes, version):
    extra_layers = []
    loc_layers = []
    conf_layers = []
    #print(base[-1].depth)
    in_channels = base[-1].depth
    #in_channels = None
    for layers, depths, box in zip(feature_layer[0], feature_layer[1], mbox):
        for layer, depth in zip(layers, depths):
            if layer == '':
                extra_layers += [ _conv_bn(in_channels, depth) ]
                in_channels = depth
            elif layer == 'B':
                #extra_layers += [ _conv_block(in_channels, depth) ]
                #in_channels = depth
                extra_layers += [ _conv_dw(in_channels, depth, stride=1, padding=1, expand_ratio=4) ]
                in_channels = depth
            elif layer == 'D1':
                stride = 1
                extra_layers += [_dilated_residual_bottleneck(in_channels, depth, stride, expand_ratio=4)]
                in_channels = depth
            elif layer == 'D2':
                stride = 2
                extra_layers += [_dilated_residual_bottleneck(in_channels, depth, stride, expand_ratio=4)]
                in_channels = depth                
            
            elif isinstance(layer, int):
                if version == 'v2':
                    extra_layers += [ _router_v2(base[layer].depth, depth) ]
                    in_channels = in_channels + depth * 4
                elif version == 'v3':
                    extra_layers += [ _router_v3(in_channels, depth) ]
                    
                    in_channels = depth + base[layer].depth
            else:
                AssertionError('undefined layer')
        loc_layers += [_conv_dw_out(in_channels,box * 4)]
        conf_layers += [_conv_dw_out(in_channels,box * num_classes)]
    
        #loc_layers += [nn.Conv2d(in_channels, box * 4, kernel_size=1)]
        #conf_layers += [nn.Conv2d(in_channels, box * num_classes, kernel_size=1)]
    return base, extra_layers, (loc_layers, conf_layers)
def _conv_dw(inp, oup, stride=1, padding=0, expand_ratio=1):
    return nn.Sequential(
        # pw
        nn.Conv2d(inp, oup * expand_ratio, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(oup * expand_ratio, oup * expand_ratio, 3, stride, padding, groups=oup * expand_ratio, bias=False),
        nn.BatchNorm2d(oup * expand_ratio),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(oup * expand_ratio, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
    )
def _conv_dw_out(inp,oup):
    return nn.Sequential(
        # pw
        nn.Conv2d(inp, inp, kernel_size = 3,padding=1,stride=1,groups=inp),
        # dw
        nn.Conv2d(inp, oup , kernel_size=1,padding=0,stride=1),
        # pw-linear
    )

class _conv_bn(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super(_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _conv_block(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=0.5):
        super(_conv_block, self).__init__()
        depth = int(oup*expand_ratio)
        self.conv = nn.Sequential(
            nn.Conv2d(inp, depth, 1, 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(depth, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _router_v2(nn.Module):
    def __init__(self, inp, oup, stride=2):
        super(_router_v2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.stride = stride

    def forward(self, x1, x2):
        # prune channel
        x2 = self.conv(x2)
        # reorg
        B, C, H, W = x2.size()
        s = self.stride
        x2 = x2.view(B, C, H // s, s, W // s, s).transpose(3, 4).contiguous()
        x2 = x2.view(B, C, H // s * W // s, s * s).transpose(2, 3).contiguous()
        x2 = x2.view(B, C, s * s, H // s, W // s).transpose(1, 2).contiguous()
        x2 = x2.view(B, s * s * C, H // s, W // s)
        return torch.cat((x1, x2), dim=1)


class _router_v3(nn.Module):
    def __init__(self, inp, oup, stride=1, bilinear=True):
        super(_router_v3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.LeakyReLU(0.1, inplace=True),
        )
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(oup, oup, 2, stride=2)

    def forward(self, x1, x2):
        # prune channel
        x1 = self.conv(x1)
        # up
        x1 = self.up(x1)
        # ideally the following is not needed
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        return torch.cat((x1, x2), dim=1)




def build_yolo_v2(base, feature_layer, mbox, num_classes):
    base_, extras_, head_ = add_extras(base(), feature_layer, mbox, num_classes, version='v2')
    return YOLO(base_, extras_, head_, feature_layer, num_classes)

def build_yolo_v3(base, feature_layer, mbox, num_classes):
    base_, extras_, head_ = add_extras(base(), feature_layer, mbox, num_classes, version='v3')
    return YOLO(base_, extras_, head_, feature_layer, num_classes)


if __name__ == '__main__':

    feature_layer_v2 = [[['', '',12, '']], [[1024, 1024, 64, 1024]]]
    mbox_v2 = [5]
    feature_layer_v3 = [[['B','B','B'], [16,'B','B','B'], [8,'B','B','B']],
                  [[36,36,32], [24, 30, 30, 30], [20, 30, 30, 30]]]
    mbox_v3 = [3, 3, 3]

    from lib.modeling.nets.darknet import *
    from lib.modeling.nets.mobilenet import *
    # yolo_v2 = build_yolo_v2(darknet_19, feature_layer_v2, mbox_v2, 81)
    # # print('yolo_v2', yolo_v2)
    # yolo_v2.eval()
    # x = torch.rand(1, 3, 416, 416)
    # x = torch.autograd.Variable(x, volatile=True) #.cuda()
    # feature_maps = yolo_v2(x, phase='feature')
    # print([(o.size()[2], o.size()[3]) for o in feature_maps])
    # out = yolo_v2(x, phase='eval')


    yolo_v3 = build_yolo_v3(mobilenet_v2, feature_layer_v3, mbox_v3, 2)
    #print('yolo_v3', yolo_v3)
    # print(yolo_v3.feature_layer, yolo_v3.feature_index)
    yolo_v3.eval()
    x = torch.rand(1, 3, 256, 256)
    x = torch.autograd.Variable(x, volatile=True) #.cuda()
    feature_maps = yolo_v3(x, phase='feature')
    #print(feature_maps)
    print([(o.size()[2], o.size()[3]) for o in feature_maps])
    out = yolo_v3(x, phase='eval')
    # print(set(yolo_v3.state_dict()))
    #print(set(yolo_v3.base[0].conv[1].state_dict()))



