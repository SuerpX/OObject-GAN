import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as torchv
import numpy as np

class Encoder(nn.Module):
    def __init__(self, code_size, pretrained=True):
        super().__init__()
        self.code_size = code_size
        self.resnet = torchv.models.resnet50(pretrained=pretrained)
        self.resnet = nn.DataParallel(nn.Sequential(*list(self.resnet.children())[1:-2]))
        self.conv1 = nn.DataParallel(nn.Conv2d(4, 64, 7, stride=2, padding=3))
        self.bottle_neck = nn.DataParallel(nn.Sequential(nn.Conv2d(2048, 1024, 1),
                                        nn.ReLU()))
        self.encoding = nn.DataParallel(nn.Sequential(nn.Linear(7*7*1024, code_size),
                                     nn.Tanh())) # tanh, sigmoid, relu??? what ever makes sense to represent code
        
        
        
    def forward(self, X):
        """
        Params:
            X: bsize x [3+1: image channel + object's mask] x h x w
        Return:
            code: bsize x code_size. where, 1st half is bg, and 2nd half is fg code respectively
        """
        X = self.resnet(self.conv1(X)) # bsize x 2048 x 7 x 7
        X = self.bottle_neck(X) # bsize x 1024 x 7 x 7
        X = X.contiguous().view(X.shape[0], -1)
        X = self.encoding(X)
        
        return X
        

class SkipBlock(nn.Module):
    def __init__(self, in_dim, hid_dims, kernel_size=3, dilation=1, upsample=None, 
                 skip_conv=True, batch_norm=True):
        super().__init__()
        
#         self.bneck = lambda x: x
        
        
#         if bottle_neck is not None:
#             self.bneck = nn.Sequential(nn.Conv2d(in_dim, bottle_neck, 1),
#                                       nn.ReLU())
#             in_dim = bottle_neck
            
        last_dim = in_dim    
        block = []
        for dim in hid_dims:
            conv = nn.Conv2d(last_dim, dim, kernel_size=kernel_size, stride=1, 
                             padding=kernel_size//2, dilation=dilation)
            bn = nn.BatchNorm2d(dim)
            block.append(conv)
            block.append(bn)
            block.append(nn.ReLU())
            
            last_dim = dim
        
        self.block = nn.Sequential(*block)
        # if dont use skip_conv, the dim of X must match dim of output
        self.skip_conv = nn.Sequential(nn.Conv2d(in_dim, hid_dims[-1], 1), 
                                       nn.BatchNorm2d(hid_dims[-1]),
                                       nn.ReLU())\
                            if skip_conv else lambda x: x 
        
        self.upsample = upsample if upsample is not None else lambda x: x
        
    
    def forward(self, X):
#         X = self.bneck(X)
        out = self.block(X)
        out = out + self.skip_conv(X)
        out = self.upsample(out)
        
        return out
        
    
class Decoder(nn.Module):
    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size
        self.decoding = nn.Sequential(nn.Linear(code_size, 7*7*512),
                                     nn.ReLU())
        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        blocks = [SkipBlock(512+1, [128, 128, 256], upsample=upsample), # 14 x 14
                SkipBlock(256+1, [64, 64, 128], upsample=upsample), # 28 x 28
                SkipBlock(128+1, [64, 64, 64], upsample=upsample), # 56 x 56
                SkipBlock(64+1, [64, 64, 64], upsample=upsample), # 112 x 112
                SkipBlock(64+1, [64, 64, 64], upsample=upsample)] # 224 x 224
        
        self.blocks = nn.ModuleList([nn.DataParallel(bl) for bl in blocks])
        #self.im_branch = nn.Conv2d(64, 3, 1)
        #self.mask_branch = nn.Conv2d(64, 1, 1)
        self.out = nn.Conv2d(64, 4, 1)
        
        
    def forward(self, code, info, fg_idx):
        """
        Params:
            code: bsize x code_size, where 1st half is bg, and 2nd half is fg code respectively
                    source imgs' codes are on the first half of the batch, and destination imgs' one are
                    on the 2nd half.
            img: bsize x 3 x h x w, images corresponding to the code
            loc_enc: bsize x 1 x h x w, location encoding - (gaussian filter) (should not be the exact mask 
                            because we only want to give hint to where the network should add new object to
            fg_idx: should be a permutation of (0, batch_size-1) to match old bg code with new fg code.
        """
        # switch foreground code between source and target image
        bsize = code.shape[0]
#         X = X.contiguous().view(bsize, 2, self.code_size//2)
        bg, fg = code[:, :self.code_size//2], code[:, self.code_size//2:]
        fg = fg[fg_idx]
        X = torch.cat([bg, fg], dim=-1)
        
        #
        X = self.decoding(code)
        X = X.contiguous().view(X.shape[0], 512, 7, 7)
#         info = torch.cat([img, loc_enc], dim=1)
        for bl in self.blocks:
            scale_info = F.upsample(info, X.shape[-2:], mode='bilinear', align_corners=True)
            X = torch.cat([X, scale_info], dim=1)
            X = bl(X)
        
        #oimg = self.im_branch(X)
        #omask = self.mask_branch(X)
        return self.out(X)#oimg, omask


    
class Discriminator(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = torchv.models.resnet18(pretrained=pretrained)
        self.conv1 = nn.Sequential(nn.Conv2d(4, 64, 7, stride=2, padding=3),
                                  nn.ReLU())
#         self.backbone.layer1.register_forward_hook(self.hook)
#         self.backbone.layer2.register_forward_hook(self.hook)
#         self.backbone.layer3.register_forward_hook(self.hook)
#         self.backbone.layer4.register_forward_hook(self.hook)
        
        self.backbone = nn.DataParallel(nn.Sequential(*list(self.backbone.children()))[1:-1])
        self.out = nn.DataParallel(nn.Linear(512, 1))
        self.hook_out = []
        
    def hook(self, module, inp, out):
        self.hook_out.append(out)
        
    def forward(self, X):
        """
        """
        out = self.conv1(X)
        out = self.backbone(out)
        out = out.contiguous().view(out.shape[0], out.shape[1])
        out = self.out(out)
        return out
