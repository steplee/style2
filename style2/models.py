import torch, torch.nn as nn, torch.nn.functional as F
import cv2, numpy as np, os, sys
from torchvision.ops.misc import FrozenBatchNorm2d

relu = lambda: nn.LeakyReLU(inplace=True)
#relu = lambda: nn.ReLU(inplace=True)

def load_adain_model(ckpt):
    d = torch.load(ckpt)
    clazz = d['clazz'].split("'")[1].split('.')[-1]
    if clazz == 'AdaInModel':
        from .adain import AdaInModel
        model = AdaInModel({}).cuda().eval()
    elif clazz == 'AdaInModelRA':
        from .adain2 import AdaInModelRA
        model = AdaInModelRA({}).cuda().eval()
    model.load_state_dict(d['sd'], strict=False)
    return model

def get_bb(meta):
    from torchvision.models.vgg import vgg16_bn
    m = vgg16_bn(True).features[:32].detach()
    m.requires_grad_(False)
    return m


def get_phi_network():
    from torchvision.models.resnet import resnet50
    #m = nn.Sequential(*list(resnet50(True,norm_layer=FrozenBatchNorm2d).children())[:5])
    from torchvision.models.vgg import vgg16_bn
    #m = vgg16_bn(True).features[:19]
    m = vgg16_bn(True).features[:26]
    #m = vgg16_bn(True).features[:3]
    #m = vgg16_bn(True).features[:6]
    #m = vgg16_bn(True).features[:13]
    return m.eval()

class BaseModule(nn.Module):
    def __init__(self, meta):
        super().__init__()
        self.meta = meta
        for k,v in meta.items(): setattr(self,k,v)
    def save(self, epoch):
        fname = 'saves/{}.{}.pt'.format(self.title, epoch)
        torch.save({'sd': self.state_dict(), 'epoch':epoch, 'meta': self.meta, 'clazz': str(self.__class__)}, fname)
        print(' - Saved', fname)



class G(BaseModule):
    def __init__(self, meta):
        meta.setdefault('title', 'unkG')
        super().__init__(meta)

class D(BaseModule):
    def __init__(self, meta):
        meta.setdefault('title', 'unkD')
        super().__init__(meta)


class G_1(G):
    def __init__(self, meta):
        meta.setdefault('title', 'unkG1')
        super().__init__(meta)

        normLayer = nn.BatchNorm2d
        #normLayer = nn.InstanceNorm2d
        bias = False
        #normLayer = lambda c: nn.Sequential()
        #bias = True

        self.net = nn.Sequential(
                #nn.Conv2d(3,32,3,1,1, bias=bias),
                #normLayer(32),
                #relu(),
                nn.Conv2d(3,64,3,1,1, bias=bias),
                #nn.Conv2d(32,64,3,1,1, bias=bias),
                normLayer(64),
                relu(),

                #nn.AvgPool2d(2,2),

                nn.Conv2d(64,256,3,2,1, bias=bias),
                normLayer(256),
                relu(),
                nn.Conv2d(256,512,3,1,1, bias=bias),
                normLayer(512),
                relu(),

                #nn.AvgPool2d(2,2),

                nn.Conv2d(512,256,3,2,1, bias=bias),
                #normLayer(256),
                relu(),
                nn.ConvTranspose2d(256,256,4,2,1),
                #normLayer(256),
                relu(),
                nn.ConvTranspose2d(256,128,4,2,1, bias=bias),
                #normLayer(128),
                relu(),

                nn.Conv2d(128,3,3,1,1))
        with torch.no_grad():
            #self.net[-1].weight.data.mul_(.1)
            #self.net[-1].bias.data.fill_(0)
            pass

    def forward(self, x):
        return self.net(x.sub(.4).div(.3)).mul(.3).add(.4).clamp(0,1)

class D_1(D):
    def __init__(self, meta):
        meta.setdefault('title', 'unkD1')
        super().__init__(meta)

        normLayer = nn.BatchNorm2d
        #normLayer = nn.InstanceNorm2d
        bias = False
        normLayer = lambda c: nn.Sequential()
        bias = True

        PAD = 0
        '''
        self.net = nn.Sequential(
                nn.Conv2d(3,256,3,3,PAD,bias=bias),
                relu(),
                nn.Conv2d(256,256,3,3,PAD,bias=bias),
                normLayer(256),
                relu(),

                nn.Conv2d(256,512,3,3,PAD,bias=bias),
                normLayer(512),
                relu(),

                nn.Conv2d(512, 1, 1,1,0,bias=False),
                #nn.AdaptiveAvgPool2d(1),
                #nn.Flatten()
                )
        '''
        '''
        from torchvision.models.resnet import resnet50
        self.net = nn.Sequential(
                *list(resnet50(True,norm_layer=FrozenBatchNorm2d).children())[:5],
                nn.Conv2d(256,1,1,bias=False))
        '''
        from torchvision.models.vgg import vgg16_bn
        self.net = nn.Sequential(
                *list(vgg16_bn(True).features[:17]),
                nn.Conv2d(256,1,1,bias=False))
        '''
        self.net = nn.Sequential(
                #*list(vgg16_bn(True).features[:6]),
                *[l for l in (vgg16_bn(True).features[:17]) if not isinstance(l, nn.MaxPool2d)],
                nn.Conv2d(256,1,1,bias=False))
        for m in self.net.children():
            if isinstance(m,nn.Conv2d):
                if m.kernel_size[0] == 3:
                    m.stride = (3,3)
        '''
        print('D:\n', self.net)

    def forward(self,x):
        x = x.sub(.4).div(.3)
        #return self.net(x)
        return self.net(x)



if __name__ == '__main__':
    with torch.no_grad():
        m = G_1({}).cuda().eval()

        img_ = cv2.imread('/home/slee/Downloads/canyon_hard1.jpg')
        img_ = img_[100:100+512,100:100+512]
        img0 = torch.from_numpy(img_).permute(2,0,1).unsqueeze_(0).cuda()
        x = img0.to(torch.float32).div(255)

        dps = [torch.randn_like(p) for p in m.parameters()]
        ops = [p.data.clone() for p in (m.parameters())]

        for i in range(1000):
            for op,p,dp in zip(ops,m.parameters(),dps):
                p.copy_(op.add_(dp*.1).sin().mul_(.049))
                #p.add_(dp*.01).cos_().mul_(.01)

            y = m(x).mul_(255).clamp(0,255).cpu().to(torch.uint8)

            dimg = torch.cat((img0[0].cpu(), y[0]), 1).permute(1,2,0).numpy()
            dimg = np.copy(dimg,'C')
            cv2.imshow('img',dimg)
            cv2.waitKey(1)

