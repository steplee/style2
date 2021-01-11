from .models import *
from torchvision.ops.misc import FrozenBatchNorm2d
import torchvision
from .datasets import *
import time


imgnet_mu = torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1).cuda()
imgnet_sd = torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1).cuda()
def stdize(x): return x.sub(imgnet_mu).div(imgnet_sd)
def unstdize(x): return x.mul(imgnet_sd).add(imgnet_mu)
class Unstdize(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return unstdize(x).clamp(0,1)

if False:
    # For vgg19_bn
    VGG = torchvision.models.vgg19_bn
    RELU1 = 3
    RELU2 = 9
    RELU3 = 17
    RELU4 = 30
else:
    # For vgg19
    VGG = torchvision.models.vgg19
    RELU1 = 2
    RELU2 = 7
    RELU3 = 12
    RELU4 = 21
    RELU5 = 30

def get_encoder_parts():
    layers = list(VGG(True).features[:RELU4])
    layers_ = []
    for layer in layers:
        if isinstance(layer,nn.BatchNorm2d):
            new_layer = FrozenBatchNorm2d(layer.weight.shape[0])
            new_layer.load_state_dict(layer.state_dict())
            layers_.append(new_layer)
        else: layers_.append(layer)

    vgg = nn.Sequential(*layers_)
    vgg.requires_grad_(False)
    phi1 = vgg[:RELU1]
    phi2 = vgg[RELU1:RELU2]
    phi3 = vgg[RELU2:RELU3]
    phi4 = vgg[RELU3:RELU4]
    return phi1,phi2,phi3,phi4



def oneway_adain(x,y):
    b,c,h,w = x.size()
    xx,yy = x.view(b,c,h*w), y.view(b,c,h*w)
    ymu,ystd = yy.mean(dim=2), yy.std(dim=2)+1e-9
    xmu,xstd = xx.mean(dim=2), xx.std(dim=2)+1e-9
    xmu = xmu.view(b,c,1,1)
    ymu = ymu.view(b,c,1,1)
    xstd = xstd.view(b,c,1,1)
    ystd = ystd.view(b,c,1,1)

    return ((x-xmu)/xstd) * ystd + ymu

def oneway_adain_with_stats(x,ymu,ystd):
    b,c,h,w = x.size()
    xx = x.view(b,c,h*w)
    xmu,xstd = xx.mean(dim=2), xx.std(dim=2)+1e-9
    xmu = xmu.view(b,c,1,1)
    xstd = xstd.view(b,c,1,1)

    #ymu = ymu + torch.randn_like(ymu) * np.sin(time.time())

    return ((x-xmu)/xstd) * ystd + ymu

UP = nn.UpsamplingBilinear2d
#UP = nn.UpsamplingNearest2d

def Decoder3():
    layers_ = list(VGG(True).eval().features)[:RELU4-1][::-1]
    layers = []
    for layer in layers_:
        if isinstance(layer,nn.BatchNorm2d):
            continue
        if isinstance(layer,nn.MaxPool2d):
            layers.append(UP(scale_factor=2))
        if isinstance(layer,nn.Conv2d):
            cout,cin,k,kk = layer.weight.shape
            layers.append(nn.Conv2d(cout,cin,k,padding=k//2))
        if isinstance(layer,nn.ReLU):
            layers.append(nn.ReLU(True))
    layers.append(Unstdize())
    return nn.Sequential(*layers)

class AdaInModel(BaseModule):
    def __init__(self, meta):
        meta.setdefault('title','unkAdain')
        meta.setdefault('lambdaContent',1.3)
        meta.setdefault('lambdaStyle',1.4)
        #meta.setdefault('lambdaTV',55)
        meta.setdefault('lambdaTV',-1)
        meta.setdefault('lambdaGan',11)
        meta.setdefault('useGan',True)
        meta.setdefault('ganNorm',None)
        #meta.setdefault('lambdaTV',-1)
        meta['lambdaTV'] = 0
        meta['lambdaGan'] = 35
        super().__init__(meta)

        self.phi1,self.phi2,self.phi3,self.phi4 = get_encoder_parts()
        self.encoder = nn.Sequential(self.phi1,self.phi2,self.phi3,self.phi4)

        #self.decoder = Decoder2()
        self.decoder = Decoder3()

        # Note: My TV loss is applied on grayscale values.
        self.kernelTV = torch.FloatTensor([
            0,-1,0,
            -1,4.,-1,
            0,-1,0]).view(1,1,3,3)
        self.kernelTV.requires_grad_(False)

        if self.useGan:
            self.discriminator = nn.Sequential(
                    #self.phi1, self.phi2, self.phi3,
                    nn.Conv2d(256, 128, 3, 1, 1),
                    #nn.Conv2d(128, 128, 3, 1, 1),
                    nn.ReLU(True),
                    nn.Conv2d(128, 64, 3, 1, 1),
                    nn.ReLU(True),
                    #nn.Conv2d(64, 1, 1, bias=False))
                    nn.Conv2d(64, 1, 1, bias=True))
            self.discriminator2 = nn.Sequential(
                    *list(torchvision.models.resnet50(True,norm_layer=FrozenBatchNorm2d).children())[:6],
                    nn.Conv2d(512,1,1,bias=False))


    def forward(self, x, y):
        n = x.size(0)
        xy = stdize(torch.cat((x,y),0))
        #fx,fy = self.encoder(xy).split(x.size(0))
        f1s = self.phi1(xy)
        f2s = self.phi2(f1s)
        f3s = self.phi3(f2s)
        f4s = self.phi4(f3s)

        fx4s = oneway_adain(f4s[:n],f4s[n:])

        dfx2 = self.decoder(fx4s)

        return dict(
                f1s=f1s, f2s=f2s, f3s=f3s, f4s=f4s,
                fx4s=fx4s,dfx2=dfx2)

    def forward_with_stats(self, x, stats, fxOld=None,oldGamma=.9):
        mu,std = stats
        fx = self.encoder(stdize(x))
        if fxOld is not None:
            fx = fxOld*oldGamma + fx*(1-oldGamma)
        fx4s = oneway_adain_with_stats(fx, mu,std)
        dfx2 = self.decoder(fx4s)
        return dfx2, fx
    def forward_get_stats(self, y):
        y = self.encoder(stdize(y))
        b,c,h,w = y.size()
        yy = y.view(b,c,h*w)
        ymu,ystd = yy.mean(dim=2), yy.std(dim=2)+1e-9
        ymu = ymu.view(b,c,1,1)
        ystd = ystd.view(b,c,1,1)
        return ymu,ystd

    def losses(self, x, y, pred):
        fx4s, dfx2 = pred['fx4s'], pred['dfx2']
        n = dfx2.size(0)
        both = stdize(torch.cat((y,dfx2),0))
        s1 = self.phi1(both)
        s2 = self.phi2(s1)
        s3 = self.phi3(s2)
        s4 = self.phi4(s3)
        loss_s = 0
        sw = self.lambdaStyle
        for s,w in zip((s1,s2,s3,s4),(sw*.8,sw*1,sw*1,sw*1)):
            #loss_s = loss_s + w * (s1[:n]-s1[n:]).norm(dim=1).mean()
            loss_s = loss_s + w * (
                    (s[:n].flatten(2).mean(2) - s[n:].flatten(2).mean(2)).norm(dim=1).mean() +
                    (s[:n].flatten(2).std(2)  - s[n:].flatten(2).std(2)).norm(dim=1).mean() )

        #loss_c = (fx4s - s4[n:]).norm(dim=1).mean() * self.lambdaContent
        loss_c = (pred['fx4s'][:n] - s4[n:]).norm(dim=1).mean() * self.lambdaContent
        loss_c = loss_c + (pred['f3s'][:n] - s3[n:]).norm(dim=1).mean() * self.lambdaContent * .3

        loss = loss_s + loss_c
        losses = dict(c=loss_c.item(), s=loss_s.item())

        #loss_tv = torch.zeros(1,device=loss_c.device)
        if self.lambdaTV > 0:
            loss_tv = self.tv_loss(dfx2) * self.lambdaTV
            loss = loss + loss_tv
            losses['tv'] = loss_tv.item()

        if self.useGan:
            #d_input = s2
            d_input = s3
            b,_,h,w = d_input.size()
            labels = torch.ones((n,1,h,w), device=x.device)
            labels2 = labels[..., :h//2, :w//2]
            d = self.discriminator(d_input[n:])
            d2 = self.discriminator2(stdize(dfx2))
            loss_dg = F.binary_cross_entropy_with_logits(d, labels) * self.lambdaGan
            loss_dg = loss_dg + F.binary_cross_entropy_with_logits(d2, labels2) * self.lambdaGan
            loss = loss + loss_dg
            losses['dg'] = loss_dg.item()

        return loss, losses

    def tv_loss(self, x):
        x = x.mean(dim=1,keepdim=True)
        d = F.conv2d(x, self.kernelTV.to(x.device))
        return d.abs().mean()

    def losses_d(self, x, y):
        with torch.no_grad():
            pred = self.forward(x,y)
            dfx2 = pred['dfx2']
            both = stdize(torch.cat((y,dfx2),0))
            s = self.phi1(both)
            s = self.phi2(s)
            s = self.phi3(s)

        d = self.discriminator(s)
        d2 = self.discriminator2(both)

        b,c,h,w = d.size()
        labels = 1-torch.arange(2,device=x.device,dtype=torch.float32).repeat_interleave(b//2).view(b,1,1,1).repeat(1,1,h,w)
        labels2 = labels[..., :h//2, :w//2]
        loss = F.binary_cross_entropy_with_logits(d, labels) * self.lambdaGan
        loss = loss + F.binary_cross_entropy_with_logits(d2, labels2) * self.lambdaGan
        return loss, dict(dd=loss.item()), d, d2


def make_viz(x,y,model, epoch, name='ada', reconsKey='dfx2'):
    with torch.no_grad():
        model = model.eval()
        x,y = x[:8],y[:8]
        z = model(x,y)[reconsKey]
        dimg = torch.cat((x,y,z),2)
        dimg = torch.cat(tuple(dimg.permute(0,2,3,1)),1)
        dimg = dimg.mul_(255).to(torch.uint8).cpu().numpy()
        dimg = cv2.cvtColor(dimg, cv2.COLOR_RGB2BGR)
        cv2.imwrite('out/{}{:06d}.jpg'.format(name,epoch), dimg)
        model = model.train()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batchSize', default=10, type=int)
    parser.add_argument('--imgSize', default=256, type=int)
    parser.add_argument('--load', default=None)
    parser.add_argument('--noGan', action='store_true')
    args = parser.parse_args()

    meta = dict(useGan=not args.noGan)

    model = AdaInModel(meta).cuda().train()
    print(model)

    if args.load is not None:
        d = torch.load(args.load)
        ii0 = d['epoch'] + 1
        model.load_state_dict(d['sd'])
    else:
        ii0 = 0

    ks = [k for k,v in model.named_parameters() if 'encoder' not in k and 'phi' not in k and 'discrim' not in k]
    print(' - Optimizing', ks)
    ps = [v for k,v in model.named_parameters() if 'encoder' not in k and 'phi' not in k and 'discrim' not in k]
    opt = torch.optim.Adam(ps, args.lr)
    #opt = torch.optim.RMSprop(ps, args.lr)
    #sched = torch.optim.lr_scheduler.ExponentialLR(opt, .9998) # .9998     ^ 10000  ~= 13%
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, .999955) # .999955 ^ 100000 ~= 1.1%

    if model.useGan:
        ks = [k for k,v in model.named_parameters() if 'discriminator' in k]
        print(' - Optimizing Discriminator', ks)
        ps = [v for k,v in model.named_parameters() if 'discriminator' in k]
        opt_d = torch.optim.Adam(ps, args.lr/2)

    c_dset = WikiartDataset('/data/places/places365_standard/train/', imgSize=args.imgSize)
    c_dloader = DataLoader(c_dset, batch_size=args.batchSize, num_workers=4, drop_last=True, worker_init_fn=worker_init_fn_)
    c_iter = iter(c_dloader)
    s_dset = WikiartDataset('/data/wikiart/', imgSize=args.imgSize)
    s_dloader = DataLoader(s_dset, batch_size=args.batchSize, num_workers=4, drop_last=True, worker_init_fn=worker_init_fn_)
    s_iter = iter(s_dloader)

    lossHistory = {}
    runningLosses = {}

    for ii in range(ii0,ii0+100000):
        try: ccls,cimgs = next(c_iter)
        except StopIteration:
            c_iter = iter(c_dloader)
            ccls,cimgs = next(c_iter)
        try: scls,simgs = next(s_iter)
        except StopIteration:
            s_iter = iter(s_dloader)
            scls,simgs = next(s_iter)

        x = cimgs.cuda().float().permute(0,3,1,2).div_(255.)
        y = simgs.cuda().float().permute(0,3,1,2).div_(255.)

        if ii % 25 == 0:
            make_viz(x,y,model,ii)
        if ii % 1000 == 0:
            model.save(ii)

        pred = model(x,y)

        loss,losses = model.losses(x,y,pred)

        loss.backward()
        opt.step()
        sched.step()
        model.zero_grad()
        #opt.zero_grad()

        if model.useGan:
            loss_d,losses_d,d_pred,d_pred2 = model.losses_d(x,y)
            loss_d.backward()
            opt_d.step()
            model.zero_grad()
            if ii % 100 == 0: print(' - avg pred_d :', d_pred.detach().cpu().flatten(1).mean(1).view(2,-1).mean(1))
            if ii % 100 == 0: print(' - avg pred_d2:', d_pred2.detach().cpu().flatten(1).mean(1).view(2,-1).mean(1))
            losses.update(losses_d)


        if ii % 100 == 0 or ii < 10:
            print(ii, runningLosses)
        for k,v in losses.items(): runningLosses[k] = runningLosses.get(k,v)*.95 + .05*v
        if ii % 25 == 0:
            for k,v in runningLosses.items():
                if k not in lossHistory: lossHistory[k] = []
                lossHistory[k].append(v)
            if ii % 100 == 0 and ii > 1:
                import matplotlib.pyplot as plt
                plt.clf()
                plt.title('Batch {}, lr {:.3}'.format(ii, float(sched.get_last_lr()[0])))
                for k,v in lossHistory.items():
                    plt.plot(v,label=k)
                plt.legend()
                plt.savefig('out/loss.jpg')
                plt.clf()




if __name__ == '__main__':
    main()
    '''
    model = AdaInModel({})

    x = torch.randn(1,3,256,256)
    y = torch.randn(1,3,256,256)
    print('in ',x.shape)
    pred = model(x,y)
    print('out',pred['dfx2'].shape)

    loss, losses = model.losses(x,y,pred)
    print('loss',loss)
    print('losses',losses)
    '''
