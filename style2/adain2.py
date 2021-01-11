from .models import *
from torchvision.ops.misc import FrozenBatchNorm2d
import torchvision
from .datasets import *

from .adain import (
        stdize, unstdize, Unstdize,
        oneway_adain, oneway_adain_with_stats, make_viz,
        VGG,RELU1,RELU2,RELU3,RELU4)

UP = nn.UpsamplingBilinear2d
#UP = nn.UpsamplingNearest2d

def getStats(y):
    b,c,h,w = y.size()
    y = y.flatten(2)
    return y.mean(dim=2).view(b,c,1,1), y.std(dim=2).view(b,c,1,1).add(1e-9)

class Discriminator2(nn.Module):
    def __init__(self):
        super().__init__()
        layers = list(torchvision.models.resnet50(True,norm_layer=FrozenBatchNorm2d).children())[:6]
        self.bot = nn.Sequential(*layers[:6])
        self.top = nn.Sequential(
                nn.Conv2d(512*3,512,1,bias=True),
                nn.ReLU(True),
                nn.Conv2d(512,256,1,bias=True),
                nn.ReLU(True),
                nn.Conv2d(256,1,1,bias=False))
    def forward(self, x):
        z = self.bot(x)
        b,c,h,w = z.size()
        zmu, zstd = getStats(z)
        zmu,zstd = zmu.repeat(1,1,h,w), zstd.repeat(1,1,h,w)
        z2 = torch.cat((z,zmu,zstd),1)
        return self.top(z2)


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
    return nn.ModuleList([phi1,phi2,phi3,phi4])

def get_decoder_parts():
    #layers_ = list(VGG(True).eval().features)[:RELU4-1][::-1]
    layers_ = list(VGG(True).eval().features)[:RELU4-1]
    layers = []
    layers.append(Unstdize())
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
    decoder = nn.Sequential(*layers)
    dec4 = decoder[:RELU1][::-1]
    dec3 = decoder[RELU1:RELU2][::-1]
    dec2 = decoder[RELU2:RELU3][::-1]
    dec1 = decoder[RELU3:][::-1]
    return nn.ModuleList([dec1,dec2,dec3,dec4])

def rigid_aligment_C(c, s, eps=1e-9):
    B,C,H,W = c.size()
    c, s = c.flatten(2), s.flatten(2)
    c_, s_ = c - c.mean(2, keepdim=True), s - s.mean(2, keepdim=True)
    cn, sn = c.flatten(1).norm(dim=1, keepdim=True).add(eps).unsqueeze_(-1), s.flatten(1).norm(dim=1, keepdim=True).add(eps).unsqueeze_(-1)
    ch, sh = c_ / cn, s_ / sn
    t = ch.permute(0,2,1) @ sh
    svd = torch.svd(t)
    Q = svd[2].permute(0,2,1) @ svd[0].permute(0,2,1)
    return (cn * sh @ Q + c_).view(B, -1, H, W)

def rigid_aligment_HW(c, s, eps=1e-9):
    B,C,H,W = c.size()
    c, s = c.flatten(2), s.flatten(2)
    c_, s_ = c - c.mean(2, keepdim=True), s - s.mean(2, keepdim=True)
    cn, sn = c.flatten(1).norm(dim=1, keepdim=True).add(eps).unsqueeze_(-1), s.flatten(1).norm(dim=1, keepdim=True).add(eps).unsqueeze_(-1)
    ch, sh = c_ / cn, s_ / sn
    svd = torch.svd(ch.permute(0,2,1) @ sh.permute(0,2,1))
    Q = svd[2].permute(0,2,1) @ svd[0].permute(0,2,1)
    return (cn * (sh @ Q) + c_).view(B, -1, H, W)

#rigid_aligment = rigid_aligment_HW
rigid_aligment = rigid_aligment_C

class AdaInModelRA(BaseModule):
    def __init__(self, meta):
        meta.setdefault('title','unkAdainRA')
        meta.setdefault('lambdaContent',1.4)
        meta.setdefault('lambdaStyle',1.0)
        #meta.setdefault('lambdaTV',55)
        meta.setdefault('lambdaTV',-1)
        meta.setdefault('lambdaGan',5)
        #meta.setdefault('useGan',True)
        meta.setdefault('useGan',False)
        meta.setdefault('ganNorm',None)
        #meta['lambdaTV'] = 0
        #meta['lambdaGan'] = 35
        meta['lambdaGan'] = 10
        meta['lambdaContent'] = 2
        super().__init__(meta)

        self.enc = get_encoder_parts()
        self.enc.requires_grad_(False)
        self.encoder = nn.Sequential(*self.enc)
        self.dec = get_decoder_parts()

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
            self.discriminator2 = Discriminator2()

        self.lastXstats = None


    def forward(self, x, y):
        n = x.size(0)
        xy = stdize(torch.cat((x,y),0))
        f1 = self.enc[0](xy)
        f2 = self.enc[1](f1)
        f3 = self.enc[2](f2)
        f4 = self.enc[3](f3)

        xx, yy = oneway_adain_with_stats(f4[:n], *getStats(f4[n:])), f4[n:]

        xx = rigid_aligment(xx, yy)

        fx4s = self.dec[0](xx)
        fx3s = self.dec[1](oneway_adain_with_stats(fx4s, *getStats(f3[n:])))
        fx2s = self.dec[2](oneway_adain_with_stats(fx3s, *getStats(f2[n:])))
        fx1s = self.dec[3](oneway_adain_with_stats(fx2s, *getStats(f1[n:])))

        return dict(
                f1=f1,f2=f2,f3=f3,f4=f4,
                fx1s=fx1s)

    def forward_with_stats(self, x, stats, fxOld=None,oldGamma=0):
        fx1 = self.enc[0](stdize(x))
        fx2 = self.enc[1](fx1)
        fx3 = self.enc[2](fx2)
        fx4 = self.enc[3](fx3)

        if fxOld is not None:
            fx1 = fxOld[0] * oldGamma + fx1 * (1-oldGamma)
            fx2 = fxOld[1] * oldGamma + fx2 * (1-oldGamma)
            fx3 = fxOld[2] * oldGamma + fx3 * (1-oldGamma)
            fx4 = fxOld[3] * oldGamma + fx4 * (1-oldGamma)

        # Blending stats helps avoid rapid, large intensity changes.
        # More could be done to help fix this problem (replace stride with dilation!)
        #xx, yy = oneway_adain_with_stats(fx4, stats[6],stats[7]), stats[8]
        yy = stats[8]
        b,c,h,w = fx4.size()
        xx = fx4.view(b,c,h*w)
        xmu,xstd = xx.mean(dim=2), xx.std(dim=2)+1e-9
        if self.lastXstats is None:
            xmu = xmu.view(b,c,1,1)
            xstd = xstd.view(b,c,1,1)
        else:
            xmu = (xmu.view(b,c,1,1) + self.lastXstats[0]*3) / 4
            xstd = (xstd.view(b,c,1,1) + self.lastXstats[1]*3) / 4
        xx = ((fx4-xmu)/xstd) * stats[7] + stats[6]
        self.lastXstats = [xmu,xstd]

        xx = rigid_aligment(xx,yy)
        #xx2 = rigid_aligment(F.avg_pool2d(xx,2,2), F.avg_pool2d(yy,2,2))
        #xx = xx + F.interpolate(xx2, scale_factor=2, mode='bilinear')

        fx4s = self.dec[0](xx)
        fx3s = self.dec[1](oneway_adain_with_stats(fx4s, stats[4],stats[5]))
        fx2s = self.dec[2](oneway_adain_with_stats(fx3s, stats[2],stats[3]))
        fx1s = self.dec[3](oneway_adain_with_stats(fx2s, stats[0],stats[1]))
        return fx1s, (fx1,fx2,fx3,fx4)
    def forward_get_stats(self, y):
        stats = []
        y = stdize(y)
        for blk in self.enc:
            y = blk(y)
            b,c = y.size(0),y.size(1)
            yy = y.flatten(2)
            ymu,ystd = yy.mean(dim=2).view(b,c,1,1), (yy.std(dim=2)+1e-9).view(b,c,1,1)
            stats.append(ymu)
            stats.append(ystd)
        stats.append(y) # Also include the 'style point cloud'
        return stats

    def losses(self, x, y, pred):
        fx1s = pred['fx1s']
        n = fx1s.size(0)
        both = stdize(torch.cat((y,fx1s),0))
        s1 = self.enc[0](both)
        s2 = self.enc[1](s1)
        s3 = self.enc[2](s2)
        s4 = self.enc[3](s3)
        loss_s = 0
        sw = self.lambdaStyle
        for s,w in zip((s1,s2,s3,s4),(sw*.8,sw*1,sw*1,sw*1)):
            loss_s = loss_s + w * (
                    (s[:n].flatten(2).mean(2) - s[n:].flatten(2).mean(2)).norm(dim=1).mean() +
                    (s[:n].flatten(2).std(2)  - s[n:].flatten(2).std(2)).norm(dim=1).mean() )

        loss_c = (pred['f4'][:n] - s4[n:]).norm(dim=1).mean() * self.lambdaContent

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
            d2 = self.discriminator2(stdize(fx1s))
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
            fx1s = pred['fx1s']
            both = stdize(torch.cat((y,fx1s),0))
            s = self.enc[0](both)
            s = self.enc[1](s)
            s = self.enc[2](s)

        d = self.discriminator(s)
        d2 = self.discriminator2(both)

        b,c,h,w = d.size()
        labels = 1-torch.arange(2,device=x.device,dtype=torch.float32).repeat_interleave(b//2).view(b,1,1,1).repeat(1,1,h,w)
        labels2 = labels[..., :h//2, :w//2]
        loss = F.binary_cross_entropy_with_logits(d, labels) * self.lambdaGan
        loss = loss + F.binary_cross_entropy_with_logits(d2, labels2) * self.lambdaGan
        return loss, dict(dd=loss.item()), d, d2


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

    model = AdaInModelRA(meta).cuda().train()
    print(model)

    if args.load is not None:
        d = torch.load(args.load)
        ii0 = d['epoch'] + 1
        model.load_state_dict(d['sd'])
    else:
        ii0 = 0

    ks = [k for k,v in model.named_parameters() if 'enc' not in k and 'phi' not in k and 'discrim' not in k]
    print(' - Optimizing', ks)
    ps = [v for k,v in model.named_parameters() if 'enc' not in k and 'phi' not in k and 'discrim' not in k]
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
            make_viz(x,y,model,ii, name='adaRA', reconsKey='fx1s')
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

