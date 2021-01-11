from ..models import *
from ..datasets import *

from ..perm import perm
def perm(x): return x

import gc
import matplotlib.pyplot as plt

SHOW = False
PERM_GX = True


def train_p2(styleImageNames):

    g = G_1({}).cuda().train()
    d = D_1({}).cuda().train()
    p = get_phi_network().cuda().eval()

    sz, N, M = 256, 12, 6
    sz, N, M = 128, 24, 12
    sz, N, M = 128+64, 24, 12
    sz, N, M = 128+64, 38, 14
    #sz, N, M = 24*9, 20, 10
    #sz, N, M = 128+64+32, 20, 10
    sz, N, M = 27*8, 20, 10

    s_dset = ListedDataset(styleImageNames, imgSize=sz)
    s_dloader = DataLoader(s_dset, batch_size=M, num_workers=8, drop_last=True, worker_init_fn=worker_init_fn_)
    s_iter = iter(s_dloader)

    c_dset = WikiartDataset('/data/places/places365_standard/train/', imgSize=sz)
    c_dloader = DataLoader(c_dset, batch_size=N, num_workers=8, drop_last=True, worker_init_fn=worker_init_fn_)
    c_iter = iter(c_dloader)

    #gopt = torch.optim.Adam(g.parameters(), lr=4e-4)
    #dopt = torch.optim.Adam(d.parameters(), lr=4e-4)
    #gopt = torch.optim.Adam(g.parameters(), lr=1e-4)
    #dopt = torch.optim.Adam(d.parameters(), lr=2e-4)
    gopt = torch.optim.RMSprop(g.parameters(), lr=1e-4)
    dopt = torch.optim.RMSprop(d.parameters(), lr=1e-4)
    #gopt = torch.optim.Adam(g.parameters(), lr=1e-5)
    #dopt = torch.optim.Adam(d.parameters(), lr=1e-5)
    #gopt = torch.optim.SGD(g.parameters(), lr=1e-3, momentum=.6)
    #dopt = torch.optim.SGD(d.parameters(), lr=1e-3, momentum=.6)

    lambdaContent = .20
    lossHistory = {}
    runningLosses = {}

    for ii in range(99999):
        with torch.no_grad():
            try: simgs = next(s_iter)
            except StopIteration: s_iter = iter(s_dloader)
            try: ccls,cimgs = next(c_iter)
            except StopIteration: c_iter = iter(c_dloader)

            with torch.no_grad():
                simgs = perm(simgs.permute(0,3,1,2)).permute(0,2,3,1)
            #cimgs = perm(cimgs.permute(0,3,1,2)).permute(0,2,3,1)


            simgs = simgs.cuda().to(torch.float32).div_(255).permute(0,3,1,2)
            cimgs = cimgs.cuda().to(torch.float32).div_(255).permute(0,3,1,2)



        # Train D
        DI = 1
        for jj in range(DI):
            gx = perm(g(cimgs[:M])) if PERM_GX else (g(cimgs[:M]))

            xx = torch.cat( (gx,simgs) , 0 )
            ng = gx.size(0)
            labels = torch.arange(2,device=xx.device).repeat_interleave(ng).to(torch.float32)

            '''
            d_xx = d(xx)[:, 0]
            b,h,w = d_xx.size()

            #loss_adv = d_xx[ng:].log().mean() + (1 - d_xx[:ng]).log().mean()
            #loss_adv_ = loss_adv.item()
            loss_adv_g = F.binary_cross_entropy_with_logits(d_xx[0:ng], labels[ng:].view(-1,1,1).repeat(1,h,w))
            loss_adv_d = F.binary_cross_entropy_with_logits(d_xx[:   ], labels.view(-1,1,1).repeat(1,h,w))
            loss_adv = loss_adv_g + loss_adv_d
            loss_adv_g_,loss_adv_d_ = loss_adv_g.item(), loss_adv_d.item()

            loss_adv.backward()
            gopt.step()
            gopt.zero_grad()
            dopt.step()
            dopt.zero_grad()
            '''
            d_xx = d(xx.detach())[:, 0]
            b,h,w = d_xx.size()
            loss_adv_d = F.binary_cross_entropy_with_logits(d_xx, labels.view(-1,1,1).repeat(1,h,w))
            loss_adv_d_ = loss_adv_d.item()
            loss_adv_d.backward()
            dopt.step()
            dopt.zero_grad()
            gopt.zero_grad()
            del d_xx, loss_adv_d
            if jj < DI-1:
                try: ccls,cimgs = next(c_iter)
                except: c_iter = iter(c_dloader)
                try: simgs = next(s_iter)
                except StopIteration: s_iter = iter(s_dloader)
                cimgs = cimgs.cuda().to(torch.float32).div_(255).permute(0,3,1,2)
                with torch.no_grad():
                    simgs = perm(simgs.permute(0,3,1,2)).permute(0,2,3,1)
                simgs = simgs.cuda().to(torch.float32).div_(255).permute(0,3,1,2)


        del gx
        try: ccls,cimgs = next(c_iter)
        except StopIteration: c_iter = iter(c_dloader)
        cimgs = cimgs.cuda().to(torch.float32).div_(255).permute(0,3,1,2)
        gx = perm(g(cimgs[:M])) if PERM_GX else (g(cimgs[:M]))

        #d = d.eval()
        d.requires_grad_(False)
        d_gx = d(gx)[:, 0]
        d.requires_grad_(True)
        #d = d.train()
        #d_gx = d(xx)[:ng, 0]
        loss_adv_g = F.binary_cross_entropy_with_logits(d_gx, labels[ng:].view(-1,1,1).repeat(1,h,w))
        loss_adv_g_ = loss_adv_g.item()
        loss_adv_g.backward()
        del d_gx, loss_adv_g

        # Train G
        gx = g(cimgs)
        phi_gx = p(gx.sub(.4).div(.3))
        phi_x = p(cimgs.sub(.4).div(.3))
        #print(phi_gx.shape, phi_x.shape)
        loss_p = (phi_gx - phi_x).norm(dim=1).mean() * lambdaContent
        loss_p_ = loss_p.item()
        loss_p.backward()
        gopt.step()


        dopt.zero_grad()
        gopt.zero_grad()

        if ii % 25 == 0:
            with torch.no_grad():
                MM = min(M,8)
                simgs_ = torch.cat(tuple(simgs[:MM].permute(0,2,3,1)),1).mul_(255).to(torch.uint8).cpu().numpy()
                cimgs_ = torch.cat(tuple(cimgs[:MM].permute(0,2,3,1)),1).mul_(255).to(torch.uint8).cpu().numpy()
                gx_ = torch.cat(tuple(gx.detach()[:MM].permute(0,2,3,1)),1).clamp_(0,1).mul_(255).to(torch.uint8).cpu().numpy()
                gx_p = torch.cat(tuple(perm(gx).detach()[:MM].permute(0,2,3,1)),1).clamp_(0,1).mul_(255).to(torch.uint8).cpu().numpy()
                dimg = cv2.cvtColor(np.copy(np.vstack((simgs_,cimgs_,gx_,gx_p)),'C'),cv2.COLOR_RGB2BGR)
                if SHOW: cv2.imshow('d',dimg);cv2.waitKey(1)
                cv2.imwrite('out/{:06d}.jpg'.format(ii),dimg)
                del gx_,gx_p,simgs_,cimgs_,


        del gx,phi_gx,phi_x,loss_p
        gc.collect()



        print(ii, loss_p_, loss_adv_g_, loss_adv_d_)
        runningLosses['d'] = runningLosses.get('d',loss_adv_d_)*.95 + .05*loss_adv_d_
        runningLosses['g'] = runningLosses.get('g',loss_adv_g_)*.95 + .05*loss_adv_g_
        runningLosses['p'] = runningLosses.get('p',loss_p_)*.95 + .05*loss_p_
        if ii % 5 == 0:
            for k,v in runningLosses.items():
                if k not in lossHistory: lossHistory[k] = []
                lossHistory[k].append(v)
            if ii % 25 == 0:
                plt.clf()
                for k,v in lossHistory.items():
                    plt.plot(v,label=k)
                plt.legend()
                plt.savefig('out/loss.jpg')
                plt.clf()



if __name__ == '__main__':
    cubism = '/data/wikiart/Synthetic_Cubism/juan-gris_fruit-bowl-with-bottle.jpg /data/wikiart/Synthetic_Cubism/juan-gris_guitar-on-the-table-1913.jpg /data/wikiart/Synthetic_Cubism/juan-gris_the-book-1913.jpg /data/wikiart/Synthetic_Cubism/juan-gris_the-guitar-1913.jpg'.split(' ')
    #cubism = '/data/wikiart/Synthetic_Cubism/juan-gris_fruit-bowl-with-bottle.jpg'.split(' ')
    imgs=cubism

    #gogh=['/data/wikiart/Post_Impressionism/vincent-van-gogh_the-starry-night-1888-2.jpg']
    #gogh=['/home/stephen/Pictures/space.png']
    #imgs=gogh

    train_p2(imgs)
