import torch, cv2, numpy as np

#def perm(x, res=9):
#def perm(x, res=16):
def perm(x, res=27):
    b,c,h,w = x.size()

    hr,wr = h // res, w // res
    #x = x.view(b,c, hr, res, wr, res)
    #x = x.permute(0,1, 2,4, 3,5).reshape(b,c, res,res, hr*wr)

    #x = x.view(b,c, res, hr, res, wr)
    #x = x.permute(0,1, 2,4, 3,5).reshape(b,c, res,res, hr*wr)

    #x = x.view(b,c, res, hr, res, wr)
    #x = x.permute(0,1, 2,4, 3,5).reshape(b,c, res,res, hr*wr)
    #x = x.permute(0, 3,5, 2,4, 1).reshape(b,hr*wr, res,res, c)
    #x = x.permute(0, 3,5, 2,4, 1).reshape(b,hr*wr, res,res, c)

    x = x.view(b,c, hr, res, wr, res)
    x = x.permute(0, 3,5, 2,4, 1).reshape(b, res,res, hr*wr, c).permute(0, 3,1,2, 4)

    # Not good: no batched mode
    #torch.randperm()

    a = torch.rand(b, hr*wr)
    inds = a.argsort(dim=1)

    x = x[torch.arange(b).repeat_interleave(a.size(1)), inds.view(-1)]
    x = x.view(b, hr,wr, res, res, c)
    #x = x.view(b, res, res, hr,wr, c)
    #x = x.reshape(b, hr,wr,res,res, c)

    #x = x.permute(0,5, 3,1, 4,2)
    x = x.permute(0,5, 1,3, 2,4)
    #x = x.permute(0,5, 4,2, 3,1)
    x = x.reshape(b,c,h,w)

    return x

if __name__ == '__main__':
    img_ = cv2.imread('/home/slee/Downloads/canyon_hard1.jpg')
    s = 512
    s = 256
    img_ = img_[100:100+s,100:100+s]
    img0 = torch.from_numpy(img_).permute(2,0,1)
    img_ = cv2.imread('/home/slee/Downloads/gorge.jpg')
    img_ = img_[100:100+s,100:100+s]
    img01 = torch.from_numpy(img_).permute(2,0,1)
    img0 = torch.stack((img0,img01))

    img = perm(img0)

    img = torch.cat((img0,img),2)
    img = img.cpu().permute(0,2,3,1).numpy()
    img = np.hstack(img)
    img = np.copy(img,'C')
    cv2.imshow('img',img)
    cv2.waitKey(0)
