import torch, torch.nn as nn, torch.nn.functional as F
import cv2, numpy as np, os, sys
from torch.utils.data import Dataset, IterableDataset, DataLoader
import random
#from torchvision.datasets.folder import DatasetFolder


class WikiartDataset(Dataset):
    def __init__(self, root, skipClasses=[], imgSize=512):
        super().__init__()
        self.imgSize = imgSize
        self.skipClasses = skipClasses

        clsdirs = [os.path.join(root, d) for d in os.listdir(root)]
        self.cls2id = {}
        self.samples = []
        for clsdir in clsdirs:
            if os.path.isdir(clsdir):
                for fi in os.listdir(clsdir):
                    #if fi.endswith('.jpeg') or fi.endswith('.jpg') or fi.endswith('.png'):
                        self.cls2id[clsdir] = len(self.cls2id)
                        self.samples.append((len(self.cls2id), os.path.join(clsdir,fi)))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cls,img = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

        h,w = (img.shape[:2])
        s = self.imgSize
        if h<w: img = cv2.resize(img, (int(s*w/h), s))
        else: img = cv2.resize(img, (s, int(s*h/w)))
        #if h<w: img = cv2.resize(img, (int(s*h/w), s))
        #else: img = cv2.resize(img, (s, int(s*w/h)))

        #print(img.shape)
        x = np.random.randint(0, img.shape[1] - s + 1)
        y = np.random.randint(0, img.shape[0] - s + 1)
        img = img[y:y+s,x:x+s]

        return cls, img

class ListedDataset(Dataset):
    def __init__(self, paths, imgSize=512, n=999999, patchRange=-1):
        super().__init__()
        self.imgSize = imgSize
        if patchRange == -1:
            patchRange = (self.imgSize, -1)
        self.patchRange = patchRange
        if len(paths)<32:
            self.paths = []
            self.imgs = [cv2.cvtColor(cv2.imread(i), cv2.COLOR_RGB2BGR) for i in paths]
        else:
            self.paths = paths
        self.n = n
        self.nn = len(paths)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        if len(self.paths) > 0: img = cv2.cvtColor(cv2.imread(self.paths[idx%self.nn]), cv2.COLOR_RGB2BGR)
        else: img = self.imgs[idx%self.nn]

        h,w = (img.shape[:2])
        mhw = min(h,w)

        if self.imgSize >= mhw:
            s = self.imgSize
        else:
            s = np.random.randint(self.imgSize, mhw)


        #s = self.imgSize # NOTE XXX
        if h<w: img = cv2.resize(img, (int(s*w/h), s))
        else: img = cv2.resize(img, (s, int(s*h/w)))
        s = self.imgSize

        if np.random.randint(0,2) == 0: img = np.copy(img[:, ::-1],'C')

        x = np.random.randint(0, img.shape[1] - s + 1)
        y = np.random.randint(0, img.shape[0] - s + 1)
        img = img[y:y+s,x:x+s]

        return img

def worker_init_fn_(wid):
    worker_info = torch.utils.data.get_worker_info()
    dset = worker_info.dataset
    seed_ = worker_info.id
    np.random.seed(seed_)
    random.seed(seed_)


if __name__ == '__main__':
    d = WikiartDataset('/data/wikiart')
    d = DataLoader(d, batch_size=4)

    for b in d:
        imgs = b[1]
        imgs = imgs.cpu().numpy()
        imgs = np.hstack(imgs)
        cv2.imshow('imgs', imgs); cv2.waitKey(0)
