

from ..adain import *
import cv2

#CKPT = 'saves/unkAdain.0.pt'
CKPT = 'saves/unkAdain.1000.pt'
CKPT = 'saves/unkAdain.2000.pt'
CKPT = 'saves/unkAdain.7000.pt'
CKPT = 'saves/unkAdain.9000.pt'



def main():
    with torch.no_grad():

        key = 0
        q = ord('q')

        model = AdaInModel({}).cuda().eval()
        d = torch.load(CKPT)
        model.load_state_dict(d['sd'])

        y = cv2.imread('/home/slee/Downloads/IMG_20210108_232459.jpg')[...,[2,1,0]]
        #y = cv2.imread('/data/wikiart/Post_Impressionism/vincent-van-gogh_the-starry-night-1888-2.jpg')
        #y = cv2.imread('/data/wikiart/Abstract_Expressionism/atsuko-tanaka_89a-1989.jpg')
        #y = cv2.imread('/home/slee/Pictures/4836.jpg')
        #y = cv2.imread('/data/wikiart/Ukiyo_e/hiroshige_a-snowy-gorge.jpg')
        #y = cv2.imread('/data/wikiart/Post_Impressionism/vincent-van-gogh_the-starry-night-1888-1.jpg')
        h,w = y.shape[:2]
        Y_SZ = 512
        #Y_SZ = 256
        y = cv2.resize(y, (Y_SZ, (int(h/w*Y_SZ))))
        y = torch.from_numpy(y).cuda().float().div_(255).permute(2,0,1).unsqueeze(0)
        print(y.shape)
        stats = model.forward_get_stats(y)
        print(stats[0].shape)

        x = cv2.imread('/home/slee/Downloads/IMG_20210108_232443.jpg')[...,[2,1,0]]
        Y_SZ = 512
        h,w = x.shape[:2]
        x = cv2.resize(x, (Y_SZ, (int(h/w*Y_SZ))))
        x = torch.from_numpy(x).cuda().float().div_(255).permute(2,0,1).unsqueeze(0)

        z = model.forward_with_stats(x, *stats)
        z = z.mul_(255).to(torch.uint8)[0].permute(1,2,0).cpu().numpy()
        #z = z.transpose(1,0,2)[::-1]
        z = z[...,[2,1,0]]
        cv2.imshow('Stylized', z)
        key = cv2.waitKey(0)

if __name__ == '__main__':
    main()

