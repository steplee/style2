from ..adain import *
import cv2

#CKPT = 'saves/unkAdain.0.pt'
CKPT = 'saves/unkAdain.1000.pt'
CKPT = 'saves/unkAdain.2000.pt'
CKPT = 'saves/unkAdain.5000.pt'
#CKPT = 'saves/unkAdain.7000.pt'
#CKPT = 'saves/unkAdain.9000.pt'
CKPT = 'saves/unkAdain.16000.pt'
CKPT = 'saves/unkAdain.24000.pt'
CKPT = 'saves/unkAdain.31000.pt'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--styleSize', default=256+128, type=int)
parser.add_argument('--gridSize', default=1000, type=int)
parser.add_argument('-K', default=6, type=int)
parser.add_argument('--temp', default=23, type=int)
parser.add_argument('--querySize', default=640, type=int)
parser.add_argument('--gamma', default=0, type=float,help='IIR features (relu4_1), if actionWipe is passed used for that too')
parser.add_argument('--ckpt', default=CKPT)
parser.add_argument('--actionWipe', action='store_true', help='detect scene change, interpolate to new scene in feature space')
parser.add_argument('--noShow', action='store_true', help='dont show window')
parser.add_argument('--output', default=None, help='save an mp4 file')
parser.add_argument('--noGrid', action='store_true', help='interpolate styles without interaction')

parser.add_argument('--src', default=0)
args = parser.parse_args()

mousex,mousey = 0,0
STYLE_SIZE = args.styleSize
GRID_SZ = args.gridSize
K = args.K
TEMP = args.temp
MAX_IMG_SIZE = args.querySize
GAMMA = args.gamma

def is_image(f):
    return f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')
def resize_crop(img,s):
        h,w = (img.shape[:2])
        if h<w: img = cv2.resize(img, (int(s*w/h), s))
        else: img = cv2.resize(img, (s, int(s*h/w)))
        x = img.shape[1]//2 - s//2
        y = img.shape[0]//2 - s//2
        return img[y:y+s,x:x+s]

def mouse_callback(ev,x,y,flags,param):
    global mousex, mousey
    mousex, mousey = x/GRID_SZ, y/GRID_SZ


def get_style_images(n, size):
    root = '/data/wikiart/'
    ds = [os.path.join(root,d) for d in os.listdir(root)]
    fs = []
    for d in ds:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if is_image(f): fs.append(os.path.join(d,f))
    random.seed(4)
    fs = random.sample(fs,n)
    if os.path.exists('/home/slee/Downloads/IMG_20210108_232459.jpg'):
        fs[0] = '/home/slee/Downloads/IMG_20210108_232459.jpg'
        fs[1] = '/home/slee/Downloads/IMG_20210108_232443.jpg'
    return np.stack([resize_crop(cv2.imread(f)[...,[2,1,0]], s=size) for f in fs])




def main():
    global mousex,mousey


    with torch.no_grad():
        print(' - Loading model.')
        model = load_adain_model(args.ckpt)
        print(' - Done loading model.')

        # Get random style images
        ys = get_style_images(K*K, STYLE_SIZE)

        if not args.noGrid:
            grid = ys.reshape(K, K*STYLE_SIZE, STYLE_SIZE, 3)
            grid = np.hstack(grid)
            grid = cv2.resize(np.copy(grid,'C'), (GRID_SZ,GRID_SZ))

            cv2.namedWindow('grid')
            cv2.setMouseCallback('grid', mouse_callback)
            cv2.imshow('grid',grid[...,[2,1,0]])
            cv2.waitKey(1)
        else:
            strip = [cv2.resize(i, (96,96))[...,[2,1,0]] for i in ys]

        key = 0
        q = ord('q')
        writer = None
        nframes = 0


        # Compute stats on grid images
        y = torch.from_numpy(ys).cuda().float().div_(255).permute(0,3,1,2)
        stats = model.forward_get_stats(y)
        # Make grid positions to be used for interpolation
        x_ = torch.linspace(0,1,K+1)[:-1]
        y_ = torch.linspace(0,1,K+1)[:-1]
        style_positions = torch.stack(torch.meshgrid(x_,y_),-1).view(-1,2) + .5/K

        screen = None
        src = args.src
        if src == 'screen':
            import mss
            screen = mss.mss()
            monitor = {"top": 135, "left": 80, "width": MAX_IMG_SIZE, "height": MAX_IMG_SIZE}
        try:
            vid_id = int(src)
        except:
            vid_id = src
        vcap = cv2.VideoCapture(vid_id)

        last_feats = None
        last_feats0 = None
        shotGamma = 0

        while key != q:
            # Compute style interpolation weights
            if args.noGrid:
                # when not using grid, do smoothed linear interpolation
                fps, seconds_per_style = 30, 2
                style1,style2 = (nframes//(seconds_per_style*fps))%(K*K), (nframes//(seconds_per_style*fps) + 1)%(K*K)
                alpha = nframes % (seconds_per_style*fps) / (seconds_per_style*fps-1)
                denom = alpha**2 + (1-alpha)**2
                statsWeighted = [(s[style1:style1+1]*(1-alpha)**2 + s[style2:style2+1]*(alpha**2))/denom for s in stats]
                overlay = np.vstack((strip[style1], strip[style2]))
                A = np.eye(3,dtype=np.float32)[:2]
                A[1,2] = -alpha * 96
                overlay = cv2.warpAffine(overlay, A, (96,96))
            else:
                mousePos = torch.FloatTensor([mousex,mousey]).view(1,2)
                w = (style_positions - mousePos).norm(dim=1)
                w[w<.1/K] = -99999
                w = torch.softmax(-w * TEMP, 0).cuda().view(-1,1,1,1)

                # Interpolate styles
                statsWeighted = [(s*w).sum(0,keepdim=True) for s in stats]
                overlay = None

            key = cv2.waitKey(1)
            if screen is not None:
                x = screen.grab(monitor).rgb
                x = np.frombuffer(x,dtype=np.uint8).reshape(MAX_IMG_SIZE,MAX_IMG_SIZE,3)
            else:
                _,x = vcap.read()
                x = x[...,[2,1,0]]
            if max(x.shape[:2]) > MAX_IMG_SIZE:
                h,w = (x.shape[:2])
                if h<w: x = cv2.resize(x, (int(MAX_IMG_SIZE*w/h), MAX_IMG_SIZE))
                else: x = cv2.resize(x, (MAX_IMG_SIZE, int(MAX_IMG_SIZE*h/w)))
            x = torch.from_numpy(x).cuda().float().div_(255).permute(2,0,1).unsqueeze(0)

            skipThisFrame = False

            if args.actionWipe:
                if shotGamma > 0 and last_feats != None:
                    z,feats = model.forward_with_stats(x, statsWeighted, last_feats, shotGamma)
                    shotGamma = shotGamma * GAMMA
                    if shotGamma < .05: shotGamma = 0
                else:
                    z,feats = model.forward_with_stats(x, statsWeighted)
                if last_feats is None:
                    last_feats = feats
                    last_feats0 = feats
                elif last_feats0 is not None and shotGamma < .3:
                    last_feats0_ = last_feats0[-1] if isinstance(last_feats0,tuple) else last_feats0
                    feats_ = feats[-1] if isinstance(feats,tuple) else feats
                    shot_d = (feats_.flatten(2).mean(2) - last_feats0_.flatten(2).mean(2)).norm(dim=1).mean()
                    if shot_d > 11:
                        print(' - NEW SHOT', shot_d.item())
                        shotGamma = .99
                        last_feats = last_feats0
                        skipThisFrame = True
                last_feats0 = feats
            else:
                if GAMMA > 0 and last_feats != None:
                    z,feats = model.forward_with_stats(x, statsWeighted, last_feats, GAMMA)
                else:
                    z,feats = model.forward_with_stats(x, statsWeighted)
                last_feats = feats

            if not skipThisFrame:
                nframes += 1
                z = z.mul_(255).to(torch.uint8)[0].permute(1,2,0).cpu().numpy()
                z = z [...,[2,1,0]]
                if overlay is not None:
                    z[:overlay.shape[0], :overlay.shape[1]] = overlay
                if not args.noShow:
                    cv2.imshow('Stylized', z)
                    key = cv2.waitKey(1)
                if args.output is not None:
                    if writer is None:
                        #fcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fcc = cv2.VideoWriter_fourcc(*'X264')
                        fcc = cv2.VideoWriter_fourcc(*'AVC1')
                        #out = 'appsrc ! videoconvert ! avenc_mpeg4 bitrate=100000 ! mp4mux ! filesink location={}'.format(args.output)
                        out = args.output
                        #writer = cv2.VideoWriter(out, cv2.CAP_FFMPEG, fcc, 25, z.shape[:2][::-1])
                        writer = cv2.VideoWriter(out, cv2.CAP_GSTREAMER, fcc, 25, z.shape[:2][::-1])
                    writer.write(z)





main()
