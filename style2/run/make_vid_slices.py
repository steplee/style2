import cv2, numpy as np
import socket, subprocess, time

SIZE = 640, 480
MIN_FRAMES = 16
MIN_FRAMES = 26
vidListFile = 'data/vidspy'
vidListFile = 'data/vidspy2'
outName = 'data/vid_1.mp4'
outName = 'data/vid_2.mp4'

# Make encoder process
sock_name = '/tmp/vid{}.sock'.format(time.time()//1)
#proc = subprocess.Popen('ffmpeg -y -f rawvideo -pix_fmt rgb8 -s:v {}x{} -listen 1 -i unix:{} -c:v libx264 data/vid_1.mp4' \
proc = subprocess.Popen('ffmpeg -y -f rawvideo -pix_fmt rgb24 -s:v {}x{} -listen 1 -i unix:{} {}' \
        .format(*SIZE, sock_name, outName).split(' '))
time.sleep(.5)
sock = socket.socket(socket.AF_UNIX)
sock.connect(sock_name)

with open(vidListFile) as fp:
    vidFiles = fp.readlines()

jj = 0
curStride = 10
try:
    for fi,fil in enumerate(vidFiles):
        vcap,good = cv2.VideoCapture(fil.strip()), True
        ii = 0
        s = max(SIZE)
        print(fi,'/',len(vidFiles),'')
        frames = []
        nframes = 0
        while good:
            good,img = vcap.read()
            if img is None: continue
            if ii % curStride == 0:
                (h,w), mhw = img.shape[:2], min(img.shape[:2])

                if h<w: img = cv2.resize(img, (int(s*w/h), s))
                else: img = cv2.resize(img, (s, int(s*h/w)))
                h,w = img.shape[:2]

                img = img[h//2-SIZE[1]//2:h//2+SIZE[1]//2, w//2-SIZE[0]//2:w//2+SIZE[0]//2]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.tobytes()
                frames.append(img)
                #sock.send(img)
                jj += 1
                nframes += 1
                if jj % 35 == 0:
                    curStride = np.random.randint(10, 50)
            ii += 1
            if len(frames) % 18 == 0 and nframes>MIN_FRAMES:
                for f in frames: sock.send(f)
                frames = []
        if len(frames) > 0 and nframes>MIN_FRAMES:
            for f in frames: sock.send(f)
except Exception as e:
    print('\n\n**** FAILED: ****\n',e,'\n*****\n\n')

print(' - DONE')
sock.close()
time.sleep(.5)
print(' - DONE2')
time.sleep(.5)
