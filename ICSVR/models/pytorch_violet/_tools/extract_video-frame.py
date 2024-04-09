
import argparse, av, base64, io, pickle

import os
from glob import glob
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', required=True, type=str)
    parser.add_argument('--sample', required=True, type=int)
    
    args = parser.parse_args()
    
    return args

if __name__=='__main__':
    args = get_args()
    
    lst = os.listdir(args.path)
    print(len(lst))
    
    pkl = {}
    for f in tqdm(lst, ascii=True):
        vid = f

        if(vid.endswith('.pkl')):
            continue
        imgs = []
        for pack in av.open(os.path.join(args.path, f)).demux():
            for buf in pack.decode():
                if str(type(buf))=="<class 'av.video.frame.VideoFrame'>":
                    imgs.append(buf.to_image().convert('RGB'))
        N = len(imgs)/(args.sample+1)
        
        
        vid = vid.split('.')[0]
        pkl[vid] = []
        for i in range(args.sample):
            buf = io.BytesIO()
            imgs[int(N*(i+1))].save(buf, format='JPEG')
            pkl[vid].append(str(base64.b64encode(buf.getvalue()))[2:-1])
    if('msrvtt' in args.path):
        pickle.dump(pkl, open('../../../data/msrvtt.pkl', 'wb'))
    elif('msvd' in args.path):
        pickle.dump(pkl, open('../../../data/msvd.pkl', 'wb'))
    elif('didemo' in args.path):
        pickle.dump(pkl, open('../../../data/didemo.pkl', 'wb'))
