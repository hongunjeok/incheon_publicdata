import torch
import torch.utils.data as data_utl

import numpy as np
import random
import cv2

import os
import lintel

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))



class HMDB(data_utl.Dataset):

    def __init__(self, split_file, root, mode='rgb', length=16, model='2d', random=True, c2i={}):
        self.class_to_id = c2i
        self.id_to_class = []
        for i in range(len(c2i.keys())):
            for k in c2i.keys():
                if c2i[k] == i:
                    self.id_to_class.append(k)
        cid = 0
        self.data = []
        self.model = model
        self.size = 200

        with open(split_file, 'r') as f:
            for l in f.readlines():
                if len(l) <= 5:
                    continue
                v,c = l.strip().split(' ')
                v = v.strip()
                if c not in self.class_to_id:
                    self.class_to_id[c] = cid
                    self.id_to_class.append(c)
                    cid += 1
                self.data.append([os.path.join(root, v), self.class_to_id[c]])

        self.split_file = split_file
        self.root = root
        self.mode = mode
        self.length = length
        self.random = random

    def __getitem__(self, index):
        vid, cls = self.data[index]

        with open(vid, 'rb') as f:
            enc_vid = f.read()

        
        df, w, h, _ = lintel.loadvid(enc_vid, should_random_seek=self.random, num_frames=self.length*2)
        df = np.frombuffer(df, dtype=np.uint8)

        w=w//2
        h=h//2
        
        # frame 크기 고정
        th = self.size
        tw = self.size

        # reshape 먼저 (원본은 h*2, w*2 해상도 기준)
        df = np.reshape(df, newshape=(self.length*2, h*2, w*2, 3))[::2,::2,::2,:]

        # 실제 크기 측정
        h, w = df.shape[1:3]

        if h < th or w < tw:
            # 🔧 작으면 강제로 리사이즈
            df = np.array([cv2.resize(frame, (tw, th)) for frame in df])
        else:
            # ✅ 충분히 크면 crop 진행
            if not self.random:
                i = (h - th) // 2
                j = (w - tw) // 2
            else:
                i = random.randint(0, h - th)
                j = random.randint(0, w - tw)
            df = df[:, i:i+th, j:j+tw, :]
            
            # 🔄 랜덤 수평 플립 (random=True일 때만 적용)
            if self.random and random.random() < 0.5:
                df = np.flip(df, axis=2).copy()  # axis=2 → width 방향 (가로)

            
        if self.mode == 'flow':
            #print(df[:,:,:,1:].mean())
            #exit()
            # only take the 2 channels corresponding to flow (x,y)
            df = df[:,:,:,1:]
            if self.model == '2d':
                # this should be redone...
                # stack 10 along channel axis
                df = np.asarray([df[:10],df[2:12],df[4:14]]) # gives 3x10xHxWx2
                df = df.transpose(0,1,4,2,3).reshape(3,20,self.size,self.size).transpose(0,2,3,1)
            
                
        df = 1-2*(df.astype(np.float32)/255)

        if self.model == '2d':
            # 2d -> return TxCxHxW
            return df.transpose([0,3,1,2]), cls
        # 3d -> return CxTxHxW
        return df.transpose([3,0,1,2]), cls


    def __len__(self):
        return len(self.data)



if __name__ == '__main__':
    DS = HMDB
    dataseta = DS('data/hmdb/split1_train.txt', '/ssd/hmdb/', model='2d', mode='flow', length=16)
    dataset = DS('data/hmdb/split1_test.txt', '/ssd/hmdb/', model='2d', mode='rgb', length=16, c2i=dataseta.class_to_id)

    for i in range(len(dataseta)):
        print(dataseta[i][0].shape)
