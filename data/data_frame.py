import json
import os
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


class Frame_Dataset(Dataset):
    def __init__(self, basedir, mode='train', img_size=[512, 512], data_size=-1):
        super().__init__()

        self.basedir = basedir
        self.mode = mode
        self.H, self.W = img_size[0], img_size[1]

        # Dataset Path
        with open(os.path.join(basedir, f"transforms_{mode}.json"), "r") as fp:
            self.meta = json.load(fp)
        self.data_list = self.meta['frames']

        # Load Intrinsic parameters
        self.focal = np.array(self.meta['intrinsics'])
        self.hwf = [self.H, self.W, self.focal]

        # Filter with Data Size
        if mode == 'train' and data_size > 0 and data_size < len(self.data_list):
            np.random.seed(1004)
            self.data_list = np.random.choice(self.data_list, data_size, replace=False)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_idx = idx

        # Load Image
        image_path = os.path.join(self.basedir, self.data_list[idx]['file_path'] + '.png')
        image = Image.open(image_path).resize((self.H, self.W))
        image = (np.array(image) / 255.0).astype(np.float32)

        # Load Pose Parameters (Expression, Transformation Matrix)
        pose = np.array(self.data_list[idx]['transform_matrix'])[:3, :4].astype(np.float32)
        exp = np.array(self.data_list[idx]['expression']).astype(np.float32)

        out = {}
        out['hwf'] = self.hwf
        out['image'], out['pose'], out['expression'], out['data_idx'] = image, pose, exp, data_idx
        #import pdb; pdb.set_trace()
        # Load Bounding Box (= Sampling Map)
        if self.mode == 'train':
            if 'bbox' in self.data_list[idx].keys():
                bbox = np.array(self.data_list[idx]['bbox'])
            else:
                bbox = np.array([0.0, 1.0, 0.0, 1.0])

            bbox[0:2], bbox[2:4] = self.H * bbox[0:2], self.W * bbox[2:4]
            bbox = torch.from_numpy(np.floor(bbox)).int()

            p = 0.9
            sampling_map = np.zeros((self.H, self.W))
            sampling_map.fill(1 - p)
            sampling_map[bbox[0]:bbox[1], bbox[2]:bbox[3]] = p
            sampling_map = (1 / sampling_map.sum()) * sampling_map
            sampling_map = sampling_map.reshape(-1)

            out['sampling_map'] = sampling_map

        return out


if __name__ == "__main__":
    # Debug Dataset
    from torch.utils.data import DataLoader
    
    basedir = '/home/nas1_userB/sunghyun/Project/Sparse-Nerface/nerface_dataset/person_1'
    dataset = Frame_Dataset(basedir=basedir, mode='train', data_size=10)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, batch in enumerate(loader):
        #import pdb; pdb.set_trace()

        if i == 3:
            break
