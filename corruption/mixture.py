# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import numpy as np
import enum
import pandas as pd 
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from .jpeg import jpeg_decode, jpeg_encode
from .blur import Deblurring
from .superresolution import build_sr_bicubic, build_sr_pool
from .inpaint import get_center_mask, load_freeform_masks

from ipdb import set_trace as debug


class AllCorrupt(enum.IntEnum):
    JPEG_5 = 0
    JPEG_10 = 1
    BLUR_UNI = 2
    BLUR_GAUSS = 3
    SR4X_POOL = 4
    SR4X_BICUBIC = 5
    INPAINT_CENTER = 6
    INPAINT_FREE1020 = 7
    INPAINT_FREE2030 = 8

class MixtureCorruptMethod:
    def __init__(self, opt, device="cpu"):

        # ===== blur ====
        self.blur_uni = Deblurring(torch.Tensor([1/9] * 9).to(device), 3, opt.image_size, device)

        sigma = 10
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
        g_kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
        self.blur_gauss = Deblurring(g_kernel / g_kernel.sum(), 3, opt.image_size, device)

        # ===== sr4x ====
        factor = 4
        self.sr4x_pool = build_sr_pool(factor, device, opt.image_size)
        self.sr4x_bicubic = build_sr_bicubic(factor, device, opt.image_size)
        self.upsample = torch.nn.Upsample(scale_factor=factor, mode='nearest')

        # ===== inpaint ====
        self.center_mask = get_center_mask([opt.image_size, opt.image_size])[None,...] # [1, 1, 256, 256]
        self.free1020_masks = torch.from_numpy((load_freeform_masks("freeform1020"))) # [10000, 1, 256, 256]
        self.free2030_masks = torch.from_numpy((load_freeform_masks("freeform2030"))) # [10000, 1, 256, 256]

    def jpeg(self, img, qf):
        return jpeg_decode(jpeg_encode(img, qf), qf)

    def blur(self, img, kernel):
        img = (img + 1) / 2
        if kernel == "uni":
            _img = self.blur_uni.H(img).reshape(*img.shape)
        elif kernel == "gauss":
            _img = self.blur_gauss.H(img).reshape(*img.shape)
        # [0,1] -> [-1,1]
        return _img * 2 - 1

    def sr4x(self, img, filter):
        b, c, w, h = img.shape
        if filter == "pool":
            _img = self.sr4x_pool.H(img).reshape(b, c, w // 4, h // 4)
        elif filter == "bicubic":
            _img = self.sr4x_bicubic.H(img).reshape(b, c, w // 4, h // 4)

        # scale to original image size for I2SB
        return self.upsample(_img)

    def inpaint(self, img, mask_type, mask_index=None):
        if mask_type == "center":
            mask = self.center_mask
        elif mask_type == "free1020":
            if mask_index is None:
                mask_index = np.random.randint(len(self.free1020_masks))
            mask = self.free1020_masks[[mask_index]]
        elif mask_type == "free2030":
            if mask_index is None:
                mask_index = np.random.randint(len(self.free2030_masks))
            mask = self.free2030_masks[[mask_index]]
        return img * (1. - mask) + mask * torch.randn_like(img)

    def mixture(self, img, corrupt_idx, mask_index=None):
        if corrupt_idx == AllCorrupt.JPEG_5:
            corrupt_img = self.jpeg(img, 5)
        elif corrupt_idx == AllCorrupt.JPEG_10:
            corrupt_img = self.jpeg(img, 10)
        elif corrupt_idx == AllCorrupt.BLUR_UNI:
            corrupt_img = self.blur(img, "uni")
        elif corrupt_idx == AllCorrupt.BLUR_GAUSS:
            corrupt_img = self.blur(img, "gauss")
        elif corrupt_idx == AllCorrupt.SR4X_POOL:
            corrupt_img = self.sr4x(img, "pool")
        elif corrupt_idx == AllCorrupt.SR4X_BICUBIC:
            corrupt_img = self.sr4x(img, "bicubic")
        elif corrupt_idx == AllCorrupt.INPAINT_CENTER:
            corrupt_img = self.inpaint(img, "center")
        elif corrupt_idx == AllCorrupt.INPAINT_FREE1020:
            corrupt_img = self.inpaint(img, "free1020", mask_index=mask_index)
        elif corrupt_idx == AllCorrupt.INPAINT_FREE2030:
            corrupt_img = self.inpaint(img, "free2030", mask_index=mask_index)
        return corrupt_img


class MixtureCorruptDatasetTrain(Dataset):
    def __init__(self, opt, dataset):
        super(MixtureCorruptDatasetTrain, self).__init__()
        self.dataset = dataset
        self.method = MixtureCorruptMethod(opt)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        clean_img, y = self.dataset[index] # clean_img: tensor [-1,1]

        rand_idx = np.random.choice(AllCorrupt)
        corrupt_img = self.method.mixture(clean_img.unsqueeze(0), rand_idx).squeeze(0)

        assert corrupt_img.shape == clean_img.shape, (clean_img.shape, corrupt_img.shape)
        return clean_img, corrupt_img, y

from pathlib import Path
class MixtureCorruptDatasetVal(Dataset):
    def __init__(self, opt, dataset):
        super(MixtureCorruptDatasetVal, self).__init__()
        self.dataset = dataset
        self.method = MixtureCorruptMethod(opt)

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, index):
        clean_img, y = self.dataset[index] # clean_img: tensor [-1,1]

        idx = index % len(AllCorrupt)
        corrupt_img = self.method.mixture(clean_img.unsqueeze(0), idx, mask_index=idx).squeeze(0)

        assert corrupt_img.shape == clean_img.shape, (clean_img.shape, corrupt_img.shape)
        return clean_img, corrupt_img, y

class floodDataset(Dataset):
    def __init__(self, opt, val=False, test=False):
        super(floodDataset, self).__init__()
        self.opt = opt
        dem_path = Path(opt.dataset_dir) / 'dem_png'
        # list all the subfolder in the dem_path, for example subfolder is 1, 2, 3 then return me [1, 2, 3]
        
        self.dem_stat = Path(opt.dataset_dir) / 'dem_png/elevation_stats.csv'
        self.dem_stat = pd.read_csv(self.dem_stat)
        # self.dem = cv2.imread(dem_path)
        # self.dem = cv2.cvtColor(self.dem, cv2.COLOR_BGR2GRAY)
        # self.dem = cv2.cvtColor(self.dem, cv2.COLOR_GRAY2BGR)
        self.test = test

        self.flood_path = Path(opt.dataset_dir) / 'DEPTH_png'
        dem_folder = [int(f) for f in os.listdir(self.flood_path) if os.path.isdir(os.path.join(self.flood_path, f))]
        rainfall_path = Path(opt.dataset_dir) / 'scenario_rainfall.csv'
        if test:
            dem_folder = [61, 62, 65, 67, 69]
            rainfall_path = Path(opt.dataset_dir) / 'scenario_rainfall.csv'
            self.flood_path = 'C:\\Users\\User\\Desktop\\dev\\test_dem'

        rainfall = pd.read_csv(rainfall_path)
        # remove first row, no 0 row 
        rainfall = rainfall.iloc[:, :]

        # Initialize lists to store cell values and their positions
        rainfall_cum_value = []
        cell_positions = []

        val = False
        # Iterate through each column
        for dem_num in dem_folder:
            for col in rainfall.columns:
                if col == 'time':
                    continue
                col_num = int(col.split("_")[1])
                if (val and col_num not in [2]) or (not val and col_num in []):
                    continue
                cell_values = []
                # Iterate through each row in the current column
                for row in range(len(rainfall)):
                    cell_value = rainfall.iloc[row][col]
                    cell_values.append(np.floor(cell_value))
                    # make it a len 24 list if not append 0 in front
                    temp = [0] * (24 - len(cell_values))
                    temp.extend(cell_values)
                    if len(temp) == 25:
                        temp = temp[1:]
                    rainfall_cum_value.append(temp)
                    # col_num is the rainfall index, and row is the time index
                    cell_positions.append((dem_num, col_num, row))   

        self.rainfall = rainfall_cum_value
        self.cell_positions = cell_positions

        # add transform , to tensor and normalize
        self.transform = T.Compose([
            T.ToTensor(),
            # T.Lambda(lambda t: (t * 2) - 1)
        ])

    def __find_flood_image(self, cell_position, flood_path):
        dem, col, row = cell_position
        dem_folder_name = str(dem)
        if col < 100:
            folder_name = f"RF{col:02d}"
        else:
            folder_name = f"RF{col}"
        if self.test:
            if col < 100:
                folder_name = f"RF{col:02d}"
            else:
                folder_name = f"RF{col}"
        image_name = f"{dem_folder_name}_{folder_name}_{row:03d}.png"
        image_path = os.path.join(flood_path, dem_folder_name, folder_name, image_name)
        return image_path
    
    def __find_dem_image(self, cell_position):
        dem_num = cell_position[0]
        dem_folder = Path(self.opt.dataset_dir) /  'dem_png'
        dem_path = os.path.join(dem_folder, f'{dem_num}.png')
        return dem_path
    
    def __len__(self):
        return len(self.cell_positions)

    def __getitem__(self, index):
        cell_position = self.cell_positions[index]
        rainfall = self.rainfall[index]
        dem_path = self.__find_dem_image(cell_position)
        dem_image = cv2.imread(dem_path, cv2.IMREAD_UNCHANGED)[:,:,0]
        dem_image = cv2.cvtColor(dem_image, cv2.COLOR_GRAY2BGR)
        dem_cur_state = self.dem_stat[self.dem_stat['Filename'] == cell_position[0]]
        min_elev = int(dem_cur_state['Min Elevation'])
        max_elev = int(dem_cur_state['Max Elevation'])
        # do a normalization with max = 410 min = -3, with current max = max_elev, min = min_elev
        real_height = dem_image / 255 * (max_elev - min_elev) + min_elev
        dem_image = (real_height - (-3)) / (51 + 3) * 255
        # clamp dem_image to 0-255
        dem_image = np.clip(dem_image, 0, 255)
        dem_image = np.array(dem_image, dtype=np.uint8)
        
        rainfall = np.array(rainfall, dtype=np.int64)
        # rainfall = rainfall.reshape(1, 24)

        image_path = self.__find_flood_image(cell_position, self.flood_path)

        flood_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        
        # remove the fourth channel
        # flood_image = flood_image[:, :, 2]
        # flood_image = cv2.cvtColor(flood_image, cv2.COLOR_BGR2GRAY)
        flood_image = cv2.cvtColor(flood_image, cv2.COLOR_GRAY2BGR)
        # IF FLOOD image not 256x256, print the shape and the image_path
        # flood_image = Image.open(image_path).convert('RGB')
        # flood_image = np.array(flood_image)
        # print(flood_image.shape, image_path)
        # convert to RGB if grayscale
        # if len(flood_image.shape) == 2:
        #     flood_image = cv2.cvtColor(flood_image, cv2.COLOR_GRAY2RGB)
        # if len(dem_image.shape) == 2:
        #     dem_image = cv2.cvtColor(dem_image, cv2.COLOR_GRAY2RGB)

        dem_image = self.transform(dem_image)
        flood_image = self.transform(flood_image)

        dem_image = (dem_image * 2) - 1
        flood_image = (flood_image * 2) - 1

        return flood_image, dem_image, rainfall, image_path
    