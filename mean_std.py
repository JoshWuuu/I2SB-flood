# read all the images in the floder and calculate mean std
import os

import cv2
import numpy as np
import pandas as pd

path = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\dem_png'

files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and not f.endswith('.csv') and not f.endswith('025_00.png')]
mean = 0
std = 0
dem_stat = 'C:\\Users\\User\\Desktop\\dev\\50PNG\\dem_png\\elevation_stats.csv'
dem_stat = pd.read_csv(dem_stat)

for file_name in files:
    image = cv2.imread(os.path.join(path, file_name), cv2.IMREAD_UNCHANGED)[:,:,0]
    filename = file_name.split('.')[0]
    dem_cur_state = dem_stat[dem_stat['Filename'] == int(filename)]
    min_elev = int(dem_cur_state['Min Elevation'])
    max_elev = int(dem_cur_state['Max Elevation'])
        # do a normalization with max = 410 min = -3, with current max = max_elev, min = min_elev
    real_height = image / 255 * (max_elev - min_elev) + min_elev
    mean += np.mean(real_height)
    std += np.std(real_height)

mean /= len(files)
std /= len(files)

print(mean, std)