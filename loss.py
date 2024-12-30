import os
import cv2
import numpy as np
from PIL import Image
from scipy.spatial import distance
from pytorch_fid import fid_score
import torch

def calculate_distances(real_folder, fake_folder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    real_images = []
    fake_images = []

    # Read real images
    real_filenames = sorted([filename for filename in os.listdir(real_folder) if filename.endswith(".jpg") or filename.endswith(".png")])
    fake_filenames = sorted([filename for filename in os.listdir(fake_folder) if filename.endswith(".jpg") or filename.endswith(".png")])

    # Read real images
    for filename in real_filenames:
        filename = os.path.join(real_folder, filename)
        flood_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        # remove the fourth channel
        flood_image = flood_image[:, :, :3]
        flood_image = cv2.cvtColor(flood_image, cv2.COLOR_BGR2GRAY)
        real_images.append(np.array(flood_image, dtype=np.float32))

    # Read fake images
    for filename in fake_filenames:
        image1 = Image.open(os.path.join(fake_folder, filename)).convert('L')
        fake_images.append(np.array(image1, dtype=np.float32))

    # Calculate distances
    l1_distances = []
    # l2_distances = []
    linf_distances = []

    for real_image, fake_image in zip(real_images, fake_images):
        l1_distances.append(np.mean(abs(real_image - fake_image)))
        # l2_distances.append(np.sqrt(np.mean((real_image - fake_image) ** 2)))
        linf_distances.append(np.max(abs(real_image - fake_image)))

    # print filename with l1 distances
    for filename, l1_distance in zip(real_filenames, l1_distances):
        print(filename, l1_distance)

    avg_l1_distance = np.mean(l1_distances)
    # avg_l2_distance = np.mean(l2_distances)
    avg_linf_distance = np.mean(linf_distances)

    pathes = [real_folder, fake_folder]
    # Calculate FID
    # fid = fid_score.calculate_fid_given_paths(pathes, 64, device, 2048, num_workers=1)
    fid = 0
    return avg_l1_distance, avg_linf_distance, fid
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == "__main__":
# Usage example
# real_folder = "/home/Josh/BrightestImageProcessing/Josh/image_generation/style_transfer/pytorch-CycleGAN-and-pix2pix/datasets/euv/image/test"
# fake_folder = "/home/Josh/BrightestImageProcessing/Josh/image_generation/style_transfer/pytorch-CycleGAN-and-pix2pix/results/p2p_wgangp_bs64_batch_pixel"
    real_folder = "C:\\Users\\User\\Desktop\\dev\\TEST\\TEST_png\\test_total"
    fake_folder = "C:\\Users\\User\\Desktop\\dev\\I2SB-flood\\results\\flood-test1\\test3_nfe200"

    avg_l1_distance, avg_linf_distance, fid = calculate_distances(real_folder, fake_folder)

    print("Average L1 distance:", avg_l1_distance)
    print("Average L-infinity distance:", avg_linf_distance)
    print("FID:", fid)
