import numpy as np

import sys
import os

import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import cv2
from skimage import io as ski_io
from skimage import color as ski_colour


def show_masks_colour(anns, image, method='median', explicit_border=False):
    img = np.zeros((*image.shape[:2], 3))
    unique_ids = np.unique(anns)
    for u_id in unique_ids:
        m_img = anns == u_id
        color_mask = segment_colour(image, m_img, method)
        img[m_img] = color_mask
    if explicit_border:
        boundary_pixels = find_label_boundaries(anns)
        img[boundary_pixels] = [0, 0, 0]
    return img.astype('uint8')


def show_anns_color(anns, image, set_background=True, method='median'):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    background_img = np.zeros(image.shape[:2])
    img = np.zeros((*image.shape[:2], 3))
    mask_img = np.zeros(image.shape[:2])
    for ann_ind, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        color_mask = segment_colour(image, m, method)
        img[m] = color_mask
        mask_img[m] = ann_ind + 1
        background_img[m] = 1
    if set_background:
        img[background_img == 0] = segment_colour(image, background_img == 0, method)
    else:
        img[background_img == 0] = image[background_img == 0]
    img = img.astype('uint8')
    mask_img = mask_img.astype('uint8')
    return img, mask_img


def segment_colour(rgb_img, mask, method):
    segmented_pixels = rgb_img[mask > 0]
    colour = np.mean(segmented_pixels, axis=0) if method == 'avg' else np.median(
        segmented_pixels, axis=0)
    return colour


def find_label_boundaries(segmentation_map):
    # Create shifts in 4 directions (up, down, left, right)
    shifts = [
        np.roll(segmentation_map, 1, axis=0),  # up
        np.roll(segmentation_map, -1, axis=0),  # down
        np.roll(segmentation_map, 1, axis=1),  # left
        np.roll(segmentation_map, -1, axis=1)  # right
    ]

    # Initialize boundary mask with all False
    boundary_mask = np.zeros_like(segmentation_map, dtype=bool)

    # For each shift, check if the value is different from the original
    for shifted in shifts:
        boundary_mask = np.logical_or(boundary_mask, segmentation_map != shifted)

    # Fix the edges (which get wrapped around due to roll)
    # Set the edge pixels to False if they appeared as boundaries due to wrapping
    boundary_mask[0, :] = False  # top edge
    boundary_mask[-1, :] = False  # bottom edge
    boundary_mask[:, 0] = False  # left edge
    boundary_mask[:, -1] = False  # right edge

    return boundary_mask


color_palette_dict = {
    'L-red': (185, 25, 70),
    'R-red': (185, 40, 35),
    'L-pink': (250, 185, 180),  #
    'R-pink': (240, 190, 215),  #
    'L-Orange': (240, 110, 50),  #
    'R-Orange': (215, 130, 20),  #
    'L-Yellow': (245, 195, 0),  #
    'R-Yellow': (235, 235, 140),  #
    'L-Brown': (120, 50, 15),  #
    'R-Brown': (95, 65, 5),  #
    'L-Green': (110, 130, 0),  #
    'R-Green': (0, 145, 105),  #
    'L-Blue': (0, 140, 150),  #
    'R-Blue': (80, 125, 205),  #
    'L-Purple': (110, 85, 160),  #
    'R-Purple': (130, 30, 90),  #

    'cream': (250, 190, 115),  #
    'khaki': (180, 145, 0),  #
    'olive': (100, 100, 15),  #

    'M-Brown': (105, 60, 25),  # L-Brown
    'Dark-Brown': (85, 30, 25),  # L-Brown

    'Gray': (128, 128, 128),
    'Dark-Gray': (64, 64, 64),
    'Light-Gray': (192, 192, 192),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}
color_palette = np.array(list(color_palette_dict.values())).astype('uint8')
color_palette_hsv = ski_colour.rgb2hsv(color_palette)

device = torch.device("cuda")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam = sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=64,
    stability_score_thresh=0.92,
)

# data dirs
things_dir = './data/things/'
things_imgs_dir = f"{things_dir}/THINGS/Images/"
things_eeg_dir = f"{things_dir}/EEG2/"
eeg_metadata = np.load(f"{things_eeg_dir}image_metadata.npy", allow_pickle=True)[()]

split = 'train'
out_dir = f"./data/EEG2/sam/sam1_pps64_sst92/{split}_masks/"
os.makedirs(out_dir, exist_ok=True)
out_dir_vis = out_dir.replace('_masks/', '_vis/')
os.makedirs(out_dir_vis, exist_ok=True)

print(sys.argv[1:])

for img_ind, img_name in enumerate(eeg_metadata[f'{split}_img_files']):
    print(img_ind)
    dir_name = img_name[:-8]

    # loading the image and resizing it
    img = ski_io.imread(f"{things_imgs_dir}/{dir_name}/{img_name}")
    img = cv2.resize(img, (500, 500))

    # Running SAM
    masks = mask_generator.generate(img)
    _, img_mask = show_anns_color(masks, img)
    img_seg = show_masks_colour(img_mask, img, explicit_border=False)
    # Saving outputs
    ski_io.imsave(f"{out_dir}{img_name[:-4]}.png", img_mask, check_contrast=False)
    ski_io.imsave(f"{out_dir_vis}{img_name[:-4]}.png", img_seg, check_contrast=False)

    img_seg = show_masks_colour(img_mask, img, explicit_border=False)
    ski_io.imsave(f"{out_dir_vis}{img_name[:-4]}.png", img_seg, check_contrast=False)
