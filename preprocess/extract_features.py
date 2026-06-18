import numpy as np
import os
import glob
import sys

from scipy.optimize import fsolve
from PIL import Image
import cv2

import torch
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import v2
import open_clip


class DirectT:
    def __init__(self):
        pass

    def __call__(self, x, U=None):
        return x


class UniformBlur:
    def __init__(self, blur_kernel_size):
        self.blur_kernel_size = blur_kernel_size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        img_np = np.array(img)
        if img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        img_blur = cv2.GaussianBlur(img_np, (self.blur_kernel_size, self.blur_kernel_size), 0)
        img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_blur)


class FoveaBlur:
    def __init__(self, h, w, blur_kernel_size, curve_type='exp', *args, **kwargs):
        self.blur_kernel_size = blur_kernel_size
        self.mask = np.zeros((h, w), np.float32)

        center = (w // 2, h // 2)
        max_distance = np.sqrt((h - center[1] - 1) ** 2 + (w - center[0] - 1) ** 2)
        c = 0.5
        center_resolution = 1 - c
        edge_resolution = 0

        initial_guess = [1.0, 1.0]

        def equations(vars):
            t, r = vars
            eq1 = r * (t - np.sin(t)) - 1  # x = 1
            eq2 = -r * (1 - np.cos(t)) + 1.0  # y = 0
            return [eq1, eq2]

        solution = fsolve(equations, initial_guess)
        t_max, r_solution = solution
        self.r = r_solution

        fun_degrade = getattr(self, curve_type, None)
        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
                x0 = min(1, distance / max_distance)
                y0 = fun_degrade(x0, **kwargs)
                self.mask[i, j] = edge_resolution + (center_resolution - edge_resolution) * y0

    def alphaBlend(self, img1, img2, mask):
        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        blended = cv2.convertScaleAbs(img1 * (1 - alpha) + img2 * alpha)
        return blended

    def __call__(self, img, blur_kernel_size=None):
        if blur_kernel_size == None:
            blur_kernel_size = self.blur_kernel_size
        img = np.array(img)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        blured = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
        blended = self.alphaBlend(img, blured, 1 - self.mask)
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        return Image.fromarray(blended)

    def linear(self, x, **kwargs):
        return 1 - x

    def exp(self, x, **kwargs):
        system_g = kwargs.get('system_g', 4)
        return np.exp(-system_g * x)

    def quadratic(self, x, **kwargs):
        return 1 - x ** 2

    def log(self, x, **kwargs):
        b = 1 / (np.e - 1)
        a = np.log(b) + 1
        return a - np.log(x + b)

    def brachistochrone(self, x, **kwargs):

        def equation(t):
            return t - np.sin(t) - (x / self.r)

        t0 = fsolve(equation, [1.0, 1.0])[0]
        y0 = -self.r * (1 - np.cos(t0)) + 1.0
        return y0


def ImageEncoder(images, clip_model, blur_transform=None):
    img_dir = ''
    clip_model.eval()

    set_images = sorted(list(set(images)))
    batch_size = 32
    image_features_list = []
    with torch.no_grad():
        for i in range(0, len(set_images), batch_size):
            batch_images = set_images[i:i + batch_size]

            device = next(clip_model.parameters()).device
            ele = [blur_transform(Image.open(os.path.join(img_dir, img)).convert("RGB")) for img in
                   batch_images]

            image_inputs = torch.stack(ele).to(device)

            batch_image_features = clip_model.encode_image(image_inputs)
            batch_image_features = batch_image_features / batch_image_features.norm(dim=-1,
                                                                                    keepdim=True)
            image_features_list.append(batch_image_features)
        image_features = torch.cat(image_features_list, dim=0)
        image_features_dict = {set_images[i].split('/')[-1]: image_features[i].float().cpu() for i
                               in range(len(set_images))}
    return image_features_dict


# Custom transform to repeat grayscale channel 3 times
class RepeatChannel3(torch.nn.Module):
    def forward(self, x):
        if isinstance(x, Image.Image):
            # Convert PIL grayscale (mode "L") to RGB
            if x.mode == "L":
                return x.convert("RGB")
            return x
        elif isinstance(x, torch.Tensor):
            # Tensor input: [C, H, W]
            if x.shape[0] == 1:
                return x.repeat(3, 1, 1)
            return x
        else:
            raise TypeError(f"Unsupported type {type(x)}")


def extract_and_save_features(model_name, pretrained, base_img_dir, is_vanilla=False,
                              grey_scale=False, out_dir=None):
    clip_model, clip_preprocess, _ = open_clip.create_model_and_transforms(
        model_name, device='cuda', pretrained=pretrained
    )
    clip_model = clip_model.eval()

    bw_transform = [v2.Grayscale(), RepeatChannel3()] if grey_scale else []
    if is_vanilla:
        # Vanilla
        notransform = transforms.Compose([
            *bw_transform,
            transforms.Resize((224, 224)),
            DirectT(),
            *clip_preprocess.transforms[2:],
        ])
    else:
        blur_transform = {}
        c = 6
        for shift, tag in zip([-c, 0, c], ['low', 'medium', 'high']):
            blur_transform[tag] = transforms.Compose([
                *bw_transform,
                transforms.Resize((224, 224)),
                FoveaBlur(224, 224, 51 + shift, system_g=3, curve_type='exp'),
                *clip_preprocess.transforms[2:],
            ])

    for db_type in ['test', 'train']:
        img_dir = f'{base_img_dir}/{db_type}_vis/'
        img_list = sorted(glob.glob(img_dir + '/*.png') + glob.glob(img_dir + '/*.jpg'))
        print(len(img_list))

        if is_vanilla:
            # extracting features
            image_features_tosave = ImageEncoder(img_list, clip_model, notransform)

            # png to jpg
            image_features_dict_jpg = dict()
            for key, val in image_features_tosave.items():
                image_features_dict_jpg[key.replace('.png', '.jpg')] = val
        else:
            # extracting features
            image_features_tosave = {}
            for key in blur_transform.keys():
                image_features_tosave[key] = ImageEncoder(img_list, clip_model, blur_transform[key])

            # jpg to png
            image_features_dict_jpg = {}
            for tag in blur_transform.keys():
                image_features_dict_jpg[tag] = dict()
                for key, val in image_features_tosave[tag].items():
                    image_features_dict_jpg[tag][key.replace('.png', '.jpg')] = val
        image_features_tosave = image_features_dict_jpg

        bwstr = '__bw' if grey_scale else ''

        suffix = '__SAM' if 'sam1_pps64_sst92' in base_img_dir else ''
        data_name = f'{model_name}__{pretrained}{suffix}'
        if out_dir is None:
            out_type = 'DirectT' if is_vanilla else 'FoveaBlur'
            out_dir_tmp = f'./data/image_features/{out_type}'
        else:
            out_dir_tmp = out_dir
        out_dir_tmp = f'{out_dir_tmp}/{data_name}{bwstr}/'
        os.makedirs(out_dir_tmp, exist_ok=True)
        file_path = f'{out_dir_tmp}/{db_type}.pt'
        torch.save({'img_features': image_features_tosave}, file_path)


if __name__ == "__main__":
    print(sys.argv[1:])
    grey_scale = True if len(sys.argv) >= 6 and sys.argv[5] == 'bw' else False
    out_dir = sys.argv[6] if len(sys.argv) >= 7 else None
    extract_and_save_features(
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == 'vanilla', grey_scale=grey_scale,
        out_dir=out_dir)
