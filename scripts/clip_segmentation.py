import argparse
import os
from typing import Optional

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import transforms, functional
from tqdm.auto import tqdm
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

DEVICE = "cuda"


def parse_args():
    parser = argparse.ArgumentParser(description="ClipSeg script.")
    parser.add_argument(
        "--sample_dir",
        type=str,
        required=True,
        help="directory where samples are located",
    )
    parser.add_argument(
        "--add_prompt",
        type=str,
        required=True,
        action="append",
        help="a prompt used to create a mask",
        dest="prompts",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='fill',
        required=False,
        help="Either replace, fill, add or subtract",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default='0.3',
        required=False,
        help="threshold for including pixels in the mask",
    )
    parser.add_argument(
        "--smooth_pixels",
        type=int,
        default=5,
        required=False,
        help="radius of a smoothing operation applied to the generated mask",
    )
    parser.add_argument(
        "--expand_pixels",
        type=int,
        default=10,
        required=False,
        help="amount of expansion of the generated mask in all directions",
    )

    args = parser.parse_args()
    return args


class MaskSample:
    def __init__(self, filename: str):
        self.image_filename = filename
        self.mask_filename = os.path.splitext(filename)[0] + "-masklabel.png"

        self.image = None
        self.mask_tensor = None

        self.height = 0
        self.width = 0

        self.image2Tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.tensor2Image = transforms.Compose([
            transforms.ToPILImage(),
        ])

    def get_image(self) -> Image:
        if self.image is None:
            self.image = Image.open(self.image_filename).convert('RGB')
            self.height = self.image.height
            self.width = self.image.width

        return self.image

    def get_mask_tensor(self) -> Tensor:
        if self.mask_tensor is None and os.path.exists(self.mask_filename):
            mask = Image.open(self.mask_filename).convert('L')
            mask = self.image2Tensor(mask)
            mask = mask.to(DEVICE)
            self.mask_tensor = mask.unsqueeze(0)

        return self.mask_tensor

    def set_mask_tensor(self, mask_tensor: Tensor):
        self.mask_tensor = mask_tensor

    def add_mask_tensor(self, mask_tensor: Tensor):
        mask = self.get_mask_tensor()
        if mask is None:
            mask = mask_tensor
        else:
            mask += mask_tensor
        mask = torch.clamp(mask, 0, 1)

        self.mask_tensor = mask

    def subtract_mask_tensor(self, mask_tensor: Tensor):
        mask = self.get_mask_tensor()
        if mask is None:
            mask = mask_tensor
        else:
            mask -= mask_tensor
        mask = torch.clamp(mask, 0, 1)

        self.mask_tensor = mask

    def save_mask(self):
        if self.mask_tensor is not None:
            mask = self.mask_tensor.cpu().squeeze()
            mask = self.tensor2Image(mask).convert('RGB')
            mask.save(self.mask_filename)


class ClipSeg:
    def __init__(self):
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model.eval()
        self.model.to(DEVICE)

        self.smoothing_kernel_radius = None
        self.smoothing_kernel = self.__create_average_kernel(self.smoothing_kernel_radius)

        self.expand_kernel_radius = None
        self.expand_kernel = self.__create_average_kernel(self.expand_kernel_radius)

    @staticmethod
    def __create_average_kernel(kernel_radius: Optional[int]):
        if kernel_radius is None:
            return None

        kernel_size = kernel_radius * 2 + 1
        kernel_weights = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        kernel = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, bias=False, padding_mode='replicate', padding=kernel_radius)
        kernel.weight.data = kernel_weights
        kernel.requires_grad_(False)
        kernel.to(DEVICE)
        return kernel

    @staticmethod
    def __get_sample_filenames(sample_dir: str) -> [str]:
        filenames = []
        for filename in os.listdir(sample_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'] and '-masklabel.png' not in filename:
                filenames.append(os.path.join(sample_dir, filename))

        return filenames

    def __process_mask(self, mask: Tensor, target_height: int, target_width: int, threshold: float) -> Tensor:
        while len(mask.shape) < 4:
            mask = mask.unsqueeze(0)

        mask = torch.sigmoid(mask)
        mask = mask.sum(1).unsqueeze(1)
        if self.smoothing_kernel is not None:
            mask = self.smoothing_kernel(mask)
        mask = functional.resize(mask, [target_height, target_width])
        mask = (mask > threshold).float()
        if self.expand_kernel is not None:
            mask = self.expand_kernel(mask)
        mask = (mask > 0).float()

        return mask

    def mask_image(self, filename: str, prompts: [str], mode: str = 'fill', threshold: float = 0.3, smooth_pixels: int = 5, expand_pixels: int = 10):
        mask_sample = MaskSample(filename)

        if mode == 'fill' and mask_sample.get_mask_tensor() is not None:
            return

        if self.smoothing_kernel_radius != smooth_pixels:
            self.smoothing_kernel = self.__create_average_kernel(smooth_pixels)
            self.smoothing_kernel_radius = smooth_pixels

        if self.expand_kernel_radius != expand_pixels:
            self.expand_kernel = self.__create_average_kernel(expand_pixels)
            self.expand_kernel_radius = expand_pixels

        inputs = self.processor(text=prompts, images=[mask_sample.get_image()] * len(prompts), padding="max_length", return_tensors="pt")
        inputs.to(DEVICE)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predicted_mask = self.__process_mask(outputs.logits, mask_sample.height, mask_sample.width, threshold)

        if mode == 'replace' or mode == 'fill':
            mask_sample.set_mask_tensor(predicted_mask)
        elif mode == 'add':
            mask_sample.add_mask_tensor(predicted_mask)
        elif mode == 'subtract':
            mask_sample.subtract_mask_tensor(predicted_mask)

        mask_sample.save_mask()

    def mask_folder(self, sample_dir: str, prompts: [str], mode: str = 'fill', threshold: float = 0.3, smooth_pixels: int = 5, expand_pixels: int = 10):
        """
        Masks all samples in a folder.

        Parameters:
            sample_dir (`str`): directory where samples are located
            prompts (`[str]`): a list of prompts used to create a mask
            mode (`str`): can be one of
                - replace: creates new masks for all samples, even if a mask already exists
                - fill: creates new masks for all samples without a mask
                - add: adds the new region to existing masks
                - subtract: subtracts the new region from existing masks
            threshold (`float`): threshold for including pixels in the mask
            smooth_pixels (`int`): radius of a smoothing operation applied to the generated mask
            expand_pixels (`int`): amount of expansion of the generated mask in all directions
        """
        filenames = self.__get_sample_filenames(sample_dir)

        for filename in tqdm(filenames):
            self.mask_image(filename, prompts, mode, threshold, smooth_pixels, expand_pixels)


def main():
    args = parse_args()
    clip_seg = ClipSeg()
    clip_seg.mask_folder(args.sample_dir, args.prompts, args.mode, args.threshold, args.smooth_pixels, args.expand_pixels)


if __name__ == "__main__":
    main()
