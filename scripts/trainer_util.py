
from typing import Iterable
import torch
import torch.utils.checkpoint
from diffusers import DiffusionPipeline
from torchvision import transforms
from PIL import Image, ImageFilter
from pathlib import Path
from tqdm import tqdm

class Depth2Img:
    def __init__(self,unet,text_encoder,revision,pretrained_model_name_or_path,accelerator):
        self.unet = unet
        self.text_encoder = text_encoder
        self.revision = revision if revision != 'no' else 'fp32'
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.accelerator = accelerator
        self.pipeline = None
    def depth_images(self,paths):
        if self.pipeline is None:
            self.pipeline = DiffusionPipeline.from_pretrained(
                self.pretrained_model_name_or_path,
                unet=self.accelerator.unwrap_model(self.unet),
                text_encoder=self.accelerator.unwrap_model(self.text_encoder),
                revision=self.revision,
                local_files_only=True,)
            self.pipeline.to(self.accelerator.device)
            self.vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels) - 1)
        non_depth_image_files = []
        image_paths_by_path = {}
        
        for path in paths:
            #if path is list
            if isinstance(path, list):
                img = Path(path[0])
            else:
                img = Path(path)
            if self.get_depth_image_path(img).exists():
                continue
            else:
                non_depth_image_files.append(img)
        image_objects = []
        for image_path in non_depth_image_files:
            image_instance = Image.open(image_path)
            if not image_instance.mode == "RGB":
                image_instance = image_instance.convert("RGB")
            image_instance = self.pipeline.feature_extractor(
                image_instance, return_tensors="pt"
            ).pixel_values
            
            image_instance = image_instance.to(self.accelerator.device)
            image_objects.append((image_path, image_instance))
        
        for image_path, image_instance in image_objects:
            path = image_path.parent
            ogImg = Image.open(image_path)
            ogImg_x = ogImg.size[0]
            ogImg_y = ogImg.size[1]
            depth_map = self.pipeline.depth_estimator(image_instance).predicted_depth
            depth_min = torch.amin(depth_map, dim=[0, 1, 2], keepdim=True)
            depth_max = torch.amax(depth_map, dim=[0, 1, 2], keepdim=True)
            depth_map = torch.nn.functional.interpolate(depth_map.unsqueeze(1),size=(ogImg_y, ogImg_x),mode="bicubic",align_corners=False,)           

            depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
            depth_map = depth_map[0,:,:]
            depth_map_image = transforms.ToPILImage()(depth_map)
            depth_map_image = depth_map_image.filter(ImageFilter.GaussianBlur(radius=1))
            depth_map_image.save(self.get_depth_image_path(image_path))
            #quit()
        return 2 ** (len(self.pipeline.vae.config.block_out_channels) - 1)
        
    def get_depth_image_path(self,image_path):
        #if image_path is a string, convert it to a Path object
        if isinstance(image_path, str):
            image_path = Path(image_path)
        return image_path.parent / f"{image_path.stem}-depth.png"
        
# Adapted from torch-ema https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14 and taken from harubaru's implementation https://github.com/harubaru/waifu-diffusion
class EMAModel:
    """
    Exponential Moving Average of models weights
    """
    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay=0.9999):
        parameters = list(parameters)
        self.shadow_params = [p.clone().detach() for p in parameters]

        self.decay = decay
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        value = (1 + optimization_step) / (10 + optimization_step)
        return 1 - min(self.decay, value)

    @torch.no_grad()
    def step(self, parameters):
        parameters = list(parameters)

        self.optimization_step += 1
        self.decay = self.get_decay(self.optimization_step)

        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                tmp = self.decay * (s_param - param)
                s_param.sub_(tmp)
            else:
                s_param.copy_(param)

        torch.cuda.empty_cache()

    def copy_to(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy current averaged parameters into given collection of parameters.
        Args:
            parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                updated with the stored moving averages. If `None`, the
                parameters with which this `ExponentialMovingAverage` was
                initialized will be used.
        """
        parameters = list(parameters)
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.data)

    def to(self, device=None, dtype=None) -> None:
        r"""Move internal buffers of the ExponentialMovingAverage to `device`.
        Args:
            device: like `device` argument to `torch.Tensor.to`
        """
        # .to() on the tensors handles None correctly
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point() else p.to(device=device)
            for p in self.shadow_params
        ]