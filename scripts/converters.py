# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import requests
import os
import os.path as osp
import torch
try:
    from omegaconf import OmegaConf
except ImportError:
    raise ImportError(
        "OmegaConf is required to convert the LDM checkpoints. Please install it with `pip install OmegaConf`."
    )

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LDMTextToImagePipeline,
    LMSDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DiffusionPipeline
)
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import LDMBertConfig, LDMBertModel
#from diffusers.pipelines.paint_by_example import PaintByExampleImageEncoder, PaintByExamplePipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor, BertTokenizerFast, CLIPTextModel, CLIPTokenizer, CLIPVisionConfig, CLIPTextConfig
import model_util

class Convert_SD_to_Diffusers():

    def __init__(self, checkpoint_path, output_path, prediction_type=None, img_size=None, original_config_file=None, extract_ema=False, num_in_channels=None,pipeline_type=None,scheduler_type=None,sd_version=None,half=None,version=None):
        self.checkpoint_path = checkpoint_path
        self.output_path = output_path
        self.prediction_type = prediction_type
        self.img_size = img_size
        self.original_config_file = original_config_file
        self.extract_ema = extract_ema
        self.num_in_channels = num_in_channels
        self.pipeline_type = pipeline_type
        self.scheduler_type = scheduler_type
        self.sd_version = sd_version
        self.half = half
        self.version = version
        self.main()
        

    def main(self):
        image_size = self.img_size
        prediction_type = self.prediction_type
        original_config_file = self.original_config_file
        num_in_channels = self.num_in_channels
        scheduler_type = self.scheduler_type
        pipeline_type = self.pipeline_type
        extract_ema = self.extract_ema
        reference_diffusers_model = None
        if self.version == 'v1':
            is_v1 = True
            is_v2 = False
        if self.version == 'v2':
            is_v1 = False
            is_v2 = True
        if is_v2 == True and prediction_type == 'vprediction':
            reference_diffusers_model = 'stabilityai/stable-diffusion-2'
        if is_v2 == True and prediction_type == 'epsilon':
            reference_diffusers_model = 'stabilityai/stable-diffusion-2-base'
        if is_v1 == True and prediction_type == 'epsilon':
            reference_diffusers_model = 'runwayml/stable-diffusion-v1-5'
        dtype = 'fp16' if self.half else None
        v2_model = True if is_v2 else False
        print(f"loading model from: {self.checkpoint_path}")
        #print(v2_model)
        text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(v2_model, self.checkpoint_path)
        print(f"copy scheduler/tokenizer config from: {reference_diffusers_model}")
        model_util.save_diffusers_checkpoint(v2_model, self.output_path, text_encoder, unet, reference_diffusers_model, vae)
        print(f"Diffusers model saved.")
        
        

class Convert_Diffusers_to_SD():
    def __init__(self,model_path=None, output_path=None):
        pass
        def main(model_path:str, output_path:str):
            #print(model_path)
            #print(output_path)
            global_step = None
            epoch = None
            dtype = torch.float16
            pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype, tokenizer=None, safety_checker=None)
            text_encoder = pipe.text_encoder
            vae = pipe.vae
            unet = pipe.unet
            v2_model = unet.config.cross_attention_dim == 1024
            original_model = None
            key_count = model_util.save_stable_diffusion_checkpoint(v2_model, output_path, text_encoder, unet,
                                                              original_model, epoch, global_step, dtype, vae)
            print(f"Saved CKPT model")
        return main(model_path, output_path)