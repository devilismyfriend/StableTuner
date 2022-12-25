"""
Copyright 2022 HuggingFace, ShivamShrirao

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import keyboard
import gradio as gr
import argparse
import random
import hashlib
import itertools
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional
import shutil
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel,DiffusionPipeline, DPMSolverMultistepScheduler,EulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, List, Generator, Tuple
from PIL import Image
from diffusers.utils.import_utils import is_xformers_available
from trainer_util import EMAModel
import gc
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
logger = get_logger(__name__)
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--aspect_mode",
        type=str,
        default='dynamic',
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--aspect_mode_action_preference",
        type=str,
        default='add',
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument('--use_ema',default=False,action="store_true", help='Use EMA for finetuning')
    parser.add_argument('--clip_penultimate',default=False,action="store_true", help='Use penultimate CLIP layer for text embedding')
    parser.add_argument("--conditional_dropout", type=float, default=None,required=False, help="Conditional dropout probability")
    parser.add_argument('--disable_cudnn_benchmark', default=False, action="store_true")
    parser.add_argument('--use_text_files_as_captions', default=False, action="store_true")
    parser.add_argument(
            "--stop_text_encoder_training",
            type=int,
            default=999999999999999,
            help=("The epoch at which the text_encoder is no longer trained"),
        )
    parser.add_argument(
        "--use_bucketing",
        default=False,
        action="store_true",
        help="Will save and generate samples before training",
    )
    parser.add_argument(
        "--regenerate_latent_cache",
        default=False,
        action="store_true",
        help="Will save and generate samples before training",
    )
    parser.add_argument(
        "--save_latents_cache",
        default=False,
        action="store_true",
        help="Will save and generate samples before training",
    )
    parser.add_argument(
        "--sample_on_training_start",
        default=False,
        action="store_true",
        help="Will save and generate samples before training",
    )

    parser.add_argument(
        "--add_class_images_to_dataset",
        default=False,
        action="store_true",
        help="will generate and add class images to the dataset without using prior reservation in training",
    )
    parser.add_argument(
        "--auto_balance_concept_datasets",
        default=False,
        action="store_true",
        help="will balance the number of images in each concept dataset to match the minimum number of images in any concept dataset",
    )
    parser.add_argument(
        "--sample_aspect_ratios",
        default=False,
        action="store_true",
        help="sample different aspect ratios for each image",
    )
    parser.add_argument(
        "--dataset_repeats",
        type=int,
        default=1,
        help="repeat the dataset this many times",
    )
    parser.add_argument(
        "--save_every_n_epoch",
        type=int,
        default=1,
        help="save on epoch finished",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--save_sample_prompt",
        type=str,
        default=None,
        help="The prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--n_save_sample",
        type=int,
        default=4,
        help="The number of samples to save.",
    )
    parser.add_argument(
        "--sample_height",
        type=int,
        default=512,
        help="The number of samples to save.",
    )
    parser.add_argument(
        "--sample_width",
        type=int,
        default=512,
        help="The number of samples to save.",
    )
    parser.add_argument(
        "--save_guidance_scale",
        type=float,
        default=7.5,
        help="CFG for save sample.",
    )
    parser.add_argument(
        "--save_infer_steps",
        type=int,
        default=30,
        help="The number of inference steps for save sample.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--sample_step_interval", type=int, default=100000000000000, help="Sample images every N steps.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--not_cache_latents", action="store_true", help="Do not precompute and cache latents from VAE.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument("--save_sample_controlled_seed", type=int, action='append', help="Set a seed for an extra sample image to be constantly saved.")
    parser.add_argument("--delete_checkpoints_when_full_drive", default=False, action="store_true", help="Delete checkpoints when the drive is full.")
    parser.add_argument("--send_telegram_updates", default=False, action="store_true", help="Send Telegram updates.")
    parser.add_argument("--telegram_chat_id", type=str, default="0", help="Telegram chat ID.")
    parser.add_argument("--telegram_token", type=str, default="0", help="Telegram token.")
    parser.add_argument("--use_deepspeed_adam", default=False, action="store_true", help="Use experimental DeepSpeed Adam 8.")
    parser.add_argument('--append_sample_controlled_seed_action', action='append')
    parser.add_argument('--add_sample_prompt', type=str, action='append')
    parser.add_argument('--use_image_names_as_captions', default=False, action="store_true")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


ASPECT_832 = [[832, 832], 
[896, 768], [768, 896], 
[960, 704], [704, 960], 
[1024, 640], [640, 1024], 
[1152, 576], [576, 1152], 
[1280, 512], [512, 1280], 
[1344, 512], [512, 1344], 
[1408, 448], [448, 1408], 
[1472, 448], [448, 1472], 
[1536, 384], [384, 1536], 
[1600, 384], [384, 1600]]

ASPECT_896 = [[896, 896],
[960, 832], [832, 960],
[1024, 768], [768, 1024],
[1088, 704], [704, 1088],
[1152, 704], [704, 1152],
[1216, 640], [640, 1216],
[1280, 640], [640, 1280],
[1344, 576], [576, 1344],
[1408, 576], [576, 1408],
[1472, 512], [512, 1472],
[1536, 512], [512, 1536], 
[1600, 448], [448, 1600], 
[1664, 448], [448, 1664]]
ASPECT_960 = [[896, 896], 
[960, 832], [832, 960], 
[1024, 768], [768, 1024], 
[1088, 704], [704, 1088], 
[1216, 640], [640, 1216], 
[1344, 576], [576, 1344], 
[1472, 512], [512, 1472], 
[1536, 512], [512, 1536], 
[1600, 448], [448, 1600], 
[1664, 448], [448, 1664]]     
ASPECT_1024 = [[1024, 1024], 
[1088, 960], [960, 1088], 
[1152, 896], [896, 1152], 
[1216, 832], [832, 1216], 
[1344, 768], [768, 1344], 
[1472, 704], [704, 1472], 
[1600, 640], [640, 1600], 
[1728, 576], [576, 1728], 
[1792, 576], [576, 1792]]
ASPECT_768 = [[768,768],     # 589824 1:1
    [832,704],[704,832],   # 585728 1.181:1
    [896,640],[640,896],   # 573440 1.4:1
    [960,576],[576,960],   # 552960 1.6:1
    [1024,576],[576,1024], # 524288 1.778:1
    [1088,512],[512,1088], # 497664 2.125:1
    [1152,512],[512,1152], # 589824 2.25:1
    [1216,448],[448,1216], # 552960 2.714:1
    [1280,448],[448,1280], # 573440 2.857:1
    [1344,384],[384,1344], # 518400 3.5:1
    [1408,384],[384,1408], # 540672 3.667:1
    [1472,320],[320,1472], # 470400 4.6:1
    [1536,320],[320,1536], # 491520 4.8:1
]

ASPECT_704 = [[704,704],     # 501,376 1:1
    [768,640],[640,768],   # 491,520 1.2:1
    [832,576],[576,832],   # 458,752 1.444:1
    [896,512],[512,896],   # 458,752 1.75:1
    [960,512],[512,960],   # 491,520 1.875:1
    [1024,448],[448,1024], # 458,752 2.286:1
    [1088,448],[448,1088], # 487,424 2.429:1
    [1152,384],[384,1152], # 442,368 3:1
    [1216,384],[384,1216], # 466,944 3.125:1
    [1280,384],[384,1280], # 491,520 3.333:1
    [1280,320],[320,1280], # 409,600 4:1
    [1408,320],[320,1408], # 450,560 4.4:1
    [1536,320],[320,1536], # 491,520 4.8:1
]

ASPECT_640 = [[640,640],     # 409600 1:1 
    [704,576],[576,704],   # 405504 1.25:1
    [768,512],[512,768],   # 393216 1.5:1
    [896,448],[448,896],   # 401408 2:1
    [1024,384],[384,1024], # 393216 2.667:1
    [1280,320],[320,1280], # 409600 4:1
    [1408,256],[256,1408], # 360448 5.5:1
    [1472,256],[256,1472], # 376832 5.75:1
    [1536,256],[256,1536], # 393216 6:1
    [1600,256],[256,1600], # 409600 6.25:1
]

ASPECT_576 = [[576,576],     # 331776 1:1
    [640,512],[512,640],   # 327680 1.25:1
    [640,448],[448,640],   # 286720 1.4286:1
    [704,448],[448,704],   # 314928 1.5625:1
    [832,384],[384,832],   # 317440 2.1667:1
    [1024,320],[320,1024], # 327680 3.2:1
    [1280,256],[256,1280], # 327680 5:1
]

ASPECTS_512 = [[512,512],      # 262144 1:1
    [576,448],[448,576],   # 258048 1.29:1
    [640,384],[384,640],   # 245760 1.667:1
    [768,320],[320,768],   # 245760 2.4:1
    [832,256],[256,832],   # 212992 3.25:1
    [896,256],[256,896],   # 229376 3.5:1
    [960,256],[256,960],   # 245760 3.75:1
    [1024,256],[256,1024], # 245760 4:1
    ]

#failsafe aspects
ASPECTS = ASPECTS_512
def get_aspect_buckets(resolution):
    if resolution < 512:
        raise ValueError("Resolution must be at least 512")
    try: 
        rounded_resolution = int(resolution / 64) * 64
        print(f" {bcolors.WARNING} Rounded resolution to: {rounded_resolution}{bcolors.ENDC}")   
        all_image_sizes = __get_all_aspects()
        aspects = next(filter(lambda sizes: sizes[0][0]==rounded_resolution, all_image_sizes), None)
        ASPECTS = aspects
        #print(aspects)
        return aspects
    except Exception as e:
        print(f" {bcolors.FAIL} *** Could not find selected resolution: {rounded_resolution}{bcolors.ENDC}")   

        raise e

def __get_all_aspects():
    return [ASPECTS_512, ASPECT_576, ASPECT_640, ASPECT_704, ASPECT_768,ASPECT_832,ASPECT_896,ASPECT_960,ASPECT_1024]

class AutoBucketing(Dataset):
    def __init__(self,
                 concepts_list,
                 tokenizer=None,
                 flip_p=0.0,
                 repeats=1,
                 debug_level=0,
                 batch_size=1,
                 set='val',
                 resolution=512,
                 center_crop=False,
                 use_image_names_as_captions=True,
                 add_class_images_to_dataset=None,
                 balance_datasets=False,
                 crop_jitter=20,
                 with_prior_loss=False,
                 use_text_files_as_captions=False,
                 conditional_dropout=None,
                 aspect_mode='dynamic',
                 action_preference='dynamic'
                 ):
        
        self.debug_level = debug_level
        self.resolution = resolution
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.concepts_list = concepts_list
        self.use_image_names_as_captions = use_image_names_as_captions
        self.num_train_images = 0
        self.num_reg_images = 0
        self.image_train_items = []
        self.image_reg_items = []
        self.add_class_images_to_dataset = add_class_images_to_dataset
        self.balance_datasets = balance_datasets
        self.crop_jitter = crop_jitter
        self.with_prior_loss = with_prior_loss
        self.use_text_files_as_captions = use_text_files_as_captions
        self.conditional_dropout = conditional_dropout
        self.aspect_mode = aspect_mode
        self.action_preference = action_preference
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        #shared_dataloader = None
        print(f" {bcolors.WARNING}Creating Auto Bucketing Dataloader{bcolors.ENDC}")   

        shared_dataloader = DataLoaderMultiAspect(concepts_list, debug_level=debug_level,resolution=self.resolution, batch_size=self.batch_size, flip_p=flip_p,use_image_names_as_captions=self.use_image_names_as_captions,add_class_images_to_dataset=self.add_class_images_to_dataset,balance_datasets=self.balance_datasets,with_prior_loss=self.with_prior_loss,use_text_files_as_captions=self.use_text_files_as_captions,aspect_mode=self.aspect_mode,action_preference=self.action_preference)
        
        #print(self.image_train_items)
        if self.with_prior_loss and self.add_class_images_to_dataset == False:
            self.image_train_items, self.class_train_items = shared_dataloader.get_all_images()
            self.num_train_images = self.num_train_images + len(self.image_train_items)
            self.num_reg_images = self.num_reg_images + len(self.class_train_items)
            self._length = max(max(math.trunc(self.num_train_images * repeats), batch_size),math.trunc(self.num_reg_images * repeats), batch_size) - self.num_train_images % self.batch_size
            self.num_train_images = self.num_train_images + self.num_reg_images
            
        else:
            self.image_train_items = shared_dataloader.get_all_images()
            self.num_train_images = self.num_train_images + len(self.image_train_items)
            self._length = max(math.trunc(self.num_train_images * repeats), batch_size) - self.num_train_images % self.batch_size
        #amoutn of images to drop due to conditional dropout
        if self.conditional_dropout != None:
            #print('conditional dropout: ' + str(self.conditional_dropout))
            #it's a float, convert to percentage
            if self.conditional_dropout < 1:
                self.conditional_dropout = self.conditional_dropout * 100
            #calculate how many images to drop
            self.num_images_to_drop = math.trunc(self.num_train_images * (self.conditional_dropout / 100))
            print()
            print(f" {bcolors.WARNING} ** Conditional Dropout will drop: {self.num_images_to_drop} captions{bcolors.ENDC}")   
            print()
            selectedRandoms = []
            for i in range(self.num_images_to_drop):
                #pick a random image to drop
                random_image = random.randint(0, len(self.image_train_items) - 1)
                while random_image in selectedRandoms:
                    random_image = random.randint(0, len(self.image_train_items) - 1)
                selectedRandoms.append(random_image)
                #remove it from the list
                train_item = self.image_train_items[random_image]
                #edit the train_item caption to be the word 'drop'
                train_item.caption = ''
                #replace the image in the list with the edited train_item
                self.image_train_items[random_image] = train_item
                #print('   ' +str(i))

        
        print()
        print(f" {bcolors.WARNING} ** Validation Set: {set}, steps: {self._length / batch_size:.0f}, repeats: {repeats} {bcolors.ENDC}")   
        print()
        
    
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        idx = i % self.num_train_images
        #print(idx)
        image_train_item = self.image_train_items[idx]
        
        example = self.__get_image_for_trainer(image_train_item,debug_level=self.debug_level)
        if self.with_prior_loss and self.add_class_images_to_dataset == False:
            idx = i % self.num_reg_images
            class_train_item = self.class_train_items[idx]
            example_class = self.__get_image_for_trainer(class_train_item,debug_level=self.debug_level,class_img=True)
            example= {**example, **example_class}
            
        #print the tensor shape
        #print(example['instance_images'].shape)
        #print(example.keys())
        return example
    def normalize8(self,I):
            mn = I.min()
            mx = I.max()

            mx -= mn

            I = ((I - mn)/mx) * 255
            return I.astype(np.uint8)
    def __get_image_for_trainer(self,image_train_item,debug_level=0,class_img=False):
        example = {}
        save = debug_level > 2
        
        if class_img==False:
            image_train_tmp = image_train_item.hydrate(crop=False, save=0, crop_jitter=self.crop_jitter)
            image_train_tmp_image = Image.fromarray(self.normalize8(image_train_tmp.image)).convert("RGB")
            example["instance_images"] = self.image_transforms(image_train_tmp_image)       
            #print(image_train_tmp.caption)     
            example["instance_prompt_ids"] = self.tokenizer(
                image_train_tmp.caption,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
            image_train_item.self_destruct()
            return example
            
        if class_img==True:
            image_train_tmp = image_train_item.hydrate(crop=False, save=4, crop_jitter=self.crop_jitter)
            image_train_tmp_image = Image.fromarray(self.normalize8(image_train_tmp.image)).convert("RGB")
            example["class_images"] = self.image_transforms(image_train_tmp_image)
            example["class_prompt_ids"] = self.tokenizer(
                image_train_tmp.caption,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids
            image_train_item.self_destruct()
            return example

_RANDOM_TRIM = 0.04
class ImageTrainItem(): 
    """
    image: Image
    identifier: caption,
    target_aspect: (width, height), 
    pathname: path to image file
    flip_p: probability of flipping image (0.0 to 1.0)
    """    
    def __init__(self, image: Image, caption: str, target_wh: list, pathname: str, flip_p=0.0):
        self.caption = caption
        self.target_wh = target_wh
        self.pathname = pathname
        self.flip_p = flip_p
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.cropped_img = None
        self.is_dupe = []
        if image is None:
            self.image = []
        else:
            self.image = image
    def self_destruct(self):
        self.image = []
        self.cropped_img = None
        self.is_dupe.append(1)
    def hydrate(self, crop=False, save=False, crop_jitter=20):
        """
        crop: hard center crop to 512x512
        save: save the cropped image to disk, for manual inspection of resize/crop
        crop_jitter: randomly shift cropp by N pixels when using multiple aspect ratios to improve training quality
        """
        if not hasattr(self, 'image') or len(self.image) == 0:
            if len(self.is_dupe) > 0:
                chance = float(len(self.is_dupe)) / 10.0
                self.flip = transforms.RandomHorizontalFlip(p=self.flip_p+chance if chance < 1.0 else 1.0)
                self.crop_jitter = crop_jitter + (len(self.is_dupe) * 10) if crop_jitter < 50 else 50
            self.image = Image.open(self.pathname).convert('RGB')

            width, height = self.image.size
            if crop: 
                cropped_img = self.__autocrop(self.image)
                self.image = cropped_img.resize((512,512), resample=Image.Resampling.LANCZOS)
            else:
                width, height = self.image.size
                jitter_amount = random.randint(0,crop_jitter)

                if self.target_wh[0] == self.target_wh[1]:
                    if width > height:
                        left = random.randint(0, width - height)
                        self.image = self.image.crop((left, 0, height+left, height))
                        width = height
                    elif height > width:
                        top = random.randint(0, height - width)
                        self.image = self.image.crop((0, top, width, width+top))
                        height = width
                    elif width > self.target_wh[0]:
                        slice = min(int(self.target_wh[0] * _RANDOM_TRIM), width-self.target_wh[0])
                        slicew_ratio = random.random()
                        left = int(slice*slicew_ratio)
                        right = width-int(slice*(1-slicew_ratio))
                        sliceh_ratio = random.random()
                        top = int(slice*sliceh_ratio)
                        bottom = height- int(slice*(1-sliceh_ratio))

                        self.image = self.image.crop((left, top, right, bottom))
                else: 
                    image_aspect = width / height                    
                    target_aspect = self.target_wh[0] / self.target_wh[1]
                    if image_aspect > target_aspect:
                        new_width = int(height * target_aspect)
                        jitter_amount = max(min(jitter_amount, int(abs(width-new_width)/2)), 0)
                        left = jitter_amount
                        right = left + new_width
                        self.image = self.image.crop((left, 0, right, height))
                    else:
                        new_height = int(width / target_aspect)
                        jitter_amount = max(min(jitter_amount, int(abs(height-new_height)/2)), 0)
                        top = jitter_amount
                        bottom = top + new_height
                        self.image = self.image.crop((0, top, width, bottom))
                        #LAZCOS resample
                self.image = self.image.resize(self.target_wh, resample=Image.Resampling.LANCZOS)

            self.image = self.flip(self.image)

        if type(self.image) is not np.ndarray:
            if save: 
                base_name = os.path.basename(self.pathname)
                if not os.path.exists("test/output"):
                    os.makedirs("test/output")
                self.image.save(f"test/output/{base_name}")
            
            self.image = np.array(self.image).astype(np.uint8)

            self.image = (self.image / 127.5 - 1.0).astype(np.float32)
        
        #print(self.image.shape)

        return self

class DataLoaderMultiAspect():
    """
    Data loader for multi-aspect-ratio training and bucketing
    data_root: root folder of training data
    batch_size: number of images per batch
    flip_p: probability of flipping image horizontally (i.e. 0-0.5)
    """
    def __init__(self,concept_list, seed=555, debug_level=0,resolution=512, batch_size=1, flip_p=0.0,use_image_names_as_captions=True,add_class_images_to_dataset=False,balance_datasets=False,with_prior_loss=False,use_text_files_as_captions=False,aspect_mode='dynamic',action_preference='add'):
        self.resolution = resolution
        self.debug_level = debug_level
        self.flip_p = flip_p
        self.use_image_names_as_captions = use_image_names_as_captions
        self.balance_datasets = balance_datasets
        self.with_prior_loss = with_prior_loss
        self.add_class_images_to_dataset = add_class_images_to_dataset
        self.use_text_files_as_captions = use_text_files_as_captions
        self.aspect_mode = aspect_mode
        self.action_preference = action_preference
        prepared_train_data = []
        
        self.aspects = get_aspect_buckets(resolution)
        #print(f"* DLMA resolution {resolution}, buckets: {self.aspects}")
        #process sub directories flag
            
        print(f" {bcolors.WARNING} Preloading images...{bcolors.ENDC}")   

        if balance_datasets:
            print(f" {bcolors.WARNING} Balancing datasets...{bcolors.ENDC}") 
            #get the concept with the least number of images in instance_data_dir
            min_concept = min(concept_list, key=lambda x: len(os.listdir(x['instance_data_dir'])))
            #get the number of images in the concept with the least number of images
            min_concept_num_images = len(os.listdir(min_concept['instance_data_dir']))
            print(" Min concept: ",min_concept['instance_data_dir']," with ",min_concept_num_images," images")
            
            balance_cocnept_list = []
            for concept in concept_list:
                #if concept has a key do not balance it
                if 'do_not_balance' in concept:
                    if concept['do_not_balance'] == True:
                        balance_cocnept_list.append(-1)
                    else:
                        balance_cocnept_list.append(min_concept_num_images)
                else:
                        balance_cocnept_list.append(min_concept_num_images)
        for concept in concept_list:
            if 'use_sub_dirs' in concept:
                if concept['use_sub_dirs'] == True:
                    use_sub_dirs = True
                else:
                    use_sub_dirs = False
            else:
                use_sub_dirs = False
            self.image_paths = []
            #self.class_image_paths = []
            min_concept_num_images = None
            if balance_datasets:
                min_concept_num_images = balance_cocnept_list[concept_list.index(concept)]
            data_root = concept['instance_data_dir']
            data_root_class = concept['class_data_dir']
            concept_prompt = concept['instance_prompt']
            concept_class_prompt = concept['class_prompt']
            if 'flip_p' in concept.keys():
                flip_p = concept['flip_p']
                if flip_p == '':
                    flip_p = 0.0
                else:
                    flip_p = float(flip_p)
            self.__recurse_data_root(self=self, recurse_root=data_root,use_sub_dirs=use_sub_dirs)
            random.Random(seed).shuffle(self.image_paths)
            prepared_train_data.extend(self.__prescan_images(debug_level, self.image_paths, flip_p,use_image_names_as_captions,concept_prompt,use_text_files_as_captions=self.use_text_files_as_captions)[0:min_concept_num_images]) # ImageTrainItem[]
            if add_class_images_to_dataset:
                self.image_paths = []
                self.__recurse_data_root(self=self, recurse_root=data_root_class,use_sub_dirs=use_sub_dirs)
                random.Random(seed).shuffle(self.image_paths)
                use_image_names_as_captions = False
                prepared_train_data.extend(self.__prescan_images(debug_level, self.image_paths, flip_p,use_image_names_as_captions,concept_class_prompt,use_text_files_as_captions=self.use_text_files_as_captions)) # ImageTrainItem[]
            
        self.image_caption_pairs = self.__bucketize_images(prepared_train_data, batch_size=batch_size, debug_level=debug_level,aspect_mode=self.aspect_mode,action_preference=self.action_preference)
        if self.with_prior_loss and add_class_images_to_dataset == False:
            self.class_image_caption_pairs = []
            for concept in concept_list:
                self.class_images_path = []
                data_root_class = concept['class_data_dir']
                concept_class_prompt = concept['class_prompt']
                self.__recurse_data_root(self=self, recurse_root=data_root_class,use_sub_dirs=use_sub_dirs,class_images=True)
                random.Random(seed).shuffle(self.image_paths)
                use_image_names_as_captions = False
                self.class_image_caption_pairs.extend(self.__prescan_images(debug_level, self.class_images_path, flip_p,use_image_names_as_captions,concept_class_prompt,use_text_files_as_captions=self.use_text_files_as_captions))
            self.class_image_caption_pairs = self.__bucketize_images(self.class_image_caption_pairs, batch_size=batch_size, debug_level=debug_level,aspect_mode=self.aspect_mode,action_preference=self.action_preference)
        if debug_level > 0: print(f" * DLMA Example: {self.image_caption_pairs[0]} images")
        #print the length of image_caption_pairs
        print(f" {bcolors.WARNING} Number of image-caption pairs: {len(self.image_caption_pairs)}{bcolors.ENDC}") 
        if len(self.image_caption_pairs) == 0:
            raise Exception("All the buckets are empty. Please check your data or reduce the batch size.")
    def get_all_images(self):
        if self.with_prior_loss == False:
            return self.image_caption_pairs
        else:
            return self.image_caption_pairs, self.class_image_caption_pairs
    def __prescan_images(self,debug_level: int, image_paths: list, flip_p=0.0,use_image_names_as_captions=True,concept=None,use_text_files_as_captions=False):
        """
        Create ImageTrainItem objects with metadata for hydration later 
        """
        decorated_image_train_items = []
        
        for pathname in image_paths:
            identifier = concept 
            if use_image_names_as_captions:
                caption_from_filename = os.path.splitext(os.path.basename(pathname))[0].split("_")[0]
                identifier = caption_from_filename
            if use_text_files_as_captions:
                txt_file_path = os.path.splitext(pathname)[0] + ".txt"

                if os.path.exists(txt_file_path):
                    try:
                        with open(txt_file_path, 'r',encoding='utf-8') as f:
                            identifier = f.readline().rstrip()
                            f.close()
                            if len(identifier) < 1:
                                raise ValueError(f" *** Could not find valid text in: {txt_file_path}")
                            
                    except:
                        print(f" {bcolors.FAIL} *** Error reading {txt_file_path} to get caption, falling back to filename{bcolors.ENDC}") 
                        identifier = caption_from_filename
                        pass
            #print("identifier: ",identifier)
            image = Image.open(pathname)
            width, height = image.size
            image_aspect = width / height

            target_wh = min(self.aspects, key=lambda aspects:abs(aspects[0]/aspects[1] - image_aspect))

            image_train_item = ImageTrainItem(image=None, caption=identifier, target_wh=target_wh, pathname=pathname, flip_p=flip_p)

            decorated_image_train_items.append(image_train_item)
        return decorated_image_train_items

    @staticmethod
    def __bucketize_images(prepared_train_data: list, batch_size=1, debug_level=0,aspect_mode='dynamic',action_preference='add'):
        """
        Put images into buckets based on aspect ratio with batch_size*n images per bucket, discards remainder
        """
        
        # TODO: this is not terribly efficient but at least linear time
        buckets = {}
        for image_caption_pair in prepared_train_data:
            target_wh = image_caption_pair.target_wh

            if (target_wh[0],target_wh[1]) not in buckets:
                buckets[(target_wh[0],target_wh[1])] = []
            buckets[(target_wh[0],target_wh[1])].append(image_caption_pair) 
        print(f" ** Number of buckets: {len(buckets)}")
        for bucket in buckets:
            bucket_len = len(buckets[bucket])
            #real_len = len(buckets[bucket])+1
            #print(real_len)
            truncate_amount = bucket_len % batch_size
            add_amount = batch_size - bucket_len % batch_size
            action = None
            #print(f" ** Bucket {bucket} has {bucket_len} images")
            if aspect_mode == 'dynamic':
                if batch_size == bucket_len:
                    action = None
                elif add_amount < truncate_amount and add_amount != 0 and add_amount != batch_size or truncate_amount == 0:
                    action = 'add'
                    #print(f'should add {add_amount}')
                elif truncate_amount < add_amount and truncate_amount != 0 and truncate_amount != batch_size and batch_size < bucket_len:
                    #print(f'should truncate {truncate_amount}')
                    action = 'truncate'
                    #truncate the bucket
                elif truncate_amount == add_amount:
                    if action_preference == 'add':
                        action = 'add'
                    elif action_preference == 'truncate':
                        action = 'truncate'
                elif batch_size > bucket_len:
                    action = 'add'
                
            elif aspect_mode == 'add':
                action = 'add'
            elif aspect_mode == 'truncate':
                action = 'truncate'
            if action == None:
                action = None
                #print('no need to add or truncate')
            if action == None:
                #print('test')
                current_bucket_size = bucket_len
                print(f"  ** Bucket {bucket} found {bucket_len}, nice!")
            elif action == 'add':
                #copy the bucket
                shuffleBucket = random.sample(buckets[bucket], bucket_len)     
                #add the images to the bucket
                current_bucket_size = bucket_len
                truncate_count = (bucket_len) % batch_size
                #how many images to add to the bucket to fill the batch
                addAmount = batch_size - truncate_count
                if addAmount != batch_size:
                    added=0
                    while added != addAmount:
                        randomIndex = random.randint(0,len(shuffleBucket)-1)
                        #print(str(randomIndex))
                        buckets[bucket].append(shuffleBucket[randomIndex])
                        added+=1
                    print(f"  ** Bucket {bucket} found {bucket_len} images, will duplicate {added} images due to batch size {batch_size}")
                else:
                    print(f"  ** Bucket {bucket} found {bucket_len}, nice!")
            elif action == 'truncate':
                truncate_count = (bucket_len) % batch_size
                current_bucket_size = bucket_len
                buckets[bucket] = buckets[bucket][:current_bucket_size - truncate_count]
                print(f"  ** Bucket {bucket} found {bucket_len} images, will drop {truncate_count} images due to batch size {batch_size}")
            

        # flatten the buckets
        image_caption_pairs = []
        for bucket in buckets:
            image_caption_pairs.extend(buckets[bucket])
        
        return image_caption_pairs

    @staticmethod
    def __recurse_data_root(self, recurse_root,use_sub_dirs=True,class_images=False):
        for f in os.listdir(recurse_root):
            current = os.path.join(recurse_root, f)

            if os.path.isfile(current):
                ext = os.path.splitext(f)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    if class_images == False:
                        self.image_paths.append(current)
                    else:
                        self.class_images_path.append(current)
        if use_sub_dirs:
            sub_dirs = []

            for d in os.listdir(recurse_root):
                current = os.path.join(recurse_root, d)
                if os.path.isdir(current):
                    sub_dirs.append(current)

            for dir in sub_dirs:
                self.__recurse_data_root(self=self, recurse_root=dir)

class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        with_prior_preservation=True,
        size=512,
        center_crop=False,
        num_class_images=None,
        use_image_names_as_captions=False,
        repeats=1,
        use_text_files_as_captions=False,
        conditional_dropout=None,
    ):
        self.use_image_names_as_captions = use_image_names_as_captions
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.with_prior_preservation = with_prior_preservation
        self.use_text_files_as_captions = use_text_files_as_captions
        self.image_paths = []
        self.class_images_path = []
        self.conditional_dropout = conditional_dropout

        for concept in concepts_list:
            if 'use_sub_dirs' in concept:
                if concept['use_sub_dirs'] == True:
                    use_sub_dirs = True
                else:
                    use_sub_dirs = False
            else:
                use_sub_dirs = False

            for i in range(repeats):
                self.__recurse_data_root(self, concept,use_sub_dirs=use_sub_dirs)

            if with_prior_preservation:
                for i in range(repeats):
                    self.__recurse_data_root(self, concept,use_sub_dirs=False,class_images=True)
        random.shuffle(self.image_paths)
        if self.conditional_dropout != None:
            #print('conditional dropout: ' + str(self.conditional_dropout))
            #it's a float, convert to percentage
            if self.conditional_dropout < 1:
                self.conditional_dropout = self.conditional_dropout * 100
            #calculate how many images to drop
            self.num_images_to_drop = math.trunc(len(self.image_paths) * (self.conditional_dropout / 100))
            print()
            print(f" {bcolors.WARNING} ** Conditional Dropout will drop: {self.num_images_to_drop} captions{bcolors.ENDC}") 
            print()
            selectedRandoms = []
            for i in range(self.num_images_to_drop):
                #pick a random image to drop
                random_image = random.randint(0, len(self.image_paths) - 1)
                while random_image in selectedRandoms:
                    random_image = random.randint(0, len(self.image_paths) - 1)
                selectedRandoms.append(random_image)
                #remove it from the list
                train_item = self.image_paths[random_image]
                #edit the train_item caption to be the word 'drop'
                train_item[1] = ''
                #replace the image in the list with the edited train_item
                self.image_paths[random_image] = train_item
                #print('   ' +str(i))
        self.num_instance_images = len(self.image_paths)
        self._length = self.num_instance_images
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        print(f" {bcolors.WARNING} ** Dataset length: {self._length}, {int(self.num_instance_images / repeats)} images using {repeats} repeats{bcolors.ENDC}") 

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    @staticmethod
    def __recurse_data_root(self, recurse_root,use_sub_dirs=True,class_images=False):
        #if recurse root is a dict
        if isinstance(recurse_root, dict):
            concept_token = recurse_root['instance_prompt']
            data = recurse_root['instance_data_dir']
            
            if class_images:
                concept_token = recurse_root['class_prompt']
                data = recurse_root['class_data_dir']
        else:
            concept_token = None
        for f in os.listdir(data):
            current = os.path.join(data, f)

            if os.path.isfile(current):
                ext = os.path.splitext(f)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    if class_images == False:
                        self.image_paths.append([current,concept_token])
                    else:
                        self.class_images_path.append([current,concept_token])
        if use_sub_dirs:
            sub_dirs = []

            for d in os.listdir(data):
                current = os.path.join(data, d)
                if os.path.isdir(current):
                    sub_dirs.append(current)

            for dir in sub_dirs:
                if class_images != False:
                    self.__recurse_data_root(self=self, recurse_root={'instance_data_dir' : dir, 'instance_prompt' : concept_token})
                else:
                    self.__recurse_data_root(self=self, recurse_root={'class_data_dir' : dir, 'class_prompt' : concept_token})
        
    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_path, instance_prompt = self.image_paths[index % self.num_instance_images]
        og_prompt = instance_prompt
        instance_image = Image.open(instance_path)
        if self.use_image_names_as_captions == True:
            instance_prompt = str(instance_path).split(os.sep)[-1].split('.')[0].split('_')[0]
        #else if there's a txt file with the same name as the image, read the caption from there
        if self.use_text_files_as_captions == True:
            #if there's a file with the same name as the image, but with a .txt extension, read the caption from there
            #get the last . in the file name
            last_dot = str(instance_path).rfind('.')
            #get the path up to the last dot
            txt_path = str(instance_path)[:last_dot] + '.txt'

            #if txt_path exists, read the caption from there
            if os.path.exists(txt_path):
                with open(txt_path, encoding='utf-8') as f:
                    instance_prompt = f.readline().rstrip()
                    f.close()
        if self.conditional_dropout != None:
            if og_prompt == '' and instance_prompt != '':
                instance_prompt = ''
                
            
        #print('identifier: ' + instance_prompt)
        instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        if self.with_prior_preservation:
            class_path, class_prompt = self.class_images_path[index % self.num_class_images]
            class_image = Image.open(class_path)
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)
            example["class_prompt_ids"] = self.tokenizer(
                class_prompt,
                padding="do_not_pad",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
            ).input_ids

        return example


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache
    def add_latent(self, latent, text_encoder):
        self.latents_cache.append(latent)
        self.text_encoder_cache.append(text_encoder)
    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index]


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

#function to format a dictionary into a telegram message
def format_dict(d):
    message = ""
    for key, value in d.items():
        #filter keys that have the word "token" in them
        if "token" in key and "tokenizer" not in key:
            value = "TOKEN"
        if 'id' in key:
            value = "ID"
        #if value is a dictionary, format it recursively
        if isinstance(value, dict):
            for k, v in value.items():
                message += f"\n- {k}:  <b>{v}</b> \n"
        elif isinstance(value, list):
            #each value is a new line in the message
            message += f"- {key}:\n\n"
            for v in value:
                    message += f"  <b>{v}</b>\n\n"
        #if value is a list, format it as a list
        else:
            message += f"- {key}:  <b>{value}</b>\n"
    return message

def send_telegram_message(message, chat_id, token):
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}&parse_mode=html&disable_notification=True"
    import requests
    req = requests.get(url)
    if req.status_code != 200:
        raise ValueError(f"Telegram request failed with status code {req.status_code}")
def send_media_group(chat_id,telegram_token, images, caption=None, reply_to_message_id=None):
        """
        Use this method to send an album of photos. On success, an array of Messages that were sent is returned.
        :param chat_id: chat id
        :param images: list of PIL images to send
        :param caption: caption of image
        :param reply_to_message_id: If the message is a reply, ID of the original message
        :return: response with the sent message
        """
        SEND_MEDIA_GROUP = f'https://api.telegram.org/bot{telegram_token}/sendMediaGroup'
        from io import BytesIO
        import requests
        files = {}
        media = []
        for i, img in enumerate(images):
            with BytesIO() as output:
                img.save(output, format='PNG')
                output.seek(0)
                name = f'photo{i}'
                files[name] = output.read()
                # a list of InputMediaPhoto. attach refers to the name of the file in the files dict
                media.append(dict(type='photo', media=f'attach://{name}'))
        media[0]['caption'] = caption
        media[0]['parse_mode'] = 'HTML'
        return requests.post(SEND_MEDIA_GROUP, data={'chat_id': chat_id, 'media': json.dumps(media),'disable_notification':True, 'reply_to_message_id': reply_to_message_id }, files=files)
def main():
    print(f" {bcolors.OKBLUE}Booting Up StableTuner{bcolors.ENDC}") 
    print(f" {bcolors.OKBLUE}Please wait a moment as we load up some stuff...{bcolors.ENDC}") 
    #torch.cuda.set_per_process_memory_fraction(0.5)
    args = parse_args()
    if args.disable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    if args.send_telegram_updates:
        send_telegram_message(f"Booting up StableTuner!\n", args.telegram_chat_id, args.telegram_token)
    logging_dir = Path(args.output_dir, "logs", args.logging_dir)
    main_sample_dir = os.path.join(args.output_dir, "samples")
    if os.path.exists(main_sample_dir):
            shutil.rmtree(main_sample_dir)
            os.makedirs(main_sample_dir)
    #create logging directory
    if not logging_dir.exists():
        logging_dir.mkdir(parents=True)
    #create output directory
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)
    

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    if args.train_text_encoder and args.gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    if args.seed is not None:
        set_seed(args.seed)

    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt": args.class_prompt,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir": args.class_data_dir
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    if args.with_prior_preservation or args.add_class_images_to_dataset:
        pipeline = None
        for concept in args.concepts_list:
            class_images_dir = Path(concept["class_data_dir"])
            class_images_dir.mkdir(parents=True, exist_ok=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < args.num_class_images:
                torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                if pipeline is None:

                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        safety_checker=None,
                        vae=AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,subfolder=None if args.pretrained_vae_name_or_path else "vae" ),
                        torch_dtype=torch_dtype,
                         
                    )
                    pipeline.set_progress_bar_config(disable=True)
                    pipeline.to(accelerator.device)
                
                #if args.use_bucketing == False:
                num_new_images = args.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(concept["class_prompt"], num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)
                sample_dataloader = accelerator.prepare(sample_dataloader)
                #else:
                    #create class images that match up to the concept target buckets
                #    instance_images_dir = Path(concept["instance_data_dir"])
                #    cur_instance_images = len(list(instance_images_dir.iterdir()))
                    #target_wh = min(self.aspects, key=lambda aspects:abs(aspects[0]/aspects[1] - image_aspect))
                #    num_new_images = cur_instance_images - cur_class_images
                
                

                with torch.autocast("cuda"):
                    for example in tqdm(
                        sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                    ):
                        with torch.autocast("cuda"):
                            images = pipeline(example["prompt"]).images
                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                            image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name )
    elif args.pretrained_model_name_or_path:
        #print(os.getcwd())
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer" )

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder" )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae" )
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet" )
    
    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
            vae.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    if args.use_ema == True:
        ema_unet = EMAModel(unet.parameters())
    vae.requires_grad_(False)
    #vae.enable_slicing()
    if not args.train_text_encoder:
        text_encoder.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam and args.use_deepspeed_adam==False:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    elif args.use_8bit_adam and args.use_deepspeed_adam==True:
        try:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
        except ImportError:
            raise ImportError(
                "To use 8-bit DeepSpeed Adam, try updating your cuda and deepspeed integrations."
            )
        
        optimizer_class = DeepSpeedCPUAdam
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    #noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", prediction_type="v_prediction")
    '''
    train_dataset = DreamBoothDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        use_image_names_as_captions=args.use_image_names_as_captions,
    )
    '''
    if args.use_bucketing:
        train_dataset = AutoBucketing(
            concepts_list=args.concepts_list,
            use_image_names_as_captions=args.use_image_names_as_captions,
            batch_size=args.train_batch_size,
            tokenizer=tokenizer,
            add_class_images_to_dataset=args.add_class_images_to_dataset,
            balance_datasets=args.auto_balance_concept_datasets,
            resolution=args.resolution,
            with_prior_loss=False,#args.with_prior_preservation,
            repeats=args.dataset_repeats,
            use_text_files_as_captions=args.use_text_files_as_captions,
            conditional_dropout=args.conditional_dropout,
            aspect_mode=args.aspect_mode,
            action_preference=args.aspect_mode_action_preference,
        )
    else:
        train_dataset = DreamBoothDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        use_image_names_as_captions=args.use_image_names_as_captions,
        repeats=args.dataset_repeats,
        use_text_files_as_captions=args.use_text_files_as_captions,
        conditional_dropout=args.conditional_dropout
    )
    def collate_fn(examples):
        #print(examples)
        #print('test')
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]
        #print('test')
        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
        ### no need to do it now when it's loaded by the multiAspectsDataset
        #if args.with_prior_preservation:
        #    input_ids += [example["class_prompt_ids"] for example in examples]
        #    pixel_values += [example["class_images"] for example in examples]
        
        #print(pixel_values)
        #unpack the pixel_values from tensor to list


        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        
        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",\
            ).input_ids
        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True
    )
    #get the length of the dataset
    train_dataset_length = len(train_dataset)

    #code to check if latent cache needs to be resaved
    #check if last_run.json file exists in logging_dir
    if os.path.exists(logging_dir / "last_run.json"):
        #if it exists, load it
        with open(logging_dir / "last_run.json", "r") as f:
            last_run = json.load(f)
            last_run_batch_size = last_run["batch_size"]
            last_run_dataset_length = last_run["dataset_length"]
            if last_run_batch_size != args.train_batch_size:
                print(f" {bcolors.WARNING}The batch_size has changed since the last run. Regenerating Latent Cache.{bcolors.ENDC}") 

                args.regenerate_latent_cache = True
                #save the new batch_size and dataset_length to last_run.json
            if last_run_dataset_length != train_dataset_length:
                print(f" {bcolors.WARNING}The dataset length has changed since the last run. Regenerating Latent Cache.{bcolors.ENDC}") 

                args.regenerate_latent_cache = True
                #save the new batch_size and dataset_length to last_run.json
        with open(logging_dir / "last_run.json", "w") as f:
            json.dump({"batch_size": args.train_batch_size, "dataset_length": train_dataset_length}, f)
                
    else:
        #if it doesn't exist, create it
        last_run = {"batch_size": args.train_batch_size, "dataset_length": train_dataset_length}
        #create the file
        with open(logging_dir / "last_run.json", "w") as f:
            json.dump(last_run, f)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema == True:
        ema_unet.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if not args.not_cache_latents:
        latent_cache_dir = Path(args.output_dir, "logs", "latent_cache")
        #check if latents_cache.pt exists in the output_dir
        if not os.path.exists(latent_cache_dir):
            os.makedirs(latent_cache_dir)
        if not os.path.exists(os.path.join(latent_cache_dir, "latents_cache.pt")) or args.regenerate_latent_cache:
            if os.path.exists(os.path.join(latent_cache_dir, "latents_cache.pt")):
                #if it exists, delete it
                os.remove(os.path.join(latent_cache_dir, "latents_cache.pt"))
            print(f" {bcolors.WARNING}Generating latents cache...{bcolors.ENDC}") 


            train_dataset = LatentsDataset([], [])
            counter = 0
            for batch in tqdm(train_dataloader, desc="Caching latents"):
                with torch.no_grad():
                    batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                    batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
                    
                    cached_latent = vae.encode(batch["pixel_values"]).latent_dist
                    if args.train_text_encoder:
                        cached_text_enc = batch["input_ids"]
                    else:
                        cached_text_enc = text_encoder(batch["input_ids"])[0]
                    train_dataset.add_latent(cached_latent, cached_text_enc)
                    del batch
                    del cached_latent
                    del cached_text_enc
                    counter += 1
                    if counter % 500 == 0:
                        gc.collect()
                        torch.cuda.empty_cache()
                        accelerator.free_memory()
            if args.save_latents_cache:
                if not latent_cache_dir.exists():
                    latent_cache_dir.mkdir(parents=True)
                torch.save(train_dataset, os.path.join(latent_cache_dir,"latents_cache.pt"))
        else:
            print(f" {bcolors.WARNING}Loading latents cache from file...{bcolors.ENDC}") 

            train_dataset = torch.load(os.path.join(latent_cache_dir,"latents_cache.pt"))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=False)

        del vae
        if not args.train_text_encoder:
            del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder and not args.use_ema:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    elif args.train_text_encoder and args.use_ema:
        unet, text_encoder, ema_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, ema_unet, optimizer, train_dataloader, lr_scheduler
        )
    elif not args.train_text_encoder and args.use_ema:
        unet, ema_unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, ema_unet, optimizer, train_dataloader, lr_scheduler
        )
    elif not args.train_text_encoder and not args.use_ema:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch * args.gradient_accumulation_steps
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    def mid_train_playground(step):
        
        print(f"{bcolors.WARNING} Booting up GUI{bcolors.ENDC}")
        epoch = step // num_update_steps_per_epoch
        if args.train_text_encoder and args.stop_text_encoder_training == True:
            text_enc_model = accelerator.unwrap_model(text_encoder,True)
        elif args.train_text_encoder and args.stop_text_encoder_training > epoch:
            text_enc_model = accelerator.unwrap_model(text_encoder,True)
        elif args.train_text_encoder == False:
            text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder" )
        elif args.train_text_encoder and args.stop_text_encoder_training <= epoch:
            if 'frozen_directory' in locals():
                text_enc_model = CLIPTextModel.from_pretrained(frozen_directory, subfolder="text_encoder")
            else:
                text_enc_model = accelerator.unwrap_model(text_encoder,True)
        scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        unwrapped_unet = accelerator.unwrap_model(unet,True)
        if args.use_ema:
            ema_unet.copy_to(unwrapped_unet.parameters())
            
        pipeline = DiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=unwrapped_unet,
            text_encoder=text_enc_model,
            vae=AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,subfolder=None if args.pretrained_vae_name_or_path else "vae" ),
            safety_checker=None,
            torch_dtype=torch.float16
        )
        pipeline.scheduler = scheduler
        if is_xformers_available():
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(
                        "Could not enable memory efficient attention. Make sure xformers is installed"
                        f" correctly and a GPU is available: {e}"
                    )
        pipeline = pipeline.to(accelerator.device)
        def inference(prompt, negative_prompt, num_samples, height=512, width=512, num_inference_steps=50,seed=-1,guidance_scale=7.5):
            with torch.autocast("cuda"), torch.inference_mode():
                if seed != -1:
                    g_cuda = torch.Generator(device='cuda')
                    g_cuda.manual_seed(int(seed))
                else:
                    seed = random.randint(0, 100000)
                    g_cuda = torch.Generator(device='cuda')
                    g_cuda.manual_seed(seed)
                    return pipeline(
                            prompt, height=int(height), width=int(width),
                            negative_prompt=negative_prompt,
                            num_images_per_prompt=int(num_samples),
                            num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                            generator=g_cuda).images, seed
        
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt", value="photo of zwx dog in a bucket")
                    negative_prompt = gr.Textbox(label="Negative Prompt", value="")
                    run = gr.Button(value="Generate")
                    with gr.Row():
                        num_samples = gr.Number(label="Number of Samples", value=4)
                        guidance_scale = gr.Number(label="Guidance Scale", value=7.5)
                    with gr.Row():
                        height = gr.Number(label="Height", value=512)
                        width = gr.Number(label="Width", value=512)
                    with gr.Row():
                        num_inference_steps = gr.Slider(label="Steps", value=25)
                        seed = gr.Number(label="Seed", value=-1)
                with gr.Column():
                    gallery = gr.Gallery()
                    seedDisplay = gr.Number(label="Used Seed:", value=0)

            run.click(inference, inputs=[prompt, negative_prompt, num_samples, height, width, num_inference_steps,seed, guidance_scale], outputs=[gallery,seedDisplay])
        
        demo.launch(share=True,prevent_thread_lock=True)
        print(f"{bcolors.WARNING}Gradio Session is active, Press 'F12' to resume training{bcolors.ENDC}")
        keyboard.wait('f12')
        demo.close()
        del demo
        del text_enc_model
        del unwrapped_unet
        del pipeline
        return
    def save_and_sample_weights(step,context='checkpoint',save_model=True):
        #check how many folders are in the output dir
        #if there are more than 5, delete the oldest one
        #save the model
        #save the optimizer
        #save the lr_scheduler
        #save the args
        height = args.sample_height
        width = args.sample_width
        if args.sample_aspect_ratios:
            #choose random aspect ratio from ASPECTS    
            aspect_ratio = random.choice(ASPECTS)
            height = aspect_ratio[0]
            width = aspect_ratio[1]
        if os.path.exists(args.output_dir):
            if args.delete_checkpoints_when_full_drive==True:
                folders = os.listdir(args.output_dir)
                #check how much space is left on the drive
                total, used, free = shutil.disk_usage("/")
                if (free // (2**30)) < 4:
                    #folders.remove("0")
                    #get the folder with the lowest number
                    oldest_folder = min(folder for folder in folders if folder.isdigit())
                    if args.send_telegram_updates:
                        try:
                            send_telegram_message(f"Deleting folder <b>{oldest_folder}</b> because the drive is full", args.telegram_chat_id, args.telegram_token)
                        except:
                            pass
                    oldest_folder_path = os.path.join(args.output_dir, oldest_folder)
                    shutil.rmtree(oldest_folder_path)
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            if context =='step':
                #what is the current epoch
                epoch = step // num_update_steps_per_epoch
            else:
                epoch = step
            if args.train_text_encoder and args.stop_text_encoder_training == True:
                text_enc_model = accelerator.unwrap_model(text_encoder,True)
            elif args.train_text_encoder and args.stop_text_encoder_training > epoch:
                text_enc_model = accelerator.unwrap_model(text_encoder,True)
            elif args.train_text_encoder == False:
                text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder" )
            elif args.train_text_encoder and args.stop_text_encoder_training <= epoch:
                if 'frozen_directory' in locals():
                    text_enc_model = CLIPTextModel.from_pretrained(frozen_directory, subfolder="text_encoder")
                else:
                    text_enc_model = accelerator.unwrap_model(text_encoder,True)
                
            #scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
            #scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", prediction_type="v_prediction")
            scheduler = DPMSolverMultistepScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
            unwrapped_unet = accelerator.unwrap_model(unet,True)
            if args.use_ema:
                ema_unet.copy_to(unwrapped_unet.parameters())
                
            pipeline = DiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=unwrapped_unet,
                text_encoder=text_enc_model,
                vae=AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,subfolder=None if args.pretrained_vae_name_or_path else "vae" ),
                safety_checker=None,
                torch_dtype=torch.float16
            )
            pipeline.scheduler = scheduler
            if is_xformers_available():
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    logger.warning(
                        "Could not enable memory efficient attention. Make sure xformers is installed"
                        f" correctly and a GPU is available: {e}"
                    )
            save_dir = os.path.join(args.output_dir, f"{context}_{step}")
            sample_dir = os.path.join(args.output_dir, f"samples/{context}_{step}")
            #if sample dir path does not exist, create it
            
            if args.stop_text_encoder_training == True:
                save_dir = frozen_directory
            if step != 0:
                if save_model:
                    pipeline.save_pretrained(save_dir)
                    with open(os.path.join(save_dir, "args.json"), "w") as f:
                            json.dump(args.__dict__, f, indent=2)
                if args.stop_text_encoder_training == True:
                    #delete every folder in frozen_directory but the text encoder
                    for folder in os.listdir(save_dir):
                        if folder != "text_encoder" and os.path.isdir(os.path.join(save_dir, folder)):
                            shutil.rmtree(os.path.join(save_dir, folder))
            imgs = []
            if args.add_sample_prompt is not None and args.stop_text_encoder_training != True:
                
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)
                #sample_dir = os.path.join(save_dir, "samples")
                #if sample_dir exists, delete it
                if os.path.exists(sample_dir):
                    shutil.rmtree(sample_dir)
                os.makedirs(sample_dir, exist_ok=True)
                with torch.autocast("cuda"), torch.inference_mode():
                    if args.send_telegram_updates:
                        try:
                            send_telegram_message(f"Generating samples for <b>{step}</b> {context}", args.telegram_chat_id, args.telegram_token)
                        except:
                            pass
                    for samplePrompt in args.add_sample_prompt:
                        sampleIndex = args.add_sample_prompt.index(samplePrompt)
                        #convert sampleIndex to number in words
                        sampleName = f"prompt_{sampleIndex+1}"
                        os.makedirs(os.path.join(sample_dir,sampleName), exist_ok=True)
                        for i in tqdm(range(args.n_save_sample) if not args.save_sample_controlled_seed else range(args.n_save_sample+len(args.save_sample_controlled_seed)), desc="Generating samples"):
                            #check if the sample is controlled by a seed
                            if i != args.n_save_sample:
                                images = pipeline(samplePrompt,height=height,width=width, guidance_scale=args.save_guidance_scale, num_inference_steps=args.save_infer_steps).images
                                images[0].save(os.path.join(sample_dir,sampleName, f"{sampleName}_{i}.png"))
                            else:
                                for seed in args.save_sample_controlled_seed:
                                    generator = torch.Generator("cuda").manual_seed(seed)
                                    images = pipeline(samplePrompt,height=height,width=width, guidance_scale=args.save_guidance_scale, num_inference_steps=args.save_infer_steps, generator=generator).images
                                    images[0].save(os.path.join(sample_dir,sampleName, f"{sampleName}_controlled_seed_{str(seed)}.png"))
                        if args.send_telegram_updates:
                            imgs = []
                            #get all the images from the sample folder
                            dir = os.listdir(os.path.join(sample_dir,sampleName))
                            for file in dir:
                                if file.endswith(".png"):
                                    #open the image with pil
                                    img = Image.open(os.path.join(sample_dir,sampleName,file))
                                    imgs.append(img)
                            try:
                                send_media_group(args.telegram_chat_id,args.telegram_token,imgs, caption=f"Samples for the <b>{step}</b> {context} using the prompt:\n\n<b>{samplePrompt}</b>")
                            except:
                                pass
                del pipeline
                del unwrapped_unet
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if save_model == True:
                print(f" {bcolors.OKGREEN}[*] Weights saved to {save_dir}{bcolors.ENDC}")
            elif save_model == False and len(imgs) > 0:
                del imgs
                print(f" {bcolors.OKGREEN}[*] Samples saved to {sample_dir}{bcolors.ENDC}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar_e = tqdm(range(args.num_train_epochs), disable=not accelerator.is_local_main_process)

    progress_bar.set_description("Steps")
    progress_bar_e.set_description("Epochs")
    global_step = 0
    loss_avg = AverageMeter()
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()
    if args.send_telegram_updates:
        try:
            send_telegram_message(f"Starting training with the following settings:\n\n{format_dict(args.__dict__)}", args.telegram_chat_id, args.telegram_token)
        except:
            pass
    try:
        print(f" {bcolors.OKBLUE}Starting Training!{bcolors.ENDC}")
        try:
            def toggle_gui(event = None):
                if keyboard.is_pressed('ctrl'):
                    print(f" {bcolors.WARNING}GUI will boot as soon as the current step is done.{bcolors.ENDC}")
                    nonlocal mid_generation
                    mid_generation = True
            def toggle_checkpoint(event = None):
                if keyboard.is_pressed('ctrl'):
                    print(f" {bcolors.WARNING}Saving the model as soon as this epoch is done.{bcolors.ENDC}")
                    #nonlocal mid_checkpoint
                    nonlocal mid_checkpoint
                    mid_checkpoint = True
            def toggle_sample(event = None):
                if keyboard.is_pressed('ctrl'):
                    print(f" {bcolors.WARNING}Sampling will begin as soon as this epoch is done.{bcolors.ENDC}")
                    #nonlocal mid_checkpoint
                    nonlocal mid_sample
                    mid_sample = True
            keyboard.on_press_key('f12',toggle_gui)
            keyboard.on_press_key('f11',toggle_checkpoint)
            keyboard.on_press_key('f10',toggle_sample)
            print(f"{bcolors.WARNING}Use 'CTRL+F12' to open up a GUI to play around with the model (will pause training){bcolors.ENDC}")
            print(f"{bcolors.WARNING}Use 'CTRL+F11' to save a checkpoint of the current epoch{bcolors.ENDC}")
            print(f"{bcolors.WARNING}Use 'CTRL+F10' to generate samples for current epoch{bcolors.ENDC}")
        except:
            pass
        mid_generation = False
        mid_checkpoint = False
        mid_sample = False
        #lambda set mid_generation to true
        
        
        for epoch in range(args.num_train_epochs):
            unet.train()
            if args.train_text_encoder:
                text_encoder.train()
            
            #save initial weights
            if args.sample_on_training_start==True and epoch==0:
                save_and_sample_weights(epoch,'start',save_model=False)
            
            if args.train_text_encoder and args.stop_text_encoder_training == epoch:
                args.stop_text_encoder_training = True
                if accelerator.is_main_process:
                    print(f" {bcolors.WARNING} Stopping text encoder training{bcolors.ENDC}")   
                    current_percentage = (epoch/args.num_train_epochs)*100
                    #round to the nearest whole number
                    current_percentage = round(current_percentage,0)
                    try:
                        send_telegram_message(f"Text encoder training stopped at epoch {epoch} which is {current_percentage}% of training. Freezing weights and saving.", args.telegram_chat_id, args.telegram_token)   
                    except:
                        pass        
                    frozen_directory=args.output_dir + "/frozen_text_encoder"
                    if os.path.exists(frozen_directory):
                        #delete the folder if it already exists
                        shutil.rmtree(frozen_directory)
                    os.mkdir(frozen_directory)
                    save_and_sample_weights(epoch,'epoch')
                    args.stop_text_encoder_training = epoch

            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    with torch.no_grad():
                        if not args.not_cache_latents:
                            latent_dist = batch[0][0]
                        else:
                            latent_dist = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist
                        latents = latent_dist.sample() * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    with text_enc_context:
                        if not args.not_cache_latents:
                            if args.train_text_encoder:
                                if args.clip_penultimate == True:
                                    encoder_hidden_states = text_encoder(batch[0][1],output_hidden_states=True)
                                    encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states['hidden_states'][-2])
                                else:
                                    encoder_hidden_states = text_encoder(batch[0][1])[0]
                            else:
                                encoder_hidden_states = batch[0][1]
                        else:
                            if args.train_text_encoder:
                                if args.clip_penultimate == True:
                                    encoder_hidden_states = text_encoder(batch["input_ids"],output_hidden_states=True)
                                    encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states['hidden_states'][-2])
                                else:
                                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                            else:
                                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    #noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    del timesteps, noise, latents, noisy_latents, encoder_hidden_states
                    if args.with_prior_preservation:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        """
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        noise, noise_prior = torch.chunk(noise, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

                        # Compute prior loss
                        prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                        """
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                        
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    loss_avg.update(loss.detach_(), bsz)
                    if args.use_ema == True:
                        ema_unet.step(unet.parameters())

                if not global_step % args.log_interval:
                    logs = {"loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)
                
                

                if global_step > 0 and not global_step % args.sample_step_interval:
                    save_and_sample_weights(global_step,'step',save_model=False)

                progress_bar.update(1)
                progress_bar_e.refresh()
                global_step += 1

                if mid_generation==True:
                    mid_train_playground(global_step)
                    mid_generation=False
                if global_step >= args.max_train_steps:
                    break
            progress_bar_e.update(1)
            if not epoch % args.save_every_n_epoch:
                    #print(epoch % args.save_every_n_epoch)
                    #print('test')
                    if epoch != 0:
                        save_and_sample_weights(epoch,'epoch')
                    else:
                        save_and_sample_weights(epoch,'epoch',False)
            if epoch % args.save_every_n_epoch and mid_checkpoint==True or mid_sample==True:
                if mid_checkpoint==True:
                    save_and_sample_weights(epoch,'epoch',True)
                    mid_checkpoint=False
                elif mid_sample==True:
                    save_and_sample_weights(epoch,'epoch',False)
                    mid_sample=False
            accelerator.wait_for_everyone()
    except Exception:
        try:
            send_telegram_message("Something went wrong while training! :(", args.telegram_chat_id, args.telegram_token)
            #save_and_sample_weights(global_step,'checkpoint')
            send_telegram_message(f"Saved checkpoint {global_step} on exit", args.telegram_chat_id, args.telegram_token)
        except Exception:
            pass
        raise
    except KeyboardInterrupt:
        send_telegram_message("Training stopped", args.telegram_chat_id, args.telegram_token)
    save_and_sample_weights(args.num_train_epochs,'epoch')
    try:
        send_telegram_message("Training finished!", args.telegram_chat_id, args.telegram_token)
    except:
        pass

    accelerator.end_training()
    


if __name__ == "__main__":
    main()
