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
from torchvision.transforms import functional
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, List, Generator, Tuple
from PIL import Image, ImageFile
from diffusers.utils.import_utils import is_xformers_available
import trainer_util as tu

from clip_segmentation import ClipSeg
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
        "--shuffle_per_epoch",
        default=False,
        action="store_true",
        help="Will shffule the dataset per epoch",
    )
    parser.add_argument(
        "--attention",
        type=str,
        choices=["xformers", "flash_attention"],
        default="xformers",
        help="Type of attention to use."
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default='base',
        required=False,
        help="Train Base/Inpaint/Depth2Img",
    )
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
            "--sample_from_batch",
            type=int,
            default=0,
            help=("Number of prompts to sample from the batch for inference"),
        )
    parser.add_argument(
        "--flatten_sample_folder",
        default=False,
        action="store_true",
        help="Will save samples in one folder instead of per-epoch",
    )
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
        "--lr_warmup_steps", type=float, default=500, help="Number of steps for the warmup in the lr scheduler."
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
        choices=["no", "fp16", "bf16","tf32"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument("--save_sample_controlled_seed", type=int, action='append', help="Set a seed for an extra sample image to be constantly saved.")
    parser.add_argument("--detect_full_drive", default=True, action="store_true", help="Delete checkpoints when the drive is full.")
    parser.add_argument("--send_telegram_updates", default=False, action="store_true", help="Send Telegram updates.")
    parser.add_argument("--telegram_chat_id", type=str, default="0", help="Telegram chat ID.")
    parser.add_argument("--telegram_token", type=str, default="0", help="Telegram token.")
    parser.add_argument("--use_deepspeed_adam", default=False, action="store_true", help="Use experimental DeepSpeed Adam 8.")
    parser.add_argument('--append_sample_controlled_seed_action', action='append')
    parser.add_argument('--add_sample_prompt', type=str, action='append')
    parser.add_argument('--use_image_names_as_captions', default=False, action="store_true")
    parser.add_argument("--masked_training", default=False, required=False, action='store_true', help="Whether to mask parts of the image during training")
    parser.add_argument("--normalize_masked_area_loss", default=False, required=False, action='store_true', help="Normalize the loss, to make it independent of the size of the masked area")
    parser.add_argument("--unmasked_probability", type=float, default=1, required=False, help="Probability of training a step without a mask")
    parser.add_argument("--max_denoising_strength", type=float, default=1, required=False, help="Max denoising steps to train on")
    parser.add_argument('--add_mask_prompt', type=str, default=None, action="append", dest="mask_prompts", help="Prompt for automatic mask creation")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
ASPECT_2048 = [[2048, 2048],
[2112, 1984],[1984, 2112],
[2176, 1920],[1920, 2176],
[2240, 1856],[1856, 2240],
[2304, 1792],[1792, 2304],
[2368, 1728],[1728, 2368],
[2432, 1664],[1664, 2432],
[2496, 1600],[1600, 2496],
[2560, 1536],[1536, 2560],
[2624, 1472],[1472, 2624]]
ASPECT_1984 = [[1984, 1984],
[2048, 1920],[1920, 2048],
[2112, 1856],[1856, 2112],
[2176, 1792],[1792, 2176],
[2240, 1728],[1728, 2240],
[2304, 1664],[1664, 2304],
[2368, 1600],[1600, 2368],
[2432, 1536],[1536, 2432],
[2496, 1472],[1472, 2496],
[2560, 1408],[1408, 2560]]
ASPECT_1920 = [[1920, 1920],
[1984, 1856],[1856, 1984],
[2048, 1792],[1792, 2048],
[2112, 1728],[1728, 2112],
[2176, 1664],[1664, 2176],
[2240, 1600],[1600, 2240],
[2304, 1536],[1536, 2304],
[2368, 1472],[1472, 2368],
[2432, 1408],[1408, 2432],
[2496, 1344],[1344, 2496]]
ASPECT_1856 = [[1856, 1856],
[1920, 1792],[1792, 1920],
[1984, 1728],[1728, 1984],
[2048, 1664],[1664, 2048],
[2112, 1600],[1600, 2112],
[2176, 1536],[1536, 2176],
[2240, 1472],[1472, 2240],
[2304, 1408],[1408, 2304],
[2368, 1344],[1344, 2368],
[2432, 1280],[1280, 2432]]
ASPECT_1792 = [[1792, 1792],
[1856, 1728],[1728, 1856],
[1920, 1664],[1664, 1920],
[1984, 1600],[1600, 1984],
[2048, 1536],[1536, 2048],
[2112, 1472],[1472, 2112],
[2176, 1408],[1408, 2176],
[2240, 1344],[1344, 2240],
[2304, 1280],[1280, 2304],
[2368, 1216],[1216, 2368]]
ASPECT_1728 = [[1728, 1728],
[1792, 1664],[1664, 1792],
[1856, 1600],[1600, 1856],
[1920, 1536],[1536, 1920],
[1984, 1472],[1472, 1984],
[2048, 1408],[1408, 2048],
[2112, 1344],[1344, 2112],
[2176, 1280],[1280, 2176],
[2240, 1216],[1216, 2240],
[2304, 1152],[1152, 2304]]
ASPECT_1664 = [[1664, 1664],
[1728, 1600],[1600, 1728],
[1792, 1536],[1536, 1792],
[1856, 1472],[1472, 1856],
[1920, 1408],[1408, 1920],
[1984, 1344],[1344, 1984],
[2048, 1280],[1280, 2048],
[2112, 1216],[1216, 2112],
[2176, 1152],[1152, 2176],
[2240, 1088],[1088, 2240]]
ASPECT_1600 = [[1600, 1600],
[1664, 1536],[1536, 1664],
[1728, 1472],[1472, 1728],
[1792, 1408],[1408, 1792],
[1856, 1344],[1344, 1856],
[1920, 1280],[1280, 1920],
[1984, 1216],[1216, 1984],
[2048, 1152],[1152, 2048],
[2112, 1088],[1088, 2112],
[2176, 1024],[1024, 2176]]
ASPECT_1536 = [[1536, 1536],
[1600, 1472],[1472, 1600],
[1664, 1408],[1408, 1664],
[1728, 1344],[1344, 1728],
[1792, 1280],[1280, 1792],
[1856, 1216],[1216, 1856],
[1920, 1152],[1152, 1920],
[1984, 1088],[1088, 1984],
[2048, 1024],[1024, 2048],
[2112, 960],[960, 2112]]
ASPECT_1472 = [[1472, 1472],
[1536, 1408],[1408, 1536],
[1600, 1344],[1344, 1600],
[1664, 1280],[1280, 1664],
[1728, 1216],[1216, 1728],
[1792, 1152],[1152, 1792],
[1856, 1088],[1088, 1856],
[1920, 1024],[1024, 1920],
[1984, 960],[960, 1984],
[2048, 896],[896, 2048]]
ASPECT_1408 = [[1408, 1408],
[1472, 1344],[1344, 1472],
[1536, 1280],[1280, 1536],
[1600, 1216],[1216, 1600],
[1664, 1152],[1152, 1664],
[1728, 1088],[1088, 1728],
[1792, 1024],[1024, 1792],
[1856, 960],[960, 1856],
[1920, 896],[896, 1920],
[1984, 832],[832, 1984]]
ASPECT_1344 = [[1344, 1344],
[1408, 1280],[1280, 1408],
[1472, 1216],[1216, 1472],
[1536, 1152],[1152, 1536],
[1600, 1088],[1088, 1600],
[1664, 1024],[1024, 1664],
[1728, 960],[960, 1728],
[1792, 896],[896, 1792],
[1856, 832],[832, 1856],
[1920, 768],[768, 1920]]
ASPECT_1280 = [[1280, 1280],
[1344, 1216],[1216, 1344],
[1408, 1152],[1152, 1408],
[1472, 1088],[1088, 1472],
[1536, 1024],[1024, 1536],
[1600, 960],[960, 1600],
[1664, 896],[896, 1664],
[1728, 832],[832, 1728],
[1792, 768],[768, 1792],
[1856, 704],[704, 1856]]
ASPECT_1216 = [[1216, 1216],
[1280, 1152],[1152, 1280],
[1344, 1088],[1088, 1344],
[1408, 1024],[1024, 1408],
[1472, 960],[960, 1472],
[1536, 896],[896, 1536],
[1600, 832],[832, 1600],
[1664, 768],[768, 1664],
[1728, 704],[704, 1728],
[1792, 640],[640, 1792]]
ASPECT_1152 = [[1152, 1152],
[1216, 1088],[1088, 1216],
[1280, 1024],[1024, 1280],
[1344, 960],[960, 1344],
[1408, 896],[896, 1408],
[1472, 832],[832, 1472],
[1536, 768],[768, 1536],
[1600, 704],[704, 1600],
[1664, 640],[640, 1664],
[1728, 576],[576, 1728]]
ASPECT_1088 = [[1088, 1088],
[1152, 1024],[1024, 1152],
[1216, 960],[960, 1216],
[1280, 896],[896, 1280],
[1344, 832],[832, 1344],
[1408, 768],[768, 1408],
[1472, 704],[704, 1472],
[1536, 640],[640, 1536],
[1600, 576],[576, 1600],
[1664, 512],[512, 1664]]
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
ASPECT_960 = [[960, 960],
[1024, 896],[896, 1024],
[1088, 832],[832, 1088],
[1152, 768],[768, 1152],
[1216, 704],[704, 1216],
[1280, 640],[640, 1280],
[1344, 576],[576, 1344],
[1408, 512],[512, 1408],
[1472, 448],[448, 1472],
[1536, 384],[384, 1536]]   
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
    [896,640],[640,896],   # 573440 1.4:1
    [832,704],[704,832],   # 585728 1.181:1
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
def get_aspect_buckets(resolution,mode=''):
    if resolution < 512:
        raise ValueError("Resolution must be at least 512")
    try: 
        rounded_resolution = int(resolution / 64) * 64
        print(f" {bcolors.WARNING} Rounded resolution to: {rounded_resolution}{bcolors.ENDC}")   
        all_image_sizes = __get_all_aspects()
        if mode == 'MJ':
            #truncate to the first 3 resolutions
            all_image_sizes = [x[0:3] for x in all_image_sizes]
        aspects = next(filter(lambda sizes: sizes[0][0]==rounded_resolution, all_image_sizes), None)
        ASPECTS = aspects
        #print(aspects)
        return aspects
    except Exception as e:
        print(f" {bcolors.FAIL} *** Could not find selected resolution: {rounded_resolution}{bcolors.ENDC}")   

        raise e

def __get_all_aspects():
    return [ASPECTS_512, ASPECT_576, ASPECT_640, ASPECT_704, ASPECT_768,ASPECT_832,ASPECT_896,ASPECT_960,ASPECT_1024,ASPECT_1088,ASPECT_1152,ASPECT_1216,ASPECT_1280,ASPECT_1344,ASPECT_1408,ASPECT_1472,ASPECT_1536,ASPECT_1600,ASPECT_1664,ASPECT_1728,ASPECT_1792,ASPECT_1856,ASPECT_1920,ASPECT_1984,ASPECT_2048]
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
                    aspect_mode='dynamic',
                    action_preference='dynamic',
                    seed=555,
                    model_variant='base',
                    extra_module=None,
                    mask_prompts=None,
                    load_mask=False,
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
        self.aspect_mode = aspect_mode
        self.action_preference = action_preference
        self.model_variant = model_variant
        self.extra_module = extra_module
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.depth_image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.seed = seed
        #shared_dataloader = None
        print(f" {bcolors.WARNING}Creating Auto Bucketing Dataloader{bcolors.ENDC}")

        shared_dataloader = DataLoaderMultiAspect(concepts_list,
         debug_level=debug_level,
         resolution=self.resolution,
         seed=self.seed,
         batch_size=self.batch_size, 
         flip_p=flip_p,
         use_image_names_as_captions=self.use_image_names_as_captions,
         add_class_images_to_dataset=self.add_class_images_to_dataset,
         balance_datasets=self.balance_datasets,
         with_prior_loss=self.with_prior_loss,
         use_text_files_as_captions=self.use_text_files_as_captions,
         aspect_mode=self.aspect_mode,
         action_preference=self.action_preference,
         model_variant=self.model_variant,
         extra_module=self.extra_module,
         mask_prompts=mask_prompts,
         load_mask=load_mask,
        )

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
            if image_train_tmp.mask is not None:
                image_train_tmp_mask = Image.fromarray(self.normalize8(image_train_tmp.mask)).convert("L")
                example["mask"] = self.mask_transforms(image_train_tmp_mask)
            if self.model_variant == 'depth2img':
                image_train_tmp_depth = Image.fromarray(self.normalize8(image_train_tmp.extra)).convert("L")
                example["instance_depth_images"] = self.depth_image_transforms(image_train_tmp_depth)
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
            if self.model_variant == 'depth2img':
                image_train_tmp_depth = Image.fromarray(self.normalize8(image_train_tmp.extra)).convert("L")
                example["class_depth_images"] = self.depth_image_transforms(image_train_tmp_depth)
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
    mask: Image
    extra: Image
    identifier: caption,
    target_aspect: (width, height), 
    pathname: path to image file
    flip_p: probability of flipping image (0.0 to 1.0)
    """    
    def __init__(self, image: Image, mask: Image, extra: Image, caption: str, target_wh: list, pathname: str, flip_p=0.0, model_variant='base', load_mask=False):
        self.caption = caption
        self.target_wh = target_wh
        self.pathname = pathname
        self.mask_pathname = os.path.splitext(pathname)[0] + "-masklabel.png"
        self.depth_pathname = os.path.splitext(pathname)[0] + "-depth.png"
        self.flip_p = flip_p
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.cropped_img = None
        self.model_variant = model_variant
        self.load_mask=load_mask
        self.is_dupe = []
        self.variant_warning = False

        self.image = image
        self.mask = mask
        self.extra = extra

    def self_destruct(self):
        self.image = None
        self.mask = None
        self.extra = None
        self.cropped_img = None
        self.is_dupe.append(1)

    def load_image(self, pathname, crop, jitter_amount, flip):
        if len(self.is_dupe) > 0:
            self.flip = transforms.RandomHorizontalFlip(p=1.0 if flip else 0.0)
        image = Image.open(pathname).convert('RGB')

        width, height = image.size
        if crop:
            cropped_img = self.__autocrop(image)
            image = cropped_img.resize((512, 512), resample=Image.Resampling.LANCZOS)
        else:
            width, height = image.size

            if self.target_wh[0] == self.target_wh[1]:
                if width > height:
                    left = random.randint(0, width - height)
                    image = image.crop((left, 0, height + left, height))
                    width = height
                elif height > width:
                    top = random.randint(0, height - width)
                    image = image.crop((0, top, width, width + top))
                    height = width
                elif width > self.target_wh[0]:
                    slice = min(int(self.target_wh[0] * _RANDOM_TRIM), width - self.target_wh[0])
                    slicew_ratio = random.random()
                    left = int(slice * slicew_ratio)
                    right = width - int(slice * (1 - slicew_ratio))
                    sliceh_ratio = random.random()
                    top = int(slice * sliceh_ratio)
                    bottom = height - int(slice * (1 - sliceh_ratio))

                    image = image.crop((left, top, right, bottom))
            else:
                image_aspect = width / height
                target_aspect = self.target_wh[0] / self.target_wh[1]
                if image_aspect > target_aspect:
                    new_width = int(height * target_aspect)
                    jitter_amount = max(min(jitter_amount, int(abs(width - new_width) / 2)), 0)
                    left = jitter_amount
                    right = left + new_width
                    image = image.crop((left, 0, right, height))
                else:
                    new_height = int(width / target_aspect)
                    jitter_amount = max(min(jitter_amount, int(abs(height - new_height) / 2)), 0)
                    top = jitter_amount
                    bottom = top + new_height
                    image = image.crop((0, top, width, bottom))
                    # LAZCOS resample
            image = image.resize(self.target_wh, resample=Image.Resampling.LANCZOS)
            # print the pixel count of the image
            # print path to image file
            # print(self.pathname)
            # print(self.image.size[0] * self.image.size[1])
            image = self.flip(image)
        return image

    def hydrate(self, crop=False, save=False, crop_jitter=20):
        """
        crop: hard center crop to 512x512
        save: save the cropped image to disk, for manual inspection of resize/crop
        crop_jitter: randomly shift cropp by N pixels when using multiple aspect ratios to improve training quality
        """

        if self.image is None:
            chance = float(len(self.is_dupe)) / 10.0
            
            flip_p = self.flip_p + chance if chance < 1.0 else 1.0
            flip = random.uniform(0, 1) < flip_p

            if len(self.is_dupe) > 0:
                crop_jitter = crop_jitter + (len(self.is_dupe) * 10) if crop_jitter < 50 else 50
                
            jitter_amount = random.randint(0, crop_jitter)

            self.image = self.load_image(self.pathname, crop, jitter_amount, flip)

            if self.model_variant == "inpainting" or self.load_mask:
                if os.path.exists(self.mask_pathname) and self.load_mask:
                    self.mask = self.load_image(self.mask_pathname, crop, jitter_amount, flip)
                else:
                    if self.variant_warning == False:
                        print(f" {bcolors.FAIL} ** Warning: No mask found for an image, using an empty mask but make sure you're training the right model variant.{bcolors.ENDC}")
                        self.variant_warning = True
                    self.mask = Image.new('RGB', self.image.size, color="white").convert("L")

            if self.model_variant == "depth2img":
                if os.path.exists(self.depth_pathname):
                    self.extra = self.load_image(self.depth_pathname, crop, jitter_amount, flip)
                else:
                    if self.variant_warning == False:
                        print(f" {bcolors.FAIL} ** Warning: No depth found for an image, using an empty depth but make sure you're training the right model variant.{bcolors.ENDC}")
                        self.variant_warning = True
                    self.extra = Image.new('RGB', self.image.size, color="white").convert("L")
        if type(self.image) is not np.ndarray:
            if save: 
                base_name = os.path.basename(self.pathname)
                if not os.path.exists("test/output"):
                    os.makedirs("test/output")
                self.image.save(f"test/output/{base_name}")
            
            self.image = np.array(self.image).astype(np.uint8)

            self.image = (self.image / 127.5 - 1.0).astype(np.float32)
        if self.mask is not None and type(self.mask) is not np.ndarray:
            self.mask = np.array(self.mask).astype(np.uint8)

            self.mask = (self.mask / 255.0).astype(np.float32)
        if self.extra is not None and type(self.extra) is not np.ndarray:
            self.extra = np.array(self.extra).astype(np.uint8)

            self.extra = (self.extra / 255.0).astype(np.float32)

        #print(self.image.shape)

        return self

class DataLoaderMultiAspect():
    """
    Data loader for multi-aspect-ratio training and bucketing
    data_root: root folder of training data
    batch_size: number of images per batch
    flip_p: probability of flipping image horizontally (i.e. 0-0.5)
    """
    def __init__(
            self,
            concept_list,
            seed=555,
            debug_level=0,
            resolution=512,
            batch_size=1,
            flip_p=0.0,
            use_image_names_as_captions=True,
            add_class_images_to_dataset=False,
            balance_datasets=False,
            with_prior_loss=False,
            use_text_files_as_captions=False,
            aspect_mode='dynamic',
            action_preference='add',
            model_variant='base',
            extra_module=None,
            mask_prompts=None,
            load_mask=False,
    ):
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
        self.seed = seed
        self.model_variant = model_variant
        self.extra_module = extra_module
        self.load_mask = load_mask
        prepared_train_data = []
        
        self.aspects = get_aspect_buckets(resolution)
        #print(f"* DLMA resolution {resolution}, buckets: {self.aspects}")
        #process sub directories flag
            
        print(f" {bcolors.WARNING} Preloading images...{bcolors.ENDC}")   

        if balance_datasets:
            print(f" {bcolors.WARNING} Balancing datasets...{bcolors.ENDC}") 
            #get the concept with the least number of images in instance_data_dir
            for concept in concept_list:
                count = 0
                if 'use_sub_dirs' in concept:
                    if concept['use_sub_dirs'] == 1:
                        tot = 0
                        for root, dirs, files in os.walk(concept['instance_data_dir']):
                            for file in files:
                                if file.endswith( ('.jpg','.jpeg','.png','.webp','.bmp','.JPG','.JPEG','.PNG','.WEBP','.BMP')):
                                    tot += 1
                        count = tot
                    else:
                        tot = 0
                        files = os.listdir(concept['instance_data_dir'])
                        for file in files:
                            if file.endswith( ('.jpg','.jpeg','.png','.webp','.bmp','.JPG','.JPEG','.PNG','.WEBP','.BMP')):
                                tot += 1
                        count = tot
                else:
                    count = len(os.listdir(concept['instance_data_dir']))
                print(f"{concept['instance_data_dir']} has count of {count}")
                concept['count'] = count
                
            min_concept = min(concept_list, key=lambda x: x['count'])
            #get the number of images in the concept with the least number of images
            min_concept_num_images = min_concept['count']
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
            random.Random(self.seed).shuffle(self.image_paths)
            if self.model_variant == 'depth2img':
                print(f" {bcolors.WARNING} ** Loading Depth2Img Pipeline To Process Dataset{bcolors.ENDC}")
                self.vae_scale_factor = self.extra_module.depth_images(self.image_paths)
            prepared_train_data.extend(self.__prescan_images(debug_level, self.image_paths, flip_p,use_image_names_as_captions,concept_prompt,use_text_files_as_captions=self.use_text_files_as_captions)[0:min_concept_num_images]) # ImageTrainItem[]
            if add_class_images_to_dataset:
                self.image_paths = []
                self.__recurse_data_root(self=self, recurse_root=data_root_class,use_sub_dirs=use_sub_dirs)
                random.Random(self.seed).shuffle(self.image_paths)
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
                if self.model_variant == 'depth2img':
                    print(f" {bcolors.WARNING} ** Depth2Img To Process Class Dataset{bcolors.ENDC}")
                    self.vae_scale_factor = self.extra_module.depth_images(self.image_paths)
                use_image_names_as_captions = False
                self.class_image_caption_pairs.extend(self.__prescan_images(debug_level, self.class_images_path, flip_p,use_image_names_as_captions,concept_class_prompt,use_text_files_as_captions=self.use_text_files_as_captions))
            self.class_image_caption_pairs = self.__bucketize_images(self.class_image_caption_pairs, batch_size=batch_size, debug_level=debug_level,aspect_mode=self.aspect_mode,action_preference=self.action_preference)
        if mask_prompts is not None:
            print(f" {bcolors.WARNING} Checking and generating missing masks...{bcolors.ENDC}")
            clip_seg = ClipSeg()
            clip_seg.mask_images(self.image_paths, mask_prompts)
            del clip_seg
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
                        with open(txt_file_path, 'r',encoding='utf-8',errors='ignore') as f:
                            identifier = f.readline().rstrip()
                            f.close()
                            if len(identifier) < 1:
                                raise ValueError(f" *** Could not find valid text in: {txt_file_path}")
                            
                    except Exception as e:
                        print(f" {bcolors.FAIL} *** Error reading {txt_file_path} to get caption, falling back to filename{bcolors.ENDC}") 
                        print(e)
                        identifier = caption_from_filename
                        pass
            #print("identifier: ",identifier)
            image = Image.open(pathname)
            width, height = image.size
            image_aspect = width / height

            target_wh = min(self.aspects, key=lambda aspects:abs(aspects[0]/aspects[1] - image_aspect))

            image_train_item = ImageTrainItem(image=None, mask=None, extra=None, caption=identifier, target_wh=target_wh, pathname=pathname, flip_p=flip_p,model_variant=self.model_variant, load_mask=self.load_mask)

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
                    print(f"  ** Bucket {bucket} found {bucket_len} images, will {bcolors.OKCYAN}duplicate {added} images{bcolors.ENDC} due to batch size {bcolors.WARNING}{batch_size}{bcolors.ENDC}")
                else:
                    print(f"  ** Bucket {bucket} found {bucket_len}, {bcolors.OKGREEN}nice!{bcolors.ENDC}")
            elif action == 'truncate':
                truncate_count = (bucket_len) % batch_size
                current_bucket_size = bucket_len
                buckets[bucket] = buckets[bucket][:current_bucket_size - truncate_count]
                print(f"  ** Bucket {bucket} found {bucket_len} images, will {bcolors.FAIL}drop {truncate_count} images{bcolors.ENDC} due to batch size {bcolors.WARNING}{batch_size}{bcolors.ENDC}")
            

        # flatten the buckets
        image_caption_pairs = []
        for bucket in buckets:
            image_caption_pairs.extend(buckets[bucket])

        return image_caption_pairs

    @staticmethod
    def __recurse_data_root(self, recurse_root,use_sub_dirs=True,class_images=False):
        progress_bar = tqdm(os.listdir(recurse_root), desc=f" {bcolors.WARNING} ** Processing {recurse_root}{bcolors.ENDC}")
        for f in os.listdir(recurse_root):
            current = os.path.join(recurse_root, f)
            if os.path.isfile(current):
                ext = os.path.splitext(f)[1].lower()
                if '-depth' in f or '-masklabel' in f:
                    progress_bar.update(1)
                    continue
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    #try to open the file to make sure it's a valid image
                    try:
                        img = Image.open(current)
                    except:
                        print(f" ** Skipping {current} because it failed to open, please check the file")
                        progress_bar.update(1)
                        continue
                    del img
                    if class_images == False:
                        self.image_paths.append(current)
                    else:
                        self.class_images_path.append(current)
            progress_bar.update(1)
        if use_sub_dirs:
            sub_dirs = []

            for d in os.listdir(recurse_root):
                current = os.path.join(recurse_root, d)
                if os.path.isdir(current):
                    sub_dirs.append(current)

            for dir in sub_dirs:
                self.__recurse_data_root(self=self, recurse_root=dir)

class NormalDataset(Dataset):
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
        seed=555,
        model_variant='base',
        extra_module=None,
        mask_prompts=None,
        load_mask=None,
    ):
        self.use_image_names_as_captions = use_image_names_as_captions
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.with_prior_preservation = with_prior_preservation
        self.use_text_files_as_captions = use_text_files_as_captions
        self.image_paths = []
        self.class_images_path = []
        self.seed = seed
        self.model_variant = model_variant
        self.variant_warning = False
        self.vae_scale_factor = None
        self.load_mask = load_mask
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
        if mask_prompts is not None:
            print(f" {bcolors.WARNING} Checking and generating missing masks{bcolors.ENDC}")
            clip_seg = ClipSeg()
            clip_seg.mask_images(self.image_paths, mask_prompts)
            del clip_seg

        random.Random(seed).shuffle(self.image_paths)
        self.num_instance_images = len(self.image_paths)
        self._length = self.num_instance_images
        self.num_class_images = len(self.class_images_path)
        self._length = max(self.num_class_images, self.num_instance_images)
        if self.model_variant == 'depth2img':
            print(f" {bcolors.WARNING} ** Loading Depth2Img Pipeline To Process Dataset{bcolors.ENDC}")
            self.vae_scale_factor = extra_module.depth_images(self.image_paths)
            if self.with_prior_preservation:
                print(f" {bcolors.WARNING} ** Loading Depth2Img Class Processing{bcolors.ENDC}")
                extra_module.depth_images(self.class_images_path)
        print(f" {bcolors.WARNING} ** Dataset length: {self._length}, {int(self.num_instance_images / repeats)} images using {repeats} repeats{bcolors.ENDC}")

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
            
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
            ])

        self.depth_image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
            ]
        )

    @staticmethod
    def __recurse_data_root(self, recurse_root,use_sub_dirs=True,class_images=False):
        #if recurse root is a dict
        if isinstance(recurse_root, dict):
            if class_images == True:
                #print(f" {bcolors.WARNING} ** Processing class images: {recurse_root['class_data_dir']}{bcolors.ENDC}")
                concept_token = recurse_root['class_prompt']
                data = recurse_root['class_data_dir']
            else:
                #print(f" {bcolors.WARNING} ** Processing instance images: {recurse_root['instance_data_dir']}{bcolors.ENDC}")
                concept_token = recurse_root['instance_prompt']
                data = recurse_root['instance_data_dir']


        else:
            concept_token = None
        #progress bar
        progress_bar = tqdm(os.listdir(data), desc=f" {bcolors.WARNING} ** Processing {data}{bcolors.ENDC}")
        for f in os.listdir(data):
            current = os.path.join(data, f)
            if os.path.isfile(current):
                if '-depth' in f or '-masklabel' in f:
                    continue
                ext = os.path.splitext(f)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    try:
                        img = Image.open(current)
                    except:
                        print(f" ** Skipping {current} because it failed to open, please check the file")
                        progress_bar.update(1)
                        continue
                    del img
                    if class_images == False:
                        self.image_paths.append([current,concept_token])
                    else:
                        self.class_images_path.append([current,concept_token])
            progress_bar.update(1)
        if use_sub_dirs:
            sub_dirs = []

            for d in os.listdir(data):
                current = os.path.join(data, d)
                if os.path.isdir(current):
                    sub_dirs.append(current)

            for dir in sub_dirs:
                if class_images == False:
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
        if self.model_variant == "inpainting" or self.load_mask:

            mask_pathname = os.path.splitext(instance_path)[0] + "-masklabel.png"
            if os.path.exists(mask_pathname) and self.load_mask:
                mask = Image.open(mask_pathname).convert("L")
            else:
                if self.variant_warning == False:
                    print(f" {bcolors.FAIL} ** Warning: No mask found for an image, using an empty mask but make sure you're training the right model variant.{bcolors.ENDC}")
                    self.variant_warning = True
                size = instance_image.size
                mask = Image.new('RGB', size, color="white").convert("L")
            example["mask"] = self.mask_transforms(mask)
        if self.model_variant == "depth2img":
            depth_pathname = os.path.splitext(instance_path)[0] + "-depth.png"
            if os.path.exists(depth_pathname):
                depth_image = Image.open(depth_pathname).convert("L")
            else:
                if self.variant_warning == False:
                    print(f" {bcolors.FAIL} ** Warning: No depth image found for an image, using an empty depth image but make sure you're training the right model variant.{bcolors.ENDC}")
                    self.variant_warning = True
                size = instance_image.size
                depth_image = Image.new('RGB', size, color="white").convert("L")
            example["instance_depth_images"] = self.depth_image_transforms(depth_image)

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

            if self.model_variant == "inpainting":
                mask_pathname = os.path.splitext(class_path)[0] + "-masklabel.png"
                if os.path.exists(mask_pathname):
                    mask = Image.open(mask_pathname).convert("L")
                else:
                    if self.variant_warning == False:
                        print(f" {bcolors.FAIL} ** Warning: No mask found for an image, using an empty mask but make sure you're training the right model variant.{bcolors.ENDC}")
                        self.variant_warning = True
                    size = instance_image.size
                    mask = Image.new('RGB', size, color="white").convert("L")
                example["class_mask"] = self.mask_transforms(mask)
            if self.model_variant == "depth2img":
                depth_pathname = os.path.splitext(class_path)[0] + "-depth.png"
                if os.path.exists(depth_pathname):
                    depth_image = Image.open(depth_pathname)
                else:
                    if self.variant_warning == False:
                        print(f" {bcolors.FAIL} ** Warning: No depth image found for an image, using an empty depth image but make sure you're training the right model variant.{bcolors.ENDC}")
                        self.variant_warning = True
                    size = instance_image.size
                    depth_image = Image.new('RGB', size, color="white").convert("L")
                example["class_depth_images"] = self.depth_image_transforms(depth_image)
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

class CachedLatentsDataset(Dataset):
    #stores paths and loads latents on the fly
    def __init__(self, cache_paths=(),batch_size=None,tokenizer=None,text_encoder=None,dtype=None,model_variant='base',shuffle_per_epoch=False,args=None):
        self.cache_paths = cache_paths
        self.tokenizer = tokenizer
        self.args = args
        self.text_encoder = text_encoder
        #get text encoder device
        text_encoder_device = next(self.text_encoder.parameters()).device
        self.empty_batch = [self.tokenizer('',padding="do_not_pad",truncation=True,max_length=self.tokenizer.model_max_length,).input_ids for i in range(batch_size)]
        #handle text encoder for empty tokens
        if self.args.train_text_encoder != True:
            self.empty_tokens = tokenizer.pad({"input_ids": self.empty_batch},padding="max_length",max_length=tokenizer.model_max_length,return_tensors="pt",).to(text_encoder_device).input_ids
            self.empty_tokens.to(text_encoder_device, dtype=dtype)
            self.empty_tokens = self.text_encoder(self.empty_tokens)[0]
        else:
            self.empty_tokens = tokenizer.pad({"input_ids": self.empty_batch},padding="max_length",max_length=tokenizer.model_max_length,return_tensors="pt",).input_ids
            self.empty_tokens.to(text_encoder_device, dtype=dtype)

        self.conditional_dropout = args.conditional_dropout
        self.conditional_indexes = []
        self.model_variant = model_variant
        self.shuffle_per_epoch = shuffle_per_epoch
    def __len__(self):
        return len(self.cache_paths)
    def __getitem__(self, index):
        if index == 0:
            if self.shuffle_per_epoch == True:
                self.cache_paths = tuple(random.sample(self.cache_paths, len(self.cache_paths)))
            if len(self.cache_paths) > 1:
                possible_indexes_extension = None
                possible_indexes = list(range(0,len(self.cache_paths)))
                #conditional dropout is a percentage of images to drop from the total cache_paths
                if self.conditional_dropout != None:
                    if len(self.conditional_indexes) == 0:
                        self.conditional_indexes = random.sample(possible_indexes, k=int(math.ceil(len(possible_indexes)*self.conditional_dropout)))
                    else:
                        #pick indexes from the remaining possible indexes
                        possible_indexes_extension = [i for i in possible_indexes if i not in self.conditional_indexes]
                        #duplicate all values in possible_indexes_extension
                        possible_indexes_extension = possible_indexes_extension + possible_indexes_extension
                        possible_indexes_extension = possible_indexes_extension + self.conditional_indexes
                        self.conditional_indexes = random.sample(possible_indexes_extension, k=int(math.ceil(len(possible_indexes)*self.conditional_dropout)))
                        #check for duplicates in conditional_indexes values
                        if len(self.conditional_indexes) != len(set(self.conditional_indexes)):
                            #remove duplicates
                            self.conditional_indexes_non_dupe = list(set(self.conditional_indexes))
                            #add a random value from possible_indexes_extension for each duplicate
                            for i in range(len(self.conditional_indexes) - len(self.conditional_indexes_non_dupe)):
                                while True:
                                    random_value = random.choice(possible_indexes_extension)
                                    if random_value not in self.conditional_indexes_non_dupe:
                                        self.conditional_indexes_non_dupe.append(random_value)
                                        break
                            self.conditional_indexes = self.conditional_indexes_non_dupe
        self.cache = torch.load(self.cache_paths[index])
        self.latents = self.cache.latents_cache[0]
        self.tokens = self.cache.tokens_cache[0]
        self.extra_cache = None
        self.mask_cache = None
        if self.cache.mask_cache is not None:
            self.mask_cache = self.cache.mask_cache[0]
        self.mask_mean_cache = None
        if self.cache.mask_mean_cache is not None:
            self.mask_mean_cache = self.cache.mask_mean_cache[0]
        if index in self.conditional_indexes:
            self.text_encoder = self.empty_tokens
        else:
            self.text_encoder = self.cache.text_encoder_cache[0]
        if self.model_variant != 'base':
            self.extra_cache = self.cache.extra_cache[0]
        del self.cache
        return self.latents, self.text_encoder, self.mask_cache, self.mask_mean_cache, self.extra_cache, self.tokens

    def add_pt_cache(self, cache_path):
        if len(self.cache_paths) == 0:
            self.cache_paths = (cache_path,)
        else:
            self.cache_paths += (cache_path,)

class LatentsDataset(Dataset):
    def __init__(self, latents_cache=None, text_encoder_cache=None, mask_cache=None, mask_mean_cache=None, extra_cache=None,tokens_cache=None):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache
        self.mask_cache = mask_cache
        self.mask_mean_cache = mask_mean_cache
        self.extra_cache = extra_cache
        self.tokens_cache = tokens_cache
    def add_latent(self, latent, text_encoder, cached_mask, cached_extra, tokens_cache):
        self.latents_cache.append(latent)
        self.text_encoder_cache.append(text_encoder)
        self.mask_cache.append(cached_mask)
        self.mask_mean_cache.append(None if cached_mask is None else cached_mask.mean())
        self.extra_cache.append(cached_extra)
        self.tokens_cache.append(tokens_cache)
    def __len__(self):
        return len(self.latents_cache)
    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index], self.mask_cache[index], self.mask_mean_cache[index], self.extra_cache[index], self.tokens_cache[index]
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
    #temp arg
    args.batch_tokens = None
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
        mixed_precision=args.mixed_precision if args.mixed_precision != 'tf32' else 'no',
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
                        vae=AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,subfolder=None if args.pretrained_vae_name_or_path else "vae" ,safe_serialization=True),
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
                            images = pipeline(example["prompt"],height=args.resolution,width=args.resolution).images
                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                            image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name )
    elif args.pretrained_model_name_or_path:
        #print(os.getcwd())
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer" )

    # Load models and create wrapper for stable diffusion
    #text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder" )
    text_encoder_cls = tu.import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, None)
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder" )
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,subfolder=None if args.pretrained_vae_name_or_path else "vae" )
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet" )
    if is_xformers_available() and args.attention=='xformers':
        try:
            unet.enable_xformers_memory_efficient_attention()
            vae.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    elif args.attention=='flash_attention':
        tu.replace_unet_cross_attn_to_flash_attention()
    if args.use_ema == True:
        ema_unet = tu.EMAModel(unet.parameters())
    if args.model_variant == "depth2img":
        d2i = tu.Depth2Img(unet,text_encoder,args.mixed_precision,args.pretrained_model_name_or_path,accelerator)
    vae.requires_grad_(False)
    vae.enable_slicing()
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
            aspect_mode=args.aspect_mode,
            action_preference=args.aspect_mode_action_preference,
            seed=args.seed,
            model_variant=args.model_variant,
            extra_module=None if args.model_variant != "depth2img" else d2i,
            mask_prompts=args.mask_prompts,
            load_mask=args.masked_training,
        )
    else:
        train_dataset = NormalDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        use_image_names_as_captions=args.use_image_names_as_captions,
        repeats=args.dataset_repeats,
        use_text_files_as_captions=args.use_text_files_as_captions,
        seed = args.seed,
        model_variant=args.model_variant,
        extra_module=None if args.model_variant != "depth2img" else d2i,
        mask_prompts=args.mask_prompts,
        load_mask=args.masked_training,
    )
    def collate_fn(examples):
        #print(examples)
        #print('test')
        input_ids = [example["instance_prompt_ids"] for example in examples]
        tokens = input_ids
        pixel_values = [example["instance_images"] for example in examples]
        mask = None
        if "mask" in examples[0]:
            mask = [example["mask"] for example in examples]
        if args.model_variant == 'depth2img':
            depth = [example["instance_depth_images"] for example in examples]

        #print('test')
        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            if "mask" in examples[0]:
                mask += [example["class_mask"] for example in examples]
            if args.model_variant == 'depth2img':
                depth = [example["class_depth_images"] for example in examples]
        mask_values = None
        if mask is not None:
            mask_values = torch.stack(mask)
            mask_values = mask_values.to(memory_format=torch.contiguous_format).float()
        if args.model_variant == 'depth2img':
            depth_values = torch.stack(depth)
            depth_values = depth_values.to(memory_format=torch.contiguous_format).float()
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

        extra_values = None
        if args.model_variant == 'depth2img':
            extra_values = depth_values

        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "extra_values": extra_values,
            "mask_values": mask_values,
            "tokens": tokens
        }

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

    
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif args.mixed_precision == "no":
        weight_dtype = torch.float32
    elif args.mixed_precision == "tf32":
        weight_dtype = torch.float32
        torch.backends.cuda.matmul.allow_tf32 = True
        #torch.set_float32_matmul_precision("medium")

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if args.use_ema == True:
        ema_unet.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.use_bucketing:
        wh = set([tuple(x.target_wh) for x in train_dataset.image_train_items])
    else:
        wh = set([tuple([args.resolution, args.resolution]) for x in train_dataset.image_paths])
    full_mask_by_aspect = {shape: vae.encode(torch.zeros(1, 3, shape[1], shape[0]).to(accelerator.device, dtype=weight_dtype)).latent_dist.mean * 0.18215 for shape in wh}

    cached_dataset = CachedLatentsDataset(batch_size=args.train_batch_size,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    dtype=weight_dtype,
    model_variant=args.model_variant,
    shuffle_per_epoch=args.shuffle_per_epoch,
    args = args,)

    gen_cache = False
    data_len = len(train_dataloader)
    latent_cache_dir = Path(args.output_dir, "logs", "latent_cache")
    #check if latents_cache.pt exists in the output_dir
    if not os.path.exists(latent_cache_dir):
        os.makedirs(latent_cache_dir)
    for i in range(0,data_len-1):
        if not os.path.exists(os.path.join(latent_cache_dir, f"latents_cache_{i}.pt")):
            gen_cache = True
            break
    if args.regenerate_latent_cache == True:
            files = os.listdir(latent_cache_dir)
            gen_cache = True
            for file in files:
                os.remove(os.path.join(latent_cache_dir,file))
    if gen_cache == False :
        print(f" {bcolors.OKGREEN}Loading Latent Cache from {latent_cache_dir}{bcolors.ENDC}")
        del vae
        if not args.train_text_encoder:
            del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        #load all the cached latents into a single dataset
        for i in range(0,data_len-1):
            cached_dataset.add_pt_cache(os.path.join(latent_cache_dir,f"latents_cache_{i}.pt"))
    if gen_cache == True:
        #delete all the cached latents if they exist to avoid problems
        print(f" {bcolors.WARNING}Generating latents cache...{bcolors.ENDC}")
        train_dataset = LatentsDataset([], [], [], [], [], [])
        counter = 0
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with torch.no_grad():
            for batch in tqdm(train_dataloader, desc="Caching latents", bar_format='%s{l_bar}%s%s{bar}%s%s{r_bar}%s'%(bcolors.OKBLUE,bcolors.ENDC, bcolors.OKBLUE, bcolors.ENDC,bcolors.OKBLUE,bcolors.ENDC,)):
                cached_extra = None
                cached_mask = None
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
                cached_latent = vae.encode(batch["pixel_values"]).latent_dist
                if batch["mask_values"] is not None:
                    cached_mask = functional.resize(batch["mask_values"], size=cached_latent.mean.shape[2:])
                if batch["mask_values"] is not None and args.model_variant == "inpainting":
                    batch["mask_values"] = batch["mask_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                    cached_extra = vae.encode(batch["pixel_values"] * (1 - batch["mask_values"])).latent_dist
                if args.model_variant == "depth2img":
                    batch["extra_values"] = batch["extra_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                    cached_extra = functional.resize(batch["extra_values"], size=cached_latent.mean.shape[2:])
                if args.train_text_encoder:
                    cached_text_enc = batch["input_ids"]
                else:
                    cached_text_enc = text_encoder(batch["input_ids"])[0]
                train_dataset.add_latent(cached_latent, cached_text_enc, cached_mask, cached_extra, batch["tokens"])
                del batch
                del cached_latent
                del cached_text_enc
                del cached_mask
                del cached_extra
                torch.save(train_dataset, os.path.join(latent_cache_dir,f"latents_cache_{counter}.pt"))
                cached_dataset.add_pt_cache(os.path.join(latent_cache_dir,f"latents_cache_{counter}.pt"))
                counter += 1
                train_dataset = LatentsDataset([], [], [], [], [], [])
                #if counter % 300 == 0:
                    #train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=False)
                #    gc.collect()
                #    torch.cuda.empty_cache()
                #    accelerator.free_memory()

        #clear vram after caching latents
        del vae
        if not args.train_text_encoder:
            del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        #load all the cached latents into a single dataset
    train_dataloader = torch.utils.data.DataLoader(cached_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=False)
    print(f" {bcolors.OKGREEN}Latents are ready.{bcolors.ENDC}")
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = len(train_dataloader)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
        
    if args.lr_warmup_steps < 1:
        args.lr_warmup_steps = math.floor(args.lr_warmup_steps * args.max_train_steps / args.gradient_accumulation_steps)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps,
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
    num_update_steps_per_epoch = len(train_dataloader)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        #print(args.max_train_steps, num_update_steps_per_epoch)
    # Afterwards we recalculate our number of training epochs
    #print(args.max_train_steps, num_update_steps_per_epoch)
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
            vae=AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,subfolder=None if args.pretrained_vae_name_or_path else "vae", safe_serialization=True),
            safety_checker=None,
            torch_dtype=weight_dtype,
            local_files_only=False,
        )
        pipeline.scheduler = scheduler
        if is_xformers_available() and args.attention=='xformers':
            try:
                unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                logger.warning(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )
        elif args.attention=='flash_attention':
            tu.replace_unet_cross_attn_to_flash_attention()
        pipeline = pipeline.to(accelerator.device)
        def inference(prompt, negative_prompt, num_samples, height=512, width=512, num_inference_steps=50,seed=-1,guidance_scale=7.5):
            with torch.autocast("cuda"), torch.inference_mode():
                if seed != -1:
                    if g_cuda is None:
                        g_cuda = torch.Generator(device='cuda')
                    else:
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
    def print_instructions():
            print(f"{bcolors.WARNING}Use 'CTRL+SHIFT+G' to open up a GUI to play around with the model (will pause training){bcolors.ENDC}")
            print(f"{bcolors.WARNING}Use 'CTRL+SHIFT+S' to save a checkpoint of the current epoch{bcolors.ENDC}")
            print(f"{bcolors.WARNING}Use 'CTRL+SHIFT+P' to generate samples for current epoch{bcolors.ENDC}")
            print(f"{bcolors.WARNING}Use 'CTRL+SHIFT+Q' to save and quit after the current epoch{bcolors.ENDC}")
            print(f"{bcolors.WARNING}Use 'CTRL+SHIFT+ALT+S' to save a checkpoint of the current step{bcolors.ENDC}")
            print(f"{bcolors.WARNING}Use 'CTRL+SHIFT+ALT+P' to generate samples for current step{bcolors.ENDC}")
            print(f"{bcolors.WARNING}Use 'CTRL+SHIFT+ALT+Q' to save and quit after the current step{bcolors.ENDC}")
            print('')
            print(f"{bcolors.WARNING}Use 'CTRL+H' to print this message again.{bcolors.ENDC}")
    def save_and_sample_weights(step,context='checkpoint',save_model=True):
        try:
            #check how many folders are in the output dir
            #if there are more than 5, delete the oldest one
            #save the model
            #save the optimizer
            #save the lr_scheduler
            #save the args
            height = args.sample_height
            width = args.sample_width
            batch_prompts = []
            if args.sample_from_batch > 0:
                num_samples = args.sample_from_batch if args.sample_from_batch < args.train_batch_size else args.train_batch_size
                batch_prompts = []
                tokens = args.batch_tokens
                if tokens != None:
                    allPrompts = list(set([tokenizer.decode(p).replace('<|endoftext|>','').replace('<|startoftext|>', '') for p in tokens]))
                    if len(allPrompts) < num_samples:
                        num_samples = len(allPrompts)
                    batch_prompts = random.sample(allPrompts, num_samples)
                        

            if args.sample_aspect_ratios:
                #choose random aspect ratio from ASPECTS    
                aspect_ratio = random.choice(ASPECTS)
                height = aspect_ratio[0]
                width = aspect_ratio[1]
            if os.path.exists(args.output_dir):
                if args.detect_full_drive==True:
                    folders = os.listdir(args.output_dir)
                    #check how much space is left on the drive
                    total, used, free = shutil.disk_usage("/")
                    if (free // (2**30)) < 4:
                        #folders.remove("0")
                        #get the folder with the lowest number
                        #oldest_folder = min(folder for folder in folders if folder.isdigit())
                        print(f"{bcolors.FAIL}Drive is almost full, Please make some space to continue training.{bcolors.ENDC}")
                        if args.send_telegram_updates:
                            try:
                                send_telegram_message(f"Drive is almost full, Please make some space to continue training.", args.telegram_chat_id, args.telegram_token)
                            except:
                                pass
                        #count time
                        import time
                        start_time = time.time()
                        import platform
                        while input("Press Enter to continue... if you're on linux we'll wait 5 minutes for you to make space and continue"):
                            #check if five minutes have passed
                            #check if os is linux
                            if 'Linux' in platform.platform():
                                if time.time() - start_time > 300:
                                    break

                        
                        #oldest_folder_path = os.path.join(args.output_dir, oldest_folder)
                        #shutil.rmtree(oldest_folder_path)
            # Create the pipeline using using the trained modules and save it.
            if accelerator.is_main_process:
                if 'step' in context:
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
                    vae=AutoencoderKL.from_pretrained(args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,subfolder=None if args.pretrained_vae_name_or_path else "vae",),
                    safety_checker=None,
                    torch_dtype=weight_dtype,
                    local_files_only=False,
                )
                pipeline.scheduler = scheduler
                if is_xformers_available() and args.attention=='xformers':
                    try:
                        unet.enable_xformers_memory_efficient_attention()
                    except Exception as e:
                        logger.warning(
                            "Could not enable memory efficient attention. Make sure xformers is installed"
                            f" correctly and a GPU is available: {e}"
                        )
                elif args.attention=='flash_attention':
                    tu.replace_unet_cross_attn_to_flash_attention()
                save_dir = os.path.join(args.output_dir, f"{context}_{step}")
                if args.flatten_sample_folder:
                    sample_dir = os.path.join(args.output_dir, "samples")
                else:
                    sample_dir = os.path.join(args.output_dir, f"samples/{context}_{step}")
                #if sample dir path does not exist, create it
                
                if args.stop_text_encoder_training == True:
                    save_dir = frozen_directory
                if step != 0:
                    if save_model:
                        pipeline.save_pretrained(save_dir,safe_serialization=True)
                        with open(os.path.join(save_dir, "args.json"), "w") as f:
                                json.dump(args.__dict__, f, indent=2)
                    if args.stop_text_encoder_training == True:
                        #delete every folder in frozen_directory but the text encoder
                        for folder in os.listdir(save_dir):
                            if folder != "text_encoder" and os.path.isdir(os.path.join(save_dir, folder)):
                                shutil.rmtree(os.path.join(save_dir, folder))
                imgs = []
                if args.add_sample_prompt is not None or batch_prompts != [] and args.stop_text_encoder_training != True:
                    prompts = []
                    if args.add_sample_prompt is not None:
                        for prompt in args.add_sample_prompt:
                            prompts.append(prompt)
                    if batch_prompts != []:
                        for prompt in batch_prompts:
                            prompts.append(prompt)

                    pipeline = pipeline.to(accelerator.device)
                    pipeline.set_progress_bar_config(disable=True)
                    #sample_dir = os.path.join(save_dir, "samples")
                    #if sample_dir exists, delete it
                    if os.path.exists(sample_dir):
                        if not args.flatten_sample_folder:
                            shutil.rmtree(sample_dir)
                    os.makedirs(sample_dir, exist_ok=True)
                    with torch.autocast("cuda"), torch.inference_mode():
                        if args.send_telegram_updates:
                            try:
                                send_telegram_message(f"Generating samples for <b>{step}</b> {context}", args.telegram_chat_id, args.telegram_token)
                            except:
                                pass
                        for samplePrompt in prompts:
                            sampleIndex = prompts.index(samplePrompt)
                            #convert sampleIndex to number in words
                            # Data to be written
                            sampleProperties = {
                                "samplePrompt" : samplePrompt
                            }
                            
                            # Serializing json
                            json_object = json.dumps(sampleProperties, indent=4)
                            
                            if args.flatten_sample_folder:
                                sampleName = f"{context}_{step}_prompt_{sampleIndex+1}"
                            else:
                                sampleName = f"prompt_{sampleIndex+1}"
                            
                            if not args.flatten_sample_folder:
                                os.makedirs(os.path.join(sample_dir,sampleName), exist_ok=True)
                            
                            if args.model_variant == 'inpainting':
                                conditioning_image = torch.zeros(1, 3, height, width)
                                mask = torch.ones(1, 1, height, width)
                            if args.model_variant == 'depth2img':
                                #pil new white image
                                test_image = Image.new('RGB', (width, height), (255, 255, 255))
                                depth_image = Image.new('RGB', (width, height), (255, 255, 255))
                                depth = np.array(depth_image.convert("L"))
                                depth = depth.astype(np.float32) / 255.0
                                depth = depth[None, None]
                                depth = torch.from_numpy(depth)
                            for i in tqdm(range(args.n_save_sample) if not args.save_sample_controlled_seed else range(args.n_save_sample+len(args.save_sample_controlled_seed)), desc="Generating samples"):
                                #check if the sample is controlled by a seed
                                if i < args.n_save_sample:
                                    if args.model_variant == 'inpainting':
                                        images = pipeline(samplePrompt, conditioning_image, mask, height=height,width=width, guidance_scale=args.save_guidance_scale, num_inference_steps=args.save_infer_steps).images
                                    if args.model_variant == 'depth2img':
                                        images = pipeline(samplePrompt,image=test_image, guidance_scale=args.save_guidance_scale, num_inference_steps=args.save_infer_steps,strength=1.0).images
                                    elif args.model_variant == 'base':
                                        images = pipeline(samplePrompt,height=height,width=width, guidance_scale=args.save_guidance_scale, num_inference_steps=args.save_infer_steps).images
                                    
                                    if not args.flatten_sample_folder:
                                        images[0].save(os.path.join(sample_dir,sampleName, f"{sampleName}_{i}.png"))
                                    else:
                                        images[0].save(os.path.join(sample_dir, f"{sampleName}_{i}.png"))
                                                                       
                                else:
                                    seed = args.save_sample_controlled_seed[i - args.n_save_sample]
                                    generator = torch.Generator("cuda").manual_seed(seed)
                                    if args.model_variant == 'inpainting':
                                        images = pipeline(samplePrompt,conditioning_image, mask,height=height,width=width, guidance_scale=args.save_guidance_scale, num_inference_steps=args.save_infer_steps, generator=generator).images
                                    if args.model_variant == 'depth2img':
                                        images = pipeline(samplePrompt,image=test_image, guidance_scale=args.save_guidance_scale, num_inference_steps=args.save_infer_steps,generator=generator,strength=1.0).images
                                    elif args.model_variant == 'base':
                                        images = pipeline(samplePrompt,height=height,width=width, guidance_scale=args.save_guidance_scale, num_inference_steps=args.save_infer_steps, generator=generator).images
                                    
                                    if not args.flatten_sample_folder:
                                        images[0].save(os.path.join(sample_dir,sampleName, f"{sampleName}_controlled_seed_{str(seed)}.png"))
                                    else:
                                        images[0].save(os.path.join(sample_dir, f"{sampleName}_controlled_seed_{str(seed)}.png"))
                            
                            if args.send_telegram_updates:
                                imgs = []
                                #get all the images from the sample folder
                                if not args.flatten_sample_folder:
                                    dir = os.listdir(os.path.join(sample_dir,sampleName))
                                else:
                                    dir = sample_dir

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
                        torch.cuda.ipc_collect()
                if save_model == True:
                    print(f"{bcolors.OKGREEN}Weights saved to {save_dir}{bcolors.ENDC}")
                elif save_model == False and len(imgs) > 0:
                    del imgs
                    print(f"{bcolors.OKGREEN}Samples saved to {sample_dir}{bcolors.ENDC}")
        except Exception as e:
            print(e)
            print(f"{bcolors.FAIL} Error occured during sampling, skipping.{bcolors.ENDC}")
            pass

    # Only show the progress bar once on each machine.
    progress_bar_inter_epoch = tqdm(range(num_update_steps_per_epoch),bar_format='%s{l_bar}%s%s{bar}%s%s{r_bar}%s'%(bcolors.OKBLUE,bcolors.ENDC, bcolors.OKGREEN, bcolors.ENDC,bcolors.OKBLUE,bcolors.ENDC,), disable=not accelerator.is_local_main_process)
    progress_bar = tqdm(range(args.max_train_steps),bar_format='%s{l_bar}%s%s{bar}%s%s{r_bar}%s'%(bcolors.OKBLUE,bcolors.ENDC, bcolors.OKBLUE, bcolors.ENDC,bcolors.OKBLUE,bcolors.ENDC,), disable=not accelerator.is_local_main_process)
    progress_bar_e = tqdm(range(args.num_train_epochs),bar_format='%s{l_bar}%s%s{bar}%s%s{r_bar}%s'%(bcolors.OKBLUE,bcolors.ENDC, bcolors.OKGREEN, bcolors.ENDC,bcolors.OKBLUE,bcolors.ENDC,), disable=not accelerator.is_local_main_process)

    progress_bar.set_description("Overall Steps")
    progress_bar_e.set_description("Overall Epochs")
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
            def toggle_gui(event=None):
                if keyboard.is_pressed("ctrl") and keyboard.is_pressed("shift") and keyboard.is_pressed("g"):
                    print(f" {bcolors.WARNING}GUI will boot as soon as the current step is done.{bcolors.ENDC}")
                    nonlocal mid_generation
                    if mid_generation == True:
                        mid_generation = False
                        print(f" {bcolors.WARNING}Cancelled GUI.{bcolors.ENDC}")
                    else:
                        mid_generation = True

            def toggle_checkpoint(event=None):
                if keyboard.is_pressed("ctrl") and keyboard.is_pressed("shift") and keyboard.is_pressed("s") and not keyboard.is_pressed("alt"):
                    print(f" {bcolors.WARNING}Saving the model as soon as this epoch is done.{bcolors.ENDC}")
                    nonlocal mid_checkpoint
                    if mid_checkpoint == True:
                        mid_checkpoint = False
                        print(f" {bcolors.WARNING}Cancelled Checkpointing.{bcolors.ENDC}")
                    else:
                        mid_checkpoint = True

            def toggle_sample(event=None):
                if keyboard.is_pressed("ctrl") and keyboard.is_pressed("shift") and keyboard.is_pressed("p") and not keyboard.is_pressed("alt"):
                    print(f" {bcolors.WARNING}Sampling will begin as soon as this epoch is done.{bcolors.ENDC}")
                    nonlocal mid_sample
                    if mid_sample == True:
                        mid_sample = False
                        print(f" {bcolors.WARNING}Cancelled Sampling.{bcolors.ENDC}")
                    else:
                        mid_sample = True
            def toggle_checkpoint_step(event=None):
                if keyboard.is_pressed("ctrl") and keyboard.is_pressed("shift") and keyboard.is_pressed("alt") and keyboard.is_pressed("s"):
                    print(f" {bcolors.WARNING}Saving the model as soon as this step is done.{bcolors.ENDC}")
                    nonlocal mid_checkpoint_step
                    if mid_checkpoint_step == True:
                        mid_checkpoint_step = False
                        print(f" {bcolors.WARNING}Cancelled Checkpointing.{bcolors.ENDC}")
                    else:
                        mid_checkpoint_step = True

            def toggle_sample_step(event=None):
                if keyboard.is_pressed("ctrl") and keyboard.is_pressed("shift") and keyboard.is_pressed("alt") and keyboard.is_pressed("p"):
                    print(f" {bcolors.WARNING}Sampling will begin as soon as this step is done.{bcolors.ENDC}")
                    nonlocal mid_sample_step
                    if mid_sample_step == True:
                        mid_sample_step = False
                        print(f" {bcolors.WARNING}Cancelled Sampling.{bcolors.ENDC}")
                    else:
                        mid_sample_step = True
            def toggle_quit_and_save_epoch(event=None):
                if keyboard.is_pressed("ctrl") and keyboard.is_pressed("shift") and keyboard.is_pressed("q") and not keyboard.is_pressed("alt"):
                    print(f" {bcolors.WARNING}Quitting and saving the model as soon as this epoch is done.{bcolors.ENDC}")
                    nonlocal mid_quit
                    if mid_quit == True:
                        mid_quit = False
                        print(f" {bcolors.WARNING}Cancelled Quitting.{bcolors.ENDC}")
                    else:
                        mid_quit = True
            def toggle_quit_and_save_step(event=None):
                if keyboard.is_pressed("ctrl") and keyboard.is_pressed("shift") and keyboard.is_pressed("alt") and keyboard.is_pressed("q"):
                    print(f" {bcolors.WARNING}Quitting and saving the model as soon as this step is done.{bcolors.ENDC}")
                    nonlocal mid_quit_step
                    if mid_quit_step == True:
                        mid_quit_step = False
                        print(f" {bcolors.WARNING}Cancelled Quitting.{bcolors.ENDC}")
                    else:
                        mid_quit_step = True
            def help(event=None):
                if keyboard.is_pressed("ctrl") and keyboard.is_pressed("h"):
                    print_instructions()
            keyboard.on_press_key("g", toggle_gui)
            keyboard.on_press_key("s", toggle_checkpoint)
            keyboard.on_press_key("p", toggle_sample)
            keyboard.on_press_key("s", toggle_checkpoint_step)
            keyboard.on_press_key("p", toggle_sample_step)
            keyboard.on_press_key("q", toggle_quit_and_save_epoch)
            keyboard.on_press_key("q", toggle_quit_and_save_step)
            keyboard.on_press_key("h", help)
            print_instructions()
        except Exception as e:
            pass

        mid_generation = False
        mid_checkpoint = False
        mid_sample = False
        mid_checkpoint_step = False
        mid_sample_step = False
        mid_quit = False
        mid_quit_step = False
        #lambda set mid_generation to true
        frozen_directory=args.output_dir + "/frozen_text_encoder"

        for epoch in range(args.num_train_epochs):
            #every 10 epochs print instructions
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
                    if os.path.exists(frozen_directory):
                        #delete the folder if it already exists
                        shutil.rmtree(frozen_directory)
                    os.mkdir(frozen_directory)
                    save_and_sample_weights(epoch,'epoch')
                    args.stop_text_encoder_training = epoch
            progress_bar_inter_epoch.set_description("Steps To Epoch")
            progress_bar_inter_epoch.reset(total=num_update_steps_per_epoch)
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    with torch.no_grad():

                        latent_dist = batch[0][0]
                        latents = latent_dist.sample() * 0.18215
                        mask = batch[0][2]
                        mask_mean = batch[0][3]
                        if args.model_variant == 'inpainting':
                            conditioning_latent_dist = batch[0][4]
                            conditioning_latents = conditioning_latent_dist.sample() * 0.18215
                        if args.model_variant == 'depth2img':
                            depth = batch[0][4]
                    if args.sample_from_batch > 0:
                        args.batch_tokens = batch[0][5]
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, int(noise_scheduler.config.num_train_timesteps * args.max_denoising_strength), (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    with text_enc_context:
                        if args.train_text_encoder:
                            if args.clip_penultimate == True:
                                encoder_hidden_states = text_encoder(batch[0][1],output_hidden_states=True)
                                encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states['hidden_states'][-2])
                            else:
                                encoder_hidden_states = text_encoder(batch[0][1])[0]
                        else:
                            encoder_hidden_states = batch[0][1]

                    if mask is not None and random.uniform(0, 1) < args.unmasked_probability:
                        # for some steps, predict the unmasked image
                        conditioning_latents = torch.stack([full_mask_by_aspect[tuple([latents.shape[3]*8, latents.shape[2]*8])].squeeze()] * bsz)
                        mask = torch.ones(bsz, 1, latents.shape[2], latents.shape[3]).to(accelerator.device, dtype=weight_dtype)
                    # Predict the noise residual
                    if args.model_variant == 'inpainting':

                        noisy_inpaint_latents = torch.concat([noisy_latents, mask, conditioning_latents], 1)
                        model_pred = unet(noisy_inpaint_latents, timesteps, encoder_hidden_states).sample
                    elif args.model_variant == 'depth2img':
                        noisy_depth_latents = torch.cat([noisy_latents, depth], dim=1)
                        model_pred = unet(noisy_depth_latents, timesteps, encoder_hidden_states, depth).sample
                    elif args.model_variant == "base":
                        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    if args.model_variant == "inpainting":
                        del timesteps, noise, latents, noisy_latents,noisy_inpaint_latents, encoder_hidden_states
                    elif args.model_variant == "depth2img":
                        del timesteps, noise, latents, noisy_latents,noisy_depth_latents, encoder_hidden_states
                    elif args.model_variant == "base":
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
                        if mask is not None and args.model_variant != "inpainting":
                            loss = tu.masked_mse_loss(model_pred.float(), target.float(), mask, reduction="none").mean([1, 2, 3]).mean()
                            prior_loss = tu.masked_mse_loss(model_pred_prior.float(), target_prior.float(), mask, reduction="mean")
                        else:
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()
                            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss

                        if mask is not None and args.normalize_masked_area_loss:
                            loss = loss / mask_mean

                    else:
                        if mask is not None and args.model_variant != "inpainting":
                            loss = tu.masked_mse_loss(model_pred.float(), target.float(), mask, reduction="none").mean([1, 2, 3])
                            loss = loss.mean()
                        else:
                            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        if mask is not None and args.normalize_masked_area_loss:
                            loss = loss / mask_mean
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
                
                

                if global_step > 0 and not global_step % args.sample_step_interval and epoch != 0:
                    save_and_sample_weights(global_step,'step',save_model=False)

                progress_bar.update(1)
                progress_bar_inter_epoch.update(1)
                progress_bar_e.refresh()
                global_step += 1

                if mid_generation==True:
                    mid_train_playground(global_step)
                    mid_generation=False
                if mid_checkpoint_step == True:
                    save_and_sample_weights(global_step,'step',save_model=True)
                    mid_checkpoint_step=False
                if mid_sample_step == True:
                    save_and_sample_weights(global_step,'step',save_model=False)
                    mid_sample_step=False
                if mid_quit_step==True:
                    accelerator.wait_for_everyone()
                    save_and_sample_weights(global_step,'quit_step')
                    quit()
                if global_step >= args.max_train_steps:
                    break
            progress_bar_e.update(1)
            if mid_quit==True:
                accelerator.wait_for_everyone()
                save_and_sample_weights(epoch,'quit_epoch')
                quit()
            if not epoch % args.save_every_n_epoch:
                if args.save_every_n_epoch == 1 and epoch == 0:
                    save_and_sample_weights(epoch,'epoch')
                if epoch != 0:
                    save_and_sample_weights(epoch,'epoch')
                else:
                    pass
                    #save_and_sample_weights(epoch,'epoch',False)
                    print_instructions()
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
