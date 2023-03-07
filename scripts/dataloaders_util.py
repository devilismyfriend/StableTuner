import random
import math
import os
import torch
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
from trainer_util import *
from clip_segmentation import ClipSeg

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

ASPECT_512 = [[512,512],     # 262144 1:1
    [576,448],[448,576],   # 258048 1.29:1
    [640,384],[384,640],   # 245760 1.667:1
    [768,320],[320,768],   # 245760 2.4:1
    [832,256],[256,832],   # 212992 3.25:1
    [896,256],[256,896],   # 229376 3.5:1
    [960,256],[256,960],   # 245760 3.75:1
    [1024,256],[256,1024], # 245760 4:1
    ]
    
ASPECT_448 = [[448,448],   # 200704 1:1
    [512,384],[384,512],   # 196608 1.33:1
    [576,320],[320,576],   # 184320 1.8:1
    [768,256],[256,768],   # 196608 3:1
    ]

ASPECT_384 = [[384,384],   # 147456 1:1
    [448,320],[320,448],   # 143360 1.4:1
    [576,256],[256,576],   # 147456 2.25:1
    [768,192],[192,768],   # 147456 4:1
    ]
    
ASPECT_320 = [[320,320],   # 102400 1:1
    [384,256],[256,384],   # 98304 1.5:1
    [512,192],[192,512],   # 98304 2.67:1
    ]
    
ASPECT_256 = [[256,256],   # 65536 1:1
    [320,192],[192,320],   # 61440 1.67:1
    [512,128],[128,512],   # 65536 4:1
    ]

#failsafe aspects
ASPECTS = ASPECT_512
def get_aspect_buckets(resolution,mode=''):
    if resolution < 256:
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
    return [ASPECT_256, ASPECT_320, ASPECT_384, ASPECT_448, ASPECT_512, ASPECT_576, ASPECT_640, ASPECT_704, ASPECT_768,ASPECT_832,ASPECT_896,ASPECT_960,ASPECT_1024,ASPECT_1088,ASPECT_1152,ASPECT_1216,ASPECT_1280,ASPECT_1344,ASPECT_1408,ASPECT_1472,ASPECT_1536,ASPECT_1600,ASPECT_1664,ASPECT_1728,ASPECT_1792,ASPECT_1856,ASPECT_1920,ASPECT_1984,ASPECT_2048]
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
                            tot += len(files)
                        count = tot
                    else:
                        count = len(os.listdir(concept['instance_data_dir']))
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
