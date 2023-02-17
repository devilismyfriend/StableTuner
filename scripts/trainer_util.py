import gradio as gr
import json
import math
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel,DiffusionPipeline, DPMSolverMultistepScheduler,EulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from torchvision import transforms
from tqdm.auto import tqdm
from typing import Dict, List, Generator, Tuple
from PIL import Image, ImageFile
from collections.abc import Iterable
from trainer_util import *
from dataloaders_util import *

# FlashAttention based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main
# /memory_efficient_attention_pytorch/flash_attention.py LICENSE MIT
# https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE constants
EPSILON = 1e-6

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
# helper functions
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

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def masked_mse_loss(predicted, target, mask, reduction="none"):
    masked_predicted = predicted * mask
    masked_target = target * mask
    return F.mse_loss(masked_predicted, masked_target, reduction=reduction)

# flash attention forwards and backwards
# https://arxiv.org/abs/2205.14135


class FlashAttentionFunction(torch.autograd.function.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """ Algorithm 2 in the paper """

        device = q.device
        dtype = q.dtype
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros(
            (*q.shape[:-1], 1), dtype=dtype, device=device)
        all_row_maxes = torch.full(
            (*q.shape[:-1], 1), max_neg_value, dtype=dtype, device=device)

        scale = (q.shape[-1] ** -0.5)

        if not exists(mask):
            mask = (None,) * math.ceil(q.shape[-2] / q_bucket_size)
        else:
            mask = rearrange(mask, 'b n -> b 1 1 n')
            mask = mask.split(q_bucket_size, dim=-1)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            mask,
            all_row_sums.split(q_bucket_size, dim=-2),
            all_row_maxes.split(q_bucket_size, dim=-2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum(
                    '... i d, ... j d -> ... i j', qc, kc) * scale

                if exists(row_mask):
                    attn_weights.masked_fill_(~row_mask, max_neg_value)

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                             device=device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(dim=-1, keepdims=True)
                attn_weights -= block_row_maxes
                exp_weights = torch.exp(attn_weights)

                if exists(row_mask):
                    exp_weights.masked_fill_(~row_mask, 0.)

                block_row_sums = exp_weights.sum(
                    dim=-1, keepdims=True).clamp(min=EPSILON)

                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_values = einsum(
                    '... i j, ... j d -> ... i d', exp_weights, vc)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
                exp_block_row_max_diff = torch.exp(
                    block_row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + \
                               exp_block_row_max_diff * block_row_sums

                oc.mul_((row_sums / new_row_sums) * exp_row_max_diff).add_(
                    (exp_block_row_max_diff / new_row_sums) * exp_values)

                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)

        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, all_row_sums, all_row_maxes)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """ Algorithm 4 in the paper """

        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, l, m = ctx.saved_tensors

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            do.split(q_bucket_size, dim=-2),
            mask,
            l.split(q_bucket_size, dim=-2),
            m.split(q_bucket_size, dim=-2),
            dq.split(q_bucket_size, dim=-2)
        )

        for ind, (qc, oc, doc, row_mask, lc, mc, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
                dk.split(k_bucket_size, dim=-2),
                dv.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc, dkc, dvc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum(
                    '... i d, ... j d -> ... i j', qc, kc) * scale

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool,
                                             device=device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                exp_attn_weights = torch.exp(attn_weights - mc)

                if exists(row_mask):
                    exp_attn_weights.masked_fill_(~row_mask, 0.)

                p = exp_attn_weights / lc

                dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
                dp = einsum('... i d, ... j d -> ... i j', doc, vc)

                D = (doc * oc).sum(dim=-1, keepdims=True)
                ds = p * scale * (dp - D)

                dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
                dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        return dq, dk, dv, None, None, None, None

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def replace_unet_cross_attn_to_flash_attention():
    print("Using FlashAttention")

    def forward_flash_attn(self, x, context=None, mask=None):
        q_bucket_size = 512
        k_bucket_size = 1024

        h = self.heads
        q = self.to_q(x)

        context = context if context is not None else x
        context = context.to(x.dtype)

        if hasattr(self, 'hypernetwork') and self.hypernetwork is not None:
            context_k, context_v = self.hypernetwork.forward(x, context)
            context_k = context_k.to(x.dtype)
            context_v = context_v.to(x.dtype)
        else:
            context_k = context
            context_v = context

        k = self.to_k(context_k)
        v = self.to_v(context_v)
        del context, x

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        out = FlashAttentionFunction.apply(q, k, v, mask, False,
                                           q_bucket_size, k_bucket_size)

        out = rearrange(out, 'b h n d -> b n (h d)')

        # diffusers 0.6.0
        if type(self.to_out) is torch.nn.Sequential:
            return self.to_out(out)

        # diffusers 0.7.0
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out

    diffusers.models.attention.CrossAttention.forward = forward_flash_attn
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