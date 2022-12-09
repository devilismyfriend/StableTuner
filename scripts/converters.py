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
)
from diffusers.pipelines.latent_diffusion.pipeline_latent_diffusion import LDMBertConfig, LDMBertModel
#from diffusers.pipelines.paint_by_example import PaintByExampleImageEncoder, PaintByExamplePipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor, BertTokenizerFast, CLIPTextModel, CLIPTokenizer, CLIPVisionConfig
class Convert_SD_to_Diffusers():

    def __init__(self, checkpoint_path, output_path, prediction_type=None, img_size=None, original_config_file=None, extract_ema=False, num_in_channels=None,pipeline_type=None,scheduler_type=None,sd_version=None):
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
        self.main()
        
    def shave_segments(self,path, n_shave_prefix_segments=1):
        """
        Removes segments. Positive values shave the first segments, negative shave the last segments.
        """
        if n_shave_prefix_segments >= 0:
            return ".".join(path.split(".")[n_shave_prefix_segments:])
        else:
            return ".".join(path.split(".")[:n_shave_prefix_segments])


    def renew_resnet_paths(self,old_list, n_shave_prefix_segments=0):
        """
        Updates paths inside resnets to the new naming scheme (local renaming)
        """
        mapping = []
        for old_item in old_list:
            new_item = old_item.replace("in_layers.0", "norm1")
            new_item = new_item.replace("in_layers.2", "conv1")

            new_item = new_item.replace("out_layers.0", "norm2")
            new_item = new_item.replace("out_layers.3", "conv2")

            new_item = new_item.replace("emb_layers.1", "time_emb_proj")
            new_item = new_item.replace("skip_connection", "conv_shortcut")

            new_item = self.shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

            mapping.append({"old": old_item, "new": new_item})

        return mapping


    def renew_vae_resnet_paths(self,old_list, n_shave_prefix_segments=0):
        """
        Updates paths inside resnets to the new naming scheme (local renaming)
        """
        mapping = []
        for old_item in old_list:
            new_item = old_item

            new_item = new_item.replace("nin_shortcut", "conv_shortcut")
            new_item = self.shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

            mapping.append({"old": old_item, "new": new_item})

        return mapping


    def renew_attention_paths(self,old_list, n_shave_prefix_segments=0):
        """
        Updates paths inside attentions to the new naming scheme (local renaming)
        """
        mapping = []
        for old_item in old_list:
            new_item = old_item

            #         new_item = new_item.replace('norm.weight', 'group_norm.weight')
            #         new_item = new_item.replace('norm.bias', 'group_norm.bias')

            #         new_item = new_item.replace('proj_out.weight', 'proj_attn.weight')
            #         new_item = new_item.replace('proj_out.bias', 'proj_attn.bias')

            #         new_item = self.shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

            mapping.append({"old": old_item, "new": new_item})

        return mapping


    def renew_vae_attention_paths(self,old_list, n_shave_prefix_segments=0):
        """
        Updates paths inside attentions to the new naming scheme (local renaming)
        """
        mapping = []
        for old_item in old_list:
            new_item = old_item

            new_item = new_item.replace("norm.weight", "group_norm.weight")
            new_item = new_item.replace("norm.bias", "group_norm.bias")

            new_item = new_item.replace("q.weight", "query.weight")
            new_item = new_item.replace("q.bias", "query.bias")

            new_item = new_item.replace("k.weight", "key.weight")
            new_item = new_item.replace("k.bias", "key.bias")

            new_item = new_item.replace("v.weight", "value.weight")
            new_item = new_item.replace("v.bias", "value.bias")

            new_item = new_item.replace("proj_out.weight", "proj_attn.weight")
            new_item = new_item.replace("proj_out.bias", "proj_attn.bias")

            new_item = self.shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

            mapping.append({"old": old_item, "new": new_item})

        return mapping


    def assign_to_checkpoint(self,
        paths, checkpoint, old_checkpoint, attention_paths_to_split=None, additional_replacements=None, config=None
    ):
        """
        This does the final conversion step: take locally converted weights and apply a global renaming
        to them. It splits attention layers, and takes into account additional replacements
        that may arise.
        Assigns the weights to the new checkpoint.
        """
        assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

        # Splits the attention layers into three variables.
        if attention_paths_to_split is not None:
            for path, path_map in attention_paths_to_split.items():
                old_tensor = old_checkpoint[path]
                channels = old_tensor.shape[0] // 3

                target_shape = (-1, channels) if len(old_tensor.shape) == 3 else (-1)

                num_heads = old_tensor.shape[0] // config["num_head_channels"] // 3

                old_tensor = old_tensor.reshape((num_heads, 3 * channels // num_heads) + old_tensor.shape[1:])
                query, key, value = old_tensor.split(channels // num_heads, dim=1)

                checkpoint[path_map["query"]] = query.reshape(target_shape)
                checkpoint[path_map["key"]] = key.reshape(target_shape)
                checkpoint[path_map["value"]] = value.reshape(target_shape)

        for path in paths:
            new_path = path["new"]

            # These have already been assigned
            if attention_paths_to_split is not None and new_path in attention_paths_to_split:
                continue

            # Global renaming happens here
            new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
            new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
            new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

            if additional_replacements is not None:
                for replacement in additional_replacements:
                    new_path = new_path.replace(replacement["old"], replacement["new"])

            # proj_attn.weight has to be converted from conv 1D to linear
            if "proj_attn.weight" in new_path:
                checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
            else:
                checkpoint[new_path] = old_checkpoint[path["old"]]


    def conv_attn_to_linear(self,checkpoint):
        keys = list(checkpoint.keys())
        attn_keys = ["query.weight", "key.weight", "value.weight"]
        for key in keys:
            if ".".join(key.split(".")[-2:]) in attn_keys:
                if checkpoint[key].ndim > 2:
                    checkpoint[key] = checkpoint[key][:, :, 0, 0]
            elif "proj_attn.weight" in key:
                if checkpoint[key].ndim > 2:
                    checkpoint[key] = checkpoint[key][:, :, 0]


    def create_unet_diffusers_config(self,original_config, image_size: int):
        """
        Creates a config for the diffusers based on the config of the LDM model.
        """
        unet_params = original_config.model.params.unet_config.params
        vae_params = original_config.model.params.first_stage_config.params.ddconfig

        block_out_channels = [unet_params.model_channels * mult for mult in unet_params.channel_mult]

        down_block_types = []
        resolution = 1
        for i in range(len(block_out_channels)):
            block_type = "CrossAttnDownBlock2D" if resolution in unet_params.attention_resolutions else "DownBlock2D"
            down_block_types.append(block_type)
            if i != len(block_out_channels) - 1:
                resolution *= 2

        up_block_types = []
        for i in range(len(block_out_channels)):
            block_type = "CrossAttnUpBlock2D" if resolution in unet_params.attention_resolutions else "UpBlock2D"
            up_block_types.append(block_type)
            resolution //= 2

        vae_scale_factor = 2 ** (len(vae_params.ch_mult) - 1)

        head_dim = unet_params.num_heads if "num_heads" in unet_params else None
        use_linear_projection = (
            unet_params.use_linear_in_transformer if "use_linear_in_transformer" in unet_params else False
        )
        if use_linear_projection:
            # stable diffusion 2-base-512 and 2-768
            if head_dim is None:
                head_dim = [5, 10, 20, 20]

        config = dict(
            sample_size=image_size // vae_scale_factor,
            in_channels=unet_params.in_channels,
            out_channels=unet_params.out_channels,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            block_out_channels=tuple(block_out_channels),
            layers_per_block=unet_params.num_res_blocks,
            cross_attention_dim=unet_params.context_dim,
            attention_head_dim=head_dim,
            use_linear_projection=use_linear_projection,
        )

        return config


    def create_vae_diffusers_config(self,original_config, image_size: int):
        """
        Creates a config for the diffusers based on the config of the LDM model.
        """
        vae_params = original_config.model.params.first_stage_config.params.ddconfig
        _ = original_config.model.params.first_stage_config.params.embed_dim

        block_out_channels = [vae_params.ch * mult for mult in vae_params.ch_mult]
        down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
        up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

        config = dict(
            sample_size=image_size,
            in_channels=vae_params.in_channels,
            out_channels=vae_params.out_ch,
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            block_out_channels=tuple(block_out_channels),
            latent_channels=vae_params.z_channels,
            layers_per_block=vae_params.num_res_blocks,
        )
        return config


    def create_diffusers_schedular(self,original_config):
        schedular = DDIMScheduler(
            num_train_timesteps=original_config.model.params.timesteps,
            beta_start=original_config.model.params.linear_start,
            beta_end=original_config.model.params.linear_end,
            beta_schedule="scaled_linear",
        )
        return schedular


    def create_ldm_bert_config(self,original_config):
        bert_params = original_config.model.parms.cond_stage_config.params
        config = LDMBertConfig(
            d_model=bert_params.n_embed,
            encoder_layers=bert_params.n_layer,
            encoder_ffn_dim=bert_params.n_embed * 4,
        )
        return config


    def convert_ldm_unet_checkpoint(self,checkpoint, config, path=None, extract_ema=False):
        """
        Takes a state dict and a config, and returns a converted checkpoint.
        """

        # extract state_dict for UNet
        unet_state_dict = {}
        keys = list(checkpoint.keys())

        unet_key = "model.diffusion_model."
        # at least a 100 parameters have to start with `model_ema` in order for the checkpoint to be EMA
        if sum(k.startswith("model_ema") for k in keys) > 100:
            print(f"Checkpoint {path} has both EMA and non-EMA weights.")
            if extract_ema:
                print(
                    "In this conversion only the EMA weights are extracted. If you want to instead extract the non-EMA"
                    " weights (useful to continue fine-tuning), please make sure to remove the `--extract_ema` flag."
                )
                for key in keys:
                    if key.startswith("model.diffusion_model"):
                        flat_ema_key = "model_ema." + "".join(key.split(".")[1:])
                        unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(flat_ema_key)
            else:
                print(
                    "In this conversion only the non-EMA weights are extracted. If you want to instead extract the EMA"
                    " weights (usually better for inference), please make sure to add the `--extract_ema` flag."
                )

        for key in keys:
            if key.startswith(unet_key):
                unet_state_dict[key.replace(unet_key, "")] = checkpoint.pop(key)

        new_checkpoint = {}

        new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
        new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
        new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
        new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

        new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
        new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

        new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
        new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
        new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
        new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

        # Retrieves the keys for the input blocks only
        num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
        input_blocks = {
            layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}" in key]
            for layer_id in range(num_input_blocks)
        }

        # Retrieves the keys for the middle blocks only
        num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
        middle_blocks = {
            layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
            for layer_id in range(num_middle_blocks)
        }

        # Retrieves the keys for the output blocks only
        num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
        output_blocks = {
            layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}" in key]
            for layer_id in range(num_output_blocks)
        }

        for i in range(1, num_input_blocks):
            block_id = (i - 1) // (config["layers_per_block"] + 1)
            layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

            resnets = [
                key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
            ]
            attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

            if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
                new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                    f"input_blocks.{i}.0.op.weight"
                )
                new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                    f"input_blocks.{i}.0.op.bias"
                )

            paths = self.renew_resnet_paths(resnets)
            meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}
            self.assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            if len(attentions):
                paths = self.renew_attention_paths(attentions)
                meta_path = {"old": f"input_blocks.{i}.1", "new": f"down_blocks.{block_id}.attentions.{layer_in_block_id}"}
                self.assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )

        resnet_0 = middle_blocks[0]
        attentions = middle_blocks[1]
        resnet_1 = middle_blocks[2]

        resnet_0_paths = self.renew_resnet_paths(resnet_0)
        self.assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

        resnet_1_paths = self.renew_resnet_paths(resnet_1)
        self.assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

        attentions_paths = self.renew_attention_paths(attentions)
        meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
        self.assign_to_checkpoint(
            attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        for i in range(num_output_blocks):
            block_id = i // (config["layers_per_block"] + 1)
            layer_in_block_id = i % (config["layers_per_block"] + 1)
            output_block_layers = [self.shave_segments(name, 2) for name in output_blocks[i]]
            output_block_list = {}

            for layer in output_block_layers:
                layer_id, layer_name = layer.split(".")[0], self.shave_segments(layer, 1)
                if layer_id in output_block_list:
                    output_block_list[layer_id].append(layer_name)
                else:
                    output_block_list[layer_id] = [layer_name]

            if len(output_block_list) > 1:
                resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
                attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

                resnet_0_paths = self.renew_resnet_paths(resnets)
                paths = self.renew_resnet_paths(resnets)

                meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}
                self.assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )

                if ["conv.weight", "conv.bias"] in output_block_list.values():
                    index = list(output_block_list.values()).index(["conv.weight", "conv.bias"])
                    new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                        f"output_blocks.{i}.{index}.conv.weight"
                    ]
                    new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                        f"output_blocks.{i}.{index}.conv.bias"
                    ]

                    # Clear attentions as they have been attributed above.
                    if len(attentions) == 2:
                        attentions = []

                if len(attentions):
                    paths = self.renew_attention_paths(attentions)
                    meta_path = {
                        "old": f"output_blocks.{i}.1",
                        "new": f"up_blocks.{block_id}.attentions.{layer_in_block_id}",
                    }
                    self.assign_to_checkpoint(
                        paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                    )
            else:
                resnet_0_paths = self.renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
                for path in resnet_0_paths:
                    old_path = ".".join(["output_blocks", str(i), path["old"]])
                    new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                    new_checkpoint[new_path] = unet_state_dict[old_path]

        return new_checkpoint


    def convert_ldm_vae_checkpoint(self,checkpoint, config):
        # extract state dict for VAE
        vae_state_dict = {}
        vae_key = "first_stage_model."
        keys = list(checkpoint.keys())
        for key in keys:
            if key.startswith(vae_key):
                vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

        new_checkpoint = {}

        new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
        new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
        new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
        new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
        new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
        new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

        new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
        new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
        new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
        new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
        new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
        new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

        new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
        new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
        new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
        new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]

        # Retrieves the keys for the encoder down blocks only
        num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
        down_blocks = {
            layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
        }

        # Retrieves the keys for the decoder up blocks only
        num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
        up_blocks = {
            layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
        }

        for i in range(num_down_blocks):
            resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

            if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
                new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                    f"encoder.down.{i}.downsample.conv.weight"
                )
                new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                    f"encoder.down.{i}.downsample.conv.bias"
                )

            paths = self.renew_vae_resnet_paths(resnets)
            meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
            self.assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

        mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
        num_mid_res_blocks = 2
        for i in range(1, num_mid_res_blocks + 1):
            resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

            paths = self.renew_vae_resnet_paths(resnets)
            meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
            self.assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

        mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
        paths = self.renew_vae_attention_paths(mid_attentions)
        meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
        self.assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
        self.conv_attn_to_linear(new_checkpoint)

        for i in range(num_up_blocks):
            block_id = num_up_blocks - 1 - i
            resnets = [
                key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
            ]

            if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
                new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                    f"decoder.up.{block_id}.upsample.conv.weight"
                ]
                new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                    f"decoder.up.{block_id}.upsample.conv.bias"
                ]

            paths = self.renew_vae_resnet_paths(resnets)
            meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
            self.assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

        mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
        num_mid_res_blocks = 2
        for i in range(1, num_mid_res_blocks + 1):
            resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

            paths = self.renew_vae_resnet_paths(resnets)
            meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
            self.assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

        mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
        paths = self.renew_vae_attention_paths(mid_attentions)
        meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
        self.assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
        self.conv_attn_to_linear(new_checkpoint)
        return new_checkpoint


    def convert_ldm_bert_checkpoint(self,checkpoint, config):
        def _copy_attn_layer(hf_attn_layer, pt_attn_layer):
            hf_attn_layer.q_proj.weight.data = pt_attn_layer.to_q.weight
            hf_attn_layer.k_proj.weight.data = pt_attn_layer.to_k.weight
            hf_attn_layer.v_proj.weight.data = pt_attn_layer.to_v.weight

            hf_attn_layer.out_proj.weight = pt_attn_layer.to_out.weight
            hf_attn_layer.out_proj.bias = pt_attn_layer.to_out.bias

        def _copy_linear(hf_linear, pt_linear):
            hf_linear.weight = pt_linear.weight
            hf_linear.bias = pt_linear.bias

        def _copy_layer(hf_layer, pt_layer):
            # copy layer norms
            _copy_linear(hf_layer.self_attn_layer_norm, pt_layer[0][0])
            _copy_linear(hf_layer.final_layer_norm, pt_layer[1][0])

            # copy attn
            _copy_attn_layer(hf_layer.self_attn, pt_layer[0][1])

            # copy MLP
            pt_mlp = pt_layer[1][1]
            _copy_linear(hf_layer.fc1, pt_mlp.net[0][0])
            _copy_linear(hf_layer.fc2, pt_mlp.net[2])

        def _copy_layers(hf_layers, pt_layers):
            for i, hf_layer in enumerate(hf_layers):
                if i != 0:
                    i += i
                pt_layer = pt_layers[i : i + 2]
                _copy_layer(hf_layer, pt_layer)

        hf_model = LDMBertModel(config).eval()

        # copy  embeds
        hf_model.model.embed_tokens.weight = checkpoint.transformer.token_emb.weight
        hf_model.model.embed_positions.weight.data = checkpoint.transformer.pos_emb.emb.weight

        # copy layer norm
        _copy_linear(hf_model.model.layer_norm, checkpoint.transformer.norm)

        # copy hidden layers
        _copy_layers(hf_model.model.layers, checkpoint.transformer.attn_layers.layers)

        _copy_linear(hf_model.to_logits, checkpoint.transformer.to_logits)

        return hf_model


    def convert_ldm_clip_checkpoint(self,checkpoint):
        text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        keys = list(checkpoint.keys())

        text_model_dict = {}

        for key in keys:
            if key.startswith("cond_stage_model.transformer"):
                text_model_dict[key[len("cond_stage_model.transformer.") :]] = checkpoint[key]

        text_model.load_state_dict(text_model_dict)

        return text_model
    '''
    def convert_paint_by_example_checkpoint(checkpoint):
        config = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14")
        model = PaintByExampleImageEncoder(config)

        keys = list(checkpoint.keys())

        text_model_dict = {}

        for key in keys:
            if key.startswith("cond_stage_model.transformer"):
                text_model_dict[key[len("cond_stage_model.transformer.") :]] = checkpoint[key]

        # load clip vision
        model.model.load_state_dict(text_model_dict)

        # load mapper
        keys_mapper = {
            k[len("cond_stage_model.mapper.res") :]: v
            for k, v in checkpoint.items()
            if k.startswith("cond_stage_model.mapper")
        }

        MAPPING = {
            "attn.c_qkv": ["attn1.to_q", "attn1.to_k", "attn1.to_v"],
            "attn.c_proj": ["attn1.to_out.0"],
            "ln_1": ["norm1"],
            "ln_2": ["norm3"],
            "mlp.c_fc": ["ff.net.0.proj"],
            "mlp.c_proj": ["ff.net.2"],
        }

        mapped_weights = {}
        for key, value in keys_mapper.items():
            prefix = key[: len("blocks.i")]
            suffix = key.split(prefix)[-1].split(".")[-1]
            name = key.split(prefix)[-1].split(suffix)[0][1:-1]
            mapped_names = MAPPING[name]

            num_splits = len(mapped_names)
            for i, mapped_name in enumerate(mapped_names):
                new_name = ".".join([prefix, mapped_name, suffix])
                shape = value.shape[0] // num_splits
                mapped_weights[new_name] = value[i * shape : (i + 1) * shape]

        model.mapper.load_state_dict(mapped_weights)

        # load final layer norm
        model.final_layer_norm.load_state_dict(
            {
                "bias": checkpoint["cond_stage_model.final_ln.bias"],
                "weight": checkpoint["cond_stage_model.final_ln.weight"],
            }
        )

        # load final proj
        model.proj_out.load_state_dict(
            {
                "bias": checkpoint["proj_out.bias"],
                "weight": checkpoint["proj_out.weight"],
            }
        )

        # load uncond vector
        model.uncond_vector.data = torch.nn.Parameter(checkpoint["learnable_vector"])
        return model

    '''
    def convert_open_clip_checkpoint(self,checkpoint):
        text_model = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="text_encoder")

        # SKIP for now - need openclip -> HF conversion script here
        #    keys = list(checkpoint.keys())
        #
        #    text_model_dict = {}
        #    for key in keys:
        #        if key.startswith("cond_stage_model.model.transformer"):
        #            text_model_dict[key[len("cond_stage_model.model.transformer.") :]] = checkpoint[key]
        #
        #    text_model.load_state_dict(text_model_dict)

        return text_model


    def main(self):
        image_size = self.img_size
        prediction_type = self.prediction_type
        original_config_file = self.original_config_file
        num_in_channels = self.num_in_channels
        scheduler_type = self.scheduler_type
        pipeline_type = self.pipeline_type
        extract_ema = self.extract_ema
        checkpoint = torch.load(self.checkpoint_path)
        #if checkpoint has 'global_step' key, it is a HF checkpoint
        if 'global_step' not in checkpoint:
            
            if image_size == 768:
                print('test')
                global_step = 999999
            if image_size == 512:
                global_step = 875000
        else:
            global_step = checkpoint["global_step"]
        checkpoint = checkpoint["state_dict"]
        dump_path = self.output_path

        if original_config_file is None:
            key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"

            if key_name in checkpoint and checkpoint[key_name].shape[-1] == 1024:
                # model_type = "v2"
                #os.system(
                #    "wget https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"
                #)
                #use requests instead of wget
                
                url = 'https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml'
                r = requests.get(url, allow_redirects=True)
                open('v2-inference-v.yaml', 'wb').write(r.content)
                original_config_file = "./v2-inference-v.yaml"
            else:
                url = 'https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml'
                r = requests.get(url, allow_redirects=True)
                open('v1-inference.yaml', 'wb').write(r.content)
                # model_type = "v1"
                #os.system(
                #    "wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
                #)
                original_config_file = "./v1-inference.yaml"

        original_config = OmegaConf.load(original_config_file)

        if num_in_channels is not None:
            original_config["model"]["params"]["unet_config"]["params"]["in_channels"] = num_in_channels

        if (
            "parameterization" in original_config["model"]["params"]
            and original_config["model"]["params"]["parameterization"] == "v"
        ):
            if prediction_type is None:
                # NOTE: For stable diffusion 2 base it is recommended to pass `prediction_type=="epsilon"`
                # as it relies on a brittle global step parameter here
                prediction_type = "epsilon" if global_step == 875000 else "v_prediction"
            if image_size is None:
                # NOTE: For stable diffusion 2 base one has to pass `image_size==512`
                # as it relies on a brittle global step parameter here
                image_size = 512 if global_step == 875000 else 768
        else:
            if prediction_type is None:
                prediction_type = "epsilon"
            if image_size is None:
                image_size = 512

        num_train_timesteps = original_config.model.params.timesteps
        beta_start = original_config.model.params.linear_start
        beta_end = original_config.model.params.linear_end

        scheduler = DDIMScheduler(
            beta_end=beta_end,
            beta_schedule="scaled_linear",
            beta_start=beta_start,
            num_train_timesteps=num_train_timesteps,
            steps_offset=1,
            clip_sample=False,
            set_alpha_to_one=False,
            prediction_type=prediction_type,
        )
        if scheduler_type == "pndm":
            config = dict(scheduler.config)
            config["skip_prk_steps"] = True
            scheduler = PNDMScheduler.from_config(config)
        elif scheduler_type == "lms":
            scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "heun":
           scheduler = HeunDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "euler":
            scheduler = EulerDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "euler-ancestral":
            scheduler = EulerAncestralDiscreteScheduler.from_config(scheduler.config)
        elif scheduler_type == "dpm":
            scheduler = DPMSolverMultistepScheduler.from_config(scheduler.config)
        elif scheduler_type == "ddim":
            scheduler = scheduler
        #else:
            #raise ValueError(f"Scheduler of type {scheduler_type} doesn't exist!")

        # Convert the UNet2DConditionModel model.
        unet_config = self.create_unet_diffusers_config(original_config, image_size=image_size)
        unet = UNet2DConditionModel(**unet_config)

        converted_unet_checkpoint = self.convert_ldm_unet_checkpoint(
            checkpoint, unet_config, path=self.checkpoint_path, extract_ema=extract_ema
        )

        unet.load_state_dict(converted_unet_checkpoint)

        # Convert the VAE model.
        vae_config = self.create_vae_diffusers_config(original_config, image_size=image_size)
        converted_vae_checkpoint = self.convert_ldm_vae_checkpoint(checkpoint, vae_config)

        vae = AutoencoderKL(**vae_config)
        vae.load_state_dict(converted_vae_checkpoint)

        # Convert the text model.
        model_type = pipeline_type
        if model_type is None:
            model_type = original_config.model.params.cond_stage_config.target.split(".")[-1]

        if model_type == "FrozenOpenCLIPEmbedder":
            text_model = self.convert_open_clip_checkpoint(checkpoint)
            tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")
            pipe = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_model,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )
        elif model_type == "FrozenCLIPEmbedder":
            text_model = self.convert_ldm_clip_checkpoint(checkpoint)
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            safety_checker = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
            feature_extractor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")
            pipe = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_model,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=safety_checker,
                feature_extractor=feature_extractor,
            )
        else:
            text_config = self.create_ldm_bert_config(original_config)
            text_model = self.convert_ldm_bert_checkpoint(checkpoint, text_config)
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
            pipe = LDMTextToImagePipeline(vqvae=vae, bert=text_model, tokenizer=tokenizer, unet=unet, scheduler=scheduler)
        '''
        elif model_type == "PaintByExample":
            vision_model = convert_paint_by_example_checkpoint(checkpoint)
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            feature_extractor = AutoFeatureExtractor.from_pretrained("CompVis/stable-diffusion-safety-checker")
            pipe = PaintByExamplePipeline(
                vae=vae,
                image_encoder=vision_model,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=feature_extractor,
            )
        '''
        pipe.save_pretrained(dump_path)

class Convert_Diffusers_to_SD():
    def __init__(self,model_path=None, output_path=None):
    # Script for converting a HF Diffusers saved pipeline to a Stable Diffusion checkpoint.
        # *Only* converts the UNet, VAE, and Text Encoder.
        # Does not convert optimizer state or any other thing.

        # =================#
        # UNet Conversion #
        # =================#

        unet_conversion_map = [
            # (stable-diffusion, HF Diffusers)
            ("time_embed.0.weight", "time_embedding.linear_1.weight"),
            ("time_embed.0.bias", "time_embedding.linear_1.bias"),
            ("time_embed.2.weight", "time_embedding.linear_2.weight"),
            ("time_embed.2.bias", "time_embedding.linear_2.bias"),
            ("input_blocks.0.0.weight", "conv_in.weight"),
            ("input_blocks.0.0.bias", "conv_in.bias"),
            ("out.0.weight", "conv_norm_out.weight"),
            ("out.0.bias", "conv_norm_out.bias"),
            ("out.2.weight", "conv_out.weight"),
            ("out.2.bias", "conv_out.bias"),
        ]

        unet_conversion_map_resnet = [
            # (stable-diffusion, HF Diffusers)
            ("in_layers.0", "norm1"),
            ("in_layers.2", "conv1"),
            ("out_layers.0", "norm2"),
            ("out_layers.3", "conv2"),
            ("emb_layers.1", "time_emb_proj"),
            ("skip_connection", "conv_shortcut"),
        ]

        unet_conversion_map_layer = []
        # hardcoded number of downblocks and resnets/attentions...
        # would need smarter logic for other networks.
        for i in range(4):
            # loop over downblocks/upblocks

            for j in range(2):
                # loop over resnets/attentions for downblocks
                hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
                sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
                unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

                if i < 3:
                    # no attention layers in down_blocks.3
                    hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                    sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                    unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

            for j in range(3):
                # loop over resnets/attentions for upblocks
                hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
                sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
                unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

                if i > 0:
                    # no attention layers in up_blocks.0
                    hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
                    sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
                    unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

            if i < 3:
                # no downsample in down_blocks.3
                hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
                sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
                unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

                # no upsample in up_blocks.3
                hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
                sd_upsample_prefix = f"output_blocks.{3*i + 2}.{1 if i == 0 else 2}."
                unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

        hf_mid_atn_prefix = "mid_block.attentions.0."
        sd_mid_atn_prefix = "middle_block.1."
        unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

        for j in range(2):
            hf_mid_res_prefix = f"mid_block.resnets.{j}."
            sd_mid_res_prefix = f"middle_block.{2*j}."
            unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))


        def convert_unet_state_dict(unet_state_dict):
            # buyer beware: this is a *brittle* function,
            # and correct output requires that all of these pieces interact in
            # the exact order in which I have arranged them.
            mapping = {k: k for k in unet_state_dict.keys()}
            for sd_name, hf_name in unet_conversion_map:
                mapping[hf_name] = sd_name
            for k, v in mapping.items():
                if "resnets" in k:
                    for sd_part, hf_part in unet_conversion_map_resnet:
                        v = v.replace(hf_part, sd_part)
                    mapping[k] = v
            for k, v in mapping.items():
                for sd_part, hf_part in unet_conversion_map_layer:
                    v = v.replace(hf_part, sd_part)
                mapping[k] = v
            new_state_dict = {v: unet_state_dict[k] for k, v in mapping.items()}
            return new_state_dict


        # ================#
        # VAE Conversion #
        # ================#

        vae_conversion_map = [
            # (stable-diffusion, HF Diffusers)
            ("nin_shortcut", "conv_shortcut"),
            ("norm_out", "conv_norm_out"),
            ("mid.attn_1.", "mid_block.attentions.0."),
        ]

        for i in range(4):
            # down_blocks have two resnets
            for j in range(2):
                hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
                sd_down_prefix = f"encoder.down.{i}.block.{j}."
                vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

            if i < 3:
                hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
                sd_downsample_prefix = f"down.{i}.downsample."
                vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

                hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
                sd_upsample_prefix = f"up.{3-i}.upsample."
                vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

            # up_blocks have three resnets
            # also, up blocks in hf are numbered in reverse from sd
            for j in range(3):
                hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
                sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
                vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

        # this part accounts for mid blocks in both the encoder and the decoder
        for i in range(2):
            hf_mid_res_prefix = f"mid_block.resnets.{i}."
            sd_mid_res_prefix = f"mid.block_{i+1}."
            vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))


        vae_conversion_map_attn = [
            # (stable-diffusion, HF Diffusers)
            ("norm.", "group_norm."),
            ("q.", "query."),
            ("k.", "key."),
            ("v.", "value."),
            ("proj_out.", "proj_attn."),
        ]


        def reshape_weight_for_sd(w):
            # convert HF linear weights to SD conv2d weights
            return w.reshape(*w.shape, 1, 1)


        def convert_vae_state_dict(vae_state_dict):
            mapping = {k: k for k in vae_state_dict.keys()}
            for k, v in mapping.items():
                for sd_part, hf_part in vae_conversion_map:
                    v = v.replace(hf_part, sd_part)
                mapping[k] = v
            for k, v in mapping.items():
                if "attentions" in k:
                    for sd_part, hf_part in vae_conversion_map_attn:
                        v = v.replace(hf_part, sd_part)
                    mapping[k] = v
            new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
            weights_to_convert = ["q", "k", "v", "proj_out"]
            for k, v in new_state_dict.items():
                for weight_name in weights_to_convert:
                    if f"mid.attn_1.{weight_name}.weight" in k:
                        print(f"Reshaping {k} for SD format")
                        new_state_dict[k] = reshape_weight_for_sd(v)
            return new_state_dict


        # =========================#
        # Text Encoder Conversion #
        # =========================#

        import re
        textenc_conversion_lst = [
            # (stable-diffusion, HF Diffusers)
            ('resblocks.','text_model.encoder.layers.'),
            ('ln_1','layer_norm1'),
            ('ln_2','layer_norm2'),
            ('.c_fc.','.fc1.'),
            ('.c_proj.','.fc2.'),
            ('.attn','.self_attn'),
            ('ln_final.','transformer.text_model.final_layer_norm.'),
            ('token_embedding.weight','transformer.text_model.embeddings.token_embedding.weight'),
            ('positional_embedding','transformer.text_model.embeddings.position_embedding.weight')
        ]
        protected = {re.escape(x[1]):x[0] for x in textenc_conversion_lst}
        textenc_pattern = re.compile("|".join(protected.keys()))

        # Ordering is from https://github.com/pytorch/pytorch/blob/master/test/cpp/api/modules.cpp
        code2idx = {'q':0,'k':1,'v':2}

        def convert_text_enc_state_dict_v20(text_enc_dict:dict[str, torch.Tensor]):
            new_state_dict = {}
            capture_qkv_weight = {}
            capture_qkv_bias = {}
            for k,v in text_enc_dict.items():
                if k.endswith('.self_attn.q_proj.weight') or k.endswith('.self_attn.k_proj.weight') or k.endswith('.self_attn.v_proj.weight'):
                    k_pre = k[:-len('.q_proj.weight')]
                    k_code = k[-len('q_proj.weight')]
                    if k_pre not in capture_qkv_weight:
                        capture_qkv_weight[k_pre] = [None,None,None]
                    capture_qkv_weight[k_pre][code2idx[k_code]] = v
                    continue

                if k.endswith('.self_attn.q_proj.bias') or k.endswith('.self_attn.k_proj.bias') or k.endswith('.self_attn.v_proj.bias'):
                    k_pre = k[:-len('.q_proj.bias')]
                    k_code = k[-len('q_proj.bias')]
                    if k_pre not in capture_qkv_bias:
                        capture_qkv_bias[k_pre] = [None,None,None]
                    capture_qkv_bias[k_pre][code2idx[k_code]] = v
                    continue

                relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k)
                #        if relabelled_key != k:
                #            print(f"{k} -> {relabelled_key}")

                new_state_dict[relabelled_key] = v

            for k_pre,tensors in capture_qkv_weight.items():
                if None in tensors:
                    raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
                relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
                new_state_dict[relabelled_key+'.in_proj_weight'] = torch.cat(tensors)

            for k_pre,tensors in capture_qkv_bias.items():
                if None in tensors:
                    raise Exception("CORRUPTED MODEL: one of the q-k-v values for the text encoder was missing")
                relabelled_key = textenc_pattern.sub(lambda m: protected[re.escape(m.group(0))], k_pre)
                new_state_dict[relabelled_key+'.in_proj_bias'] = torch.cat(tensors)

            return new_state_dict


        def convert_text_enc_state_dict(text_enc_dict:dict[str, torch.Tensor]):
            return text_enc_dict
        
        #IS_V20_MODEL = True
        #model_path = self.input_model_path_entry.get()
        def main(model_path:str, output_path:str):
            assert model_path is not None, "Must provide a model path!"

            assert output_path is not None, "Must provide a checkpoint path!"
            #create a progress bar
            progress = 0
            #tk inter progress bar
            # load the model
            unet_path = osp.join(model_path, "unet", "diffusion_pytorch_model.bin")
            vae_path = osp.join(model_path, "vae", "diffusion_pytorch_model.bin")
            text_enc_path = osp.join(model_path, "text_encoder", "pytorch_model.bin")

            # Convert the UNet model
            unet_state_dict = torch.load(unet_path, map_location="cpu")
            unet_state_dict = convert_unet_state_dict(unet_state_dict)
            unet_state_dict = {"model.diffusion_model." + k: v for k, v in unet_state_dict.items()}

            # Convert the VAE model
            vae_state_dict = torch.load(vae_path, map_location="cpu")
            vae_state_dict = convert_vae_state_dict(vae_state_dict)
            vae_state_dict = {"first_stage_model." + k: v for k, v in vae_state_dict.items()}

            # Convert the text encoder model
            text_enc_dict = torch.load(text_enc_path, map_location="cpu")

        # Easiest way to identify v2.0 model seems to be that the text encoder (OpenCLIP) is deeper
            is_v20_model = "text_model.encoder.layers.22.layer_norm2.bias" in text_enc_dict

            if is_v20_model:
                # Need to add the tag 'transformer' in advance so we can knock it out from the final layer-norm
                text_enc_dict = {"transformer." + k: v for k, v in text_enc_dict.items()} 
                text_enc_dict = convert_text_enc_state_dict_v20(text_enc_dict)
                text_enc_dict = {"cond_stage_model.model." + k: v for k, v in text_enc_dict.items()}
            else:
                text_enc_dict = convert_text_enc_state_dict(text_enc_dict)
                text_enc_dict = {"cond_stage_model.transformer." + k: v for k, v in text_enc_dict.items()}

            # Put together new checkpoint
            state_dict = {**unet_state_dict, **vae_state_dict, **text_enc_dict}
            #if args.half:
            #    state_dict = {k: v.half() for k, v in state_dict.items()}
            state_dict = {"state_dict": state_dict}
            torch.save(state_dict, output_path)
        return main(model_path, output_path)