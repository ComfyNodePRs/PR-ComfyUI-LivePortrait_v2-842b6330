import os

import torch
import yaml
import folder_paths
import comfy.model_management as mm
import comfy.utils
import numpy as np
import cv2
from tqdm import tqdm
import gc
from PIL import Image
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
import shutil

import os
import os.path as osp
import tyro
import subprocess
from .src.config.argument_config import ArgumentConfig
from .src.config.inference_config import InferenceConfig
from .src.config.crop_config import CropConfig#xpose
from .src.live_portrait_pipeline_animal import LivePortraitPipelineAnimal



class InferenceConfig:
    def __init__(
        self,
        
        flag_use_half_precision=True,
        flag_lip_zero=True,
        lip_zero_threshold=0.03,
        flag_eye_retargeting=False,
        flag_lip_retargeting=False,
        flag_stitching=True,
        input_shape=(256, 256),
        device_id=0,
        flag_do_rot=True,
        **kwargs,
    ):
        self.flag_use_half_precision = flag_use_half_precision
        self.flag_lip_zero = flag_lip_zero
        self.lip_zero_threshold = lip_zero_threshold
        self.flag_eye_retargeting = flag_eye_retargeting
        self.flag_lip_retargeting = flag_lip_retargeting
        self.flag_stitching = flag_stitching
        self.input_shape = input_shape
        self.device_id = device_id
        self.flag_do_rot = flag_do_rot
        
        
def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})
 
class LivePortraitProcess_animal:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {

            
            "source_image": ("IMAGE",),
            "video": ("STRING", {"placeholder": "X://insert/path/here.mp4",}),
            
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        
        
    )
    RETURN_NAMES = (
        "images output",
        
    )
    FUNCTION = "process"
    CATEGORY = "LivePortrait"

    def process(
        self,
        source_image,
        
        **kwargs
        
    ):
        
        a = kwargs['video']
        a = a + ".mp4"
        a = "custom_nodes/ComfyUI-LivePortrait_v2/assets/examples/driving/" + a
        
        
        source_mp4_path = a
        destination_mp4_path = "custom_nodes/ComfyUI-LivePortrait_v2/assets/examples/driving/d9.mp4"
        shutil.copy(source_mp4_path, destination_mp4_path)
        
        for (batch_number, image) in enumerate(source_image):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
        
            metadata = PngInfo()
                # if prompt is not None:
                #     metadata.add_text("prompt", json.dumps(prompt))
                # if extra_pnginfo is not None:
                #     for x in extra_pnginfo:
                #         metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            #filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            #file = f"{filename_with_batch_num}_{counter:05}_.png"
            
            img.save("custom_nodes/ComfyUI-LivePortrait_v2/assets/examples/source/s12.jpg", pnginfo=metadata, compress_level=4)
        #raise ValueError(source_image)
        
        
        
        
        #raise ValueError(drive_images)
        # drive_images = drive_images * 256
        # drive_images.to(torch.int8)
        # drive_images.numpy()
        # list = drive_images.tolist()
        #raise ValueError(drive_images)
        
        #raise ValueError(list[0])
        #raise ValueError(list[0])
        
        
        
        
        
        tyro.extras.set_accent_color("bright_cyan")
        
        args = tyro.cli(ArgumentConfig)
        #raise ValueError(1)
        
        
        
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)
        crop_cfg = partial_fields(CropConfig, args.__dict__)
        
        live_portrait_pipeline_animal = LivePortraitPipelineAnimal(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )
    # run
        
        result = live_portrait_pipeline_animal.execute(args)
        
        #result = torch.tensor(result, dtype=torch.int16)
        
        #raise ValueError(result)
        
        
        
        
        
        return (result,)
    

        
        

NODE_CLASS_MAPPINGS = { 
    "LivePortraitProcess_animal": LivePortraitProcess_animal,
  

}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LivePortraitProcess_animal": "LivePortraitProcess_animal",

    }
