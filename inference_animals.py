# coding: utf-8


import os
import os.path as osp
import tyro
import subprocess
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig#xpose
from src.live_portrait_pipeline_animal import LivePortraitPipelineAnimal


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})




def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)
    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)
    live_portrait_pipeline_animal = LivePortraitPipelineAnimal(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )
    # run
    live_portrait_pipeline_animal.execute(args)

if __name__ == "__main__":
    main()
