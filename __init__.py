import os
import json

from gradio_client import Client
from PIL import Image, ImageOps
from time import perf_counter
from dotenv import load_dotenv

import numpy as np
import torch

load_dotenv()

HF_KEY = os.getenv("HF_KEY")
HF_FLUX_DEV = "black-forest-labs/FLUX.1-dev"
HF_FLUX_SCHNELL = "black-forest-labs/FLUX.1-schnell"
BASE_PATH = os.path.dirname(os.path.realpath(__file__))


class HFFlux:
    '''
    Generate image from text with FLUX.1 model
    '''
    models = ['DEV', 'SCHNELL']
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (s.models, {"default": "DEV"}),
                "seed": ("INT", {"default": 0}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 1}),
                "CFG": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 16.0, "step":0.1, "round": 0.01}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 50, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    OUTPUT_NODE = True

    CATEGORY = "NX_Nodes"

    def run(self, prompt='', model='DEV', seed=0, width=1024, height=1024, CFG=3.5, steps=25): 
        model_path = HF_FLUX_DEV
        if model == 'SCHNELL':
            model_path = HF_FLUX_SCHNELL
        client = Client(model_path, hf_token=HF_KEY)

        params = {"prompt": prompt, "seed": seed, "randomize_seed": False, "width": width, "height": height, "num_inference_steps": steps, "api_name": "/infer"}
        if model == 'DEV':
            params["guidance_scale"] = CFG

        # print(client.view_api())

        # result = client.predict(
        #     prompt=prompt,
        #     seed=seed,
        #     randomize_seed=False,
        #     width=width,
        #     height=height,
        #     guidance_scale=CFG,
        #     num_inference_steps=steps,
        #     api_name="/infer"
        # )
        result = client.predict(**params)

        img = Image.open(result[0])
        # img = Image.open(BASE_PATH + "/exemple.png")
        img = ImageOps.exif_transpose(img)
        image = img.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]


        return (image,)


NODE_CLASS_MAPPINGS = {
    "HFFlux": HFFlux
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HFFlux": "🔀 Hugging Face Flux",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]