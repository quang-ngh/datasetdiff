#   Update 3/2/2023
import torch
from diffusers import StableDiffusionPipeline

if __name__ == '__main__':

    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"


    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)  

    