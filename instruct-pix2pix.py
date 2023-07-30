import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import os
import time
import secrets

# https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/2144
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

model_id = "timbrooks/instruct-pix2pix"
pipe = ""
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

def download_image(path):
    #image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.Image.open(path)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

while True:
    path = "D:/Code/ml_tools/instruct-pix2pix/in/in.jpg"
    new_name = secrets.token_hex(16)
    path_processed = "D:/Code/ml_tools/instruct-pix2pix/processed/" + new_name + ".jpg"

    sec = 5
    if not os.path.isfile(path):
        print("sleeping for " + str(sec) + "s, place image now at in/in.jpg")
        time.sleep(sec)
    else:
        prompt = input("prompt> ") # "turn him into cyborg"
        if prompt == "exit":
            break
        inference_steps = int(input("inference steps> ")) # 10
        guidance_scale = float(input("guidance scale> ")) # 1.0

        image = download_image(path)
        images = pipe(prompt, image=image, num_inference_steps=inference_steps, image_guidance_scale=guidance_scale).images
        images[0].save("D:/Code/ml_tools/instruct-pix2pix/out/" + new_name + ".jpg")
        os.replace(path, path_processed)
        time.sleep(1)
        print("moved input file to processed/")
