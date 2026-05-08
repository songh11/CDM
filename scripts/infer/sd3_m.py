import torch
from diffusers import StableDiffusion3Pipeline


prompt = "A photo of cat saying hello."


print("Loading teacher model...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    "byliutao/stable-diffusion-3-medium-turbo", 
)
pipe.to("cuda:0", torch.bfloat16)

image = pipe(
    prompt,
    height=1024,
    width=1024,
    num_inference_steps=4,
    sigmas=[1.0, 0.75, 0.5, 0.25],
    guidance_scale=1.0,
    generator = torch.Generator("cuda:0").manual_seed(43)
).images[0]

image.save("output.png")

