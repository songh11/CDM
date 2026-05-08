import torch
from diffusers import LongCatImagePipeline

device = torch.device('cuda:0')

pipe = LongCatImagePipeline.from_pretrained(pretrained_model_name_or_path="byliutao/Longcat-Image-Turbo", torch_dtype= torch.bfloat16 )
pipe.to(device, torch.bfloat16)
prompt = 'a photo of a cat.'
image = pipe(
    prompt,
    height=1024,
    width=1024,
    num_inference_steps=4,
    sigmas=[1.0, 0.75, 0.5, 0.25],
    guidance_scale=1.0,
    generator = torch.Generator("cuda:0").manual_seed(42),
    enable_prompt_rewrite=False
).images[0]
image.save("output.png")
