import torch
from diffusers import DreamLitePipeline

pipe = DreamLitePipeline.from_pretrained(
    "carlofkl/DreamLite-base", torch_dtype=torch.bfloat16
).to("cuda")
image = pipe("a corgi astronaut", num_inference_steps=28).images[0]
