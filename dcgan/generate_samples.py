from __future__ import print_function
import argparse
import os
import random

import torch
import torchvision.utils as vutils

from dcgan_model import Generator, validate_image_size


parser = argparse.ArgumentParser()
parser.add_argument("--netG", required=True, help="path to Generator checkpoint")
parser.add_argument("--outf", default="./generated_samples", help="folder to output generated images")
parser.add_argument("--num-samples", type=int, default=10, help="number of samples to generate")
parser.add_argument("--manualSeed", type=int, help="manual seed")
parser.add_argument("--nz", type=int, default=100, help="size of the latent z vector")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters")
parser.add_argument("--imageSize", type=int, default=64, help="generated image size, kept for interface consistency")
opt = parser.parse_args()
print(opt)

if opt.num_samples < 1:
    raise ValueError("--num-samples must be at least 1")

os.makedirs(opt.outf, exist_ok=True)
validate_image_size(opt.imageSize)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU for sample generation")

nc = 3
nz = int(opt.nz)
ngf = int(opt.ngf)
print(f"Building DCGAN sampler for imageSize={opt.imageSize}")

netG = Generator(opt.imageSize, nz, ngf, nc).to(device)
state_dict = torch.load(opt.netG, map_location=device)
netG.load_state_dict(state_dict)
netG.eval()

with torch.no_grad():
    noise = torch.randn(opt.num_samples, nz, 1, 1, device=device)
    generated = netG(noise)

grid_cols = min(opt.num_samples, 5)
grid_path = os.path.join(opt.outf, "generated_grid.png")
vutils.save_image(generated, grid_path, nrow=grid_cols, normalize=True)
print(f"Saved generated grid to {grid_path}")

for index, image in enumerate(generated):
    sample_path = os.path.join(opt.outf, f"sample_{index:03d}.png")
    vutils.save_image(image.unsqueeze(0), sample_path, normalize=True)

print(f"Saved {opt.num_samples} individual generated images to {opt.outf}")
