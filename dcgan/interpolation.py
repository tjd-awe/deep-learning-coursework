from __future__ import print_function
import argparse
import os
import random

import torch
import torchvision.utils as vutils

from dcgan_model import Generator, validate_image_size


parser = argparse.ArgumentParser()
parser.add_argument("--netG", required=True, help="path to Generator checkpoint")
parser.add_argument("--outf", default="./interpolation_results", help="folder to output interpolation images")
parser.add_argument("--interpolate-steps", type=int, default=10, help="number of interpolation steps")
parser.add_argument("--manualSeed", type=int, help="manual seed")
parser.add_argument("--nz", type=int, default=100, help="size of the latent z vector")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters")
parser.add_argument("--imageSize", type=int, default=64, help="generated image size, kept for interface consistency")
parser.add_argument("--save-individual", action="store_true", help="save each interpolated sample as an individual image")
opt = parser.parse_args()
print(opt)

if opt.interpolate_steps < 2:
    raise ValueError("--interpolate-steps must be at least 2")

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
    print("CUDA not available, using CPU for interpolation")

nc = 3
nz = int(opt.nz)
ngf = int(opt.ngf)


def interpolate_latents(latent_a, latent_b, steps):
    interpolated = []
    for step in range(steps):
        alpha = step / (steps - 1)
        latent = (1.0 - alpha) * latent_a + alpha * latent_b
        interpolated.append(latent)
    return torch.cat(interpolated, dim=0)


print(f"Building DCGAN interpolation model for imageSize={opt.imageSize}")

netG = Generator(opt.imageSize, nz, ngf, nc).to(device)
state_dict = torch.load(opt.netG, map_location=device)
netG.load_state_dict(state_dict)
netG.eval()

with torch.no_grad():
    z1 = torch.randn(1, nz, 1, 1, device=device)
    z2 = torch.randn(1, nz, 1, 1, device=device)
    interpolated_z = interpolate_latents(z1, z2, opt.interpolate_steps)
    generated = netG(interpolated_z)

grid_path = os.path.join(opt.outf, "interpolation_grid.png")
vutils.save_image(generated, grid_path, nrow=opt.interpolate_steps, normalize=True)
print(f"Saved interpolation grid to {grid_path}")

if opt.save_individual:
    for index, image in enumerate(generated):
        sample_path = os.path.join(opt.outf, f"interpolation_{index:03d}.png")
        vutils.save_image(image.unsqueeze(0), sample_path, normalize=True)
    print(f"Saved {opt.interpolate_steps} individual interpolation images to {opt.outf}")
