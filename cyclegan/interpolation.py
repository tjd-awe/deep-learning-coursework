import argparse
import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable

from models import GeneratorResNet
from datasets import ImageDataset

from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="latest", help="checkpoint tag to load, e.g. latest or 199")
parser.add_argument("--dataset_name", type=str, default="archive", help="name of the dataset")
parser.add_argument("--dataroot", type=str, required=True, help="path to local dataset directory")
parser.add_argument("--input_a", type=str, default=None, help="optional path to the first input image")
parser.add_argument("--input_b", type=str, default=None, help="optional path to the second input image")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--interpolate_steps", type=int, default=10, help="number of interpolation steps")
parser.add_argument("--seed", type=int, default=42, help="random seed")
opt = parser.parse_args()
print(opt)


cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

input_shape = (opt.channels, opt.img_height, opt.img_width)


G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)

if cuda:
    G_AB = G_AB.cuda()


def checkpoint_path(model_name):
    return "saved_models/%s/%s_%s.pth" % (opt.dataset_name, model_name, opt.checkpoint)


G_AB.load_state_dict(torch.load(checkpoint_path("G_AB")))

G_AB.eval()


transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

input_transforms = transforms.Compose(
    [
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


os.makedirs("interpolation_results/%s" % opt.dataset_name, exist_ok=True)
output_path = "interpolation_results/%s/interpolation_%s.png" % (opt.dataset_name, opt.checkpoint)


def interpolate_images(img1, img2, steps=10):
    interpolated = []
    for i in range(steps):
        alpha = i / (steps - 1)
        interpolated_img = (1 - alpha) * img1 + alpha * img2
        interpolated.append(interpolated_img)
    return interpolated


def load_single_image(path):
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = input_transforms(image).unsqueeze(0).type(Tensor)
    return Variable(tensor)


def test_interpolation():
    if (opt.input_a is None) != (opt.input_b is None):
        raise ValueError("--input_a and --input_b must be provided together")

    if opt.input_a is not None and opt.input_b is not None:
        img1 = load_single_image(opt.input_a)
        img2 = load_single_image(opt.input_b)
    else:
        dataloader = DataLoader(
            ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=False, mode="test"),
            batch_size=2,
            shuffle=True,
            num_workers=1,
        )

        batch = next(iter(dataloader))
        img1 = Variable(batch["A"][0:1].type(Tensor))
        img2 = Variable(batch["A"][1:2].type(Tensor))
    
    fake_img1 = G_AB(img1)
    fake_img2 = G_AB(img2)
    
    interpolated_fake = interpolate_images(fake_img1, fake_img2, opt.interpolate_steps)
    interpolated_real = interpolate_images(img1, img2, opt.interpolate_steps)
    
    all_images = torch.cat(interpolated_real + interpolated_fake, dim=0)
    grid = make_grid(all_images, nrow=opt.interpolate_steps, normalize=True)
    save_image(grid, output_path, normalize=False)
    
    print("Interpolation results saved to %s" % output_path)


if __name__ == "__main__":
    test_interpolation()
