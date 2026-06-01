import argparse
import glob
import os
from pathlib import Path

from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from models import GeneratorResNet


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="latest", help="checkpoint tag to load, e.g. latest or 199")
parser.add_argument("--dataset_name", type=str, default="archive", help="name of the dataset")
parser.add_argument("--dataroot", type=str, required=True, help="path to local dataset directory")
parser.add_argument("--outdir", type=str, default=None, help="optional output directory")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
opt = parser.parse_args()
print(opt)


cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

input_shape = (opt.channels, opt.img_height, opt.img_width)

G_AB = GeneratorResNet(input_shape, opt.n_residual_blocks)
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()


def checkpoint_path(model_name):
    return "saved_models/%s/%s_%s.pth" % (opt.dataset_name, model_name, opt.checkpoint)


G_AB.load_state_dict(torch.load(checkpoint_path("G_AB")))
G_BA.load_state_dict(torch.load(checkpoint_path("G_BA")))
G_AB.eval()
G_BA.eval()


transform = transforms.Compose(
    [
        transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def collect_image_paths(split_name):
    split_dir = os.path.join(opt.dataroot, split_name)
    if not os.path.isdir(split_dir):
        raise RuntimeError("Expected dataset directory '%s'" % split_dir)

    image_paths = sorted(glob.glob(os.path.join(split_dir, "*.*")))
    if not image_paths:
        raise RuntimeError("No images found in '%s'" % split_dir)
    return image_paths


def load_tensor(path):
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image).unsqueeze(0).type(Tensor)


def resolve_output_dirs():
    root = opt.outdir or os.path.join("visualization_results", opt.dataset_name, opt.checkpoint)
    rows_dir = os.path.join(root, "rows")
    os.makedirs(rows_dir, exist_ok=True)
    return root, rows_dir


def main():
    _, rows_dir = resolve_output_dirs()
    test_a_paths = collect_image_paths("testA")
    test_b_paths = collect_image_paths("testB")

    with torch.no_grad():
        pair_count = min(len(test_a_paths), len(test_b_paths))
        for index in range(pair_count):
            real_A = load_tensor(test_a_paths[index])
            fake_B = G_AB(real_A)
            recov_A = G_BA(fake_B)
            id_A = G_BA(real_A)

            row_image = torch.cat(
                (
                    real_A[0].cpu(),
                    fake_B[0].cpu(),
                    recov_A[0].cpu(),
                    id_A[0].cpu(),
                ),
                dim=2,
            )

            a_stem = Path(test_a_paths[index]).stem
            b_stem = Path(test_b_paths[index]).stem
            output_name = f"{index:04d}_{a_stem}__{b_stem}.png"
            output_path = os.path.join(rows_dir, output_name)
            save_image(row_image, output_path, normalize=True)

    print("Visualization rows saved to %s" % rows_dir)


if __name__ == "__main__":
    main()
