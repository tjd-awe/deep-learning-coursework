import argparse
import os
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from models import GeneratorResNet
from datasets import ImageDataset

from torch.utils.data import DataLoader
from scipy import linalg
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="latest", help="checkpoint tag to load, e.g. latest or 199")
parser.add_argument("--dataset_name", type=str, default="archive", help="name of the dataset")
parser.add_argument("--dataroot", type=str, required=True, help="path to local dataset directory")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--seed", type=int, default=42, help="random seed")
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


transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]


os.makedirs("evaluation_results/%s" % opt.dataset_name, exist_ok=True)
output_path = "evaluation_results/%s/evaluation_%s.txt" % (opt.dataset_name, opt.checkpoint)


def load_inception_model():
    from torchvision.models import inception_v3
    model = inception_v3(pretrained=True, transform_input=False)
    model = model.eval()
    if cuda:
        model = model.cuda()
    return model


def calculate_fid(real_features, fake_features):
    mu1 = np.mean(real_features, axis=0)
    mu2 = np.mean(fake_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def evaluate_quality():
    dataloader = DataLoader(
        ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=False, mode="test"),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )

    inception_model = load_inception_model()

    real_a_features = []
    real_b_features = []
    fake_a_features = []
    fake_b_features = []

    cycle_l1_a_total = 0.0
    cycle_l1_b_total = 0.0
    count = 0

    for batch in dataloader:
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)
        recov_A = G_BA(fake_B)
        recov_B = G_AB(fake_A)

        resized_real_A = F.interpolate(real_A, size=(299, 299), mode="bilinear", align_corners=False)
        resized_real_B = F.interpolate(real_B, size=(299, 299), mode="bilinear", align_corners=False)
        resized_fake_A = F.interpolate(fake_A, size=(299, 299), mode="bilinear", align_corners=False)
        resized_fake_B = F.interpolate(fake_B, size=(299, 299), mode="bilinear", align_corners=False)

        with torch.no_grad():
            real_a_feat = inception_model(resized_real_A)
            real_b_feat = inception_model(resized_real_B)
            fake_a_feat = inception_model(resized_fake_A)
            fake_b_feat = inception_model(resized_fake_B)

        real_a_features.append(real_a_feat.cpu().numpy().flatten())
        real_b_features.append(real_b_feat.cpu().numpy().flatten())
        fake_a_features.append(fake_a_feat.cpu().numpy().flatten())
        fake_b_features.append(fake_b_feat.cpu().numpy().flatten())

        cycle_l1_a_total += F.l1_loss(recov_A, real_A, reduction="mean").item()
        cycle_l1_b_total += F.l1_loss(recov_B, real_B, reduction="mean").item()
        count += 1

    real_a_features = np.array(real_a_features)
    real_b_features = np.array(real_b_features)
    fake_a_features = np.array(fake_a_features)
    fake_b_features = np.array(fake_b_features)

    fid_a2b = calculate_fid(real_b_features, fake_b_features)
    fid_b2a = calculate_fid(real_a_features, fake_a_features)
    cycle_l1_a = cycle_l1_a_total / max(count, 1)
    cycle_l1_b = cycle_l1_b_total / max(count, 1)

    print("FID_A2B: %.4f" % fid_a2b)
    print("FID_B2A: %.4f" % fid_b2a)
    print("cycle_L1_A: %.6f" % cycle_l1_a)
    print("cycle_L1_B: %.6f" % cycle_l1_b)

    with open(output_path, "w") as f:
        f.write("FID_A2B: %.4f\n" % fid_a2b)
        f.write("FID_B2A: %.4f\n" % fid_b2a)
        f.write("cycle_L1_A: %.6f\n" % cycle_l1_a)
        f.write("cycle_L1_B: %.6f\n" % cycle_l1_b)

    print("Evaluation results saved to %s" % output_path)


if __name__ == "__main__":
    evaluate_quality()
