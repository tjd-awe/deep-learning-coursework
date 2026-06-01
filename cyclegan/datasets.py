import glob
import os
import random

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        dir_A = os.path.join(root, f"{mode}A")
        dir_B = os.path.join(root, f"{mode}B")

        if not os.path.isdir(dir_A) or not os.path.isdir(dir_B):
            raise RuntimeError(
                f"Expected dataset directories '{dir_A}' and '{dir_B}'"
            )

        self.files_A = sorted(glob.glob(os.path.join(dir_A, "*.*")))
        self.files_B = sorted(glob.glob(os.path.join(dir_B, "*.*")))

        if len(self.files_A) == 0 or len(self.files_B) == 0:
            raise RuntimeError(
                f"No images found in '{dir_A}' or '{dir_B}'"
            )

        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)

    def __getitem__(self, index):
        image_A = Image.open(self.files_A[index % self.len_A])

        if self.unaligned:
            image_B = Image.open(self.files_B[random.randint(0, self.len_B - 1)])
        else:
            image_B = Image.open(self.files_B[index % self.len_B])

        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)
        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(self.len_A, self.len_B)
