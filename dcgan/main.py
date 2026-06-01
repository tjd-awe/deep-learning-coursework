from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

from dcgan_model import Generator, Discriminator, validate_image_size


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | mnist | imagenet | folder | lfw | fake | celeba')
parser.add_argument('--dataroot', required=False, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64, help='number of generator filters')
parser.add_argument('--ndf', type=int, default=64, help='number of discriminator filters')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--dry-run', action='store_true', help='check a single training cycle works')
parser.add_argument('--ngpu', type=int, default=1, help='single GPU mode only, keep this value as 1')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')
parser.add_argument('--eval-only', action='store_true', help='only run evaluation')
parser.add_argument('--eval-freq', type=int, default=5, help='evaluation frequency (every N epochs)')
parser.add_argument('--fid-samples', type=int, default=5000, help='number of samples for FID evaluation')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This DCGAN script now runs in single-GPU mode only. "
        "Please install a CUDA-enabled PyTorch build and run it on a machine with an NVIDIA GPU."
    )

if opt.ngpu != 1:
    print(f"Single-GPU mode enabled, ignoring --ngpu={opt.ngpu} and using cuda:0")

device = torch.device("cuda:0")
torch.cuda.set_device(device)
print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA device count detected: {torch.cuda.device_count()}")

validate_image_size(opt.imageSize)
print(f"Building DCGAN for imageSize={opt.imageSize}")

if opt.dataroot is None and str(opt.dataset).lower() != 'fake':
    raise ValueError("`dataroot` parameter is required for dataset \"%s\"" % opt.dataset)

nc = 3

# 平铺结构图片数据集类
class FlatImageFolder(torch.utils.data.Dataset):
    """处理平铺结构的图片数据集（如CelebA的img_align_celeba）"""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root)
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if len(self.image_files) == 0:
            raise RuntimeError(f"No image files found in {root}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    classes = [ c + '_train' for c in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                         transform=transforms.Compose([
                             transforms.Resize(opt.imageSize),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,)),
                         ]))
    nc = 1
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
elif opt.dataset == 'celeba':
    dataset = FlatImageFolder(
        root=opt.dataroot,
        transform=transforms.Compose([
            transforms.Resize(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
else:
    raise ValueError(f"Unknown dataset: {opt.dataset}")

assert dataset

def split_dataset(full_dataset, seed, train_ratio=0.9):
    total_size = len(full_dataset)
    if total_size == 0:
        raise RuntimeError("Dataset is empty, cannot create train/eval split")
    if total_size == 1:
        single_index = [0]
        return (
            torch.utils.data.Subset(full_dataset, single_index),
            torch.utils.data.Subset(full_dataset, single_index),
        )

    train_size = int(total_size * train_ratio)
    train_size = max(1, min(train_size, total_size - 1))
    eval_size = total_size - train_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_size, generator=generator).tolist()

    train_indices = indices[:train_size]
    eval_indices = indices[train_size:train_size + eval_size]

    return (
        torch.utils.data.Subset(full_dataset, train_indices),
        torch.utils.data.Subset(full_dataset, eval_indices),
    )


train_dataset, eval_dataset = split_dataset(dataset, opt.manualSeed)

print(f"Dataset size: {len(dataset)} images")
print(f"Train split size: {len(train_dataset)} images")
print(f"Eval split size: {len(eval_dataset)} images")

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    pin_memory=True)

eval_loader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers),
    pin_memory=True)

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


netG = Generator(opt.imageSize, nz, ngf, nc).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG, map_location=device))
print(netG)

netD = Discriminator(opt.imageSize, ndf, nc).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD, map_location=device))
print(netD)


criterion = nn.BCELoss()

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


class Evaluator:
    """FID和IS评估器，带可视化"""
    def __init__(self, device):
        self.device = device
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.fc = nn.Identity()
        self.inception.eval()
        self.inception.to(device)
        self.fid_history = []
        self.is_history = []
        self.epoch_history = []

    def get_activations(self, dataloader, num_samples=5000):
        """提取InceptionV3特征（输入需Resize到299x299）"""
        activations = []
        count = 0
        with torch.no_grad():
            for batch in dataloader:
                if count >= num_samples:
                    break
                img = batch[0].to(self.device)
                img = nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
                pred = self.inception(img)
                activations.append(pred.cpu().numpy())
                count += img.size(0)
        if len(activations) == 0:
            return np.zeros((num_samples, 2048))
        return np.concatenate(activations, axis=0)[:num_samples]

    def calculate_fid(self, real_acts, fake_acts):
        """计算FID分数"""
        mu1, sigma1 = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)
        mu2, sigma2 = fake_acts.mean(axis=0), np.cov(fake_acts, rowvar=False)
        if sigma1.ndim == 0:
            sigma1 = np.array([[sigma1]])
        if sigma2.ndim == 0:
            sigma2 = np.array([[sigma2]])
        if sigma1.shape[0] != sigma1.shape[1]:
            sigma1 = np.eye(sigma1.shape[0])
        if sigma2.shape[0] != sigma2.shape[1]:
            sigma2 = np.eye(sigma2.shape[0])
        ssdiff = np.sum((mu1 - mu2) ** 2)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    def calculate_is(self, activations):
        """计算IS分数"""
        p_yx = torch.nn.functional.softmax(torch.from_numpy(activations).float(), dim=1)
        p_y = p_yx.mean(dim=0)
        kl_div = p_yx * (torch.log(p_yx + 1e-10) - torch.log(p_y.unsqueeze(0) + 1e-10))
        is_score = torch.exp(kl_div.sum(dim=1).mean())
        return is_score.item()

    def evaluate(self, real_loader, fake_loader, epoch, num_samples=5000):
        """执行评估并记录历史"""
        real_acts = self.get_activations(real_loader, num_samples)
        fake_acts = self.get_activations(fake_loader, num_samples)

        fid = self.calculate_fid(real_acts, fake_acts)
        is_score = self.calculate_is(fake_acts)

        self.fid_history.append(fid)
        self.is_history.append(is_score)
        self.epoch_history.append(epoch)

        return fid, is_score

    def plot_metrics(self, save_path):
        """绘制评估指标可视化图表（FID/IS双柱状图）"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.bar(self.epoch_history, self.fid_history, color='steelblue', alpha=0.85)
        plt.xlabel('Epoch')
        plt.ylabel('FID')
        plt.title('FID Score Bar Chart (Lower is Better)')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.bar(self.epoch_history, self.is_history, color='seagreen', alpha=0.85)
        plt.xlabel('Epoch')
        plt.ylabel('IS')
        plt.title('Inception Score Bar Chart (Higher is Better)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        csv_path = save_path.replace('.png', '.csv')
        with open(csv_path, 'w') as f:
            f.write('epoch,fid,is\n')
            for e, fid, is_val in zip(self.epoch_history, self.fid_history, self.is_history):
                f.write(f'{e},{fid:.4f},{is_val:.4f}\n')


def generate_fake_samples(netG, num_samples, device, batch_size=64):
    """生成fake图像用于评估"""
    fake_images = []
    modelG = netG.module if hasattr(netG, 'module') else netG
    modelG.eval()
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            current_batch = min(batch_size, num_samples - i)
            noise = torch.randn(current_batch, nz, 1, 1, device=device)
            fake = modelG(noise)
            fake_images.append(fake.cpu())
    modelG.train()
    fake_dataset = torch.utils.data.TensorDataset(torch.cat(fake_images, dim=0))
    return torch.utils.data.DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)


# 仅评估模式
if opt.eval_only:
    if opt.netG == '':
        raise ValueError("--netG is required for evaluation mode")

    print("Running evaluation only...")

    evaluator = Evaluator(device)
    fake_loader = generate_fake_samples(netG, opt.fid_samples, device, opt.batchSize)

    fid, is_score = evaluator.evaluate(eval_loader, fake_loader, 0, opt.fid_samples)

    print(f"FID: {fid:.4f}")
    print(f"IS: {is_score:.4f}")
    evaluator.plot_metrics(os.path.join(opt.outf, 'eval_metrics.png'))
    exit(0)


evaluator = Evaluator(device)

if opt.dry_run:
    opt.niter = 1

for epoch in range(opt.niter):
    for i, data in enumerate(train_loader, 0):
        netD.zero_grad()
        real_cpu = data[0].to(device, non_blocking=True)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, dtype=real_cpu.dtype, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(train_loader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            modelG = netG.module if hasattr(netG, 'module') else netG
            fake_display = modelG(fixed_noise)
            vutils.save_image(fake_display.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

        if opt.dry_run:
            break

    # 评估
    if (epoch + 1) % opt.eval_freq == 0:
        print(f"Evaluating at epoch {epoch + 1}...")
        fake_loader = generate_fake_samples(netG, opt.fid_samples, device, opt.batchSize)
        fid, is_score = evaluator.evaluate(eval_loader, fake_loader, epoch + 1, opt.fid_samples)
        print(f"Epoch {epoch + 1} - FID: {fid:.4f}, IS: {is_score:.4f}")
        evaluator.plot_metrics(os.path.join(opt.outf, 'metrics_plot.png'))

    # 保存模型
    modelG = netG.module if hasattr(netG, 'module') else netG
    modelD = netD.module if hasattr(netD, 'module') else netD
    torch.save(modelG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(modelD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
