# 深度学习课程作业

这个仓库放的是我在深度学习课程里完成的几部分作业和实验，内容包括基础的 MLP、小型生成模型实验，以及最后整理的课程报告。

## 我做了什么

`assignment1` 里是一个比较基础的多层感知机实现，对应课程早期作业。

`dcgan` 里我实现了基于 PyTorch 的 DCGAN 训练流程，支持数据集读取、训练、生成样本、插值和简单评估，代码里还加了对 CelebA 平铺目录的处理。

`cyclegan` 里我实现了 CycleGAN 的训练代码，包含双生成器、双判别器、循环一致性损失、identity loss、样本保存和模型 checkpoint 保存。

`report` 目录里保留了我最后写的课程报告、Latex 源文件以及实验结果图。

## 仓库内容

- `assignment1/`：MLP 作业
- `dcgan/`：DCGAN 训练与生成代码
- `cyclegan/`：CycleGAN 训练与可视化代码
- `report/`：课程报告和结果图

## 运行说明

DCGAN 的入口是 `dcgan/main.py`，CycleGAN 的入口是 `cyclegan/cyclegan.py`。两个实验都需要自己准备数据集路径，并按各自的 `requirements` 配置环境。
