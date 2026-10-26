"""ImageNet-compatible ResNet-18 encoder and a small mask velocity UNet.

Half-width legacy checkpoints remain readable; new experiments use the standard
encoder width so pretrained weights can be used without slicing parameters.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision.models import resnet18, ResNet18_Weights


class HalfResNet18(ResNet):
    def __init__(self):
        super().__init__(BasicBlock, [2, 2, 2, 2])
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(BasicBlock, 32, 2)
        self.layer2 = self._make_layer(BasicBlock, 64, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 256, 2, stride=2)
        del self.fc, self.avgpool

    def forward(self, image):
        x = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        a = self.layer1(x)
        b = self.layer2(a)
        c = self.layer3(b)
        return [a, b, c, self.layer4(c)]


class Block(nn.Module):
    def __init__(self, input_channels, output_channels, time_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, output_channels)
        self.time = nn.Linear(time_channels, output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, output_channels)
        self.skip = nn.Conv2d(input_channels, output_channels, 1)

    def forward(self, x, time):
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time(time)[:, :, None, None]
        return F.silu(self.norm2(self.conv2(h)) + self.skip(x))


class RoadModel(nn.Module):
    def __init__(self, discriminative=False, encoder_kind='resnet18', pretrained=False):
        super().__init__()
        self.discriminative = discriminative
        self.encoder_kind = encoder_kind
        if encoder_kind == 'half_resnet18':
            if pretrained:
                raise ValueError('Half-width legacy encoder has no ImageNet weights')
            self.encoder = HalfResNet18()
        elif encoder_kind == 'resnet18':
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            # Retain the usual named layers for checkpoint and freeze compatibility.
            del backbone.fc, backbone.avgpool
            self.encoder = backbone
        else:
            raise ValueError(f'Unknown encoder: {encoder_kind}')
        self.time_mlp = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 64))
        channels = [16, 16, 32, 64, 128, 128]
        conditions = [0, 0, 32, 64, 128, 256] if encoder_kind == 'half_resnet18' else [0, 0, 64, 128, 256, 512]
        self.down = nn.ModuleList()
        previous = 1
        for ch, cond in zip(channels, conditions):
            self.down.append(Block(previous + cond, ch))
            previous = ch
        self.up = nn.ModuleList([Block(channels[i + 1] + channels[i], channels[i])
                                 for i in reversed(range(5))])
        self.head = nn.Conv2d(channels[0], 1, 1)

    def encode(self, image):
        if self.encoder_kind == 'half_resnet18':
            return self.encoder(image)
        mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
        std = image.new_tensor([.229, .224, .225])[None, :, None, None]
        x = self.encoder.maxpool(self.encoder.relu(self.encoder.bn1(self.encoder.conv1((image - mean) / std))))
        a = self.encoder.layer1(x)
        b = self.encoder.layer2(a)
        c = self.encoder.layer3(b)
        return [a, b, c, self.encoder.layer4(c)]

    def velocity(self, state, time, features):
        frequency = torch.exp(torch.arange(32, device=state.device) * (-math.log(10000) / 31))
        angles = time.float()[:, None] * frequency[None] * 1000
        embedding = self.time_mlp(torch.cat([angles.sin(), angles.cos()], 1))
        skips = []
        x = state
        for i, block in enumerate(self.down):
            if i:
                x = F.avg_pool2d(x, 2)
            if i >= 2:
                condition = F.interpolate(features[i - 2], size=x.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, condition], 1)
            x = block(x, embedding)
            skips.append(x)
        for block, skip in zip(self.up, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat([x, skip], 1), embedding)
        return self.head(x)

    def forward(self, image, state=None, time=None):
        if state is None:
            state = torch.zeros_like(image[:, :1])
        if time is None:
            time = torch.zeros(image.shape[0], device=image.device)
        return self.velocity(state, time, self.encode(image))

    def freeze_encoder(self):
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        return self


def soft_dice_loss(prediction, target):
    dims = tuple(range(1, prediction.ndim))
    return (1 - (2 * (prediction * target).sum(dims) + 1e-6) /
            (prediction.sum(dims) + target.sum(dims) + 1e-6)).mean()


def supervised_loss(model, image, mask):
    if model.discriminative:
        logits = model(image)
        return F.binary_cross_entropy_with_logits(logits, mask) + soft_dice_loss(logits.sigmoid(), mask)
    y = 2 * mask - 1
    z = torch.randn_like(y)
    time = torch.rand(len(image), device=image.device)
    t = time[:, None, None, None]
    state = (1 - t) * y + t * z
    velocity = model(image, state, time)
    endpoint = ((state - t * velocity + 1) / 2).clamp(0, 1)
    return F.mse_loss(velocity, z - y) + soft_dice_loss(endpoint, mask)
