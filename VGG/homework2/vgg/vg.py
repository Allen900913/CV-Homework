
import torch.nn as nn
from torchsummary import summary
from torchvision import models
import matplotlib.pyplot as plt

class VGG19Model(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        net = models.vgg19_bn(weights=weights)
        net.classifier = self.classifier()
        self.vgg19_model = net

    def classifier(self):
        return nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        return self.vgg19_model(x)