import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import numpy as np

from base import BaseModel

from model.modeling import VisionTransformer, CONFIGS


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class VIT(BaseModel):
    def __init__(self, model_type, img_size, num_classes=40):
        super().__init__()
        config = CONFIGS[model_type]
        self.model = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes)
        self.model.load_from(np.load("ViT-B_16.npz"))

    def forward(self, x):
        x = self.model(x)
        return x

