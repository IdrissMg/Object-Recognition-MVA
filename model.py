import torch
import torch.nn as nn
import torchvision


class VGGCustom(nn.Module):
    def __init__(self, output_dim=2):
        super(VGGCustom, self).__init__()

        vgg_model = torchvision.models.vgg19(pretrained=True)
        self.output_dim = output_dim

        self.Conv1 = nn.Sequential(*list(vgg_model.features.children())[0:7])

        for param in self.Conv1.parameters():
            param.requires_grad = False

        self.Conv2 = nn.Sequential(*list(vgg_model.features.children())[7:37])

        self.Dense1 = nn.Sequential(nn.Linear(25088, 2048), nn.ReLU(), torch.nn.Dropout(p=0.5))
        self.Dense2 = nn.Sequential(nn.Linear(2048, 64), nn.ReLU(), torch.nn.Dropout(p=0.5))
        self.Classifier = nn.Linear(64, self.output_dim)

    def forward(self, x):
        output = self.Conv1(x)
        output = self.Conv2(output)

        output = output.view(-1, 25088)

        output = self.Dense1(output)
        output = self.Dense2(output)

        output = self.Classifier(output)

        return output
