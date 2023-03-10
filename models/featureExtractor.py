import models.Models as Models
from torchstat import stat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchinfo import summary

# model = models.vgg16(pretrained=True)
# print(model)
drn = Models.drn_c_58(pretrained=True)
print(drn)


# print(stat(drn, input_size=(3, 224, 224)))
class CnnDnnLstm(nn.Module):
    def __init__(self, num_classes=2):
        super(CnnDnnLstm, self).__init__()
        drn_model = Models.drn_c_58(pretrained=True)
        self.drn = torch.nn.Sequential(*(list(drn_model.children())[:-1]))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(512, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=256, num_layers=3)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x_3d):
        hidden = None
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.drn(x_3d[:, t, :, :, :])
                x = self.flatten(x)
            x = self.fc(x)
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        x = self.fc1(out[-1, :, :])
        x = F.relu(x)
        x = self.fc2(x)
        return x


model_drn = CnnDnnLstm()
# print(summary(model_drn, input_size=(32, 30, 3, 224, 224)))
print(model_drn)
