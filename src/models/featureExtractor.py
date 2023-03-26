import src.models.Models as Models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
INPUT_SIZE = 9
HIDDEN_SIZE = 64
NUM_CLASSES = 1
LEARNING_RATE = .001
Decay = .9


class CnnDnnLstm(nn.Module):
    def __init__(self, num_classes=14, hidden_size=HIDDEN_SIZE):
        super(CnnDnnLstm, self).__init__()
        drn_model = Models.drn_c_58(pretrained=True)
        self.drn = torch.nn.Sequential(*(list(drn_model.children())[:-1]))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(512, 300))
        self.lstm = nn.LSTM(input_size=300, hidden_size=HIDDEN_SIZE, num_layers=2)
        self.fc1 = nn.Linear(HIDDEN_SIZE, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid()

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
        x = self.sigmoid(self.output(x))
        return x


# model_drn = CnnDnnLstm()
model = CnnDnnLstm(hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# print(summary(model_drn, input_size=(32, 30, 3, 224, 224)))
# print(model_drn)
