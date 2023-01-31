import torch
import torch.nn as nn
import torch.functional as F


class MRNA_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=128,
                kernel_size=40,
                stride=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),  # num_features = channel num (N,C,L)
            nn.Dropout(p=0.02),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(
                in_channels=128, out_channels=32, kernel_size=30, stride=1, dilation=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=32),
            nn.Dropout(p=0.02),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=30,
                stride=1,
                dilation=4,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Dropout(p=0.02),
        )

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=58240, out_features=128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Dropout(p=0.02),
        )

        self.layer5 = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=32),
            nn.Dropout(p=0.02),
        )

        self.layer_out = nn.Sequential(nn.Linear(in_features=32, out_features=1))

        self.flatter = nn.Flatten()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatter(x)
        print(x.size())
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer_out(x)

        return x
