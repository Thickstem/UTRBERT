import torch
import torch.nn as nn
import torch.functional as F


class MRNA_CNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.model
        self.layer1 = nn.Sequential(
            nn.Conv1d(
                in_channels=4,
                out_channels=self.cfg.conv1_ch,
                kernel_size=self.cfg.conv1_kernel,
                stride=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(
                num_features=self.cfg.conv1_ch
            ),  # num_features = channel num (Batch,Channel,Length)
            nn.Dropout(p=0.02),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.cfg.conv1_ch,
                out_channels=self.cfg.conv2_ch,
                kernel_size=self.cfg.conv2_kernel,
                stride=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.cfg.conv2_ch),
            nn.Dropout(p=0.02),
            # nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.cfg.conv2_ch,
                out_channels=self.cfg.conv3_ch,
                kernel_size=self.cfg.conv3_kernel,
                stride=1,
                dilation=1,
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.cfg.conv3_ch),
            nn.Dropout(p=0.02),
        )

        self.linear_input_dim = int(
            self.cfg.conv3_ch
            * (
                (self.cfg.max_len - self.cfg.conv1_kernel) // 2
                - (self.cfg.conv2_kernel - 1)
                - (self.cfg.conv3_kernel - 1)
            )
        )

        self.layer4 = nn.Sequential(
            nn.Linear(
                in_features=self.linear_input_dim, out_features=self.cfg.linear1_dim
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.cfg.linear1_dim),
            nn.Dropout(p=0.02),
        )

        self.layer5 = nn.Sequential(
            nn.Linear(
                in_features=self.cfg.linear1_dim, out_features=self.cfg.linear2_dim
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.cfg.linear2_dim),
            nn.Dropout(p=0.02),
        )

        self.layer_out = nn.Sequential(
            nn.Linear(in_features=self.cfg.linear2_dim, out_features=1)
        )

        self.flatter = nn.Flatten()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatter(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer_out(x)

        return x
