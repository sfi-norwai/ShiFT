from torch import nn
import torch
from src.models.inceptiontime import *
from src.models.resnet1D_raw import *

"""Two contrastive encoders"""
class TFC(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(TFC, self).__init__()

        self.feature_extractor_t = InceptionTime(input_dims,output_dims)

        self.projector_t = nn.Sequential(
            nn.Linear(output_dims, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dims)
        )

       # self.feature_extractor_f = InceptionTime(input_dims,output_dims)
        self.feature_extractor_f = ResNet1D(input_dims,output_dims)

        self.projector_f = nn.Sequential(
            nn.Linear(output_dims, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dims)
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x = self.feature_extractor_t(x_in_t)
    
        h_time = self.gap(x.transpose(1,2)).squeeze()

        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        f = self.feature_extractor_f(x_in_f)
        h_freq = self.gap(f.transpose(1,2)).squeeze()
        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq
