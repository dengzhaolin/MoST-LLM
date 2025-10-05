import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class SA(nn.Module):
    def __init__(self, channels):
        super(SA, self).__init__()
        self.channels = channels
        self.Wq = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1)),
            nn.ReLU())
        self.Wk = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1)),
            nn.ReLU())
        self.Wv = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1)),
            nn.ReLU())
        self.FC = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1)),
            nn.ReLU())

    def forward(self, rep):
        query = self.Wq(rep).permute(0, 3, 2,  1)
        key = self.Wk(rep).permute(0, 3, 2,  1)
        value = self.Wv(rep).permute(0, 3, 2, 1)
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= (self.channels ** 0.5)
        attention = F.softmax(attention, dim=-1)
        rep = torch.matmul(attention, value)
        rep = self.FC(rep.permute(0, 3 , 2,1))
        del query, key, value, attention
        return rep



## Residual Block
########################################
class MultiFactedEncoder(nn.Module):
    def __init__(self, channels, in_dim):
        super(MultiFactedEncoder, self).__init__()

        self.channels = channels

        self.num = 2
        # Spatial-Attention Layer

        self.proj1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=int(channels // 2), kernel_size=(1, 1)),
            nn.Tanh(),
            nn.Conv2d(in_channels=int(channels // 2), out_channels=channels, kernel_size=(1, 1)),
            nn.Tanh()
        )
        self.sa = SA(self.channels)

        self.filter_convs = nn.Conv2d(in_channels=self.num * self.channels,
                                      out_channels=self.channels,
                                      kernel_size=(1, 1))
        self.gate_convs = nn.Conv2d(in_channels=self.num * self.channels,
                                    out_channels=self.channels,
                                    kernel_size=(1, 1))
        self.residual_convs = nn.Sequential(nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1)),
                                            nn.Tanh(),)

    def forward(self, rep):

        rep = self.proj1(rep)
        rep_list = []
        # Spatial-Attention Layer
        rep_spa = self.sa(rep)
        rep_list.append(rep_spa)
        # Modality-Attention Layer
        #rep_sou = self.ma(rep)
        #rep_list.append(rep_sou)
        rep_list.append(rep)
        rep = torch.cat(rep_list, dim=1)
        # Temporal Convolution (TC)
        filter = self.filter_convs(rep)
        b, _, t,  n = filter.shape
        filter = torch.tanh(filter)
        gate = self.gate_convs(rep)
        gate = torch.sigmoid(gate)
        rep = filter * gate
        rep = self.residual_convs(rep)
        return rep


class MultiFactedDecoder(nn.Module):
    def __init__(self, channels, out_dim):
        super(MultiFactedDecoder, self).__init__()

        self.channels = channels

        self.num = 2

        self.sa = SA(self.channels)

        self.filter_convs = nn.Conv2d(in_channels=self.num * self.channels,
                                      out_channels=self.channels,
                                      kernel_size=(1, 1))

        self.gate_convs = nn.Conv2d(in_channels=self.num * self.channels,
                                    out_channels=self.channels,
                                    kernel_size=(1, 1))

        self.residual_convs = nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1))


        self.our_proj=nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=int(self.channels // 2), kernel_size=(1, 1)),
            nn.Tanh(),
            nn.Conv2d(in_channels=int(self.channels // 2), out_channels=out_dim, kernel_size=(1, 1)),
            nn.Tanh()
        )



    def forward(self, rep):
        rep_list = []

        rep_spa = self.sa(rep)

        rep_list.append(rep_spa)

        rep_list.append(rep)

        rep = torch.cat(rep_list, dim=1)
        # Temporal Convolution (TC)
        filter = self.filter_convs(rep)
        b, _, t, n = filter.shape
        filter = torch.tanh(filter)
        gate = self.gate_convs(rep)
        gate = torch.sigmoid(gate)
        rep = filter * gate
        rep = self.residual_convs(rep)

        rep = self.our_proj(rep)
        return rep


class MultiFactedVAE(nn.Module):
    def __init__(self,in_channels=4, hidden_dim=128):
        super(MultiFactedVAE, self).__init__()

        out_channels = in_channels
        self.encoder = MultiFactedEncoder(hidden_dim,in_channels)

        self.decoder = MultiFactedDecoder(hidden_dim, out_channels)

    def forward(self, x):
        x=x.squeeze(-1)

        multi_facted_embedding = self.encoder(x)

        reconstructed = self.decoder(multi_facted_embedding).unsqueeze(-1)

        return reconstructed , multi_facted_embedding






model =MultiFactedVAE(4,128)

x=torch.randn(64,4,16,98,1)

reconstructed , multi_facted_embedding=model(x)

print(reconstructed.shape, multi_facted_embedding.shape)