import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import sys
import numpy as np
from models.layers import MDA, GSSL, MSSL
from torchsummary import summary


class SA(nn.Module):
    def __init__(self, channels):
        super(SA, self).__init__()
        self.channels = channels
        self.Wq = nn.Sequential(
            nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1, 1)),
            nn.ReLU())
        self.Wk = nn.Sequential(
            nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1, 1)),
            nn.ReLU())
        self.Wv = nn.Sequential(
            nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1, 1)),
            nn.ReLU())
        self.FC = nn.Sequential(
            nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1, 1)),
            nn.ReLU())

    def forward(self, rep):
        query = self.Wq(rep).permute(0, 2, 4, 3, 1)
        key = self.Wk(rep).permute(0, 2, 4, 3, 1)
        value = self.Wv(rep).permute(0, 2, 4, 3, 1)
        attention = torch.matmul(query, key.transpose(3, 4))
        attention /= (self.channels ** 0.5)
        attention = F.softmax(attention, dim=-1)
        rep = torch.matmul(attention, value)
        rep = self.FC(rep.permute(0, 4, 1, 3, 2))
        del query, key, value, attention
        return rep


########################################
## Modality-Attention (MA) Layer
########################################
class MA(nn.Module):
    def __init__(self, channels):
        super(MA, self).__init__()
        self.channels = channels
        self.Wq = nn.Sequential(
            nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1, 1)),
            nn.ReLU())
        self.Wk = nn.Sequential(
            nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1, 1)),
            nn.ReLU())
        self.Wv = nn.Sequential(
            nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1, 1)),
            nn.ReLU())
        self.FC = nn.Sequential(
            nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1, 1)),
            nn.ReLU())

    def forward(self, rep):
        query = self.Wq(rep).permute(0, 3, 4, 2, 1)
        key = self.Wk(rep).permute(0, 3, 4, 2, 1)
        value = self.Wv(rep).permute(0, 3, 4, 2, 1)
        attention = torch.matmul(query, key.transpose(3, 4))
        attention /= (self.channels ** 0.5)
        attention = F.softmax(attention, dim=-1)
        rep = torch.matmul(attention, value)
        rep = self.FC(rep.permute(0, 4, 3, 1, 2))
        del query, key, value, attention
        return rep

    ########################################


## Residual Block
########################################
class ResidualBlock(nn.Module):
    def __init__(self, num_modals, num_nodes, channels, dilation, kernel_size):
        super(ResidualBlock, self).__init__()
        self.num_modals = num_modals
        self.num_nodes = num_nodes
        self.channels = channels
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.num = 3
        # Spatial-Attention Layer
        self.sa = SA(self.channels)
        # Modality-Attention Layer
        self.ma = MA(self.channels)
        # Temporal Convolution
        self.filter_convs = nn.Conv3d(in_channels=self.num * self.channels,
                                      out_channels=self.num_modals * self.channels,
                                      kernel_size=(self.num_modals, 1, self.kernel_size),
                                      dilation=(1, 1, self.dilation))
        self.gate_convs = nn.Conv3d(in_channels=self.num * self.channels,
                                    out_channels=self.num_modals * self.channels,
                                    kernel_size=(self.num_modals, 1, self.kernel_size),
                                    dilation=(1, 1, self.dilation))
        self.residual_convs = nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1, 1))
        # Skip Connection
        self.skip_convs = nn.Conv3d(in_channels=self.channels, out_channels=self.channels, kernel_size=(1, 1, 1))

    def forward(self, rep):
        rep_list = []
        # Spatial-Attention Layer
        rep_spa = self.sa(rep)
        rep_list.append(rep_spa)
        # Modality-Attention Layer
        rep_sou = self.ma(rep)
        rep_list.append(rep_sou)
        rep_list.append(rep)
        rep = torch.cat(rep_list, dim=1)
        # Temporal Convolution (TC)
        filter = self.filter_convs(rep)
        b, _, _, n, t = filter.shape
        filter = torch.tanh(filter).reshape(b, -1, self.num_modals, n, t)
        gate = self.gate_convs(rep)
        gate = torch.sigmoid(gate).reshape(b, -1, self.num_modals, n, t)
        rep = filter * gate
        # Parametrized skip connection
        save_rep = rep
        sk = rep
        sk = self.skip_convs(sk)
        rep = self.residual_convs(rep)
        return rep, sk, gate


########################################
## MoST Encoder
########################################
class ST_Encoder(nn.Module):
    def __init__(self, layers, num_modals, num_nodes, channels, kernel_size):
        super(ST_Encoder, self).__init__()
        self.layers = layers
        # Residual Blocks
        self.residualblocks = nn.ModuleList()
        dilation = 1
        for i in range(self.layers):
            self.residualblocks.append(ResidualBlock(num_modals, num_nodes, channels, dilation, kernel_size))
            dilation *= 2

    def forward(self, rep):
        skip = 0
        for i in range(self.layers):
            residual = rep
            rep, sk, gate = self.residualblocks[i](rep)
            rep = rep + residual[:, :, :, :, -rep.size(4):]
            try:
                skip = sk + skip[:, :, :, :, -sk.size(4):]
            except:
                skip = sk
        return skip


class SignalFactedEncoder(nn.Module):
    def __init__(self, pre_train_vae_encoder, modality,channel):
        super(SignalFactedEncoder, self).__init__()

        self.modality=modality

        self.shareEncoder = pre_train_vae_encoder
        for param in self.shareEncoder.parameters():
            param.requires_grad = False


        self.ST_Encoder = ST_Encoder(layers=4, num_modals=modality, num_nodes=98, channels=channel, kernel_size=2)


    def forward(self, x):

        b,c,n,l=x.shape


        rep = self.shareEncoder(x)

        rep=rep.reshape(rep.shape[0],-1,self.modality,n,l)

        encoded = self.ST_Encoder(rep)

        return encoded

class SignalFactedDecoder(nn.Module):

    def __init__(self, channels,input_len,pred_len):
        super(SignalFactedDecoder, self).__init__()

        self.decoder_recover = nn.Sequential(
            nn.Conv3d(in_channels=channels, out_channels=channels //2, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=channels //2, out_channels=channels//4, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=channels//4, out_channels=input_len, kernel_size=(1, 1, 1)),
        )

        self.decoder_pred = nn.Sequential(
            nn.Conv3d(in_channels=channels, out_channels=channels // 2, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=channels // 2, out_channels=channels // 4, kernel_size=(1, 1, 1)),
            nn.ReLU(),
            nn.Conv3d(in_channels=channels // 4, out_channels=input_len, kernel_size=(1, 1, 1)),
        )

    def forward(self, rep):

        rep_pred = self.decoder_pred(rep)

        rep_recover = self.decoder_recover(rep)

        return rep_pred, rep_recover



class SignalFactedVAE(nn.Module):
    def __init__(self,multi_facted_encoder,modal_nuim, hidden_dim=48):
        super(SignalFactedVAE, self).__init__()

        pred_len = 3

        self.encoder = SignalFactedEncoder(multi_facted_encoder,modal_nuim,hidden_dim)

        self.decoder = SignalFactedDecoder(hidden_dim,16, pred_len)

    def forward(self, x):

        x=x.squeeze(-1).permute(0,3,2,1)

        multi_facted_embedding = self.encoder(x)

        pred,reconstructed = self.decoder(multi_facted_embedding)

        return pred.permute(0,1,3,2,4),reconstructed.permute(0,1,3,2,4) , multi_facted_embedding.squeeze(-1).permute(0,3,2,1)

"""

from ST_facted.multi_facted.MultiFacted_model import MultiFacedVAE

vae =MultiFacedVAE(4)

model = SignalFactedVAE(vae.encoder,48)

x=torch.randn(64,16,98,4,1)

pred,reconstructed , signal_facted_embedding=model(x)

print(pred.shape,reconstructed.shape, signal_facted_embedding.shape)

"""

