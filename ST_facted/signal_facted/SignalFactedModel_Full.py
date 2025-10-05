import torch
import torch.nn as nn
import torch.nn.functional as F

class SignalFactedEncoder(nn.Module):
    def __init__(self,num_modals, hidden_dim, multifactedencoder):
        super(SignalFactedEncoder, self).__init__()

        self.share_encoder = multifactedencoder

        self.num_modals=num_modals

        for param in self.share_encoder.parameters():
            param.requires_grad = False

        self.encoder = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim*2, kernel_size=(1,1,1), stride=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(hidden_dim*2, hidden_dim *2, kernel_size=(1,1,1), stride=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(hidden_dim *2, hidden_dim , kernel_size=(1,1,1), stride=(1,1,1)),
            nn.ReLU(),
            nn.Conv3d(hidden_dim , hidden_dim, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.ReLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Conv3d(hidden_dim , hidden_dim , kernel_size=1),
        )

    def forward(self, x):

        rep = self.share_encoder(x)

        rep = rep.reshape(rep.shape[0], -1, self.num_modals, rep.shape[2], rep.shape[3])

        encoded = self.encoder(rep)

        mu = self.fc_mu(encoded)

        return mu


class SignalFactedDecoder(nn.Module):
    def __init__(self, channels, out_dim):
        super(SignalFactedDecoder, self).__init__()

        self.decoder=nn.Sequential(
            nn.Conv3d(in_channels=channels, out_channels=channels*2, kernel_size=(1, 1, 5)),
            nn.ReLU(),
            nn.Conv3d(in_channels=channels*2, out_channels=channels, kernel_size=(1, 1, 5)),
            nn.ReLU(),
            nn.Conv3d(in_channels=channels, out_channels=out_dim, kernel_size=(1, 1, 6)),
        )

    def forward(self, rep):

        rep = self.decoder(rep)

        return rep


class SignalFactedVAE(nn.Module):
    def __init__(self,multi_facted_encoder, hidden_dim=192):
        super(SignalFactedVAE, self).__init__()

        out_channels =1

        self.encoder = SignalFactedEncoder(4,hidden_dim,multi_facted_encoder)

        self.decoder = SignalFactedDecoder(hidden_dim, out_channels)

    def forward(self, x):

        x=x.squeeze(-1).permute(0,3,2,1)

        multi_facted_embedding = self.encoder(x)

        reconstructed = self.decoder(multi_facted_embedding).permute(0,4,3,2,1)


        return reconstructed , multi_facted_embedding.squeeze(-1)

#
from ST_facted.multi_facted.MultiFacted_model import MultiFacedVAE

vae =MultiFacedVAE(4,4,192)

model = SignalFactedVAE(vae.encoder,48)

x=torch.randn(64,16,98,4,1)

reconstructed , signal_facted_embedding=model(x)

print(reconstructed.shape, signal_facted_embedding.shape)

