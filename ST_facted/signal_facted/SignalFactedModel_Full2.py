import torch
import torch.nn as nn
import torch.nn.functional as F

class SignalFactedEncoder(nn.Module):
    def __init__(self,num_modals, hidden_dim, multifactedencoder):
        super(SignalFactedEncoder, self).__init__()

        self.share_encoder = multifactedencoder

        self.num_modals=num_modals

        for param in self.share_encoder.parameters():
            param.requires_grad = True

        self.encoder = nn.Sequential(
            nn.Conv2d(hidden_dim//num_modals, hidden_dim//4, kernel_size=(1,2), stride=(1,1)),
            nn.Tanh(),
            nn.Conv2d(hidden_dim//4, hidden_dim //2, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.Conv2d(hidden_dim //2, hidden_dim//2 , kernel_size=(1,2), stride=(1,4)),
            nn.Tanh(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=(1, 2), stride=(1, 8)),
            nn.Tanh()
        )

    def forward(self, x):

        rep = self.share_encoder(x)

        rep = rep.reshape(rep.shape[0]*self.num_modals, -1,  rep.shape[3], rep.shape[2])

        encoded = self.encoder(rep)


        return encoded


class SignalFactedDecoder(nn.Module):
    def __init__(self, channels, out_dim):
        super(SignalFactedDecoder, self).__init__()

        self.decoder=nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//2, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=channels//2, out_channels=channels//4, kernel_size=1),
            nn.Tanh(),
            nn.Conv2d(in_channels=channels//4, out_channels=out_dim, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, rep):

        rep = self.decoder(rep)

        return rep


class SignalFactedVAE(nn.Module):
    def __init__(self,multi_facted_encoder, hidden_dim=128):
        super(SignalFactedVAE, self).__init__()

        out_channels = 16

        self.modality=4

        self.encoder = SignalFactedEncoder(self.modality,hidden_dim,multi_facted_encoder)

        self.decoder = SignalFactedDecoder(hidden_dim, out_channels)

    def forward(self, x):

        b=x.shape[0]

        x=x.squeeze(-1).permute(0,3,1,2)

        multi_facted_embedding = self.encoder(x)



        reconstructed = self.decoder(multi_facted_embedding.reshape(b,-1,self.modality,multi_facted_embedding.shape[2])).permute(0,1,3,2).unsqueeze(-1)

        return reconstructed , multi_facted_embedding.squeeze(-1)


from ST_facted.multi_facted.MultiFacted_model_MA import MultiFacedVAE

vae =MultiFacedVAE(4,1,48)

model = SignalFactedVAE(vae.encoder,128)

x=torch.randn(64,16,98,4,1)

reconstructed , signal_facted_embedding=model(x)

print(reconstructed.shape, signal_facted_embedding.shape)

