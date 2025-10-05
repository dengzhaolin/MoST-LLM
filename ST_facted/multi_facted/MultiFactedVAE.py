import torch
from torch import nn
class MultiFacedEncoder(nn.Module):
    def __init__(self,in_channels, hidden_dim=1024):
        super(MultiFacedEncoder, self).__init__()

        self.in_project = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1),

        )

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim*2, hidden_dim *2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim *2, hidden_dim , kernel_size=1),
            nn.ReLU()
        )

        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim*2, hidden_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
            nn.ReLU()
        )
        self.fc_mu = nn.Sequential(
            nn.Conv1d(hidden_dim , hidden_dim , kernel_size=1),

        )

    def forward(self, x):

        x=x.reshape(x.shape[0],-1,x.shape[-1])



        encoded = self.in_project(x)

        spatial_encoded = self.spatial_encoder(encoded)
        temporal_encoded = self.temporal_encoder(spatial_encoded)



        mu = self.fc_mu(temporal_encoded)

        return mu

class MultiFacedDecoder(nn.Module):
    def __init__(self, modality_channel,seq_len=16, hidden_dim=512, out_channels=4):
        super(MultiFacedDecoder, self).__init__()

        self.modality_channel = modality_channel

        self.out_channels = out_channels

        self.dim=48

        self.decoder_fc = nn.Conv1d(hidden_dim, seq_len*self.dim, kernel_size=1)


        # Decoder layers
        self.decoder_pred = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.dim,self.dim//2, kernel_size=(5,1)),
            nn.ReLU(),
            nn.Conv2d(self.dim//2, self.dim //4,kernel_size=(5,1)),
            nn.ReLU(),
            nn.Conv2d(self.dim //4, out_channels , kernel_size=(6,1)),
        )

        self.decoder_recover = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim // 2, kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(self.dim// 2, self.dim // 4, kernel_size=(1,1)),
            nn.ReLU(),
            nn.Conv2d(self.dim // 4, out_channels, kernel_size=(1,1)),

        )
    def forward(self, z):

        z=self.decoder_fc(z).reshape(z.shape[0],self.dim,-1,z.shape[-1])

        decoded_pred = self.decoder_pred(z)

        decoded_recover = self.decoder_recover(z)

        return decoded_pred,decoded_recover


class MultiFacedVAE(nn.Module):
    def __init__(self,modality_channel,seq_len=16, in_channels=4, hidden_dim=512):
        super(MultiFacedVAE, self).__init__()
        self.encoder = MultiFacedEncoder(in_channels*seq_len, hidden_dim)
        self.decoder = MultiFacedDecoder(modality_channel,seq_len, hidden_dim, out_channels=modality_channel)

    def forward(self, x):

        x=x.squeeze(-1).permute(0,3,1,2)

        mu = self.encoder(x)   # 16,128,16,98


        embedding = mu   # torch.Size([64, 128, 16, 98])

        pred ,reconstructed = self.decoder(mu)
        return pred.permute(0,2,3,1).unsqueeze(-1) ,reconstructed.permute(0,2,3,1).unsqueeze(-1), embedding



# x=torch.rand(64,16,98,4,1)
# model=MultiFacedVAE(4,16,4,192)
# pred,reconstructed, embedding=model(x)
# print(pred.shape)
# print(reconstructed.shape)
# print(embedding.shape)



