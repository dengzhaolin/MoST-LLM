import torch
import torch.nn as nn
import sys

sys.path.append('../multifaced_mask')

from ST_facted.multi_facted.multifacted_model_放弃 import MultiFactedVAE
class SingleMaskEncoder(nn.Module):
    def __init__(self, pre_trained_encoder, hidden_dim,  modality):
        super(SingleMaskEncoder, self).__init__()
        self.shared_encoder = pre_trained_encoder
        for param in self.shared_encoder.parameters():
            param.requires_grad = True

        self.modality=modality


        self.encoder = nn.Sequential(
            nn.Conv3d( hidden_dim//self.modality, hidden_dim//2, kernel_size=1),
            nn.Tanh(),
            nn.Conv3d(hidden_dim//2, hidden_dim , kernel_size=1),
            nn.Tanh(),
            nn.Conv3d(hidden_dim , hidden_dim , kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):


        shared_embedding = self.shared_encoder(x) #batch_size,channels, time,regions

        batch_size, c, time , regions   = shared_embedding.shape

        shared_embedding = shared_embedding.view(batch_size,-1,self.modality, time,regions)  #batch 48,4,16,98


        embedding = self.encoder(shared_embedding)  # batch,  final_embed_dim, modality  , time node

        return embedding

class SingleMaskDecoder(nn.Module):
    def __init__(self, hidden_dim=32, out_channels=1):
        super(SingleMaskDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv3d( hidden_dim, hidden_dim//4, kernel_size=1),
            nn.Tanh(),
            nn.Conv3d(hidden_dim//4, hidden_dim//8, kernel_size=1),
            nn.Tanh(),
            nn.Conv3d(hidden_dim//8 , out_channels, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, z):

        reconstructed = self.decoder(z)

        return reconstructed


class SingleMaskVAE(nn.Module):
    def __init__(self, pre_trained_vae,modality, hidden_dim,  out_channels):
        super(SingleMaskVAE, self).__init__()
        self.encoder = SingleMaskEncoder(pre_trained_vae.encoder,hidden_dim,  modality )

        self.decoder = SingleMaskDecoder(hidden_dim, out_channels)
    def forward(self, x):

        x=x.squeeze(-1)


        embedding = self.encoder(x)



        reconstructed_flat = self.decoder(embedding).permute(0,3,4,2,1)


        return reconstructed_flat  , embedding



VAE=MultiFactedVAE(4,128)

Svae=SingleMaskVAE(VAE,4,128,1)

x=torch.randn(64,4,16,98,1)

reconstructed_flat ,embedding=Svae(x)

print(reconstructed_flat.shape)
print(embedding.shape)



